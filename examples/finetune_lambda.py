import os
import re
import time
import glob
import numpy as np
import pandas as pd
import torch
from contextlib import redirect_stdout, redirect_stderr

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, MAE, RMSE, MAPE

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

# -----------------------------
# SETTINGS
# -----------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

LOG_FILE = "finetune_polaris_h20_runs.log"   # append-only
REPEATS = 5
HORIZON = 20

# Where pretrained GCE checkpoints live:
PRETRAINED_DIR = f"checkpoints_no_static/horizon_{HORIZON}"

# Where to save finetuned checkpoints:
FINETUNE_BASE_DIR = "ckpts_finetuned_polaris_h20_repeats"

# Data / model config 
BATCH_SIZE = 64
EPOCHS = 50
MAX_ENCODER_LENGTH = 50  
static_categoricals = []
static_reals = []
time_varying_known_reals = [
    "bytes_op0","bytes_op1","bytes_sum","io_count",
    "read_ops_count","write_ops_count",
    "bytes_sum_ema_short","bytes_sum_ema_long",
    "bytes_sum_macd","bytes_sum_macd_signal",
]
time_varying_unknown_reals = ["duration_sum"]
group_ids = ["series_id"]
target = "duration_sum"

model_kwargs = dict(
    learning_rate=1e-4,
    hidden_size=128,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=128,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    output_size=3,
    logging_metrics=[SMAPE(), MAE(), RMSE(), MAPE()],
    log_interval=10,
    reduce_on_plateau_patience=4,
)

trainer_kwargs = dict(
    max_epochs=EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    enable_progress_bar=False,
    logger=False,
)

# -----------------------------
# HELPERS
# -----------------------------
def find_single_ckpt(folder: str) -> str:
    ckpts = [f for f in os.listdir(folder) if f.endswith(".ckpt")]
    if len(ckpts) != 1:
        raise RuntimeError(f"Expected exactly one .ckpt in {folder}, found {len(ckpts)}")
    return os.path.join(folder, ckpts[0])

def parse_epoch_from_ckpt_name(ckpt_path: str) -> int:
    """
    Supports filename patterns like:
      finetuned-07-0.1234.ckpt
      finetuned-epoch=07-val_loss=0.1234.ckpt
    """
    base = os.path.basename(ckpt_path)
    m = re.search(r"epoch=([0-9]+)", base)
    if m:
        return int(m.group(1))
    m = re.search(r"finetuned-([0-9]+)-", base)
    if m:
        return int(m.group(1))
    m = re.search(r"-([0-9]{1,3})-", base)
    return int(m.group(1)) if m else -1

def compute_metrics(actuals: np.ndarray, preds: np.ndarray):
    mse   = mean_squared_error(actuals, preds)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(actuals, preds)
    mape  = mean_absolute_percentage_error(actuals, preds)
    smape = 100 * np.mean(2 * np.abs(actuals - preds) / (np.abs(actuals) + np.abs(preds) + 1e-8))
    return mse, rmse, mae, mape, smape

def log_header(f, run_id: int):
    f.write("\n" + "=" * 110 + "\n")
    f.write(f"START FINETUNE | horizon={HORIZON} | run={run_id} | ts={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 110 + "\n")
    f.flush()


# -----------------------------
# CALLBACK TO MEASURE EPOCH TIMES (train + val)
# -----------------------------
class EpochTimeCollector(Callback):
    """
    Measures wall time from start of train epoch to end of validation epoch,
    so dt ~= train+val time for that epoch (when val_dataloader is provided).
    """
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self._t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t0 = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self._t0 is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = time.time() - self._t0
        self.epoch_times.append(dt)
        print(f"Epoch={trainer.current_epoch:02d} epoch_time(train+val)={dt:.3f}s")
        self._t0 = None


#-----------------------------
# LOAD LAMBDA DATA 
# -----------------------------
df = pd.read_csv("data/timexer_polaris.csv", parse_dates=["date"])
df = df.sort_values("date")
df["time_idx"] = (df["date"] - df["date"].min()).dt.total_seconds().astype(int)
df["series_id"] = 0

num_train = int(len(df) * 0.7)
num_test  = int(len(df) * 0.2)
num_vali  = len(df) - num_train - num_test

train_df = df.iloc[:num_train]
val_df   = df.iloc[num_train : num_train + num_vali]
test_df  = df.iloc[num_train + num_vali :]

covariate_scalers = {var: GroupNormalizer(groups=["series_id"]) for var in time_varying_known_reals}

# -----------------------------
# GET PRETRAINED CHECKPOINT (GCE)
# -----------------------------
pretrained_ckpt = find_single_ckpt(PRETRAINED_DIR)

# -----------------------------
# RUN 5 FINETUNE REPEATS (H=20)
# -----------------------------
all_rows = []

for run_id in range(1, REPEATS + 1):
    os.makedirs(FINETUNE_BASE_DIR, exist_ok=True)
    run_ckpt_dir = os.path.join(FINETUNE_BASE_DIR, f"run_{run_id}")
    os.makedirs(run_ckpt_dir, exist_ok=True)

    # append-only logging for this run
    with open(LOG_FILE, "a", buffering=1) as log_f, redirect_stdout(log_f), redirect_stderr(log_f):
        log_header(log_f, run_id)
        print(f"Pretrained checkpoint: {pretrained_ckpt}")
        print(f"Fine-tune ckpt dir: {run_ckpt_dir}")

        # -------------------------
        # Build datasets
        # -------------------------
        training = TimeSeriesDataSet(
            train_df,
            time_idx="time_idx",
            target=target,
            group_ids=group_ids,
            max_encoder_length=MAX_ENCODER_LENGTH,
            max_prediction_length=HORIZON,
            static_categoricals=static_categoricals,
            static_reals=static_reals,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=group_ids),
            scalers=covariate_scalers,
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        validation = TimeSeriesDataSet.from_dataset(training, val_df, stop_randomization=True)
        test = TimeSeriesDataSet.from_dataset(training, test_df, stop_randomization=True)

        train_loader = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
        val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
        test_loader  = test.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

        # -------------------------
        # Load model from pretrained GCE checkpoint
        # -------------------------
        model = TemporalFusionTransformer.load_from_checkpoint(pretrained_ckpt)

        # Optional: ensure hyperparams are set as desired (often not needed if in ckpt)
        # If you want to override LR etc:
        # model.hparams.update(model_kwargs)

        # -------------------------
        # Callbacks
        # -------------------------
        early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        checkpoint = ModelCheckpoint(
            dirpath=run_ckpt_dir,
            filename="finetuned-{epoch:02d}-{val_loss:.6f}",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            verbose=True,
        )
        time_cb = EpochTimeCollector()

        # -------------------------
        # Train + timing
        # -------------------------
        trainer = Trainer(callbacks=[early_stop, checkpoint, time_cb], **trainer_kwargs)

        fit_t0 = time.time()
        trainer.fit(model, train_loader, val_loader)
        fit_total_time = time.time() - fit_t0

        # Best checkpoint + best epoch
        best_ckpt = checkpoint.best_model_path
        if not best_ckpt:
            # fallback: get any ckpt in run dir
            ckpts = sorted(glob.glob(os.path.join(run_ckpt_dir, "*.ckpt")))
            if not ckpts:
                raise RuntimeError(f"No checkpoint written to {run_ckpt_dir}")
            best_ckpt = ckpts[0]

        best_epoch = parse_epoch_from_ckpt_name(best_ckpt)
        epoch_times = time_cb.epoch_times

        avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else float("nan")

        if epoch_times and best_epoch >= 0 and best_epoch < len(epoch_times):
            e2e_until_best = float(np.sum(epoch_times[: best_epoch + 1]))
        else:
            # fallback if parsing fails
            e2e_until_best = float(fit_total_time)

        print(f"\n--- TRAINING TIME SUMMARY (run={run_id}) ---")
        print(f"epochs_ran={len(epoch_times)} | avg_epoch_time(train+val)={avg_epoch_time:.4f}s")
        print(f"best_epoch={best_epoch} | e2e_time_until_best={e2e_until_best:.4f}s")
        print(f"total_fit_time_until_stop={fit_total_time:.4f}s")
        print(f"best_ckpt={best_ckpt}")

        # -------------------------
        # Inference timing + metrics
        # -------------------------
        best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        preds = best_model.predict(test_loader, return_x=False).detach().cpu().numpy().flatten()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_time = time.time() - t0

        actuals = np.concatenate([y.detach().cpu().numpy() for _, (y, _) in iter(test_loader)], axis=0).flatten()

        mse, rmse, mae, mape, smape = compute_metrics(actuals, preds)

        print(f"\n--- INFERENCE + METRICS (run={run_id}) ---")
        print(f"inference_time_total={inference_time:.4f}s | per_batch={inference_time/len(test_loader):.6f}s")
        print(f"MSE={mse:.6f} | SMAPE={smape:.6f}% | RMSE={rmse:.6f} | MAE={mae:.6f} | MAPE={mape:.6%}")

        # store summary row (goes into table later)
        row = dict(
            run=run_id,
            horizon=HORIZON,
            avg_epoch_time_s=avg_epoch_time,
            e2e_until_best_s=e2e_until_best,
            total_fit_time_s=fit_total_time,
            inference_time_s=inference_time,
            mse=mse,
            smape=smape,
            rmse=rmse,
            mae=mae,
            mape=mape,
            best_epoch=best_epoch,
            best_ckpt=best_ckpt,
        )
        all_rows.append(row)

        print(f"\nEND RUN {run_id}")


# -----------------------------
# WRITE FINAL SUMMARY TABLES (append to same log)
# -----------------------------
summary_df = pd.DataFrame(all_rows).sort_values(["run"])

agg_df = summary_df.agg({
    "avg_epoch_time_s": ["mean", "std"],
    "e2e_until_best_s": ["mean", "std"],
    "total_fit_time_s": ["mean", "std"],
    "inference_time_s": ["mean", "std"],
    "mse": ["mean", "std"],
    "mae": ["mean", "std"],
    "smape": ["mean", "std"],
}).T

with open(LOG_FILE, "a", buffering=1) as f:
    f.write("\n" + "#" * 110 + "\n")
    f.write("FINAL SUMMARY (per run)\n")
    f.write("#" * 110 + "\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n" + "#" * 110 + "\n")
    f.write("FINAL SUMMARY (mean/std over 5 runs)\n")
    f.write("#" * 110 + "\n")
    f.write(agg_df.to_string())
    f.write("\n")