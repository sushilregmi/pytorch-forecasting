import os
import sys
import time
import glob
import re
import numpy as np
import pandas as pd
import torch
from contextlib import redirect_stdout, redirect_stderr

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss, SMAPE, MAE, RMSE, MAPE

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# -----------------------
# CONFIG
# -----------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

LOG_FILE = "experiment_logs_original.txt"     # single file, append only
BASE_CKPT_DIR = "ckpts_tft_sweep_original"

REPEATS = 4
ENCODER_LENGTHS = [350, 450]

HORIZONS = [5]

BATCH_SIZE = 64
EPOCHS = 50

df = pd.read_csv("data/timexer.csv", parse_dates=["date"])
df = df.sort_values("date")
df["time_idx"] = (df["date"] - df["date"].min()).dt.total_seconds().astype(int)
df["series_id"] = 0

num_train = int(len(df) * 0.7)
num_test  = int(len(df) * 0.2)
num_vali  = len(df) - num_train - num_test

train_df = df.iloc[:num_train]
val_df   = df.iloc[num_train : num_train + num_vali]
test_df  = df.iloc[num_train + num_vali :]


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

covariate_scalers = {var: GroupNormalizer(groups=["series_id"]) for var in time_varying_known_reals}

model_kwargs = dict(
    learning_rate=1e-3,
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
    accelerator="gpu",
    devices=1,
    gradient_clip_val=0.1,
    enable_progress_bar=False,
    logger=False,  # avoids lightning log dirs
)

# -----------------------
# METRICS
# -----------------------
def compute_metrics(actuals: np.ndarray, preds: np.ndarray):
    mse   = mean_squared_error(actuals, preds)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(actuals, preds)
    mape  = mean_absolute_percentage_error(actuals, preds)
    smape = 100 * np.mean(2 * np.abs(preds - actuals) / (np.abs(actuals) + np.abs(preds) + 1e-8))
    return mse, rmse, mae, mape, smape


def find_best_ckpt(ckpt_dir: str) -> str:
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return ckpts[0]


def parse_epoch_from_ckpt_path(path: str) -> int:
    """
    Expects filename like: best-epoch=07-val_loss=0.123456.ckpt
    or your: best-07-0.123456.ckpt
    We'll support both.
    """
    base = os.path.basename(path)
    # pattern 1: "...epoch=07..."
    m = re.search(r"epoch=([0-9]+)", base)
    if m:
        return int(m.group(1))
    # pattern 2: "best-07-..."
    m = re.search(r"best-([0-9]+)-", base)
    if m:
        return int(m.group(1))
    # fallback: try any -DD- in name
    m = re.search(r"-([0-9]{1,3})-", base)
    if m:
        return int(m.group(1))
    return -1


# -----------------------
# CALLBACKS THAT TRACK TIMES
# -----------------------
class EpochTimeCollector(Callback):
    """
    Collects per-epoch wall time for train+val.
    Stores in shared dict: shared["epoch_times"] = [t0, t1, ...]
    """
    def __init__(self, shared: dict, enc_len: int, run_id: int, horizon: int):
        super().__init__()
        self.shared = shared
        self.enc_len = enc_len
        self.run_id = run_id
        self.horizon = horizon
        self.t0 = None

    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.time()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.t0 is None:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        dt = time.time() - self.t0
        epoch = trainer.current_epoch
        self.shared.setdefault("epoch_times", []).append(dt)

        print(
            f"[enc={self.enc_len} run={self.run_id} horizon={self.horizon}] "
            f"Epoch={epoch:02d} epoch_time(train+val)={dt:.3f}s"
        )
        self.t0 = None


class PrintLossCallback(Callback):
    def __init__(self, enc_len: int, run_id: int, horizon: int):
        super().__init__()
        self.enc_len = enc_len
        self.run_id = run_id
        self.horizon = horizon

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss")
        val_loss   = metrics.get("val_loss")

        train_loss_val = float(train_loss.detach().cpu()) if train_loss is not None else float("nan")
        val_loss_val   = float(val_loss.detach().cpu()) if val_loss is not None else float("nan")

        print(
            f"[enc={self.enc_len} run={self.run_id} horizon={self.horizon}] "
            f"Epoch={epoch:02d} train_loss={train_loss_val:.6f} val_loss={val_loss_val:.6f}"
        )

# -----------------------
# LOG HEADER
# -----------------------
def append_log_header(f, enc_len, run_id, horizon):
    f.write("\n" + "=" * 100 + "\n")
    f.write(
        f"START | encoder_length={enc_len} | run={run_id} | horizon={horizon} | "
        f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    f.write("=" * 100 + "\n")
    f.flush()


# -----------------------
# SWEEP + SUMMARY
# -----------------------
all_run_rows = [] 

for enc_len in ENCODER_LENGTHS:
    for run_id in range(1, REPEATS + 1):
        for horizon in HORIZONS:

            with open(LOG_FILE, "a", buffering=1) as log_f, \
                 redirect_stdout(log_f), redirect_stderr(log_f):

                append_log_header(log_f, enc_len, run_id, horizon)
                print(f"Training for enc_len={enc_len}, run={run_id}, horizon={horizon}")

                ckpt_dir = os.path.join(
                    BASE_CKPT_DIR,
                    f"enc_{enc_len}",
                    f"horizon_{horizon}",
                    f"run_{run_id}",
                )
                os.makedirs(ckpt_dir, exist_ok=True)

                # Shared dict to collect epoch times
                shared = {}

                # ---- datasets ----
                training = TimeSeriesDataSet(
                    train_df,
                    time_idx="time_idx",
                    group_ids=group_ids,
                    target=target,
                    max_encoder_length=enc_len,
                    max_prediction_length=horizon,
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
                validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=False)
                test = TimeSeriesDataSet.from_dataset(training, test_df, predict=False, stop_randomization=False)

                train_loader = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
                val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
                test_loader  = test.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

                # ---- callbacks ----
                early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

                checkpoint = ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename="best-{epoch:02d}-{val_loss:.6f}",  # your style
                    save_top_k=1,
                    monitor="val_loss",
                    mode="min",
                    verbose=True,
                )

                epoch_time_cb = EpochTimeCollector(shared=shared, enc_len=enc_len, run_id=run_id, horizon=horizon)
                print_cb = PrintLossCallback(enc_len=enc_len, run_id=run_id, horizon=horizon)

                # ---- train ----
                tft = TemporalFusionTransformer.from_dataset(training, **model_kwargs)
                trainer = Trainer(callbacks=[early_stop, checkpoint, print_cb, epoch_time_cb], **trainer_kwargs)

                fit_t0 = time.time()
                trainer.fit(tft, train_loader, val_loader)
                fit_total_time = time.time() - fit_t0  # total wall time until training stops (not necessarily best)

                # ---- best checkpoint + best epoch ----
                best_ckpt = checkpoint.best_model_path
                if not best_ckpt:
                    # fallback
                    best_ckpt = find_best_ckpt(ckpt_dir)

                best_epoch = parse_epoch_from_ckpt_path(best_ckpt)
                print(f"Best checkpoint: {best_ckpt}")
                print(f"Best epoch (parsed): {best_epoch}")

                epoch_times = shared.get("epoch_times", [])
                avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else float("nan")

                # end-to-end until best checkpoint epoch (sum epoch times up to best epoch)
                # epoch_times list is in order epoch0, epoch1, ... as training progressed
                if epoch_times and best_epoch >= 0 and best_epoch < len(epoch_times):
                    e2e_until_best = float(np.sum(epoch_times[: best_epoch + 1]))
                else:
                    # fallback: if parsing fails, use full fit time as approximation
                    e2e_until_best = float(fit_total_time)

                print(f"Avg epoch time (train+val): {avg_epoch_time:.4f}s")
                print(f"End-to-end time until best checkpoint: {e2e_until_best:.4f}s")
                print(f"Total fit time (until stop): {fit_total_time:.4f}s")

                # ---- evaluate + inference time ----
                best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
                _ = trainer.test(best_model, dataloaders=test_loader, verbose=False)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                preds = best_model.predict(test_loader).detach().cpu().numpy()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_time = time.time() - t0

                median = preds.flatten()
                actuals = np.concatenate(
                    [y.detach().cpu().numpy() for _, (y, _) in iter(test_loader)],
                    axis=0
                ).flatten()

                mse, rmse, mae, mape, smape = compute_metrics(actuals, median)

                print(f"Inference time total: {inference_time:.4f}s  | per batch: {inference_time/len(test_loader):.6f}s")
                print(f"MSE={mse:.6f}  SMAPE={smape:.6f}%")

                # ---- store summary row for final print (also goes into log because stdout is redirected) ----
                row = dict(
                    encoder_length=enc_len,
                    horizon=horizon,
                    run=run_id,
                    avg_epoch_time_s=avg_epoch_time,
                    e2e_until_best_s=e2e_until_best,
                    inference_time_s=inference_time,
                    mse=mse,
                    smape=smape,
                    best_epoch=best_epoch,
                    best_ckpt=best_ckpt,
                )
                all_run_rows.append(row)

                print(f"END | encoder_length={enc_len} | run={run_id} | horizon={horizon}")

# -----------------------
# FINAL SUMMARY PRINT (prints to console normally if you run outside redirect,
# but in your script it will print to terminal. If you want it ALSO in the log,
# just wrap this block with open(LOG_FILE,"a") + redirect_stdout again.
# -----------------------
summary_df = pd.DataFrame(all_run_rows).sort_values(["encoder_length", "horizon", "run"])

# pretty aggregate across repeats
agg_df = (
    summary_df
    .groupby(["encoder_length", "horizon"], as_index=False)
    .agg(
        avg_epoch_time_s_mean=("avg_epoch_time_s", "mean"),
        e2e_until_best_s_mean=("e2e_until_best_s", "mean"),
        inference_time_s_mean=("inference_time_s", "mean"),
        mse_mean=("mse", "mean"),
        smape_mean=("smape", "mean"),
        mse_std=("mse", "std"),
        smape_std=("smape", "std"),
    )
    .sort_values(["encoder_length", "horizon"])
)

# Print BOTH per-run and aggregated at the very end into the same log file (append)
with open(LOG_FILE, "a", buffering=1) as log_f:
    log_f.write("\n" + "#" * 100 + "\n")
    log_f.write("FINAL SUMMARY (per run)\n")
    log_f.write("#" * 100 + "\n")
    log_f.write(summary_df.to_string(index=False))
    log_f.write("\n\n" + "#" * 100 + "\n")
    log_f.write("FINAL SUMMARY (mean/std across repeats)\n")
    log_f.write("#" * 100 + "\n")
    log_f.write(agg_df.to_string(index=False))
    log_f.write("\n")