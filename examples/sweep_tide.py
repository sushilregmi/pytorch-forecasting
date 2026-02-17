import os
import re
import time
import glob
import numpy as np
import pandas as pd
import torch

from contextlib import redirect_stdout, redirect_stderr

from pytorch_forecasting import TimeSeriesDataSet, TiDEModel
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE, MAPE

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, Callback

from torchmetrics import MeanSquaredError
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


# ============================
# CONFIG
# ============================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

RUNS = 4
HORIZONS = [25, 30, 35, 40]

LOG_FILE = "tide_multirun_lon.log"               # append-only
CKPT_BASE_DIR = "checkpoints_tide_multirun_lon"

BATCH_SIZE = 64
EPOCHS = 50
MAX_ENCODER_LENGTH = 50

trainer_kwargs = dict(
    max_epochs=EPOCHS,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    gradient_clip_val=0.1,
    enable_progress_bar=False,
    logger=False,
)

# ============================
# DATA
# ============================
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

# ============================
# FEATURES
# ============================
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

# TiDE: you had no covariate_scalers here; keep it that way unless you want it
# (If you do want scalers, you can pass scalers=covariate_scalers like in TFT/TimeXer)

# ============================
# HELPERS
# ============================
def compute_metrics(y_true, y_pred):
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = float(np.sqrt(mse))
    mae   = mean_absolute_error(y_true, y_pred)
    mape  = mean_absolute_percentage_error(y_true, y_pred)
    smape = 100.0 * float(np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    ))
    return mse, rmse, mae, mape, smape

def parse_epoch_from_ckpt_name(ckpt_path: str) -> int:
    base = os.path.basename(ckpt_path)
    m = re.search(r"epoch=([0-9]+)", base)
    if m:
        return int(m.group(1))
    m = re.search(r"best-([0-9]+)-", base)
    if m:
        return int(m.group(1))
    m = re.search(r"-([0-9]{1,3})-", base)
    return int(m.group(1)) if m else -1

def log_header(f, run_id: int, horizon: int):
    f.write("\n" + "=" * 120 + "\n")
    f.write(f"MODEL=TiDE | RUN={run_id} | HORIZON={horizon} | ts={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 120 + "\n")
    f.flush()

# ============================
# CALLBACK: epoch_time(train+val)
# ============================
class EpochTimeCollector(Callback):
    def __init__(self, run_id: int, horizon: int):
        super().__init__()
        self.run_id = run_id
        self.horizon = horizon
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
        print(f"[TiDE run={self.run_id} H={self.horizon}] Epoch={trainer.current_epoch:02d} time(train+val)={dt:.3f}s")
        self._t0 = None


# ============================
# MAIN
# ============================
summary_rows = []

for run_id in range(1, RUNS + 1):
    for horizon in HORIZONS:

        ckpt_dir = os.path.join(CKPT_BASE_DIR, f"run_{run_id}", f"horizon_{horizon}")
        os.makedirs(ckpt_dir, exist_ok=True)

        with open(LOG_FILE, "a", buffering=1) as log_f, redirect_stdout(log_f), redirect_stderr(log_f):
            log_header(log_f, run_id, horizon)
            print(f"Checkpoint dir: {ckpt_dir}")

            # ---- datasets (train/val) ----
            training = TimeSeriesDataSet(
                train_df,
                time_idx="time_idx",
                group_ids=group_ids,
                target=target,
                max_encoder_length=MAX_ENCODER_LENGTH,
                max_prediction_length=horizon,
                static_categoricals=static_categoricals,
                static_reals=static_reals,
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=GroupNormalizer(groups=group_ids),
                add_relative_time_idx=False,     # keep as you had for TiDE training
                add_target_scales=True,
                add_encoder_length=True,
            )
            validation = TimeSeriesDataSet.from_dataset(training, val_df, predict=False, stop_randomization=False)

            train_loader = training.to_dataloader(train=True,  batch_size=BATCH_SIZE, num_workers=0)
            val_loader   = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

            # ---- callbacks ----
            early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
            checkpoint = ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best-{epoch:02d}-{val_loss:.4f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                verbose=True,
            )
            time_cb = EpochTimeCollector(run_id, horizon)

            # ---- model ----
            model = TiDEModel.from_dataset(
                training,
                input_chunk_length=MAX_ENCODER_LENGTH,
                output_chunk_length=horizon,
                num_encoder_layers=2,
                num_decoder_layers=2,
                decoder_output_dim=16,
                hidden_size=128,
                temporal_width_future=4,
                temporal_hidden_size_future=32,
                temporal_decoder_hidden=32,
                use_layer_norm=False,
                dropout=0.1,
                loss=MeanSquaredError(),
                logging_metrics=[SMAPE(), MAE(), RMSE(), MAPE()],
                learning_rate=1e-3,
            )

            trainer = Trainer(callbacks=[early_stop, checkpoint, time_cb], **trainer_kwargs)

            # ---- fit timing ----
            fit_t0 = time.time()
            trainer.fit(model, train_loader, val_loader)
            total_fit_time = time.time() - fit_t0

            best_ckpt = checkpoint.best_model_path
            if not best_ckpt:
                ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
                if not ckpts:
                    raise RuntimeError(f"No checkpoint found in {ckpt_dir}")
                best_ckpt = ckpts[0]

            best_epoch = parse_epoch_from_ckpt_name(best_ckpt)

            epoch_times = time_cb.epoch_times
            avg_epoch_time = float(np.mean(epoch_times)) if epoch_times else float("nan")
            e2e_until_best = float(np.sum(epoch_times[:best_epoch + 1])) if (best_epoch >= 0 and best_epoch < len(epoch_times)) else float(total_fit_time)

            print("\n--- TRAINING TIME SUMMARY ---")
            print(f"epochs_ran={len(epoch_times)}")
            print(f"avg_epoch_time(train+val)={avg_epoch_time:.4f}s")
            print(f"best_epoch={best_epoch}")
            print(f"e2e_time_until_best={e2e_until_best:.4f}s")
            print(f"total_fit_time_until_stop={total_fit_time:.4f}s")
            print(f"best_ckpt={best_ckpt}")

            # ---- EVAL ----
            # Use a dataset definition consistent with your eval snippet
            training_for_test = TimeSeriesDataSet(
                train_df,
                time_idx="time_idx",
                group_ids=group_ids,
                target=target,
                max_encoder_length=MAX_ENCODER_LENGTH,
                max_prediction_length=horizon,
                static_categoricals=static_categoricals,
                static_reals=static_reals,
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=GroupNormalizer(groups=group_ids),
                add_relative_time_idx=True,      # your eval snippet used True
                add_target_scales=True,
                add_encoder_length=True,
            )
            test = TimeSeriesDataSet.from_dataset(training_for_test, test_df, predict=False, stop_randomization=False)
            test_loader = test.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)

            best_model = TiDEModel.load_from_checkpoint(best_ckpt)

            # inference timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            raw = best_model.predict(test_loader)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.time() - t0

            preds = raw.detach().cpu().numpy() if isinstance(raw, torch.Tensor) else raw
            y_pred = np.array(preds).flatten()

            y_true = np.concatenate(
                [y.detach().cpu().numpy() for _, (y, _) in iter(test_loader)],
                axis=0
            ).flatten()

            mse, rmse, mae, mape, smape = compute_metrics(y_true, y_pred)

            print("\n--- INFERENCE + METRICS ---")
            print(f"inference_time_total={inference_time:.4f}s | per_batch={inference_time/len(test_loader):.6f}s")
            print(f"MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.6%}, SMAPE={smape:.6f}%")

            summary_rows.append(dict(
                model="TiDE",
                run=run_id,
                horizon=horizon,
                avg_epoch_time_s=avg_epoch_time,
                e2e_until_best_s=e2e_until_best,
                total_fit_time_s=total_fit_time,
                best_epoch=best_epoch,
                best_ckpt=best_ckpt,
                inference_time_s=inference_time,
                mse=mse, rmse=rmse, mae=mae, mape=mape, smape=smape,
            ))

            print(f"\nEND | TiDE run={run_id} horizon={horizon}")

# ---- summaries ----
summary_df = pd.DataFrame(summary_rows).sort_values(["horizon", "run"])
agg_df = (
    summary_df.groupby(["horizon"], as_index=False)
    .agg(
        avg_epoch_time_s_mean=("avg_epoch_time_s", "mean"),
        avg_epoch_time_s_std=("avg_epoch_time_s", "std"),
        e2e_until_best_s_mean=("e2e_until_best_s", "mean"),
        e2e_until_best_s_std=("e2e_until_best_s", "std"),
        inference_time_s_mean=("inference_time_s", "mean"),
        inference_time_s_std=("inference_time_s", "std"),
        mse_mean=("mse", "mean"), mse_std=("mse", "std"),
        smape_mean=("smape", "mean"), smape_std=("smape", "std"),
    )
    .sort_values(["horizon"])
)

summary_df.to_csv("tide_multirun_per_run.csv", index=False)
agg_df.to_csv("tide_multirun_by_horizon.csv", index=False)

with open(LOG_FILE, "a", buffering=1) as f:
    f.write("\n" + "#" * 120 + "\n")
    f.write("FINAL SUMMARY (TiDE) - per run\n")
    f.write("#" * 120 + "\n")
    f.write(summary_df.to_string(index=False))
    f.write("\n\n" + "#" * 120 + "\n")
    f.write("FINAL SUMMARY (TiDE) - mean/std by horizon\n")
    f.write("#" * 120 + "\n")
    f.write(agg_df.to_string(index=False))
    f.write("\n")
