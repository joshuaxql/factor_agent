from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from .config import require_optional
from .qlib_data import split_xy

tqdm.monitor_interval = 0
_TORCH_RUNTIME_CONFIGURED = False


ALPHA158_TS_20 = [
    "RESI5",
    "WVMA5",
    "RSQR5",
    "KLEN",
    "RSQR10",
    "CORR5",
    "CORD5",
    "CORR10",
    "ROC60",
    "RESI10",
    "VSTD5",
    "RSQR60",
    "CORR60",
    "WVMA60",
    "STD5",
    "RSQR20",
    "CORD60",
    "CORD10",
    "CORR20",
    "KLOW",
]


def finite_or_nan(x: pd.DataFrame) -> pd.DataFrame:
    return x.replace([np.inf, -np.inf], np.nan)


def select_sequence_features(x: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.sequence_feature_preset == "all":
        return x
    cols = [col for col in ALPHA158_TS_20 if col in x.columns]
    if len(cols) != len(ALPHA158_TS_20):
        missing = sorted(set(ALPHA158_TS_20) - set(cols))
        raise RuntimeError(f"Missing Alpha158 time-series feature columns: {missing}")
    return x.loc[:, cols]


class TorchScaler:
    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x: pd.DataFrame) -> "TorchScaler":
        arr = x.to_numpy(dtype=np.float32, copy=False)
        finite = np.isfinite(arr)
        count = finite.sum(axis=0, dtype=np.float32)
        safe_count = np.maximum(count, 1.0)
        clean = np.where(finite, arr, 0.0)
        self.mean = (clean.sum(axis=0) / safe_count).astype(np.float32)
        diff = np.where(finite, arr - self.mean, 0.0)
        self.std = np.sqrt((diff * diff).sum(axis=0) / safe_count).astype(np.float32)
        self.mean[count == 0] = 0.0
        self.std[(count == 0) | ~np.isfinite(self.std) | (self.std < 1e-6)] = 1.0
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fitted.")
        arr = x.to_numpy(dtype=np.float32, copy=True)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        scaled = (arr - self.mean) / self.std
        return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def torch_device(args: argparse.Namespace):
    global _TORCH_RUNTIME_CONFIGURED
    torch = require_optional("torch")
    if not _TORCH_RUNTIME_CONFIGURED:
        if os.name == "nt" and torch.cuda.is_available():
            torch.backends.cudnn.enabled = False
            logger.info("[torch] disabled cuDNN on Windows for stable CUDA RNN shutdown")
        _TORCH_RUNTIME_CONFIGURED = True
    if torch.cuda.is_available():
        return torch.device("cuda")
    if args.allow_cpu:
        return torch.device("cpu")
    raise RuntimeError("CUDA is not available. Add --allow-cpu only for debugging; production runs are expected on GPU.")


def progress_bar(*args, **kwargs):
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("mininterval", 1.0)
    kwargs.setdefault("leave", True)
    return tqdm(*args, **kwargs)


def train_linear(train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame, valid_y: pd.Series, test_x: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    torch = require_optional("torch")
    logger.info("[linear] fitting scaler...")
    scaler = TorchScaler().fit(train_x)
    device = torch_device(args)
    model = torch.nn.Linear(train_x.shape[1], 1).to(device)
    return train_torch_tabular("linear", model, scaler, train_x, train_y, valid_x, valid_y, test_x, args, device)


def train_mlp(train_x: pd.DataFrame, train_y: pd.Series, valid_x: pd.DataFrame, valid_y: pd.Series, test_x: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    torch = require_optional("torch")
    logger.info("[mlp] fitting scaler...")
    scaler = TorchScaler().fit(train_x)
    device = torch_device(args)
    model = torch.nn.Sequential(
        torch.nn.Linear(train_x.shape[1], args.hidden_size),
        torch.nn.BatchNorm1d(args.hidden_size),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(args.hidden_size, args.hidden_size // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(args.hidden_size // 2, 1),
    ).to(device)
    return train_torch_tabular("mlp", model, scaler, train_x, train_y, valid_x, valid_y, test_x, args, device)


def train_torch_tabular(name: str, model, scaler: TorchScaler, train_x, train_y, valid_x, valid_y, test_x, args, device) -> pd.Series:
    torch = require_optional("torch")

    logger.info(f"[{name}] preparing tensors on {device}...")
    train_x_t = torch.from_numpy(scaler.transform(train_x)).to(device)
    train_y_t = torch.from_numpy(train_y.to_numpy(dtype=np.float32)).view(-1, 1).to(device)
    valid_x_t = torch.from_numpy(scaler.transform(valid_x)).to(device)
    valid_y_t = torch.from_numpy(valid_y.to_numpy(dtype=np.float32)).view(-1, 1).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    best_state = None
    best_loss = float("inf")
    train_num = train_y_t.shape[0]
    steps_per_epoch = max(1, int(np.ceil(train_num / args.batch_size)))
    max_steps = max(1, args.epochs * steps_per_epoch)
    eval_steps = steps_per_epoch
    stop_steps = 0
    train_loss_sum = 0.0

    model.train()
    with progress_bar(range(1, max_steps + 1), desc=f"[{name}] train", unit="step") as pbar:
        for step in pbar:
            choice = torch.randint(train_num, (args.batch_size,), device=device)
            bx = train_x_t.index_select(0, choice)
            by = train_y_t.index_select(0, choice)
            optim.zero_grad(set_to_none=True)
            loss = loss_fn(model(bx), by)
            loss.backward()
            optim.step()
            loss_value = float(loss.detach().cpu())
            train_loss_sum += loss_value
            pbar.set_postfix(
                train_loss=f"{loss_value:.6f}",
                best_val=f"{best_loss:.6f}" if np.isfinite(best_loss) else "nan",
                epoch=f"{int(np.ceil(step / steps_per_epoch))}/{args.epochs}",
            )

            if step % eval_steps == 0 or step == max_steps:
                model.eval()
                with torch.no_grad():
                    val_loss = float(loss_fn(model(valid_x_t), valid_y_t).detach().cpu())
                epoch = int(np.ceil(step / steps_per_epoch))
                avg_train_loss = train_loss_sum / min(eval_steps, step)
                logger.info(
                    f"[{name}] step {step}/{max_steps} epoch {epoch}/{args.epochs} "
                    f"train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f} best={best_loss:.6f}"
                )
                train_loss_sum = 0.0
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stop_steps = 0
                else:
                    stop_steps += 1
                    if stop_steps >= 4:
                        logger.info(f"[{name}] early stopping at step {step}")
                        break
                model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds: list[np.ndarray] = []
    test_arr = scaler.transform(test_x)
    with torch.no_grad():
        steps = range(0, len(test_arr), args.batch_size)
        for start in progress_bar(steps, total=(len(test_arr) + args.batch_size - 1) // args.batch_size, desc=f"[{name}] predict", unit="batch"):
            bx = torch.from_numpy(test_arr[start : start + args.batch_size]).to(device)
            preds.append(model(bx).detach().cpu().numpy().reshape(-1))
    return pd.Series(np.concatenate(preds), index=test_x.index, name="score")


def train_lightgbm(train_x, train_y, valid_x, valid_y, test_x, args) -> pd.Series:
    lightgbm = require_optional("lightgbm")
    logger.info("[lightgbm] preprocessing data...")
    train_x, valid_x, test_x = finite_or_nan(train_x), finite_or_nan(valid_x), finite_or_nan(test_x)
    dtrain = lightgbm.Dataset(train_x.values, label=np.squeeze(train_y.values), free_raw_data=False)
    dvalid = lightgbm.Dataset(valid_x.values, label=np.squeeze(valid_y.values), reference=dtrain, free_raw_data=False)
    params = dict(
        objective="regression",
        metric="l2",
        learning_rate=0.03,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        seed=args.seed,
        num_threads=-1,
        device_type="cpu",
        verbosity=-1,
    )
    logger.info("[lightgbm] using device_type=cpu")
    pbar = progress_bar(total=args.tree_n_estimators, desc="[lightgbm] train", unit="iter")

    def progress_callback(env):
        target = env.iteration + 1
        if target > pbar.n:
            pbar.update(target - pbar.n)
        if env.evaluation_result_list:
            _, metric_name, metric_value, *_ = env.evaluation_result_list[-1]
            pbar.set_postfix({metric_name: f"{metric_value:.6f}"})

    progress_callback.order = 20
    progress_callback.before_iteration = False
    try:
        model = lightgbm.train(
            params,
            dtrain,
            num_boost_round=args.tree_n_estimators,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[progress_callback, lightgbm.early_stopping(args.early_stopping_rounds, verbose=False)],
        )
    finally:
        pbar.close()
    logger.info("[lightgbm] predicting...")
    return pd.Series(model.predict(test_x.values), index=test_x.index, name="score")


def xgboost_device(args: argparse.Namespace) -> str:
    requested = getattr(args, "xgboost_device", "auto")
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_xgboost(train_x, train_y, valid_x, valid_y, test_x, args) -> pd.Series:
    xgboost = require_optional("xgboost")
    logger.info("[xgboost] preprocessing data...")
    train_x, valid_x, test_x = finite_or_nan(train_x), finite_or_nan(valid_x), finite_or_nan(test_x)
    dtrain = xgboost.DMatrix(train_x.values, label=np.squeeze(train_y.values))
    dvalid = xgboost.DMatrix(valid_x.values, label=np.squeeze(valid_y.values))

    class XGBoostProgress(xgboost.callback.TrainingCallback):
        def __init__(self, total: int):
            self.pbar = progress_bar(total=total, desc="[xgboost] train", unit="iter")

        def after_iteration(self, model, epoch, evals_log):
            target = epoch + 1
            if target > self.pbar.n:
                self.pbar.update(target - self.pbar.n)
            if evals_log:
                dataset = next(reversed(evals_log))
                metric = next(reversed(evals_log[dataset]))
                values = evals_log[dataset][metric]
                if values:
                    self.pbar.set_postfix({metric: f"{values[-1]:.6f}"})
            return False

        def after_training(self, model):
            self.pbar.close()
            return model

    device = xgboost_device(args)
    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        max_depth=7,
        eta=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device=device,
        seed=args.seed,
        nthread=-1,
    )
    logger.info(f"[xgboost] using device={device}")
    model = xgboost.train(
        params,
        dtrain=dtrain,
        num_boost_round=args.tree_n_estimators,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=False,
        callbacks=[XGBoostProgress(args.tree_n_estimators)],
    )
    if device == "cuda":
        # Pandas/numpy prediction data lives on CPU.
        # Switching the booster avoids XGBoost's DMatrix device mismatch warning.
        model.set_param({"device": "cpu"})
    logger.info("[xgboost] predicting...")
    return pd.Series(model.predict(xgboost.DMatrix(test_x.values)), index=test_x.index, name="score")


class PanelSequenceDataset:
    def __init__(self, x: pd.DataFrame, y: pd.Series, scaler: TorchScaler, step_len: int):
        x = x.sort_index(level=["instrument", "datetime"])
        y = y.reindex(x.index)
        self.x = scaler.transform(x)
        self.y = y.to_numpy(dtype=np.float32)
        self.index = x.index
        self.step_len = step_len
        self.positions = self._valid_positions()

    def _valid_positions(self) -> np.ndarray:
        instruments = self.index.get_level_values("instrument")
        valid: list[int] = []
        start = 0
        values = instruments.to_numpy()
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[start]:
                begin = start + self.step_len - 1
                if begin < i:
                    valid.extend(range(begin, i))
                start = i
        return np.asarray(valid, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int):
        pos = self.positions[idx]
        seq = self.x[pos - self.step_len + 1 : pos + 1]
        return seq, np.float32(self.y[pos])

    def predict_index(self) -> pd.MultiIndex:
        return self.index[self.positions]


class SequenceModelFactory:
    @staticmethod
    def make(name: str, input_size: int, args: argparse.Namespace):
        torch = require_optional("torch")
        if name in {"gru", "lstm"}:
            rnn_cls = torch.nn.GRU if name == "gru" else torch.nn.LSTM
            return RNNRegressor(rnn_cls, input_size, args.hidden_size, args.num_layers)
        if name == "tra":
            return TRARegressor(input_size, args.hidden_size, args.num_layers)
        if name == "transformer":
            return TransformerRegressor(input_size, args.hidden_size, args.num_layers)
        raise ValueError(name)


class RNNRegressor:
    def __new__(cls, rnn_cls, input_size: int, hidden_size: int, num_layers: int):
        torch = require_optional("torch")

        class _RNN(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = rnn_cls(
                    input_size,
                    hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0.0,
                )
                self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden_size), torch.nn.Linear(hidden_size, 1))

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.head(out[:, -1])

        return _RNN()


class TRARegressor:
    def __new__(cls, input_size: int, hidden_size: int, num_layers: int):
        torch = require_optional("torch")

        class _TRA(torch.nn.Module):
            """A compact temporal routing adapter on top of GRU states."""

            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.GRU(
                    input_size,
                    hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=0.1 if num_layers > 1 else 0.0,
                )
                self.routes = torch.nn.ModuleList(
                    [
                        torch.nn.Sequential(torch.nn.LayerNorm(hidden_size), torch.nn.Linear(hidden_size, hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, 1))
                        for _ in range(3)
                    ]
                )
                self.router = torch.nn.Sequential(torch.nn.LayerNorm(hidden_size), torch.nn.Linear(hidden_size, len(self.routes)))

            def forward(self, x):
                states, _ = self.encoder(x)
                h = states[:, -1]
                route_scores = torch.cat([route(h) for route in self.routes], dim=1)
                weights = torch.softmax(self.router(h), dim=1)
                return (route_scores * weights).sum(dim=1, keepdim=True)

        return _TRA()


class TransformerRegressor:
    def __new__(cls, input_size: int, hidden_size: int, num_layers: int):
        torch = require_optional("torch")

        class _Transformer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(input_size, hidden_size)
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=4,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.1,
                    batch_first=True,
                    activation="gelu",
                )
                self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
                self.head = torch.nn.Sequential(torch.nn.LayerNorm(hidden_size), torch.nn.Linear(hidden_size, 1))

            def forward(self, x):
                h = self.proj(x)
                h = self.encoder(h)
                return self.head(h[:, -1])

        return _Transformer()


def train_sequence(name: str, train_x, train_y, valid_x, valid_y, test_x, test_y, args) -> pd.Series:
    torch = require_optional("torch")
    from torch.utils.data import DataLoader

    logger.info(f"[{name}] selecting sequence features...")
    train_x = select_sequence_features(train_x, args)
    valid_x = select_sequence_features(valid_x, args)
    test_x = select_sequence_features(test_x, args)
    logger.info(f"[{name}] fitting scaler and building sequence datasets...")
    scaler = TorchScaler().fit(train_x)
    train_ds = PanelSequenceDataset(train_x, train_y, scaler, args.step_len)
    valid_ds = PanelSequenceDataset(valid_x, valid_y, scaler, args.step_len)
    test_ds = PanelSequenceDataset(test_x, test_y, scaler, args.step_len)
    if len(train_ds) == 0 or len(valid_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError(f"{name} has empty sequence dataset. Reduce --step-len.")

    device = torch_device(args)
    model = SequenceModelFactory.make(name, train_x.shape[1], args).to(device)
    drop_train = len(train_ds) >= args.batch_size
    drop_valid = len(valid_ds) >= args.batch_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda", drop_last=drop_train)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda", drop_last=drop_valid)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    best_state = None
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        batches = 0
        with progress_bar(train_loader, desc=f"[{name}] epoch {epoch + 1}/{args.epochs}", unit="batch") as pbar:
            for bx, by in pbar:
                bx = bx.to(device, non_blocking=True)
                by = by.view(-1, 1).to(device, non_blocking=True)
                optim.zero_grad(set_to_none=True)
                loss = loss_fn(model(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)
                optim.step()
                train_loss += float(loss.detach().cpu())
                batches += 1
                pbar.set_postfix(train_loss=f"{train_loss / batches:.6f}", best_val=f"{best_loss:.6f}" if np.isfinite(best_loss) else "nan")
        model.eval()
        losses = []
        with torch.no_grad():
            for bx, by in progress_bar(valid_loader, desc=f"[{name}] valid {epoch + 1}/{args.epochs}", unit="batch", leave=False):
                bx = bx.to(device, non_blocking=True)
                by = by.view(-1, 1).to(device, non_blocking=True)
                losses.append(float(loss_fn(model(bx), by).detach().cpu()))
        val_loss = float(np.mean(losses))
        logger.info(f"[{name}] epoch {epoch + 1}/{args.epochs} val_loss={val_loss:.6f} best={best_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 4:
                logger.info(f"[{name}] early stopping after epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for bx, _ in progress_bar(test_loader, desc=f"[{name}] predict", unit="batch"):
            bx = bx.to(device, non_blocking=True)
            preds.append(model(bx).detach().cpu().numpy().reshape(-1))
    return pd.Series(np.concatenate(preds), index=test_ds.predict_index(), name="score")


def train_model(name: str, frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], args: argparse.Namespace) -> pd.Series:
    train_df, valid_df, test_df = frames
    train_x, train_y = split_xy(train_df)
    valid_x, valid_y = split_xy(valid_df)
    test_x, test_y = split_xy(test_df)
    if args.fast_dev:
        train_x, train_y = train_x.iloc[-50000:], train_y.iloc[-50000:]
        valid_x, valid_y = valid_x.iloc[-10000:], valid_y.iloc[-10000:]
        test_x, test_y = test_x.iloc[:10000], test_y.iloc[:10000]
        args.epochs = min(args.epochs, 2)

    if name == "linear":
        return train_linear(train_x, train_y, valid_x, valid_y, test_x, args)
    if name == "mlp":
        return train_mlp(train_x, train_y, valid_x, valid_y, test_x, args)
    if name == "lightgbm":
        return train_lightgbm(train_x, train_y, valid_x, valid_y, test_x, args)
    if name == "xgboost":
        return train_xgboost(train_x, train_y, valid_x, valid_y, test_x, args)
    if name in {"gru", "tra", "lstm", "transformer"}:
        return train_sequence(name, train_x, train_y, valid_x, valid_y, test_x, test_y, args)
    raise ValueError(name)
