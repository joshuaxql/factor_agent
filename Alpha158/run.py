from __future__ import annotations

from contextlib import contextmanager
import json
import os
import sys
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from .config import parse_args, selected_models, set_seed


@contextmanager
def _import_qlib_modules_without_project_env():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="qlib_import_") as tmp:
        try:
            os.chdir(tmp)
            yield
        finally:
            os.chdir(cwd)


with _import_qlib_modules_without_project_env():
    from .metrics import evaluate_model
    from .models import train_model
    from .plots import cleanup_top_level_outputs, write_model_report
    from .qlib_data import LABEL_EXPR, auto_split, init_qlib, load_frames, split_xy


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    models = selected_models(args.models)
    out_dir = Path(args.output_dir)
    if args.fast_dev and out_dir == Path("outputs"):
        out_dir = out_dir / "fast_dev"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    provider_path = init_qlib(args.provider_uri, kernels=args.qlib_kernels)
    split = auto_split(args)
    run_info = {"provider_uri": str(provider_path), "market": args.market, "label": LABEL_EXPR, "segments": split.as_dict()}
    cleanup_top_level_outputs(out_dir, args.market)
    logger.info(f"Segments: {split.as_dict()}")
    logger.info(f"Models: {models}")

    data_started = time.time()
    frames = load_frames(args, split, out_dir)
    logger.info(f"[data] done in {(time.time() - data_started) / 60:.1f} min")
    test_label = split_xy(frames[2])[1]

    for name in models:
        started = time.time()
        logger.info(f"[{name}] training...")
        model_dir = out_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "split.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
        pred = train_model(name, frames, args)
        pred.to_frame("score").to_pickle(model_dir / "pred.pkl")
        row, returns, ic_frame = evaluate_model(name, pred, test_label, args.topk_ratio)
        metrics = pd.DataFrame([row])
        metrics.to_csv(model_dir / "metrics.csv", index=False)
        returns.to_pickle(model_dir / "daily_returns.pkl")
        ic_frame.to_pickle(model_dir / "ic.pkl")
        write_model_report(name, metrics, returns, ic_frame, model_dir, args.market)
        logger.info(
            f"[{name}] done in {(time.time() - started) / 60:.1f} min | "
            f"IC={row['IC']:.4f} RankIC={row['Rank IC']:.4f} ARR={row['ARR (%)']:.2f}%",
        )

    logger.info(f"Done. Outputs written to: {out_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logger.error(f"ERROR: {exc}")
        raise
