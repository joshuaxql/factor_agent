from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

from .config import parse_args, set_seed
from .expressions import parse_many
from .pool import MseAlphaPool
from .qlib_data import QlibAlphaCalculator, auto_split, init_qlib, write_json
from .search import AlphaGenerator


def parse_weights(raw: str | None, n: int) -> list[float]:
    if raw is None:
        return [1.0 / n for _ in range(n)]
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if len(weights) != n:
        raise ValueError(f"Expected {n} weights, got {len(weights)}")
    return weights


def resolve_eval_range(args, split):
    if (args.start is None) != (args.end is None):
        raise ValueError("--start and --end must be provided together.")
    if args.start and args.end:
        return args.start, args.end
    return split.test


def resolve_mine_range(args, split):
    if args.mine_years <= 0:
        return split.train
    import pandas as pd
    from qlib.data import D

    calendar = pd.DatetimeIndex(D.calendar(freq=args.freq))
    train_start = pd.Timestamp(split.train[0])
    train_end = pd.Timestamp(split.train[1])
    calendar = calendar[(calendar >= train_start) & (calendar <= train_end)]
    if len(calendar) == 0:
        return split.train
    wanted_start = train_end - pd.DateOffset(years=args.mine_years)
    pos = calendar.searchsorted(wanted_start, side="left")
    return calendar[min(pos, len(calendar) - 1)].strftime("%Y-%m-%d"), split.train[1]


def evaluate_existing(args, split) -> None:
    start, end = resolve_eval_range(args, split)
    instruments = args.instruments or args.market
    calc = QlibAlphaCalculator(
        instruments,
        start,
        end,
        target=args.target,
        freq=args.freq,
        normalize_alpha=not args.no_normalize,
        sample_instruments=args.sample_instruments,
    )
    raw_exprs = list(args.expr or [])
    weights = None
    if args.pool_json:
        raw = json.loads(Path(args.pool_json).read_text(encoding="utf-8"))
        raw_exprs.extend(raw["exprs"])
        weights = [float(x) for x in raw["weights"]]
    exprs = parse_many(raw_exprs)
    if weights is None and args.weights:
        weights = parse_weights(args.weights, len(exprs))
    rows = []
    for expr in exprs:
        summary = calc.calc_summary(calc.evaluate_alpha(expr))
        rows.append({"name": str(expr), **summary.to_dict()})
    if weights is not None:
        summary = calc.calc_summary(calc.make_ensemble_alpha(exprs, weights))
        rows.append({"name": "pool", **summary.to_dict()})
    result = pd.DataFrame(rows)
    print(result.to_string(index=False))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "metrics.csv", index=False)
    write_json(
        out_dir / "split.json",
        {
            "mode": "evaluate",
            "provider_uri": str(Path(args.provider_uri).expanduser().resolve()),
            "market": args.market,
            "eval": (start, end),
            "segments": split.as_dict(),
            "sample_instruments": args.sample_instruments,
        },
    )


def run_generation(args, split) -> None:
    instruments = args.instruments or args.market
    out_dir = Path(args.output_dir).resolve() / "alphagen"
    out_dir.mkdir(parents=True, exist_ok=True)
    mine_range = resolve_mine_range(args, split)
    train_calc = QlibAlphaCalculator(
        instruments,
        mine_range[0],
        mine_range[1],
        target=args.target,
        freq=args.freq,
        normalize_alpha=not args.no_normalize,
        sample_instruments=args.sample_instruments,
    )
    valid_calc = QlibAlphaCalculator(
        instruments,
        split.valid[0],
        split.valid[1],
        target=args.target,
        freq=args.freq,
        normalize_alpha=not args.no_normalize,
        sample_instruments=args.sample_instruments,
    )
    test_calc = QlibAlphaCalculator(
        instruments,
        split.test[0],
        split.test[1],
        target=args.target,
        freq=args.freq,
        normalize_alpha=not args.no_normalize,
        sample_instruments=args.sample_instruments,
    )
    pool = MseAlphaPool(
        args.pool_capacity,
        train_calc,
        ic_lower_bound=args.ic_lower_bound,
        l1_alpha=args.l1_alpha,
        optimize_max_steps=args.pool_opt_steps,
        optimize_tolerance=args.pool_opt_tolerance,
    )
    write_json(
        out_dir / "split.json",
        {
            "mode": "generate",
            "provider_uri": str(Path(args.provider_uri).expanduser().resolve()),
            "market": args.market,
            "label": args.target,
            "segments": split.as_dict(),
            "mine": mine_range,
            "sample_instruments": args.sample_instruments,
            "pool_capacity": args.pool_capacity,
            "steps": args.steps,
        },
    )
    generator = AlphaGenerator(
        pool,
        include_csrank=args.include_csrank,
        max_expr_length=args.max_expr_length,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        device=args.device,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        ppo_clip=args.ppo_clip,
        ppo_epochs=args.ppo_epochs,
        print_expr=args.print_expr,
    )
    logger.info(f"[alphagen] segments: {split.as_dict()}")
    logger.info(f"[alphagen] qlib-native masked PPO generation: steps={args.steps}")
    generator.run(args.steps, args.episode_batch_size, out_dir, log_every=args.log_every)
    metrics = {"pool_size": pool.size, "train_best_ic": pool.best_ic_ret}
    if pool.size > 0:
        metrics["valid_ic"], metrics["valid_rank_ic"] = pool.test_ensemble(valid_calc)
        metrics["test_ic"], metrics["test_rank_ic"] = pool.test_ensemble(test_calc)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)
    logger.info(f"[alphagen] done. Outputs written to: {out_dir}")


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    init_qlib(args.provider_uri, kernels=args.qlib_kernels)
    split = auto_split(args)
    if args.expr or args.pool_json:
        evaluate_existing(args, split)
    else:
        run_generation(args, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
