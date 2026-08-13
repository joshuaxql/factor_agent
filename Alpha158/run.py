from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import qlib
from qlib.backtest import backtest
from qlib.backtest.decision import TradeDecisionWO
from qlib.backtest.executor import SimulatorExecutor
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.eva.alpha import calc_ic
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.report.alpha import write_metrics_html
from qlib.contrib.report.analysis_model import model_performance_graph
from qlib.contrib.report.analysis_position import (
    report_graph,
    risk_analysis_graph,
    score_ic_graph,
)
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from Alpha158.config import CONFIG, Config


def validate_config(config: Config) -> None:
    dates = [
        pd.Timestamp(config.train_start),
        pd.Timestamp(config.train_end),
        pd.Timestamp(config.valid_start),
        pd.Timestamp(config.valid_end),
        pd.Timestamp(config.test_start),
        pd.Timestamp(config.test_end),
    ]
    if not dates[0] <= dates[1] < dates[2] <= dates[3] < dates[4] <= dates[5]:
        raise ValueError(
            "train, valid, and test dates must be ordered and non-overlapping"
        )
    if config.topk <= 0:
        raise ValueError("topk must be positive")
    if not 0 <= config.n_drop <= config.topk:
        raise ValueError("n_drop must be between 0 and topk")
    if config.num_boost_round <= 0 or config.early_stopping_rounds <= 0:
        raise ValueError("LightGBM iteration counts must be positive")
    if config.rebalance_interval <= 0:
        raise ValueError("rebalance_interval must be positive")
    if config.logging_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"Unsupported QLIB_LOGGING_LEVEL: {config.logging_level}")


def init_qlib(config: Config) -> str:
    kwargs: dict[str, object] = {
        "region": REG_CN,
        "logging_level": config.logging_level,
    }
    if config.provider_uri:
        provider_path = Path(config.provider_uri).expanduser().resolve()
        if not provider_path.is_dir():
            raise FileNotFoundError(
                f"Qlib provider directory does not exist: {provider_path}"
            )
        kwargs["provider_uri"] = str(provider_path)
    if config.qlib_kernels > 0:
        kwargs["kernels"] = config.qlib_kernels
    qlib.init(**kwargs)
    return str(kwargs.get("provider_uri", "Qlib default provider"))


def build_dataset(config: Config) -> DatasetH:
    handler = Alpha158(
        instruments=config.market,
        start_time=config.train_start,
        end_time=config.test_end,
        fit_start_time=config.train_start,
        fit_end_time=config.train_end,
        label=([config.label_expr], [config.label_name]),
    )
    return DatasetH(
        handler=handler,
        segments={
            "train": (config.train_start, config.train_end),
            "valid": (config.valid_start, config.valid_end),
            "test": (config.test_start, config.test_end),
        },
    )


class LGBModel:
    """Qlib's official LGBModel core implementation, kept local to Alpha158."""

    def __init__(
        self, loss="mse", early_stopping_rounds=50, num_boost_round=1000, **kwargs
    ):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError(f"Unsupported LightGBM loss: {loss}")
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.model = None
        self.evals_result: dict = {}

    def _prepare_data(self, dataset: DatasetH):
        import lightgbm as lgb

        datasets = []
        for segment in ("train", "valid"):
            frame = dataset.prepare(
                segment, col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
            )
            if frame.empty:
                raise ValueError(f"Empty {segment} data from dataset")
            features, labels = frame["feature"], frame["label"]
            if labels.values.ndim != 2 or labels.values.shape[1] != 1:
                raise ValueError("LightGBM doesn't support multi-label training")
            datasets.append(
                (
                    lgb.Dataset(
                        features.values,
                        label=np.squeeze(labels.values),
                        free_raw_data=False,
                    ),
                    segment,
                )
            )
        return datasets

    def fit(self, dataset: DatasetH, verbose_eval=20):
        import lightgbm as lgb

        prepared = self._prepare_data(dataset)
        datasets, names = zip(*prepared)
        self.evals_result = {}
        with tqdm(
            total=self.num_boost_round,
            desc="LightGBM training",
            unit="round",
            dynamic_ncols=True,
        ) as progress:

            def update_progress(_env):
                progress.update()

            update_progress.order = 25
            update_progress.before_iteration = False
            self.model = lgb.train(
                self.params,
                datasets[0],
                num_boost_round=self.num_boost_round,
                valid_sets=datasets,
                valid_names=names,
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds),
                    lgb.log_evaluation(period=verbose_eval),
                    lgb.record_evaluation(self.evals_result),
                    update_progress,
                ],
            )
        return self

    def predict(self, dataset: DatasetH, segment="test") -> pd.Series:
        if self.model is None:
            raise ValueError("model is not fitted yet")
        features = dataset.prepare(
            segment, col_set="feature", data_key=DataHandlerLP.DK_I
        )
        return pd.Series(
            self.model.predict(features.values), index=features.index, name="score"
        )


def calculate_ic(
    pred: pd.Series, label: pd.Series
) -> tuple[pd.DataFrame, dict[str, float]]:
    pred_label = pd.concat(
        [pred.rename("score"), label.rename("label")], axis=1
    ).dropna()
    if pred_label.empty:
        raise RuntimeError(
            "No aligned prediction and label rows are available for IC calculation"
        )
    ic, rank_ic = calc_ic(pred_label["score"], pred_label["label"], dropna=True)
    ic_frame = pd.concat([ic.rename("IC"), rank_ic.rename("Rank IC")], axis=1)

    ic_std = ic.std(ddof=1)
    rank_ic_std = rank_ic.std(ddof=1)
    summary = {
        "IC": float(ic.mean()),
        "ICIR": float(ic.mean() / ic_std) if ic_std and np.isfinite(ic_std) else np.nan,
        "Rank IC": float(rank_ic.mean()),
        "Rank ICIR": (
            float(rank_ic.mean() / rank_ic_std)
            if rank_ic_std and np.isfinite(rank_ic_std)
            else np.nan
        ),
    }
    return ic_frame, summary


class IntervalTopkDropoutStrategy(TopkDropoutStrategy):
    def __init__(self, *, rebalance_interval: int, **kwargs):
        super().__init__(**kwargs)
        self.rebalance_interval = rebalance_interval
        self._last_rebalance_step: int | None = None

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self._last_rebalance_step = None

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        if (
            self._last_rebalance_step is not None
            and trade_step - self._last_rebalance_step < self.rebalance_interval
        ):
            return TradeDecisionWO([], self)
        decision = super().generate_trade_decision(execute_result)
        if self._last_rebalance_step is not None or decision.get_decision():
            self._last_rebalance_step = trade_step
        return decision


def run_topk_dropout(pred: pd.Series, config: Config):
    benchmark: str | list[str] = config.benchmark
    if config.benchmark.lower() == "market":
        benchmark = D.list_instruments(
            D.instruments(config.market),
            start_time=config.test_start,
            end_time=config.test_end,
            as_list=True,
        )
        if not benchmark:
            raise RuntimeError(
                f"The benchmark market '{config.market}' has no instruments in the test period"
            )
    strategy = IntervalTopkDropoutStrategy(
        signal=pred.dropna(),
        topk=config.topk,
        n_drop=config.n_drop,
        rebalance_interval=config.rebalance_interval,
        limit_status_enabled=config.limit_status_enabled,
    )
    executor = SimulatorExecutor(
        time_per_step="day",
        generate_portfolio_metrics=True,
    )
    portfolio_metrics, indicator_metrics = backtest(
        start_time=config.test_start,
        end_time=config.test_end,
        strategy=strategy,
        executor=executor,
        benchmark=benchmark,
        account=config.account,
        exchange_kwargs={
            "freq": "day",
            "deal_price": config.deal_price,
            "open_cost": config.open_cost,
            "close_cost": config.close_cost,
            "min_cost": config.min_cost,
        },
    )
    if "1day" not in portfolio_metrics:
        raise RuntimeError(
            f"Qlib backtest did not return daily portfolio metrics: {list(portfolio_metrics)}"
        )
    report, positions = portfolio_metrics["1day"]
    indicators, _ = indicator_metrics["1day"]
    if report.empty:
        raise RuntimeError("Qlib TopkDropout backtest returned an empty report")

    excess_without_cost = report["return"] - report["bench"]
    excess_with_cost = excess_without_cost - report["cost"]
    risk = pd.DataFrame(
        {
            "excess_without_cost": risk_analysis(excess_without_cost, freq="day")[
                "risk"
            ],
            "excess_with_cost": risk_analysis(excess_with_cost, freq="day")["risk"],
        }
    ).T
    return report, positions, indicators, risk


def json_number(value: float | int) -> float | int | None:
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value.item() if isinstance(value, np.generic) else value


def write_performance_html(
    output_dir: Path,
    pred: pd.Series,
    label: pd.Series,
    report: pd.DataFrame,
    risk: pd.DataFrame,
) -> None:
    pred_label = pd.concat(
        [pred.rename("score"), label.rename("label")], axis=1
    ).dropna()
    analysis = risk.stack().rename("risk").to_frame()
    analysis.index.names = [None, None]
    sections = [
        ("Portfolio Performance", list(report_graph(report, show_notebook=False))),
        ("Risk Analysis", list(risk_analysis_graph(analysis, show_notebook=False))),
        ("IC and Rank IC", list(score_ic_graph(pred_label, show_notebook=False))),
        (
            "Model Performance",
            list(
                model_performance_graph(
                    pred_label,
                    graph_names=["group_return", "pred_autocorr"],
                    show_notebook=False,
                )
            ),
        ),
    ]
    write_metrics_html(
        sections,
        output_dir / "performance.html",
        title="LightGBM + Alpha158 Performance",
    )


def save_results(
    output_dir: Path,
    model,
    pred: pd.Series,
    label: pd.Series,
    ic_frame: pd.DataFrame,
    ic_summary: dict[str, float],
    report: pd.DataFrame,
    positions: dict,
    indicators: pd.DataFrame,
    risk: pd.DataFrame,
    provider_uri: str,
    config: Config,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.model.save_model(str(output_dir / "lightgbm.txt"))
    pred.to_frame().to_pickle(output_dir / "pred.pkl")
    ic_frame.to_csv(output_dir / "ic.csv")
    report.to_csv(output_dir / "backtest_report.csv")
    pd.to_pickle(positions, output_dir / "positions.pkl")
    indicators.to_csv(output_dir / "trade_indicators.csv")
    risk.to_csv(output_dir / "risk_analysis.csv")
    write_performance_html(output_dir, pred, label, report, risk)

    metrics = {
        "provider_uri": provider_uri,
        "market": config.market,
        "benchmark": config.benchmark,
        "label": config.label_expr,
        "best_iteration": model.model.best_iteration,
        "training_objective": config.loss,
        "early_stopping_metric": "l2",
        "implementation": "qlib_official_lgbmodel_core",
        "num_leaves": config.num_leaves,
        "max_depth": config.max_depth,
        "learning_rate": config.learning_rate,
        "colsample_bytree": config.colsample_bytree,
        "subsample": config.subsample,
        "lambda_l1": config.lambda_l1,
        "lambda_l2": config.lambda_l2,
        "topk": config.topk,
        "n_drop": config.n_drop,
        "rebalance_interval": config.rebalance_interval,
        "limit_status_enabled": config.limit_status_enabled,
        **ic_summary,
    }
    metrics.update(
        {
            f"{portfolio}_{metric}": value
            for portfolio, row in risk.iterrows()
            for metric, value in row.items()
        }
    )
    serializable = {key: json_number(value) for key, value in metrics.items()}
    (output_dir / "metrics.json").write_text(
        json.dumps(serializable, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def main(config: Config = CONFIG) -> int:
    validate_config(config)
    provider_uri = init_qlib(config)
    dataset = build_dataset(config)
    model = LGBModel(
        loss=config.loss,
        early_stopping_rounds=config.early_stopping_rounds,
        num_boost_round=config.num_boost_round,
        colsample_bytree=config.colsample_bytree,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        lambda_l1=config.lambda_l1,
        lambda_l2=config.lambda_l2,
        max_depth=config.max_depth,
        num_leaves=config.num_leaves,
        num_threads=config.num_threads,
    )
    model.fit(dataset)
    pred = model.predict(dataset)

    test_frame = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_I)
    if test_frame.empty or config.label_name not in test_frame.columns:
        raise RuntimeError(f"The test segment is missing {config.label_name}")
    test_y = test_frame[config.label_name].rename("label")

    ic_frame, ic_summary = calculate_ic(pred, test_y)
    report, positions, indicators, risk = run_topk_dropout(pred, config)
    output_dir = Path(config.output_dir).expanduser().resolve()
    save_results(
        output_dir,
        model,
        pred,
        test_y,
        ic_frame,
        ic_summary,
        report,
        positions,
        indicators,
        risk,
        provider_uri,
        config,
    )

    print(pd.Series(ic_summary).to_string())
    print("\nTopkDropout risk analysis:")
    print(risk.to_string())
    print(f"\nResults written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
