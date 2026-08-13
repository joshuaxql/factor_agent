# Qlib-reloaded

[Microsoft Qlib](https://github.com/microsoft/qlib) 的重新构建精简版本。
目前支持Windows和Linux

## 环境要求

- Python >= 3.12
- [Miniconda](https://docs.conda.io/projects/miniconda/) / Conda
- make（仅用于编译 Cython C++ 扩展）
- Qlib 数据（见[数据准备](#数据准备)）

### 配置环境变量

复制 `.env.example` 为 `.env`，按需填写：

```bash
cp .env.example .env
```

支持的环境变量：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `QLIB_DATA` | Qlib 数据目录 | `~/.qlib/qlib_data/cn_data` |
| `TUSHARE_TOKEN` | Tushare Pro Token（下载阶段至少需要 2000 积分） | 无 |
| `QLIB_LOGGING_LEVEL` | Qlib 日志级别 | `INFO` |

程序入口会自动加载 `.env` 文件，无需手动 `export`。

## 快速开始

```bash
# 创建并激活环境
conda create -n qlib-reloaded python=3.12
conda activate qlib-reloaded

# 安装通用依赖
python -m pip install -r requirements.txt

# 按本机 CUDA 版本安装 PyTorch（见下方说明）
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 编译 Cython C++ 扩展
make build-ext

# 验证安装
make verify
```

`requirements.txt` 不包含 `torch`、`torchvision`、`torchaudio`。这些包需要按本机 CUDA 版本选择 PyTorch 官方 pip wheel 源安装：

```bash
nvcc --version
```

| CUDA Toolkit | PyTorch wheel 源 |
|--------------|------------------|
| CUDA 12.8 | `https://download.pytorch.org/whl/cu128` |
| CUDA 12.6 | `https://download.pytorch.org/whl/cu126` |
| CUDA 11.8 | `https://download.pytorch.org/whl/cu118` |
| CPU only | `https://download.pytorch.org/whl/cpu` |

安装后检查：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 数据准备

### 使用 Tushare 构建数据

`data/` 提供可断点续传的 Tushare 全量 A 股构建流程。日期、目录、指数、限频和阶段开关统一在 `data/config.py` 配置；Token 写入项目根目录 `.env` 的 `TUSHARE_TOKEN`。默认构建 2000-01-01 至 2026-07-31 数据、申万 2021 一级历史行业，以及沪深 300、中证 500、中证 800、中证 1000 和中证全指股票池。

```bash
python -m data.run
```

原始分块和标准化 Parquet 保存在 `.data/tushare/`，最终 provider 写入 `QLIB_DATA`。价格按 Tushare 复权因子处理并将每只股票首个有效收盘价归一到 `1`；`factor` 使用 Qlib 的 `adjusted_price / original_price` 语义。可分别运行 `python -m data.download`、`python -m data.normalize` 和 `python -m data.provider`。

## 使用示例

```python
import os
import qlib
from qlib.constant import REG_CN
from qlib.data import D

qlib.init(provider_uri=os.environ.get("QLIB_DATA", "~/.qlib/qlib_data/cn_data"), region=REG_CN)

# 获取带有滚动/展开运算符的特征数据
df = D.features(
    ["SH600000"],
    ["$close", "Mean($close, 5)", "Slope($close, 5)"],
    start_time="2020-01-02",
    end_time="2020-01-10",
    freq="day",
)
print(df)
```

## LightGBM + Alpha158

`Alpha158/run.py` 使用与 `test.py` 相同的 `QlibDataLoader` 表达式加载方式计算 Alpha158 因子，训练 LightGBM，并调用 Qlib 的 `calc_ic` 计算 IC/RankIC。组合回测使用 Qlib 的 `TopkDropoutStrategy`、`SimulatorExecutor`、`backtest` 和 `risk_analysis`。

```bash
conda activate qlib-reloaded
python -m Alpha158.run
```

所有运行参数都集中在 `Alpha158/config.py`，运行前直接修改 `Config` 中的默认值。数据路径和 Qlib 日志级别分别从项目根目录 `.env` 的 `QLIB_DATA`、`QLIB_LOGGING_LEVEL` 读取。默认股票池为 `csi300`，基准为该股票池的每日等权收益，5 日标签为 `Ref($close, -6)/Ref($close, -1) - 1`。训练流程按照官方 Qlib LightGBM Alpha158 benchmark：`DatasetH + Alpha158 + LGBModel(loss="mse")`，使用 LightGBM 内置 L2 early stopping；模型参数也采用官方 workflow 配置。回测仍使用 `topk=50`、`n_drop=5`、每 5 个交易日换仓。

```python
@dataclass(frozen=True)
class Config:
    market: str = "csi300"
    benchmark: str = "market"
    topk: int = 50
    n_drop: int = 5
    rebalance_interval: int = 5
```

结果默认写入 `outputs/alpha158_lightgbm/`，包括 LightGBM 模型、预测、每日 IC、回测报告、持仓、交易指标、风险分析和汇总指标。`performance.html` 使用 Qlib 原生报告函数生成，集中展示组合收益与回撤、风险分析、IC/RankIC、分组收益和预测稳定性。

## 许可证

MIT License — 版权所有 (c) Microsoft Corporation。详见 [LICENSE](LICENSE)。
