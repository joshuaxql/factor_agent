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
| `TUSHARE_TOKEN` | Tushare API token | 无 |
| `GM_TOKEN` | 掘金 API token（指数历史成分股） | 无 |
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

### 使用社区数据源

可以使用这个[数据源](https://github.com/chenditc/investment_data/releases)将 Qlib 中国股市数据下载到 `$QLIB_DATA` 目录（默认 `~/.qlib/qlib_data/cn_data`）：

```bash
export QLIB_DATA=~/.qlib/qlib_data/cn_data
mkdir -p $QLIB_DATA
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C $QLIB_DATA --strip-components=1
rm -f qlib_bin.tar.gz
```

### 通过掘金自行构建

当前数据管线使用掘金量化下载 A 股日线、复权因子、市值和 CSI 指数历史成分股，并缓存到 `.env` 中的 `$QLIB_DATA/cache`。转换完成后会在 `$QLIB_DATA` 下生成 qlib provider 结构：`calendars/`、`instruments/`、`features/`。

```bash
# .env 中设置 QLIB_DATA、GM_TOKEN

# 一键下载、处理并转换为 qlib 格式
python -m data.build_qlib --start 2010-01-01 --end 2026-06-30

# 如果 cache/daily 已经存在，只做 qlib 格式转换
python -m data.dump_bin --provider-uri D:\data\qlib

# 也可以沿用 data.run 入口
python -m data.run --phase dump --provider-uri D:\data\qlib
```

相关文件：
- `data/collector.py` —— 掘金原始数据下载到 `cache/raw/`
- `data/processor.py` —— 原始缓存处理为 `cache/daily/`
- `data/dump_bin.py` —— `cache/daily/` 转换为 qlib `.bin` provider
- `data/build_qlib.py` —— 一键下载、处理、转换入口

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

## Alpha158 模型实验

一键运行 Linear、XGBoost、LightGBM、MLP、GRU、TRA、LSTM、Transformer with Alpha158：

```bash
conda activate qlib-reloaded
python -m Alpha158.run --provider-uri "$QLIB_DATA" --output-dir outputs --models all --cache-data
```

默认读取 `$QLIB_DATA`（未设置时回退到 `~/.qlib/qlib_data/cn_data`），股票池为 `csi500`，根据本地交易日历自动划分训练、验证、测试集。结果输出到 `outputs/`：

- `cache/`：Alpha158 处理后数据缓存
- `linear/`、`xgboost/`、`lightgbm` 等：各模型独立结果目录，包含该模型自己的 `metrics.csv`、`metrics.html`、`daily_returns.pkl`、`ic.pkl`、`pred.pkl`

实验代码位于 `Alpha158/`：

- `run.py`：一键总入口
- `qlib_data.py`：qlib 初始化、日期切分、Alpha158 数据加载
- `models.py`：Linear、XGBoost、LightGBM、MLP、GRU、TRA、LSTM、Transformer
- `metrics.py`：调用 `qlib.contrib.evaluate_alpha` 计算 IC、RankIC、ARR、IR、MDD 等指标；IC/RankIC 的 label 统一为未来 5 个交易日收益 `Ref($close, -6)/Ref($close, -1) - 1`
- `plots.py`：兼容转发层，实际调用 `qlib.contrib.report.alpha` 生成表格图和收益曲线图
- `config.py`：命令行参数和公共配置

`qlib.contrib.report` 已恢复为 upstream Qlib 的完整 report 目录，并新增 `qlib.contrib.report.alpha` 作为 Alpha/model 实验的 HTML 报告接口；绘图使用 Plotly 并输出 HTML 报告，不依赖 Kaleido。

如需调试小样本：

```bash
python -m Alpha158.run --models linear mlp --sample-instruments 50 --fast-dev --allow-cpu
```

并行度可通过参数调整：

```bash
python -m Alpha158.run --processor-n-jobs -1 --qlib-kernels 16 --models linear
```

`--processor-n-jobs` 控制 Alpha158 processor 的 joblib 并行度，`-1` 表示使用全部 CPU；`--qlib-kernels` 控制 qlib 特征读取进程数，`0` 表示使用 qlib 默认值。

速度相关默认值参考 upstream Qlib benchmark：

- `--processor-preset upstream`：默认值，跳过慢的 `ProcessInf`，和 upstream Alpha158 树模型配置一致。
- `--processor-preset safe`：保留 `ProcessInf + Fillna`，更保守但明显更慢。
- `--sequence-feature-preset alpha20`：默认值，GRU/LSTM/TRA/Transformer 使用 upstream 时序模型配置里的 20 个 Alpha158 特征。
- `--tree-n-estimators 800 --early-stopping-rounds 50`：默认比原来的 2000 轮更快，仍保留 early stopping。

## 许可证

MIT License — 版权所有 (c) Microsoft Corporation。详见 [LICENSE](LICENSE)。
