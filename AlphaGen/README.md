# AlphaGen

这是面向当前精简 qlib 框架的原生 AlphaGen 实现，不 import 或依赖 `ICT-FinD-Lab/alphagen` 源码。实现逻辑仿照原 AlphaGen：

- 用栈式 token 序列生成公式表达式。
- 用合法动作 mask 保证生成过程只产生语法有效的表达式。
- 用 LSTM 策略网络生成表达式，并用 masked PPO clipped objective 根据因子池 reward 更新策略。
- 用 MSE AlphaPool 维护协同因子集合，按单因子 IC 和因子间 mutual IC 优化线性权重。
- 所有因子值计算走 `qlib.data.D.features()`。
- 所有 IC、Rank IC、ICIR 等指标走 `qlib.contrib.evaluate_alpha`。

默认运行 RL 因子生成：

```bash
python -m AlphaGen.run
```

常用调试命令：

```bash
python -m AlphaGen.run --sample-instruments 20 --steps 200 --include-csrank
```

当前默认参数偏向可直接在 qlib 本地数据上训练：

- `--steps` 默认 `5000`，避免一启动就跑原论文级别的长训练。
- `--mine-years` 默认 `2`，RL reward 只用训练集尾部 2 年；最终报告仍用完整 train/valid/test。
- 因子池权重优化默认 `--pool-opt-steps 300 --pool-opt-tolerance 50`。
- RL 搜索默认排除 `Pow`，因为随机生成的大幂次表达式很容易触发 pandas/numpy overflow 并严重拖慢 qlib 计算；手工 `--expr` 仍可解析和评估 `Pow/Power`。
- 因子值进入指标计算前会清理 `inf` 和极端大值，并使用 qlib 的 `CSZScoreNorm`、`Fillna` 做截面标准化和缺失填充。

更快的开发验证命令：

```bash
python -m AlphaGen.run --sample-instruments 20 --steps 200 --mine-years 1 --pool-opt-steps 100 --include-csrank
```

评估已有表达式时才传 `--expr`：

```bash
python -m AlphaGen.run --expr "CSRank($close)" --sample-instruments 50
```

默认参数与 `Alpha158` 对齐：`QLIB_DATA`、`csi500`、`outputs`、`qlib-kernels=8`，并使用相同的 train/valid/test 自动切分参数。

输出位于 `outputs/alphagen/`：

- `pool.json`：当前因子池表达式、权重、单因子 IC。
- `episodes.csv`：每轮生成表达式和 reward。
- `metrics.csv`：最终 train/valid/test 组合表现。
- `split.json`：数据路径、市场、label 和切分信息。
