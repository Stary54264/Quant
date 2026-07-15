# Quant — 基于 H 值、L 信号与吸引子的量化模型

> 一个融合 **Hurst 指数（H 值）**、**L 信号（Level / Liquidity 信号）** 与 **动力系统吸引子（Attractor）** 三大因子的多周期量化交易研究框架。

---

## 1. 项目简介

本项目尝试跳出传统的"技术指标 + 机器学习"范式，改从**非线性动力学**与**分形市场假说（FMH）**的视角构建交易模型。核心假设是：

> 金融时间序列并非纯随机游走，而是一个**低维、有噪、带记忆**的混沌系统。当系统在相空间中靠近某个吸引子时，价格路径存在可预测的结构，此时结合分形维度（H 值）与流动性/结构信号（L 信号）可以获得稳定的正期望。

三大核心模块：

| 模块 | 作用 | 核心问题 |
| --- | --- | --- |
| **H 值** — Hurst 指数 | 判定市场处于**趋势 / 均值回归 / 随机**哪种状态 | *当下市场记忆有多强？* |
| **L 信号** — Level 信号 | 捕捉多周期流动性/结构位（挂单簇、成交量重心、支撑压力） | *当下钱在哪里？* |
| **吸引子** — Phase-space Attractor | 在重构相空间中定位当前状态与"引力中心"的相对位置 | *系统正在被拉向何处？* |

三者形成 **状态识别 → 位置定位 → 触发下单** 的完整闭环。

---

## 2. 理论基础

### 2.1 Hurst 指数（H 值）
使用 **R/S 分析**、**DFA（去趋势波动分析）** 和 **小波估计**三种方法互相校验：

- `H ≈ 0.5` — 随机游走（不交易）
- `H > 0.5` — 长记忆 / 趋势持续（顺势策略）
- `H < 0.5` — 反持续 / 均值回归（逆势策略）

关键实现：滚动窗口 `w ∈ {64, 128, 256, 512}`，避免单一窗口带来的估计偏差。

### 2.2 L 信号
L 信号是一族**结构化流动性因子**的统称，包含：

1. **L-Volume**：VWAP、成交量重心（VPOC）与 HVN/LVN 节点
2. **L-Order**：订单簿失衡 OBI、大单/冰山探测
3. **L-Structure**：多周期 Swing High/Low、供给需求区
4. **L-Time**：日内周期节律、开收盘窗口效应

输出为归一化的 `L ∈ [-1, 1]`，正值代表买方主导，负值代表卖方主导。

### 2.3 吸引子（Attractor）
基于 **Takens 嵌入定理**将一维价格序列重构到 `m` 维相空间：

```
x(t) → [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
```

- 嵌入维度 `m` 由 **False Nearest Neighbors (FNN)** 确定
- 延迟 `τ` 由**互信息法（AMI）**首个极小值确定
- 通过 **最大 Lyapunov 指数** 判定系统是否处于混沌可预测区
- 用 **RQA（递归量化分析）** 或 **k-medoids** 聚类识别"引力中心"

信号 `A`：当前状态到最近吸引子的方向 + 距离。

---

## 3. 目录结构（规划）

```
Quant/
├── README.md
├── configs/                 # 品种/周期/参数 YAML
├── data/
│   ├── raw/                 # tick / K 线原始数据
│   └── processed/           # 清洗、对齐后的 parquet
├── src/
│   ├── features/
│   │   ├── hurst.py         # H 值三估计器
│   │   ├── l_signal.py      # L 信号族
│   │   └── attractor.py     # 相空间重构 + 吸引子识别
│   ├── models/
│   │   ├── regime.py        # H → 状态机
│   │   ├── fusion.py        # H × L × A 融合打分
│   │   └── risk.py          # 波动率目标 & 凯利仓位
│   ├── backtest/
│   │   ├── engine.py        # 事件驱动回测
│   │   └── metrics.py       # Sharpe / Calmar / MAR / 最大 DD
│   ├── live/
│   │   └── executor.py      # 实盘执行 & 风控闸门
│   └── utils/
├── notebooks/               # 研究/可视化
├── tests/
└── reports/                 # 回测报告与图表
```

---

## 4. 模型流程

```
        ┌─────────────┐      ┌─────────────┐      ┌──────────────┐
 K线 →  │  H 值估计   │      │  L 信号族   │      │  相空间重构  │
        │ (R/S,DFA,W) │      │ (Vol/OB/…)  │      │  + 吸引子 A  │
        └──────┬──────┘      └──────┬──────┘      └──────┬───────┘
               ▼                    ▼                    ▼
        ┌─────────────────────────────────────────────────────┐
        │  Regime Router: 由 H 决定使用"顺势/回归/静默"模板   │
        └────────────────────────┬────────────────────────────┘
                                 ▼
        ┌─────────────────────────────────────────────────────┐
        │  Fusion Score  S = w_H·f(H) + w_L·L + w_A·A         │
        │  (权重按 regime 动态切换)                            │
        └────────────────────────┬────────────────────────────┘
                                 ▼
        ┌─────────────────────────────────────────────────────┐
        │  Sizing & Risk: 波动率目标 + 分数级凯利 + 熔断       │
        └────────────────────────┬────────────────────────────┘
                                 ▼
                            下单 / 平仓
```

融合打分示例：

- **趋势态**（`H > 0.55`）：`S = 0.5·sign(trend) + 0.3·L + 0.2·A`
- **回归态**（`H < 0.45`）：`S = -0.4·zscore(price) + 0.3·L + 0.3·A`
- **随机态**：不开仓，仅监控吸引子跃迁

---

## 5. 快速开始

```bash
# 1. 环境
conda create -n quant python=3.11 -y
conda activate quant
pip install -r requirements.txt

# 2. 拉取样例数据
python -m src.data.fetch --symbol BTCUSDT --tf 1h --since 2022-01-01

# 3. 计算特征
python -m src.features.build --config configs/btc_1h.yaml

# 4. 回测
python -m src.backtest.run --config configs/btc_1h.yaml --report reports/btc_1h

# 5. 查看结果
open reports/btc_1h/summary.html
```

---

## 6. 评估指标

| 指标 | 目标 |
| --- | --- |
| 年化收益 | ≥ 30% |
| Sharpe | ≥ 1.5 |
| Calmar | ≥ 1.0 |
| 最大回撤 | ≤ 20% |
| 胜率 | 参考项，不作为主目标 |
| 盈亏比 | ≥ 1.5 |

回测使用**Purged K-Fold + Embargo**（López de Prado）避免信息泄漏，参数选择用 **组合对称交叉验证（CPCV）**。

---

## 7. 路线图

- [ ] **M0** — 数据管道 & 特征库骨架
- [ ] **M1** — H 值三估计器 + 单元测试
- [ ] **M2** — L 信号族（V1：VWAP/VPOC/OBI）
- [ ] **M3** — Takens 重构 + 吸引子聚类
- [ ] **M4** — Regime Router & Fusion 融合
- [ ] **M5** — 事件驱动回测引擎
- [ ] **M6** — 参数鲁棒性 & CPCV
- [ ] **M7** — Paper Trading（模拟盘）
- [ ] **M8** — 实盘小仓位灰度

---

## 8. 参考文献

1. Peters, E. E. *Fractal Market Analysis*, 1994.
2. Takens, F. *Detecting strange attractors in turbulence*, 1981.
3. López de Prado, M. *Advances in Financial Machine Learning*, 2018.
4. Kantz & Schreiber, *Nonlinear Time Series Analysis*, 2004.
5. Mandelbrot, B. *The (Mis)Behavior of Markets*, 2004.

---

## 9. 声明

本仓库仅用于**学术研究与个人策略验证**，不构成任何投资建议。金融市场存在极高风险，实盘交易请自行承担全部后果。

