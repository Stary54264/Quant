# Quant — 基于 H 值、L 信号与吸引子的量化模型

> 一个融合 **Hurst 指数（H）**、**Lyapunov 指数（L）** 与 **动力系统吸引子（Attractor）** 三大非线性动力学因子的多周期量化交易研究框架。

---

## 1. 项目简介

本项目尝试跳出传统的"技术指标 + 机器学习"范式，改从**非线性动力学**与**分形市场假说（FMH）**的视角构建交易模型。核心假设是：

> 金融时间序列并非纯随机游走，而是一个**低维、有噪、带记忆**的混沌系统。当系统在相空间中靠近某个吸引子时，价格路径存在可预测的结构，此时结合分形维度（H 值）与流动性/结构信号（L 信号）可以获得稳定的正期望。

三大核心模块：

| 模块 | 作用 | 核心问题 |
| --- | --- | --- |
| **H** — Hurst 指数 | 判定市场处于**趋势 / 均值回归 / 随机**哪种状态 | *当下市场记忆有多强？* |
| **L** — Lyapunov 指数 | 度量相邻轨道的发散率，量化**可预测视界**与混沌程度 | *当下预测还能走多远？* |
| **A** — Phase-space Attractor | 在重构相空间中定位当前状态与"引力中心"的相对位置 | *系统正在被拉向何处？* |

三者形成 **记忆强度 → 可预测视界 → 位置定位 → 触发下单** 的完整闭环，共享同一套相空间语言（Takens 嵌入），因子之间几何一致、可以直接拼接。

---

## 2. 理论基础不对

### 2.1 Hurst 指数（H 值）
使用 **R/S 分析**、**DFA（去趋势波动分析）** 和 **小波估计**三种方法互相校验：

- `H ≈ 0.5` — 随机游走（不交易）
- `H > 0.5` — 长记忆 / 趋势持续（顺势策略）
- `H < 0.5` — 反持续 / 均值回归（逆势策略）

关键实现：滚动窗口 `w ∈ {64, 128, 256, 512}`，避免单一窗口带来的估计偏差。

### 2.2 Lyapunov 指数（L 信号）
在重构的相空间中，两条初始距离为 `δ₀` 的邻近轨道，经时间 `t` 后距离演化为：

```
δ(t) ≈ δ₀ · exp(λ₁ · t)
```

`λ₁` 即**最大 Lyapunov 指数（MLE）**，是 L 信号的核心：

- `λ₁ > 0` — 混沌系统，存在**可预测视界** `T_max ≈ 1/λ₁`；超过该时长的预测无意义
- `λ₁ ≈ 0` — 边缘可预测（周期/准周期临界）
- `λ₁ < 0` — 系统收敛到不动点/吸引子，可预测性强

估计方法采用 **Rosenstein / Kantz 算法**（对短金融序列更鲁棒），滚动窗口 `w ∈ {256, 512, 1024}`。

L 信号输出为**归一化的可预测视界**：

```
L_t = clip(1 / (λ₁·Δt · N_ahead), 0, 1)
```

即"在未来 `N_ahead` 根 K 线内，模型还剩多少可预测性"。`L` 越接近 1 → 越值得开仓；越接近 0 → 混沌主导，应缩减仓位或不交易。

### 2.3 吸引子（Attractor，A 信号）
基于 **Takens 嵌入定理**将一维价格序列重构到 `m` 维相空间：

```
x(t) → [x(t), x(t-τ), x(t-2τ), ..., x(t-(m-1)τ)]
```

- 嵌入维度 `m` 由 **False Nearest Neighbors (FNN)** 确定
- 延迟 `τ` 由 **互信息法（AMI）** 首个极小值确定
- 用 **RQA（递归量化分析）** + **k-medoids / DBSCAN** 在历史相空间聚类，识别"引力中心" `{C_k}`
- 关联维数 `D₂` 作为吸引子复杂度的辅助校验

A 信号：当前状态点 `x_t` 到最近吸引子中心 `C_k*` 的**方向单位向量**与**归一化距离**，据此判断价格正被拉向哪个历史平衡态。

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
│   │   ├── hurst.py         # H 值三估计器 (R/S, DFA, Wavelet)
│   │   ├── lyapunov.py      # L 信号 (Rosenstein / Kantz MLE)
│   │   ├── embedding.py     # Takens 嵌入 (FNN + AMI)
│   │   └── attractor.py     # 吸引子聚类 + A 信号
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
                    ┌──────────────────────┐
             K线 →  │  Takens 相空间重构   │ ── τ (AMI), m (FNN)
                    └──────────┬───────────┘
                               ▼
        ┌──────────────┬──────────────────┬──────────────┐
        │   H 值估计   │  Lyapunov 估计   │  吸引子聚类  │
        │ (R/S/DFA/W)  │ (Rosenstein/Kantz)│ (RQA/k-med) │
        └──────┬───────┴────────┬─────────┴──────┬───────┘
               ▼                ▼                ▼
             记忆 H         可预测视界 L      距离/方向 A
                             (0~1)
                               ▼
        ┌─────────────────────────────────────────────────┐
        │  Regime Router: 由 H 决定"顺势/回归/静默"模板   │
        │  Gate       : L 作为总开关，L < L_min 则不开仓   │
        └────────────────────────┬────────────────────────┘
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  Fusion Score  S = L · ( w_H·f(H) + w_A·A )     │
        │  L 作乘性置信度，H/A 决定方向与强度              │
        └────────────────────────┬────────────────────────┘
                                 ▼
        ┌─────────────────────────────────────────────────┐
        │  Sizing & Risk: 波动率目标 + 分数级凯利 + 熔断   │
        └────────────────────────┬────────────────────────┘
                                 ▼
                            下单 / 平仓
```

融合打分示例（**L 作为乘性置信闸**，避免混沌区盲目下单）：

- **趋势态**（`H > 0.55`）：`S = L · ( 0.6·sign(trend) + 0.4·A )`
- **回归态**（`H < 0.45`）：`S = L · ( -0.5·zscore(price) + 0.5·A )`
- **随机态**（`0.45 ≤ H ≤ 0.55` 或 `L < L_min`）：不开仓，仅监控吸引子跃迁

**持仓视界**与 L 挂钩：单笔最大持仓时间 `≤ α · 1/λ₁`（默认 `α = 0.3`），超过则强制平仓——即"不在自己看不清的时段留仓位"。

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
- [ ] **M2** — Takens 嵌入（AMI + FNN）
- [ ] **M3** — Lyapunov 估计（Rosenstein/Kantz）+ 视界基准测试
- [ ] **M4** — 吸引子聚类 + A 信号
- [ ] **M5** — Regime Router & Fusion 融合
- [ ] **M6** — 事件驱动回测引擎
- [ ] **M7** — 参数鲁棒性 & CPCV
- [ ] **M8** — Paper Trading（模拟盘）
- [ ] **M9** — 实盘小仓位灰度

---

## 8. 参考文献

1. Peters, E. E. *Fractal Market Analysis*, 1994.
2. Takens, F. *Detecting strange attractors in turbulence*, 1981.
3. Rosenstein, M. T. et al. *A practical method for calculating largest Lyapunov exponents from small data sets*, Physica D, 1993.
4. Kantz, H. *A robust method to estimate the maximal Lyapunov exponent of a time series*, Phys. Lett. A, 1994.
5. Kantz & Schreiber, *Nonlinear Time Series Analysis*, 2004.
6. López de Prado, M. *Advances in Financial Machine Learning*, 2018.
7. Mandelbrot, B. *The (Mis)Behavior of Markets*, 2004.

---

## 9. 声明

本仓库仅用于**学术研究与个人策略验证**，不构成任何投资建议。金融市场存在极高风险，实盘交易请自行承担全部后果。

