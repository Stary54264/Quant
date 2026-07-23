#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子评价脚本
用法: python <skill-path>/scripts/evaluate_factor.py <因子文件路径> <JSON参数字典>
示例: python ~/.claude/skills/factor-writer/scripts/evaluate_factor.py factors/rsi '{"N": 14}'
功能: 计算因子的IC、ICIR、胜率等评价指标，在多个预测周期上评价，并输出markdown格式的评价结果
"""

import os
import sys
import glob
import json
import importlib.util
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免需要显示
import matplotlib.pyplot as plt


def find_data_csv(assets_dir: str) -> str:
    """在 assets 目录下查找 000905_yyyymmdd_yyyymmdd.csv 数据文件。

    使用 glob 模式匹配，避免因数据区间变化而改动脚本。
    如果匹配到多个文件，选取字典序最后一个（即区间终止日期最新的）。
    """
    pattern = os.path.join(assets_dir, '000905_*.csv')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"未在 {assets_dir} 找到 000905_yyyymmdd_yyyymmdd.csv 数据文件"
        )
    return matches[-1]


def calc_spearmanr(x, y):
    """计算Spearman秩相关系数和p值"""
    # 去除nan
    mask = ~x.isna() & ~y.isna()
    if mask.sum() < 2:
        return np.nan, np.nan
    rho, p_value = spearmanr(x[mask], y[mask])
    return rho, p_value


def calc_ic(factor: pd.Series, forward_return: pd.Series) -> dict:
    """计算IC相关指标"""
    # 去除nan
    mask = ~factor.isna() & ~forward_return.isna()
    if mask.sum() < 10:
        return {
            'ic_overall': np.nan,
            'ic_overall_pvalue': np.nan,
            'ic_mean': np.nan,
            'ic_std': np.nan,
            'icir': np.nan,
            'count': int(mask.sum())
        }
    ic, p_value = calc_spearmanr(factor[mask], forward_return[mask])

    # 滚动IC (252天窗口)
    ic_series = []
    window = 252
    factor_masked = factor[mask]
    ret_masked = forward_return[mask]
    for i in range(window, len(factor_masked)):
        ic_i, _ = spearmanr(
            factor_masked.iloc[i-window:i],
            ret_masked.iloc[i-window:i]
        )
        ic_series.append(ic_i)
    ic_series = pd.Series(ic_series)

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std if ic_std != 0 else np.nan

    return {
        'ic_overall': ic,
        'ic_overall_pvalue': p_value,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'count': int(mask.sum())
    }


def calc_max_drawdown(cum_returns: pd.Series) -> float:
    """计算最大回撤
    Parameters
    ----------
    cum_returns : pd.Series
        累积收益率序列 (1 + daily_returns).cumprod()
    Returns
    -------
    float
        最大回撤 (百分比，负数表示回撤，比如 -0.20 表示最大回撤 20%)
    """
    # 累积收益曲线的峰值
    peak = cum_returns.cummax()
    # 计算回撤 = (当前值 - 峰值) / 峰值
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min()
    return max_dd


def quantile_analysis(factor_values: pd.Series, forward_return: pd.Series, close: pd.Series, n_quantiles=5, horizon=1) -> dict:
    """分层收益分析"""
    valid_mask = ~factor_values.isna() & ~forward_return.isna()
    factor_clean = factor_values[valid_mask]
    return_clean = forward_return[valid_mask]
    close_clean = close[valid_mask]  # 对应日期的收盘价
    original_index = factor_clean.index  # 保存原始索引用于输出累积收益

    if len(factor_clean) < n_quantiles * 5:
        print(f"警告: 分层分析跳过 - 有效数据数量({len(factor_clean)})少于最小要求({n_quantiles * 5})")
        return {}

    # 检查唯一值数量，自动调整分组数量
    n_unique = factor_clean.nunique()
    # 如果唯一值少于要求的分层数量，自动调整为按唯一值数量分组
    actual_n_quantiles = min(n_quantiles, n_unique)
    if actual_n_quantiles < 2:
        print(f"警告: 分层分析跳过 - 因子唯一值数量({n_unique})太少，至少需要2个唯一值才能分组")
        return {}
    if actual_n_quantiles < n_quantiles:
        print(f"提示: 因子唯一值数量({n_unique})少于要求的分层数量({n_quantiles})，自动调整为{actual_n_quantiles}组")

    # 根据唯一值数量选择分组方法
    if n_unique <= n_quantiles:
        # 唯一值数量小于等于要求分组数，使用cut等距分组
        factor_q = pd.cut(factor_clean, actual_n_quantiles, labels=range(1, actual_n_quantiles+1))
        factor_q_full = pd.cut(factor_clean, actual_n_quantiles, labels=False)
    else:
        # 唯一值足够，优先使用qcut（分位数分组）
        factor_q = pd.qcut(factor_clean, actual_n_quantiles, labels=range(1, actual_n_quantiles+1))
        factor_q_full = pd.qcut(factor_clean, actual_n_quantiles, labels=False)

    group_returns = {}
    group_returns_std = {}
    group_counts = {}
    for q in range(1, actual_n_quantiles+1):
        mask_q = factor_q == q
        # return_clean 是 horizon 日累计收益，转换为单日平均收益存储
        group_returns[q] = (return_clean[mask_q].mean()) / horizon
        group_returns_std[q] = return_clean[mask_q].std()
        group_counts[q] = int(mask_q.sum())

    # 多空收益 (多在因子值最大组，空在最小组)
    # group_returns 现在存储的是单日平均收益
    ls_daily_return_mean = group_returns[actual_n_quantiles] - group_returns[1]

    # 1. 多空策略: 做多最大组 + 做空最小组
    ls_weights = np.zeros_like(factor_clean)
    ls_weights[factor_q_full == 0] = -1 / horizon  # 空最小组，转换为单日
    ls_weights[factor_q_full == actual_n_quantiles-1] = 1 / horizon  # 多最大组，转换为单日
    ls_returns = forward_return[valid_mask] * ls_weights
    # 计算多空策略累积收益率曲线
    daily_ls_returns = ls_returns
    ls_cumulative = (1 + daily_ls_returns).cumprod()
    ls_cumulative_returns = (ls_cumulative - 1) * 100
    ls_cumulative_returns = ls_cumulative_returns.reset_index()
    ls_cumulative_returns.columns = ['date', 'cumulative_return']
    # 计算多空最大回撤
    ls_max_drawdown = calc_max_drawdown(ls_cumulative) * 100  # 转换为百分比

    # 2. 只做多策略: 只做多最大分组
    long_only_weights = np.zeros_like(factor_clean)
    long_only_weights[factor_q_full == actual_n_quantiles-1] = 1 / horizon  # 只多最大组
    long_only_returns = forward_return[valid_mask] * long_only_weights
    daily_long_only_returns = long_only_returns
    long_only_cumulative = (1 + daily_long_only_returns).cumprod()
    long_only_cumulative_returns = (long_only_cumulative - 1) * 100
    long_only_cumulative_returns = long_only_cumulative_returns.reset_index()
    long_only_cumulative_returns.columns = ['date', 'cumulative_return']
    # 计算只多的统计指标
    long_only_daily_mean = group_returns[actual_n_quantiles]
    long_only_daily_std = group_returns_std[actual_n_quantiles]
    long_only_annual = long_only_daily_mean * 252 * 100
    if long_only_daily_std != 0 and not np.isnan(long_only_daily_std):
        long_only_sharpe = (long_only_daily_mean / long_only_daily_std) * np.sqrt(252)
    else:
        long_only_sharpe = np.nan
    # 计算只做多最大回撤
    long_only_max_drawdown = calc_max_drawdown(long_only_cumulative) * 100

    # 3. 只做空策略: 只做空最小分组
    short_only_weights = np.zeros_like(factor_clean)
    short_only_weights[factor_q_full == 0] = -1 / horizon  # 只空最小组
    short_only_returns = forward_return[valid_mask] * short_only_weights
    daily_short_only_returns = short_only_returns
    short_only_cumulative = (1 + daily_short_only_returns).cumprod()
    short_only_cumulative_returns = (short_only_cumulative - 1) * 100
    short_only_cumulative_returns = short_only_cumulative_returns.reset_index()
    short_only_cumulative_returns.columns = ['date', 'cumulative_return']
    # 计算只空的统计指标
    short_only_daily_mean = -group_returns[1]
    short_only_daily_std = group_returns_std[1]
    short_only_annual = short_only_daily_mean * 252 * 100
    if short_only_daily_std != 0 and not np.isnan(short_only_daily_std):
        short_only_sharpe = (short_only_daily_mean / short_only_daily_std) * np.sqrt(252)
    else:
        short_only_sharpe = np.nan
    # 计算只做空最大回撤
    short_only_max_drawdown = calc_max_drawdown(short_only_cumulative) * 100

    # 4. 基准: 买入持有指数
    daily_bh_returns = close_clean.pct_change().fillna(0)
    bh_cumulative = (1 + daily_bh_returns).cumprod()
    bh_cumulative_returns = (bh_cumulative - 1) * 100
    bh_cumulative_returns = bh_cumulative_returns.reset_index()
    bh_cumulative_returns.columns = ['date', 'cumulative_return']
    # 计算基准统计指标
    bh_daily_mean = daily_bh_returns.mean()
    bh_daily_std = daily_bh_returns.std()
    bh_annual = bh_daily_mean * 252 * 100
    if bh_daily_std != 0 and not np.isnan(bh_daily_std):
        bh_sharpe = (bh_daily_mean / bh_daily_std) * np.sqrt(252)
    else:
        bh_sharpe = np.nan
    # 计算基准最大回撤
    benchmark_max_drawdown = calc_max_drawdown(bh_cumulative) * 100

    ls_daily_return_std = daily_ls_returns.std()

    # 年化收益 (假设252交易日)
    # 多空单日平均收益 × 252交易日 = 年化收益
    ls_annual_return = ls_daily_return_mean * 252 * 100
    # 夏普比率 (年化)
    # 年化夏普 = (单日平均收益 / 单日标准差) × sqrt(252)
    if ls_daily_return_std != 0 and not np.isnan(ls_daily_return_std):
        ls_sharpe = (ls_daily_return_mean / ls_daily_return_std) * np.sqrt(252)
    else:
        ls_sharpe = np.nan

    # 检验单调性：计算分组收益与分组序号的相关系数
    group_ids = list(group_returns.keys())
    returns = list(group_returns.values())
    if len(returns) >= 3:
        mono_corr, _ = calc_spearmanr(pd.Series(group_ids), pd.Series(returns))
    else:
        mono_corr = np.nan

    return {
        'group_returns': group_returns,
        'group_returns_std': group_returns_std,
        'group_counts': group_counts,
        # 多空策略
        'long_short_return': ls_daily_return_mean,
        'long_short_annual': ls_annual_return,
        'long_short_sharpe': ls_sharpe,
        'long_short_max_drawdown': ls_max_drawdown,
        # 只做多策略
        'long_only_return': long_only_daily_mean,
        'long_only_annual': long_only_annual,
        'long_only_sharpe': long_only_sharpe,
        'long_only_max_drawdown': long_only_max_drawdown,
        # 只做空策略
        'short_only_return': short_only_daily_mean,
        'short_only_annual': short_only_annual,
        'short_only_sharpe': short_only_sharpe,
        'short_only_max_drawdown': short_only_max_drawdown,
        # 基准买入持有
        'benchmark_return': bh_daily_mean,
        'benchmark_annual': bh_annual,
        'benchmark_sharpe': bh_sharpe,
        'benchmark_max_drawdown': benchmark_max_drawdown,
        # 累积收益曲线
        'monotonic_corr': mono_corr,
        'cumulative_returns': ls_cumulative_returns,
        'long_only_cumulative': long_only_cumulative_returns,
        'short_only_cumulative': short_only_cumulative_returns,
        'benchmark_cumulative': bh_cumulative_returns,
    }


def evaluate_horizon(factor_values: pd.Series, close: pd.Series, horizon: int) -> dict:
    """评价单个预测周期"""
    # 计算未来horizon日的收益
    forward_return = close.pct_change(horizon).shift(-horizon)
    ic_metrics = calc_ic(factor_values, forward_return)
    # 每个周期都做分层分析
    quantile_res = quantile_analysis(factor_values, forward_return, close, n_quantiles=5, horizon=horizon)
    return {**ic_metrics, **quantile_res}


def evaluate_by_year(factor_values: pd.Series, close: pd.Series, horizon: int) -> dict:
    """按年份分组评价"""
    forward_return = close.pct_change(horizon).shift(-horizon)
    results_by_year = {}

    # 按年份分组
    for year in factor_values.index.year.unique():
        mask = (factor_values.index.year == year) & ~factor_values.isna() & ~forward_return.isna()
        if mask.sum() < 10:  # 数据太少跳过
            continue

        ic, _ = calc_spearmanr(factor_values[mask], forward_return[mask])
        results_by_year[str(year)] = {
            'ic': ic,
            'count': int(mask.sum())
        }

    return results_by_year


def evaluate_factor(factor_values: pd.Series, df_data: pd.DataFrame, params: dict = None) -> dict:
    """
    综合评价因子，在多个预测周期上评价

    参数
    ----------
    factor_values : pd.Series
        因子值，索引与df_data一致
    df_data : pd.DataFrame
        原始数据，包含close列
    params : dict, optional
        因子计算使用的参数，默认 None

    返回
    -------
    dict
        评价指标字典，按周期组织
    """
    close = df_data['close']

    # 评价多个预测周期
    horizons = [1, 5, 10, 20]
    results_by_horizon = {}

    # 因子描述性统计
    valid_factor = factor_values.dropna()
    factor_stats = {
        'mean': valid_factor.mean(),
        'std': valid_factor.std(),
        'min': valid_factor.min(),
        'max': valid_factor.max(),
        'skew': valid_factor.skew(),
        'kurt': valid_factor.kurtosis(),
        'count': int(len(valid_factor)),
        'na_count': factor_values.isna().sum()
    }

    # 因子一阶自相关性（衡量因子稳定性）
    if len(valid_factor) > 2:
        lag1 = factor_values.shift(1)
        autocorr, _ = calc_spearmanr(factor_values, lag1)
    else:
        autocorr = np.nan
    factor_stats['autocorr_lag1'] = autocorr

    for h in horizons:
        results_by_horizon[h] = evaluate_horizon(factor_values, close, h)
        # 添加按年评价
        results_by_horizon[h]['by_year'] = evaluate_by_year(factor_values, close, h)

    # 数据区间信息
    data_start = str(valid_factor.index[0].date())
    data_end = str(valid_factor.index[-1].date())
    years = round(len(valid_factor) / 252, 1)

    if len(valid_factor) < 20:
        return {
            'error': '有效数据过少，无法进行评价'
        }

    return {
        'horizons': results_by_horizon,
        'factor_stats': factor_stats,
        'data_period': {
            'start': data_start,
            'end': data_end,
            'years': years
        },
        'params': params if params is not None else {}
    }


def generate_evaluation_markdown(eval_result: dict, factor_name: str, output_dir: str) -> str:
    """生成markdown格式的评价结果"""
    if 'error' in eval_result:
        return f"""## 因子评价

**错误**: {eval_result['error']}
"""

    md = f"""# 因子{factor_name}评价

## 基本信息

- 数据区间: {eval_result['data_period']['start']} 至 {eval_result['data_period']['end']}
- 有效样本数: {eval_result['factor_stats']['count']}
- 缺失值数量: {eval_result['factor_stats']['na_count']}

"""

    # 添加参数信息 - 始终显示
    params = eval_result.get('params', {})
    md += "## 因子参数\n\n"
    if params:
        for key, value in params.items():
            md += f"- `{key}` = {value}\n"
    else:
        md += "- 该因子没有参数\n"
    md += "\n"

    # 添加因子分布统计，需要保持f-string格式化
    md += f"""## 因子分布统计

| 统计量 | 数值 |
| -------- | ------ |
| 均值 | {eval_result['factor_stats']['mean']:.4f} |
| 标准差 | {eval_result['factor_stats']['std']:.4f} |
| 最小值 | {eval_result['factor_stats']['min']:.4f} |
| 最大值 | {eval_result['factor_stats']['max']:.4f} |
| 偏度 | {eval_result['factor_stats']['skew']:.4f} |
| 峰度 | {eval_result['factor_stats']['kurt']:.4f} |
| 一阶自相关性 | {eval_result['factor_stats']['autocorr_lag1']:.4f} |
"""
    md += """
> **指标说明**:
>
> - **偏度 (Skewness)**: 衡量分布不对称程度
>   - 偏度 > 0 → 右偏，长尾在右，大多数因子值偏小
>   - 偏度 < 0 → 左偏，长尾在左，大多数因子值偏大
>   - 偏度 ≈ 0 → 分布对称，接近正态分布
> - **峰度 (Kurtosis)**: 超额峰度（以正态分布为基准），衡量尾部厚度
>   - 峰度 > 0 → 比正态分布更陡峭，尾部更厚，更容易出现极端值
>   - 峰度 < 0 → 比正态分布更平缓，尾部更薄，极端值较少
>   - 峰度 ≈ 0 → 尾部厚度与正态分布相当
> - **一阶自相关性 (Lag-1 Autocorrelation)**: 因子值与其滞后1日值的Spearman秩相关系数
>   - 自相关性越高 → 因子值变化越缓慢 → 因子换手率越低 → 交易成本越小
>   - 自相关性 > 0.9 → 因子非常稳定，适合低频交易
>   - 自相关性 < 0.5 → 因子变化很快，需要较高换手率

---

## IC分析 (秩相关系数)，按预测周期

| 周期 | 整体IC | 滚动IC均值 | ICIR | 样本数 |
| ------ | -------- | ------------ | ------ | -------- |
"""

    # 找出各指标绝对值最大的周期，只给它加粗
    max_ic_overall_abs = 0
    max_ic_overall_horizon = None
    max_ic_mean_abs = 0
    max_ic_mean_horizon = None
    max_icir_abs = 0
    max_icir_horizon = None

    for horizon, res in eval_result['horizons'].items():
        if not np.isnan(res['ic_overall']):
            ic_overall_abs = abs(res['ic_overall'])
            if ic_overall_abs > max_ic_overall_abs:
                max_ic_overall_abs = ic_overall_abs
                max_ic_overall_horizon = horizon
        if not np.isnan(res['ic_mean']):
            ic_mean_abs = abs(res['ic_mean'])
            if ic_mean_abs > max_ic_mean_abs:
                max_ic_mean_abs = ic_mean_abs
                max_ic_mean_horizon = horizon
        if not np.isnan(res['icir']):
            icir_abs = abs(res['icir'])
            if icir_abs > max_icir_abs:
                max_icir_abs = icir_abs
                max_icir_horizon = horizon

    for horizon, res in eval_result['horizons'].items():
        # 整体IC，只有最大值加粗
        if not np.isnan(res['ic_overall']):
            if horizon == max_ic_overall_horizon:
                ic_overall = f"**{res['ic_overall']:.4f}**"
            else:
                ic_overall = f"{res['ic_overall']:.4f}"
        else:
            ic_overall = "-"

        # 滚动IC均值，只有最大值加粗
        if not np.isnan(res['ic_mean']):
            if horizon == max_ic_mean_horizon:
                ic_mean = f"**{res['ic_mean']:.4f}**"
            else:
                ic_mean = f"{res['ic_mean']:.4f}"
        else:
            ic_mean = "-"

        # ICIR，只有最大值加粗
        if not np.isnan(res['icir']):
            if horizon == max_icir_horizon:
                icir = f"**{res['icir']:.3f}**"
            else:
                icir = f"{res['icir']:.3f}"
        else:
            icir = "-"

        md += f"| {horizon}日 | {ic_overall} | {ic_mean} | {icir} | {res['count']} |\n"

    md += """
> **指标说明**:
>
> - **整体IC**: **信息系数 (Information Coefficient)**，因子值与未来horizon日收益的Spearman秩相关系数
>   - IC ∈ [-1, 1]，绝对值越大，因子预测能力越强
>   - IC > 0 → 因子值越大，未来收益越高；IC < 0 → 因子值越大，未来收益越低（适合做空）
>   - 判断标准: |IC| < 0.03 很弱；0.03~0.05 较弱；0.05~0.1 良好；|IC| > 0.1 很强
> - **滚动IC均值**: 使用 **252交易日**（约1年）滚动窗口计算的IC均值，衡量因子预测能力的时间稳定性
> - **ICIR**: 信息系数比率 = `滚动IC均值 / 滚动IC标准差`，衡量预测能力的稳定性
>   - 绝对值越大，预测能力越稳定
>   - 判断标准: |ICIR| < 0.3 不稳定；0.3~0.5 一般；**|ICIR| > 0.5** → **良好**；|ICIR| > 0.7 → 优秀

---

## 分组收益汇总

| 预测周期 | 第1组(最小因子) | 第2组 | 第3组 | 第4组 | 第5组(最大因子) | 收益单调性 |
| ------- | -------------- | ----- | ----- | ----- | -------------- | ---------- |
"""

    for horizon, res in sorted(eval_result['horizons'].items()):
        if 'group_returns' in res and res['group_returns']:
            row = f"| {horizon}日 |"
            # 找出收益最高的分组
            returns = [(q, ret) for q, ret in res['group_returns'].items()]
            max_q = max(returns, key=lambda x: x[1])[0]
            for q in sorted(res['group_returns'].keys()):
                ret = res['group_returns'][q]*100
                cnt = res['group_counts'][q]
                if q == max_q:
                    row += f" **{ret:.4f}% (n={cnt})** |"
                else:
                    row += f" {ret:.4f}% (n={cnt}) |"
            if 'monotonic_corr' in res and not pd.isna(res['monotonic_corr']):
                row += f" {res['monotonic_corr']:.3f} |"
            else:
                row += " - |"
            md += row + "\n"

    md += """
> **指标说明**:
>
> - **第1~5组**: 将因子值从小到大等分为5组，表格中显示每组平均单日收益率 (样本数量)
> - **收益单调性**: 分组序号与组内平均收益的Spearman秩相关系数
>   - 越接近+1 → 收益随因子值增大单调递增
>   - 越接近-1 → 收益随因子值增大单调递减
>   - 接近0 → 收益没有明显单调性

---

## 多空策略收益汇总

| 预测周期 | 单日收益 | 年化收益 | 夏普比率 | 最大回撤 |
| -------- | -------- | -------- | -------- | -------- |
"""

    # 先收集所有有效值，找出每列的最大值（越大越好，最大回撤也是越大越好因为-10% > -20%）
    daily_vals = []
    annual_vals = []
    sharpe_vals = []
    drawdown_vals = []
    for horizon, res in sorted(eval_result['horizons'].items()):
        if 'group_returns' in res and res['group_returns']:
            val = res['long_short_return']*100 if not pd.isna(res['long_short_return']) else None
            if val is not None:
                daily_vals.append(val)
            val = res['long_short_annual'] if 'long_short_annual' in res and not pd.isna(res['long_short_annual']) else None
            if val is not None:
                annual_vals.append(val)
            val = res['long_short_sharpe'] if 'long_short_sharpe' in res and not pd.isna(res['long_short_sharpe']) else None
            if val is not None:
                sharpe_vals.append(val)
            val = res['long_short_max_drawdown'] if 'long_short_max_drawdown' in res and not pd.isna(res['long_short_max_drawdown']) else None
            if val is not None:
                drawdown_vals.append(val)

    max_daily = max(daily_vals) if daily_vals else None
    max_annual = max(annual_vals) if annual_vals else None
    max_sharpe = max(sharpe_vals) if sharpe_vals else None
    max_drawdown = max(drawdown_vals) if drawdown_vals else None

    # 输出表格，最大值加粗
    for horizon, res in sorted(eval_result['horizons'].items()):
        if 'group_returns' in res and res['group_returns']:
            row = f"| {horizon}日 |"
            # 单日收益
            val = res['long_short_return']*100 if not pd.isna(res['long_short_return']) else None
            if val is not None:
                if max_daily is not None and abs(val - max_daily) < 1e-8:
                    row += f" **{val:.4f}%** |"
                else:
                    row += f" {val:.4f}% |"
            else:
                row += " - |"
            # 年化收益
            val = res['long_short_annual'] if 'long_short_annual' in res and not pd.isna(res['long_short_annual']) else None
            if val is not None:
                if max_annual is not None and abs(val - max_annual) < 1e-8:
                    row += f" **{val:.2f}%** |"
                else:
                    row += f" {val:.2f}% |"
            else:
                row += " - |"
            # 夏普比率
            val = res['long_short_sharpe'] if 'long_short_sharpe' in res and not pd.isna(res['long_short_sharpe']) else None
            if val is not None:
                if max_sharpe is not None and abs(val - max_sharpe) < 1e-8:
                    row += f" **{val:.2f}** |"
                else:
                    row += f" {val:.2f} |"
            else:
                row += " - |"
            # 最大回撤
            val = res['long_short_max_drawdown'] if 'long_short_max_drawdown' in res and not pd.isna(res['long_short_max_drawdown']) else None
            if val is not None:
                if max_drawdown is not None and abs(val - max_drawdown) < 1e-8:
                    row += f" **{val:.2f}%** |"
                else:
                    row += f" {val:.2f}% |"
            else:
                row += " - |"
            md += row + "\n"

    md += """
> **指标说明**:
>
> - **多空策略**: 做多最大因子分组 + 做空最小因子分组
> - **单日收益**: 平均单日收益率
> - **年化收益**: 年化收益率 = 单日收益 × 252交易日
> - **夏普比率**: 年化夏普比率 = (单日平均收益 / 单日标准差) × √252
> - **最大回撤**: 策略持有期内的最大回撤百分比
>
"""

    # 为每个有累积收益数据的周期生成图表
    for horizon, res in sorted(eval_result['horizons'].items()):
        if 'cumulative_returns' in res and res['cumulative_returns'] is not None and not res['cumulative_returns'].empty:
            try:
                ls_cum_ret = res['cumulative_returns']
                lo_cum_ret = res.get('long_only_cumulative', None)
                so_cum_ret = res.get('short_only_cumulative', None)
                bh_cum_ret = res['benchmark_cumulative']

                # 创建图表
                fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
                # 画四种曲线
                # 1. 多空策略
                ax.plot(ls_cum_ret['date'], ls_cum_ret['cumulative_return'],
                       linewidth=2, color='#2c3e50', label='Long-Short')
                # 2. 只做多策略 (红色)
                if lo_cum_ret is not None and not lo_cum_ret.empty:
                    ax.plot(lo_cum_ret['date'], lo_cum_ret['cumulative_return'],
                           linewidth=1.5, color='#e74c3c', linestyle='-', label='Long Only')
                # 3. 只做空策略 (绿色)
                if so_cum_ret is not None and not so_cum_ret.empty:
                    ax.plot(so_cum_ret['date'], so_cum_ret['cumulative_return'],
                           linewidth=1.5, color='#27ae60', linestyle='-', label='Short Only')
                # 4. 基准（买入持有指数）(蓝色虚线)
                if bh_cum_ret is not None and not bh_cum_ret.empty:
                    ax.plot(bh_cum_ret['date'], bh_cum_ret['cumulative_return'],
                           linewidth=1.5, color='#3498db', linestyle='--', label='Benchmark (Buy & Hold)')

                # 美化图表
                ax.set_title(f'Cumulative Return Comparison (Horizon = {horizon} days)', fontsize=12)
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('Cumulative Return (%)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')

                # 旋转x轴标签避免重叠
                plt.xticks(rotation=30)
                plt.tight_layout()

                # 保存图片
                chart_path = os.path.join(output_dir, f'cumulative_return_h{horizon}.png')
                plt.savefig(chart_path, dpi=150, bbox_inches='tight')
                plt.close()

                # 在markdown中添加图片引用
                md += f'### 累积收益率曲线对比 ({horizon}日周期)\n\n'
                md += f'![累积收益率曲线](cumulative_return_h{horizon}.png)\n\n'
            except Exception as e:
                # 画图失败不中断整个评价流程
                print(f"警告: 画图失败 (horizon={horizon}): {e}")
                pass

    md += """## 分年度IC分析 (最佳预测周期)

| 年份 | IC | 样本数 |
| ------ | ---- | -------- |
"""

    # 获取第一个有数据的周期的分年结果
    best_h = None
    max_icir_abs = 0
    for h, res in eval_result['horizons'].items():
        if not np.isnan(res['icir']) and abs(res['icir']) > max_icir_abs:
            max_icir_abs = abs(res['icir'])
            best_h = h

    if best_h is not None and 'by_year' in eval_result['horizons'][best_h]:
        by_year = eval_result['horizons'][best_h]['by_year']
        # 找出IC绝对值最大的年份，只给它加粗
        max_ic_abs = 0
        max_year = None
        for year, stats in by_year.items():
            if not np.isnan(stats['ic']):
                if abs(stats['ic']) > max_ic_abs:
                    max_ic_abs = abs(stats['ic'])
                    max_year = year

        for year, stats in sorted(by_year.items()):
            if not np.isnan(stats['ic']):
                if str(year) == str(max_year):
                    ic_val = f"**{stats['ic']:.4f}**"
                else:
                    ic_val = f"{stats['ic']:.4f}"
            else:
                ic_val = "-"
            md += f"| {year} | {ic_val} | {stats['count']} |\n"

    md += """
## 评价总结

"""

    if best_h is not None:
        best_icir = eval_result['horizons'][best_h]['icir']
        if abs(best_icir) >= 0.5:
            quality = "**良好**"
        elif abs(best_icir) >= 0.3:
            quality = "**中等**"
        else:
            quality = "较弱"
        md += f"- 最佳预测周期为{best_h}日，ICIR = {best_icir:.3f}，因子预测能力{quality}\n"

    return md


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='因子评价脚本：计算因子IC、ICIR、分层收益、累积收益率曲线等评价指标',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例: python scripts/evaluate_factor.py factors/rsi --params \'{"N": 14}\' -b 20140101 -e 20201231'
    )
    parser.add_argument('factor_file', help='因子文件路径，例如 factors/rsi 或 factors/rsi/rsi.py')
    parser.add_argument('-p', '--params', default='{}', help='JSON格式的参数字典，例如 \'{"N": 14}\'，默认为空字典')
    parser.add_argument('-b', '--begin', default='20140101', help='开始日期，格式 YYYYMMDD，默认为 20140101')
    parser.add_argument('-e', '--end', default='20201231', help='结束日期，格式 YYYYMMDD，默认为 20201231')

    args = parser.parse_args()

    factor_file = args.factor_file
    if not factor_file.endswith('.py'):
        factor_file += '.py'

    params = json.loads(args.params)

    # 脚本目录 = 技能目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    skill_dir = os.path.dirname(script_dir)

    # 读取测试数据（在技能目录的 assets 下），文件命名格式：000905_yyyymmdd_yyyymmdd.csv
    csv_path = find_data_csv(os.path.join(skill_dir, 'assets'))
    df_test = pd.read_csv(csv_path, parse_dates=['date'])
    df_test = df_test.set_index('date')

    # 根据日期区间过滤
    begin_date = pd.to_datetime(args.begin, format='%Y%m%d')
    end_date = pd.to_datetime(args.end, format='%Y%m%d')
    df_test = df_test[(df_test.index >= begin_date) & (df_test.index <= end_date)]

    # 动态导入因子模块
    cwd = os.getcwd()
    factor_module_path = os.path.join(cwd, factor_file)
    if not os.path.exists(factor_module_path):
        print(f"错误: 因子文件不存在 {factor_module_path}")
        sys.exit(1)

    spec = importlib.util.spec_from_file_location(
        os.path.basename(factor_file).replace('.py', ''),
        factor_module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # 获取第一个类作为因子类
    factor_classes = [cls for name, cls in module.__dict__.items() if isinstance(cls, type)]
    if not factor_classes:
        print("错误: 未找到因子类")
        sys.exit(1)

    FactorClass = factor_classes[0]
    factor_name = FactorClass.__name__

    # 计算因子
    calculator = FactorClass(df_test)
    result = calculator.cal_continue(params)

    # 评价因子（多周期）
    eval_result = evaluate_factor(result, df_test, params)

    # 根据参数创建独立输出目录
    # 目录名称格式: eval_{begin}_{end}_params{params_hash}
    # 对params生成一个短hash，避免目录名过长
    import hashlib
    params_str = json.dumps(params, sort_keys=True)  # 排序保证相同参数生成相同hash
    params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]  # 取前8位足够区分
    eval_dir_name = f"eval_{args.begin}_{args.end}_{params_hash}"
    base_output_dir = os.path.dirname(factor_module_path)
    output_dir = os.path.join(base_output_dir, eval_dir_name)

    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 生成markdown，所有图表会自动保存到output_dir
    eval_md = generate_evaluation_markdown(eval_result, factor_name, output_dir)

    # 保存markdown文件到输出目录
    output_filename = os.path.basename(factor_file).replace('.py', '_evaluation.md')
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(eval_md)

    print(f"\n✓ 评价结果已保存到目录: {output_dir}")
    print(f"  - 评价文件: {output_path}")


if __name__ == "__main__":
    main()