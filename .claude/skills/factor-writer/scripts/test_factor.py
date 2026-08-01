#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用因子测试脚本
用法: python <skill-path>/scripts/test_factor.py <factor-file-path> <JSON参数字典>
示例: python ~/.claude/skills/factor-writer/scripts/test_factor.py factors/rsi '{"N": 14}'
"""

import os
import sys
import glob
import json
import importlib.util
import pandas as pd


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

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='通用因子测试脚本：验证因子代码可运行并保存因子值',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='示例: python scripts/test_factor.py factors/rsi --params \'{"N": 14}\' -b 20140101 -e 20201231'
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

    # 动态导入因子模块（因子在用户当前目录的 factors 下）
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
    factor_classes = [cls for name, cls in module.__dict__.items()
                      if isinstance(cls, type)]
    if not factor_classes:
        print("错误: 未找到因子类")
        sys.exit(1)

    FactorClass = factor_classes[0]

    # 计算因子
    calculator = FactorClass(df_test)
    result = calculator.cal_continue(params)

    # 打印结果
    print(f"✓ 因子 '{FactorClass.__name__}' 计算完成")
    print(f"  参数: {params}")
    print(f"  数据总行数: {len(df_test)}")
    print(f"  非空因子值: {result.dropna().shape[0]}")
    print("\n前10行结果:")
    print(result.dropna().head(10))

    # 将因子值保存到CSV文件，日期格式使用YYYYMMDD整数
    factor_name = os.path.basename(factor_file).replace('.py', '')
    output_csv = os.path.join(os.path.dirname(factor_module_path), f"{factor_name}.csv")

    # 重建DataFrame，将日期转换为YYYYMMDD整数格式
    df_result = result.reset_index()
    df_result['date'] = df_result['date'].dt.strftime('%Y%m%d').astype(int)
    df_result = df_result.set_index('date')
    df_result.columns = [factor_name]
    df_result.to_csv(output_csv, header=True)

    print(f"\n✓ 因子值已保存到: {output_csv} (日期格式: YYYYMMDD)")

if __name__ == "__main__":
    main()
