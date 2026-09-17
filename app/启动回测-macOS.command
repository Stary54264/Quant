#!/bin/bash
# macOS 启动器（在 Finder 中双击运行）。
# 自动切到仓库根目录，检查数据与依赖后启动小程序并打开浏览器。

# 仓库根目录 = 本文件所在目录（app/）的上一级
cd "$(dirname "$0")/.." || exit 1

echo "=== 姜砚尊最聪明最帅 (macOS) ==="

# 1. 检查数据文件（不随 git 分发，需手动拷贝）
if [ ! -f "data/backtest/a_share/daily_stocks.parquet" ] || \
   [ ! -f "data/backtest/a_share/daily_indices.parquet" ]; then
  echo ""
  echo "[错误] 未找到数据文件。请把两个 parquet 放到："
  echo "  data/backtest/a_share/daily_stocks.parquet"
  echo "  data/backtest/a_share/daily_indices.parquet"
  echo ""
  read -r -p "按回车键退出…" _
  exit 1
fi

# 2. 选择可用的 Python 3
if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  echo "[错误] 未找到 python3，请先安装 Python 3.10+（https://www.python.org/downloads/）"
  read -r -p "按回车键退出…" _
  exit 1
fi

# 3. 缺依赖时自动安装
if ! "$PY" -c "import streamlit, duckdb, pyarrow, pandas, numpy, scipy, reportlab" 2>/dev/null; then
  echo "首次运行，正在安装依赖（pip install -r requirements.txt）…"
  "$PY" -m pip install -r requirements.txt || {
    echo "[错误] 依赖安装失败，请手动执行：$PY -m pip install -r requirements.txt"
    read -r -p "按回车键退出…" _
    exit 1
  }
fi

# 4. 启动（Ctrl+C 或关闭本窗口停止服务）
echo "启动中，浏览器将自动打开 http://localhost:8501 …"
"$PY" -m streamlit run app/app.py
