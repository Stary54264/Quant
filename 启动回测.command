#!/bin/bash
# 双击启动回测小程序：自动切到仓库目录、起 Streamlit 并打开浏览器。
cd "$(dirname "$0")"

if ! python3 -c "import streamlit" 2>/dev/null; then
  echo "首次运行，正在安装 streamlit…"
  python3 -m pip install streamlit || { echo "安装失败，请手动执行: python3 -m pip install streamlit"; read -r -k 1; exit 1; }
fi

python3 -m streamlit run app/app.py
