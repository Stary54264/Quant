# Python 环境（Windows）
- **解释器**: `C:\Users\86133\AppData\Local\Programs\Python\Python311\python.exe`（Python 3.11.9，winget 装的）
- **pip 镜像**: 默认可用清华源 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
- **PATH**: 未加到系统 PATH，命令行执行统一用绝对路径或 `python -m pkg`

## 已安装包
numpy · pandas · matplotlib · scipy · yfinance

## 运行本项目脚本
```bash
"C:/Users/86133/AppData/Local/Programs/Python/Python311/python.exe" tests/test_hurst.py
```

## 添加新依赖
```bash
"C:/Users/86133/AppData/Local/Programs/Python/Python311/python.exe" -m pip install <pkg> -i https://pypi.tuna.tsinghua.edu.cn/simple
```
