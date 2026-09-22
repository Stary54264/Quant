@echo off
rem Windows launcher (double-click in Explorer).
rem Switches to repo root, checks data and dependencies, then starts the app.

rem Switch console to UTF-8 so the Chinese app name renders correctly
chcp 65001 >nul

rem Repo root = parent of this file's directory (app\)
cd /d "%~dp0\.."

echo === 姜砚尊最聪明最帅 (Windows) ===

rem 1. Check data files (not distributed via git; copy them manually)
if not exist "data\backtest\a_share\daily_stocks.parquet" goto nodata
if not exist "data\backtest\a_share\daily_indices.parquet" goto nodata
goto findpy

:nodata
echo.
echo [ERROR] Data files not found. Please copy the two parquet files to:
echo   data\backtest\a_share\daily_stocks.parquet
echo   data\backtest\a_share\daily_indices.parquet
echo.
pause
exit /b 1

rem 2. Locate a usable Python (try `python`, then the `py` launcher)
:findpy
set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY (
  where py >nul 2>nul && set "PY=py -3"
)
if not defined PY (
  echo [ERROR] Python not found. Install Python 3.10+ from https://www.python.org/downloads/
  echo         (tick "Add python.exe to PATH" during installation)
  pause
  exit /b 1
)

rem 3. Install dependencies on first run
%PY% -c "import streamlit, duckdb, pyarrow, pandas, numpy, scipy, reportlab, matplotlib" >nul 2>nul
if errorlevel 1 (
  echo Installing dependencies on first run: pip install -r requirements.txt
  %PY% -m pip install -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Run manually:
    echo         %PY% -m pip install -r requirements.txt
    pause
    exit /b 1
  )
)

rem 4. Start (press Ctrl+C or close this window to stop)
echo Starting... browser will open http://localhost:8501
%PY% -m streamlit run app\app.py
pause
