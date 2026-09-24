@echo off
rem Windows launcher (double-click in Explorer).
rem Switches to repo root, checks data and dependencies, then starts the app.

rem Switch console to UTF-8 so the Chinese app name renders correctly
chcp 65001 >nul

rem Repo root = parent of this file's directory (app\)
cd /d "%~dp0\.."

echo === 姜砚尊最聪明最帅 (Windows) ===

rem 1. Check data files (not distributed via git; any one complete market suffices)
set "HAVE_A="
set "HAVE_US="
if exist "data\backtest\a_share\daily_stocks.parquet" if exist "data\backtest\a_share\daily_indices.parquet" set "HAVE_A=1"
if exist "data\backtest\us\daily_stocks.parquet" if exist "data\backtest\us\daily_indices.parquet" set "HAVE_US=1"
if defined HAVE_A goto findpy
if defined HAVE_US goto findpy
goto nodata

:nodata
echo.
echo [ERROR] No market data found. Copy the two parquet files of either market to:
echo   data\backtest\a_share\daily_stocks.parquet
echo   data\backtest\a_share\daily_indices.parquet
echo   or:
echo   data\backtest\us\daily_stocks.parquet
echo   data\backtest\us\daily_indices.parquet
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

rem 3. Install dependencies on first run. Import the app modules themselves:
rem    their import chain covers every runtime dependency, so adding a new
rem    dependency only requires updating requirements.txt -- not this launcher.
%PY% -c "import sys; sys.path.insert(0, 'app'); import streamlit; import generate_report; import report_pdf" >nul 2>nul
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
