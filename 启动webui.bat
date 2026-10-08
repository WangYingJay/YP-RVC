@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "ROOT=%~dp0"
set "PORT=7860"
set "PYTHONIOENCODING=utf-8"
set "NO_PROXY=localhost,127.0.0.1,0.0.0.0"
set "no_proxy=localhost,127.0.0.1,0.0.0.0"
set "PYTHONUTF8=1"
set "PYTHONPATH=%ROOT%"
set "RVC_DATASET_DIR=%ROOT%datasets"
set "RVC_LOG_DIR=%ROOT%logs"

if exist "%ROOT%.venv\Scripts\python.exe" (
  set "PY=%ROOT%.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo 训练页 http://127.0.0.1:%PORT%
echo 数据集 %RVC_DATASET_DIR%
echo 实验输出 %RVC_LOG_DIR%
"%PY%" "%ROOT%webui.py"
if errorlevel 1 (
  echo.
  echo 启动失败。请先在本目录安装依赖，或建立 .venv 后再运行。
  pause
)
