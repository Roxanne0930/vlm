@echo off
chcp 65001 > nul

set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

echo ====================================
echo      🚀 正在启动 VLM 图文助手...
echo ====================================

:: 自动激活虚拟环境
call venv\Scripts\activate
cd /d "%PROJECT_ROOT%"
set DASHSCOPE_API_KEY=sk-自己的密钥

python ui/app.py

pause