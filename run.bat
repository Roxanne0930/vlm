@echo off
chcp 65001 > nul

set PROJECT_ROOT=%~dp0
cd /d "%PROJECT_ROOT%"

echo ====================================
echo   🚀 正在启动 VLM 图文助手...

:: 自动激活虚拟环境
call .venv\Scripts\activate
cd /d "%PROJECT_ROOT%"
set DASHSCOPE_API_KEY=sk-API密钥

echo   ✨ 系统正在后台初始化大模型与界面组件...
echo   🔗 提示：系统启动后将自动为您打开网页
echo   🌐 手动访问网址: http://127.0.0.1:7860
echo ==================================================
echo.

:: 延迟 8 秒后，自动在默认浏览器中打开系统网址
start /b cmd /c "ping 127.0.0.1 -n 8 >nul && start http://127.0.0.1:7860"

python ui/app.py

pause