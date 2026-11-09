@echo off
cd /d "%~dp0ComfyUI"
echo ============================================
echo  MiniMax H3 Director (Qwen Prompt)
echo  Port: 8200
echo  Optimizer: Qwen3.8-VL GGUF
echo  Attention: Sage
echo ============================================
echo.
"%~dp0python\python.exe" main.py --listen 127.0.0.1 --port 8200 --use-sage-attention --auto-launch
pause
