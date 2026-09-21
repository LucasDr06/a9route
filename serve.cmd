@echo off
REM ============================================================================
REM  双击这个文件 = 起本地 Web 窗口（http://127.0.0.1:8790/）
REM
REM  它其实就干一件事：调 main.py —— 也就是
REM      a9route.cli.main() -> cmd_serve() -> web.app.main() -> app.run()
REM  **服务就跑在这个窗口的进程里**（阻塞），所以：
REM    * 这个窗口别关；关了服务就没了，浏览器会变成"拒绝连接"
REM    * 想停就 Ctrl+C
REM    * 换端口：在本窗口里手敲  python main.py serve --port 8800
REM
REM  为什么需要这个文件：从"别的程序/会话"里（比如 AI 助手的命令执行环境）
REM  起的进程，会在那条命令结束时被**连整棵进程树一起结束** ——
REM  表现就是"刚才还能用，突然拒绝连接"。自己这个窗口起的最稳。
REM ============================================================================
cd /d "%~dp0"
set "PY=C:\Users\Admin\miniconda3\envs\alphash9auto\python.exe"
if not exist "%PY%" set "PY=python"
echo 用 %PY% 启动 a9route（端口 8790）…
echo.
"%PY%" main.py
echo.
echo 服务已退出。按任意键关窗口。
pause >nul
