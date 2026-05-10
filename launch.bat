@echo off
cd /d "%~dp0"

echo.
echo =======================================
echo   ODC Markets Research Agent
echo =======================================
echo.

:: Activate venv
if not exist ".venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found. Run this from the project folder.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat

:: Check streamlit is installed
where streamlit >nul 2>&1
if errorlevel 1 (
    echo Streamlit not found. Installing...
    pip install streamlit plotly
)

echo Checking app.py for errors...
python -c "import ast; ast.parse(open('app.py').read()); print('app.py OK')"
if errorlevel 1 (
    echo ERROR: app.py has a syntax error.
    pause
    exit /b 1
)

echo.
echo Starting Streamlit...
echo.

python -m streamlit run app.py --server.port 8501 --server.headless false 2>&1

echo.
echo Server stopped. See any errors above.
pause
