@echo off
setlocal
cd /d "%~dp0"
echo Starting SkinGuardAI Streamlit Interface on Port 8501...
.venv\Scripts\streamlit.exe run app.py --server.port 8501

