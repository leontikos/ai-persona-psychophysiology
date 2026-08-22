@echo off
REM Launches the research chatbot on Windows from the conda environment named
REM "chatbot". Create it once with:
REM     conda create -n chatbot python=3.11
REM     conda activate chatbot
REM     pip install -r requirements.txt

echo Starting Research Chatbot...

cd /d "%~dp0"

REM Locate Anaconda / Miniconda: registry first, then the usual install paths.
FOR /F "tokens=2* skip=2" %%a in ('reg query "HKCU\Software\Anaconda\Navigator" /v "InstallPath" 2^>nul') do set "ANACONDA_PATH=%%b"
if not defined ANACONDA_PATH (
    FOR /F "tokens=2* skip=2" %%a in ('reg query "HKLM\Software\Anaconda\Navigator" /v "InstallPath" 2^>nul') do set "ANACONDA_PATH=%%b"
)
if not defined ANACONDA_PATH (
    if exist "%USERPROFILE%\Anaconda3" (
        set "ANACONDA_PATH=%USERPROFILE%\Anaconda3"
    ) else if exist "%USERPROFILE%\miniconda3" (
        set "ANACONDA_PATH=%USERPROFILE%\miniconda3"
    ) else if exist "C:\ProgramData\Anaconda3" (
        set "ANACONDA_PATH=C:\ProgramData\Anaconda3"
    ) else if exist "C:\Anaconda3" (
        set "ANACONDA_PATH=C:\Anaconda3"
    )
)

if not defined ANACONDA_PATH (
    echo Error: Could not find an Anaconda or Miniconda installation.
    pause
    exit /b 1
)

echo Found Anaconda at: %ANACONDA_PATH%

call "%ANACONDA_PATH%\Scripts\activate.bat"
call conda activate chatbot
streamlit run streamlit_app.py

REM Keep the window open if Streamlit exits unexpectedly.
pause
