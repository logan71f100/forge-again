@echo off
rem ============================================================================
rem forge-again launcher (Windows 10/11) -- fully self-contained.
rem First run downloads a portable Python 3.12, builds the venv and installs
rem PyTorch (CUDA 12.6) + all dependencies. Later runs skip finished steps.
rem Usage:  start.bat [sd^|xl^|flux]
rem Env:    FORGE_MODELS_DIR  models folder   (default: .\models)
rem         FORGE_PORT        UI port         (default: 7860)
rem ============================================================================
setlocal EnableExtensions
cd /d "%~dp0"

if "%FORGE_MODELS_DIR%"=="" set "FORGE_MODELS_DIR=%~dp0models"
if "%FORGE_PORT%"=="" set "FORGE_PORT=7860"
set "PYDIR=%~dp0python"
set "VENV=%~dp0venv"
set "STAMP=%VENV%\.deps_installed"
set "PYURL=https://github.com/astral-sh/python-build-standalone/releases/download/20260718/cpython-3.12.13+20260718-x86_64-pc-windows-msvc-install_only.tar.gz"
set "TORCH_CMD=torch==2.13.0+cu126 torchvision==0.28.0+cu126 --index-url https://download.pytorch.org/whl/cu126"

rem ------------------------------------------------------------- mode select
set "MODE=%~1"
if "%MODE%"=="" if exist "%~dp0current_mode.txt" set /p MODE=<"%~dp0current_mode.txt"
rem no argument and no saved mode: default to sd -- modes switch in one click
rem from the UI once running, so there is no reason to block on a menu here
if "%MODE%"=="" set "MODE=sd"
if "%MODE%"=="1" goto sd
if /i "%MODE%"=="sd" goto sd
if "%MODE%"=="3" goto flux
if /i "%MODE%"=="flux" goto flux
goto xl

:sd
echo   -^> SD 1.5 mode
set "MODENAME=sd"
set "REPLACER_DEF_SAMPLER=DPM++ 2M"
set "REPLACER_DEF_SCHEDULER=Karras"
set "REPLACER_DEF_WIDTH=512"
set "REPLACER_DEF_HEIGHT=512"
set "REPLACER_DEF_STEPS=25"
set "REPLACER_DEF_CFG=7.0"
set "REPLACER_DEF_DENOISE=0.5"
set "REPLACER_FLUX_GUIDANCE=3.5"
goto common

:xl
echo   -^> SDXL mode
set "MODENAME=xl"
set "REPLACER_DEF_SAMPLER=DPM++ 2M"
set "REPLACER_DEF_SCHEDULER=Karras"
set "REPLACER_DEF_WIDTH=1024"
set "REPLACER_DEF_HEIGHT=1024"
set "REPLACER_DEF_STEPS=25"
set "REPLACER_DEF_CFG=5.0"
set "REPLACER_DEF_DENOISE=0.75"
set "REPLACER_FLUX_GUIDANCE=3.5"
goto common

:flux
echo   -^> Flux Fill mode
set "MODENAME=flux"
set "REPLACER_DEF_SAMPLER=Euler"
set "REPLACER_DEF_SCHEDULER=Simple"
set "REPLACER_DEF_WIDTH=1024"
set "REPLACER_DEF_HEIGHT=1024"
set "REPLACER_DEF_STEPS=20"
set "REPLACER_DEF_CFG=1.0"
set "REPLACER_DEF_DENOISE=1.0"
set "REPLACER_FLUX_GUIDANCE=30"
goto common

:common
set "REPLACER_DEF_MASK_EXPAND=15"
set "REPLACER_DEF_BOX_THRESHOLD=0.35"
set "REPLACER_DEF_MASK_BLUR=6"
set "REPLACER_DEF_PADDING=48"
set "REPLACER_DEF_FILL=original"

rem --------------------------------------------------------------- bootstrap
if not exist "%PYDIR%\python.exe" (
    echo [bootstrap] Downloading portable Python 3.12 ...
    curl -L --fail -o "%~dp0_py.tar.gz" "%PYURL%" || goto :fail
    echo [bootstrap] Extracting Python ...
    if exist "%~dp0_pytmp" rmdir /s /q "%~dp0_pytmp"
    mkdir "%~dp0_pytmp"
    tar -xzf "%~dp0_py.tar.gz" -C "%~dp0_pytmp" || goto :fail
    move "%~dp0_pytmp\python" "%PYDIR%" >nul || goto :fail
    rmdir /s /q "%~dp0_pytmp"
    del "%~dp0_py.tar.gz"
)

if not exist "%VENV%\Scripts\python.exe" (
    echo [bootstrap] Creating virtual environment ...
    "%PYDIR%\python.exe" -m venv "%VENV%" || goto :fail
)

if not exist "%STAMP%" (
    echo [bootstrap] Upgrading pip ...
    "%VENV%\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
    echo [bootstrap] Installing PyTorch for CUDA 12.6, large download ...
    "%VENV%\Scripts\python.exe" -m pip install %TORCH_CMD% || goto :fail
    echo [bootstrap] Installing requirements ...
    "%VENV%\Scripts\python.exe" -m pip install --no-build-isolation -r "%~dp0requirements_versions.txt" || goto :fail
    echo ok> "%STAMP%"
    echo [bootstrap] Environment ready.
)

rem --------------------------------------------------------------- configure
"%VENV%\Scripts\python.exe" "%~dp0set_mode.py" %MODENAME%

rem AI assistant vision model (~18GB, first run only; set FORGE_NO_LLM=1 to skip)
"%VENV%\Scripts\python.exe" "%~dp0download_llm.py"

rem extra launch arguments: one line in extra-args.txt (optional, next to this
rem script) and/or the FORGE_EXTRA_ARGS environment variable
set "EXTRA_ARGS="
if exist "%~dp0extra-args.txt" set /p EXTRA_ARGS=<"%~dp0extra-args.txt"

rem open the UI in the default browser once it is up (set FORGE_NO_BROWSER=1
rem to suppress, e.g. for headless/service use)
set "AUTOLAUNCH=--autolaunch"
if defined FORGE_NO_BROWSER set "AUTOLAUNCH="

:launch
set /p CKMODE=<"%~dp0current_mode.txt"
set "SD_WEBUI_RESTART=1"
set "HF_HOME=%FORGE_MODELS_DIR%\hf-cache"
"%VENV%\Scripts\python.exe" "%~dp0launch.py" --listen --port %FORGE_PORT% --api --cuda-malloc --no-half-vae --disable-xformers --skip-python-version-check --ckpt-dir "%FORGE_MODELS_DIR%\checkpoints\%CKMODE%" --lora-dir "%FORGE_MODELS_DIR%\Lora" --vae-dir "%FORGE_MODELS_DIR%\VAE" --text-encoder-dir "%FORGE_MODELS_DIR%\text_encoder" --esrgan-models-path "%FORGE_MODELS_DIR%\ESRGAN" %AUTOLAUNCH% %EXTRA_ARGS% %FORGE_EXTRA_ARGS%

rem UI-triggered restarts relaunch through this loop: mark them so the server
rem does not open another browser tab each time
if exist "%~dp0tmp\restart" ( del /q "%~dp0tmp\restart" & set "SD_WEBUI_RESTARTING=1" & goto launch )
exit /b 0

:fail
echo.
echo [bootstrap] SETUP FAILED. Fix the error above and re-run. Partial state is
echo [bootstrap] kept so a re-run resumes where it stopped.
exit /b 1
