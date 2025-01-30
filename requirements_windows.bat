@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo Starting installation process...
echo ========================================
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv gaze_mouse_env
if not exist "gaze_mouse_env\Scripts\activate.bat" (
    echo Error: Virtual environment creation failed.
    exit /b 1
)
echo Virtual environment created successfully.
echo.

REM Activate virtual environment using full path
echo Activating virtual environment...
call "%~dp0gaze_mouse_env\Scripts\activate.bat"
if errorlevel 1 (
    echo Error: Failed to activate virtual environment.
    exit /b 1
)
echo Virtual environment activated successfully.
echo.

REM Use python -m pip to ensure we're using the correct pip
echo Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
echo Base packages upgraded successfully.
echo.

REM Install core dependencies with version specifications
echo Installing core dependencies...
echo.

echo Installing numpy...
python -m pip install numpy>=1.21.0
echo.

echo Installing OpenCV...
python -m pip install opencv-python>=4.5.0
echo.

echo Installing mediapipe...
python -m pip install mediapipe
echo.

echo Installing pyautogui...
python -m pip install pyautogui>=0.9.53
echo.

echo Installing albumentations...
python -m pip install albumentations==1.3.0
echo.

echo Installing PyTorch and torchvision...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo.

echo Installing ultralytics...
python -m pip install ultralytics>=8.0.0
echo.

echo Installing roboflow...
python -m pip install roboflow>=1.1.0
echo.

echo Installing statistics...
python -m pip install statistics>=1.0.3.5
echo.

echo Installing python-dotenv...
python -m pip install python-dotenv
echo.

echo Installing matplotlib...
python -m pip install matplotlib==3.4.3
echo.

echo Core dependencies installed.
echo.

REM Install additional dependencies for better performance
echo Installing additional dependencies...
echo.

echo Installing OpenCV contrib modules...
python -m pip install opencv-contrib-python>=4.5.0
echo.

echo Installing Pillow...
python -m pip install pillow>=8.0.0
echo.

echo Additional dependencies installed.
echo.

REM Create necessary directories
echo Creating project directories...
if not exist "models" mkdir models
if not exist "data" mkdir data
echo Project directories created.
echo.

REM Check if installation was successful
echo Checking dependencies...
python -c "import cv2; import mediapipe; import torch; import albumentations" && (
    echo.
    echo ========================================
    echo Dependency check passed.
    echo Installation completed successfully.
    echo ========================================
) || (
    echo.
    echo ========================================
    echo Error: Some dependencies failed to install correctly.
    echo Please check the error messages above.
    echo ========================================
)

echo.
echo Important instructions:
echo - To activate the environment, use: gaze_mouse_env\Scripts\activate
echo - To start the eye tracking, run: python eye_tracking_module.py
echo.
echo Press any key to exit...
pause >nul
