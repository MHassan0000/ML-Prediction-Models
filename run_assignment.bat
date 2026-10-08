@echo off
echo ============================================================
echo  ML Assignment 2 - Run All Pipelines & Apps
echo  Author: Muhammad Hassan Yousaf (FA24-BSE-082)
echo ============================================================
echo.

echo [Step 1] Running Task 1 Pipeline (Titanic)...
python pipelines/Task1_Titanic_Survival_Prediction.py
if errorlevel 1 (
    echo ERROR: Task 1 failed!
    pause
    exit /b 1
)
echo Task 1 complete!
echo.

echo [Step 2] Running Task 2 Pipeline (Heart Disease)...
python pipelines/Task2_Heart_Disease_Prediction.py
if errorlevel 1 (
    echo ERROR: Task 2 failed!
    pause
    exit /b 1
)
echo Task 2 complete!
echo.

echo [Step 3] Regenerating Clean Jupyter Notebooks...
python generate_notebooks.py
echo Notebooks ready!
echo.

echo [Step 4] Launching Unified Streamlit App Hub...
python -m streamlit run app.py

pause
