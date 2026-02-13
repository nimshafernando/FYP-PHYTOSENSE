@echo off
REM PhytoSense Application - Complete Testing Suite Runner
REM This batch file starts Flask app and runs all tests with evidence generation

echo ========================================
echo PHYTOSENSE TESTING SUITE LAUNCHER 
echo ========================================
echo.

REM Change to project directory
cd /d "C:\Users\Asus\OneDrive\Desktop\medicinal_leaf_classifier"

echo 1️⃣ STARTING FLASK APPLICATION...
echo ----------------------------------------
start "Flask App" python flask_app.py

echo Waiting for Flask app to start...
timeout /t 5 /nobreak > nul

echo.
echo 2️⃣ EXECUTING COMPREHENSIVE TESTING SUITE...
echo ----------------------------------------
cd tests
python run_all_tests.py

echo.
echo 3️⃣ TESTING COMPLETE - CHECK RESULTS IN tests/test_reports/
echo ----------------------------------------
echo.
echo 📊 Key Evidence Files Generated:
echo   • comprehensive_testing_report.json
echo   • testing_evidence_summary.md  
echo   • testing_limitations_report.json
echo   • testing_limitations_summary.md
echo.
echo Press any key to view test reports folder...
pause >nul
explorer test_reports

echo.
echo ✅ PhytoSense Testing Suite Complete!
echo Check test_reports/ folder for all evidence files.
echo.
pause