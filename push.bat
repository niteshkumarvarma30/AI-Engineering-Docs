@echo off
echo =============================================
echo   AI Engineering Docs - Sync to GitHub
echo =============================================
echo.

set /p msg="Enter commit message (or press Enter for 'Update documentation'): "
if "%msg%"=="" set msg=Update documentation

echo.
echo [1/3] Adding changes...
git add -A

echo.
echo [2/3] Committing changes...
git commit -m "%msg%"

echo.
echo [3/3] Pushing to GitHub...
git push origin main

echo.
echo =============================================
echo   Sync Complete!
echo =============================================
pause
