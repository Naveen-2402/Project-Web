@echo off
echo Creating necessary directories...
if not exist "app\static\css" mkdir app\static\css
if not exist "app\static\js" mkdir app\static\js
if not exist "app\static\uploads" mkdir app\static\uploads
if not exist "app\static\results" mkdir app\static\results

echo Starting Vehicle Speed Detection Dashboard...
python run.py 