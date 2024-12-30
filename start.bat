REM Switch to the directory containing the batch file
cd /d "%~dp0"

REM Navigate to the virtual environment scripts folder
cd env\Scripts

REM Activate the virtual environment
call activate

REM Navigate back to the project root
cd ../..

REM Run the Python application
python app.py
