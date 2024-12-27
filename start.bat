@echo off
cd /d "%~dp0"               :: Switch to the directory containing the batch file
cd env\Scripts              :: Navigate to the virtual environment scripts folder
call activate               :: Activate the virtual environment
cd ../..                    :: Navigate back to the project root
python app.py               :: Run the Python application@echo off
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
