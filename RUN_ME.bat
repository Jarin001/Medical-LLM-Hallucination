@echo off
REM Double-click this file. It runs everything and saves the output to a text file.
REM No typing required.

setlocal
set PROJ=%~dp0
set PY=%PROJ%.venv\Scripts\python.exe
set OUT=%PROJ%results\FULL_OUTPUT.txt

if not exist "%PY%" (
  echo.
  echo ERROR: Python environment not found at:
  echo   %PY%
  echo.
  pause
  exit /b 1
)

echo Running MedHallu-lite. This takes about a minute.
echo Output is being saved to:
echo   %OUT%
echo.

> "%OUT%" (
  echo ==========================================================
  echo  STEP 1 of 4 - THE DATASET
  echo  What it shows: 10,000 question/answer pairs downloaded,
  echo  and the counts of easy/medium/hard lies.
  echo ==========================================================
)
echo [1 of 4] Looking at the dataset...
"%PY%" "%PROJ%src\data.py" --config pqa_labeled >> "%OUT%" 2>&1

>> "%OUT%" (
  echo.
  echo ==========================================================
  echo  STEP 2 of 4 - HOW A LIE GETS ITS DIFFICULTY LABEL
  echo  What it shows: one question walked through all four
  echo  steps of the pipeline that built the dataset.
  echo ==========================================================
)
echo [2 of 4] Showing how the dataset was built...
"%PY%" "%PROJ%src\generate_demo.py" --dry-run --n 2 >> "%OUT%" 2>&1

>> "%OUT%" (
  echo.
  echo ==========================================================
  echo  STEP 3 of 4 - THE DUMB BASELINE
  echo  What it shows: a program that ignores the question and
  echo  always outputs "fake" still scores F1 = 0.667.
  echo  Most models in the paper score BELOW this.
  echo ==========================================================
)
echo [3 of 4] Running the cheating baseline...
"%PY%" "%PROJ%src\detect.py" --backend constant --limit 300 >> "%OUT%" 2>&1
"%PY%" "%PROJ%src\detect.py" --backend lexical --limit 300 --both >> "%OUT%" 2>&1

>> "%OUT%" (
  echo.
  echo ==========================================================
  echo  STEP 4 of 4 - THE PAPER'S CLAIM THAT DID NOT HOLD UP
  echo  What it shows: the paper says hard lies sound MORE like
  echo  the truth. Measured three ways on the released data, it
  echo  comes out the opposite. See the "NO, reversed" lines.
  echo ==========================================================
)
echo [4 of 4] Checking the paper's semantic claim...
"%PY%" "%PROJ%src\semantics.py" --config pqa_labeled --embed >> "%OUT%" 2>&1

echo.
echo ==========================================================
echo  DONE.
echo.
echo  Full output saved to:
echo    %OUT%
echo ==========================================================
echo.
echo Opening the results file now...
timeout /t 2 >nul
start "" notepad "%OUT%"

echo.
pause
