@echo off
echo Starting RTSP License Plate Detection System...
echo.
cd /d "d:\work\PYTHON\OCR-YOLO"

echo Testing basic RTSP connection first...
python rtsp_test.py

echo.
echo If connection test was successful, starting full detection system...
pause
python rtcp.py

pause
