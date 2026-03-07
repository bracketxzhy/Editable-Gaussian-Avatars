@echo off
cd /d "d:\project\Projects-zhy\Editable Gaussian Avatars"
if exist mmcv_build.log del /f /q mmcv_build.log
call C:\Users\brack\anaconda3\condabin\conda.bat activate ega-gpu
python -m pip install --no-build-isolation --no-cache-dir mmcv-full==1.7.1 > mmcv_build.log 2>&1
exit /b %errorlevel%
