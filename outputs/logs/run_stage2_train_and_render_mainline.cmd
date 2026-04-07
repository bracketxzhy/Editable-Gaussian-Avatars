@echo off
cd /d "d:\project\Projects-zhy\Editable Gaussian Avatars"
set PYTHONPATH=d:\project\Projects-zhy\Editable Gaussian Avatars\src
if not exist "outputs\logs" mkdir "outputs\logs"
echo [%date% %time%] START stage2 train>>"outputs\logs\stage2_train_and_render_mainline.log"
"C:\Users\brack\anaconda3\envs\ega-gpu\python.exe" -m avatar_gs.cli.stage2_train_gaussian --config configs/stage2_gaussian.yaml >>"outputs\logs\stage2_train_and_render_mainline.log" 2>&1
if errorlevel 1 (
  echo [%date% %time%] TRAIN FAILED code=%errorlevel%>>"outputs\logs\stage2_train_and_render_mainline.log"
  exit /b %errorlevel%
)
echo [%date% %time%] START render>>"outputs\logs\stage2_train_and_render_mainline.log"
"C:\Users\brack\anaconda3\envs\ega-gpu\python.exe" -m avatar_gs.cli.render --config configs/render.yaml >>"outputs\logs\stage2_train_and_render_mainline.log" 2>&1
if errorlevel 1 (
  echo [%date% %time%] RENDER FAILED code=%errorlevel%>>"outputs\logs\stage2_train_and_render_mainline.log"
  exit /b %errorlevel%
)
echo [%date% %time%] DONE>>"outputs\logs\stage2_train_and_render_mainline.log"
