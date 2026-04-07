@echo off
cd /d "d:\project\Projects-zhy\Editable Gaussian Avatars"
set PYTHONPATH=d:\project\Projects-zhy\Editable Gaussian Avatars\src
if not exist "outputs\logs" mkdir "outputs\logs"
"C:\Users\brack\anaconda3\envs\ega-gpu\python.exe" -m avatar_gs.cli.stage2_train_gaussian --config configs/stage2_gaussian.yaml 1>>"outputs\logs\stage2_inner_live.stdout.log" 2>>"outputs\logs\stage2_inner_live.stderr.log"
