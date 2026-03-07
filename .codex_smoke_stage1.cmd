@echo off
call C:\Users\brack\anaconda3\condabin\conda.bat activate ega-gpu
cd /d "d:\project\Projects-zhy\Editable Gaussian Avatars"
set PYTHONPATH=%CD%\src
python -m py_compile third_party\SMPLer-X\main\inference.py src\avatar_gs\cli\_run_smplerx_refine_frames.py src\avatar_gs\cli\stage1_fit_smplerx.py
if errorlevel 1 exit /b 1
python -m avatar_gs.cli.stage1_fit_smplerx --project_root . --frames_dir data/frames/input --assets_dir assets --out_dir outputs/stage1_smplerx_smoke --smplerx_frame_indices 75 --single_person
