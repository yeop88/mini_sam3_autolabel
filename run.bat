@echo off
rem SAM 3 auto label GUI (pythonw, no console window)
rem   run.bat                          -> last used camera and resolution (chapter05\01_camera_recognized)
rem   run.bat --camera 1 --size 1280x720
rem   run.bat --video D:\videos\desk.mp4
rem   run.bat --no-camera --tab review
rem   run.bat --no-sam                 -> do not auto-load SAM 3 at start (saves ~3.4 GB GPU memory)
cd /d "%~dp0"
start "" "C:\Users\USER\anaconda3\envs\physical_ai\pythonw.exe" main.py %*
