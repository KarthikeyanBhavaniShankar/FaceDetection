# Real-Time Sleepiness Detection App

## Download the MediaPipe model

This project requires the `face_landmarker.task` model file.

On Windows PowerShell, run:

```powershell
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task" `
  -OutFile "face_landmarker.task"

