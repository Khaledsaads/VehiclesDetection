from tracking import ObjectTracker
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
from pathlib import Path





app = FastAPI(title='Tracker')
model = YOLO(r'Models\best.pt')
target_classes = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']

@app.post('/track')
async def track(file:UploadFile = File(...)):
    video_path = f'input_vidoes/{file.filename}'
    with open(video_path, 'wb')as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    output_path = Path('output_vidoes')
    tracker = ObjectTracker(model, target_classes, max_age= 5, gpu=True )
    tracker.process_video(video_path, output_path)
    return {
        "message": "Video processed successfully"
    }
    
    