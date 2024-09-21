import io

import uvicorn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from ultralytics import YOLO

from utils import get_boxes_list

app = FastAPI(
    debug=True,
    title="Predictions service",
    version="0.0.1",
)
model = YOLO("yolov8m.pt")


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_content = await file.read()
    image = Image.open(io.BytesIO(image_content))
    results = model(source=image, conf=0.3)
    predictions = await get_boxes_list(results)
    return {"predictions": predictions}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
