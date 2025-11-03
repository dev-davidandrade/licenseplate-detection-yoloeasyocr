import cv2
import numpy as np
import os
from ultralytics import YOLO
import easyocr
from services.plate_utils import clean_text


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "weights", "weights_best.pt")
model = YOLO(model_path)
reader = easyocr.Reader(["pt"], gpu=False)


def process_plate_image(image_path: str):
    image = cv2.imread(image_path)
    results = model(image)

    if len(results[0].boxes) == 0:
        return {"plate_text": "Não detectado", "confidence": 0.0}

    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])

    h = y2 - y1
    y1n = y1 + int(h * 0.25)
    y2n = y2 - int(h * 0.05)
    crop = image[y1n:y2n, x1:x2]

    ocr_results = reader.readtext(crop)
    texts = [clean_text(t[1]) for t in ocr_results if t[1]]

    best = max(texts, key=len, default="Não detectado")

    return {"plate_text": best, "confidence": conf}
