from __future__ import annotations

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from backend.models.database import init_db, save_plate_result
from backend.services.ocr_service import process_plate_array


MAX_UPLOAD_BYTES = 15 * 1024 * 1024

app = FastAPI(title="API Leitor de Placas", version="0.2.0")
init_db()


@app.get("/")
def home():
    return {
        "message": "API de leitura de placas funcionando",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}


def validate_upload_content_type(content_type: str | None) -> None:
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Envie um arquivo de imagem.")


@app.post("/ocr")
async def detect_plate(file: UploadFile = File(...)):
    validate_upload_content_type(file.content_type)

    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="A imagem excede o limite de 15 MB.")

    image_bgr = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="A imagem não pôde ser decodificada.")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = await run_in_threadpool(process_plate_array, image_rgb)
    if result["valid_brazilian_plate"]:
        save_plate_result(result)
    return result
