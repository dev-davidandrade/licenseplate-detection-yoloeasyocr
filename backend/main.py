from fastapi import FastAPI, UploadFile, File
from services.ocr_service import process_plate_image
from models.database import init_db, save_plate_result

app = FastAPI(title="API Leitor de Placas")

init_db()


@app.get("/")
def home():
    return {"message": "API de Leitura de Placas funcionando!"}


@app.post("/ocr")
async def detect_plate(file: UploadFile = File(...)):
    contents = await file.read()
    image_path = f"uploads/{file.filename}"
    with open(image_path, "wb") as f:
        f.write(contents)

    result = process_plate_image(image_path)

    save_plate_result(result)

    return result
