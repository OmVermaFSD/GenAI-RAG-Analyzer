from fastapi import FastAPI, UploadFile, File
import pytesseract
from PIL import Image
import io

app = FastAPI()

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    text = pytesseract.image_to_string(image)
    return {"text": text}
