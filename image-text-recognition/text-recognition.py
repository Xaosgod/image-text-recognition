from fastapi import FastAPI, File, UploadFile
import easyocr
app = FastAPI()
@app.post("/text-recognition")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    reader = easyocr.Reader(['ru', 'en'])
    a=reader.readtext(contents, detail=0, paragraph=True)
    return str(a)
    



