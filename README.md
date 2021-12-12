# Веб-сервис image-text-recognition на базе OCR фреймворка easyocr
## Работа с OCR фреймворком easyocr
### Для использования EasyOCR на Windows необходимо: 
- Установить torch и torchvision
```
pip install torch
pip install torchvision
```
- установить EasyOCR
```
pip install easyocr
```
- Загрузить из Python модель нужного языка и использовать.
```
import easyocr
reader = easyocr.Reader(['ru', 'en'])
reader.readtext(img, detail=0, paragraph=True)
```
- Также проведена работа по созданию собственного OSR об этом в файле READMEOSR.md
## Создание REST API с помощью FastApi
- Устанавливаем необходимые компоненты
```
pip install wheel -U
pip install uvicorn fastapi pydantic 
```
- Создаем основу приложения с расширением .py
```
from fastapi import FastAPI, File, UploadFile
app = FastAPI()
```
- Проверим работу приложение для этого открывает терминал и прописываепм команду:
```
uvicorn <имя_вашего_файла>:app

```
- В браузере открываем страницу по адресу http://127.0.0.1:8000/docs
- Если запуск происходил через вируальную машину, то узнаем связанный через сетевой мост IP для этого прописываем в терминале команду
```
ifconfig

```
- Далее прописываем команду
```
uvicorn <имя_вашего_файла>:app --host <ваше_IP>

```
- Добавляем эндпоинт POST, на входе подается изображение
```
@app.post("/text-recognition")
async def create_upload_file(file: UploadFile = File(...)):

```
- Далее считываем файл и обрабатываем его с помощью OCR фреймворка easyocr, на выходе имеем распознанный текст

```
@app.post("/text-recognition")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    reader = easyocr.Reader(['ru', 'en'])
    a=reader.readtext(contents, detail=0, paragraph=True)
    return str(a)

```
- Полный код:
```
from fastapi import FastAPI, File, UploadFile
import easyocr
app = FastAPI()
@app.post("/text-recognition")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    reader = easyocr.Reader(['ru', 'en'])
    a=reader.readtext(contents, detail=0, paragraph=True)
    return str(a)

```








