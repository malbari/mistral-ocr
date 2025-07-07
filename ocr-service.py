import os
import time
import uuid
import base64
import io
import uuid
from typing import List, Dict, Any
from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi import FastAPI, UploadFile, File, Path
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

# OCR libs
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Simulatore Mistral OCR")

# Mappa id UUID -> filename
id_to_filename = {}

class OCRRequest(BaseModel):
    file: str  # id del file

@app.post("/v1/files")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        file_id = str(uuid.uuid4())
        id_to_filename[file_id] = file.filename

        response = {
            "id": file_id,
            "object": "file",
            "bytes": len(contents),
            "created_at": int(time.time()),
            "filename": file.filename,
            "purpose": "ocr",
            "sample_type": "pretrain",
            "num_lines": 0,
            "source": "upload"
        }
        print(f"➡️ File caricato: {file.filename}, id: {file_id}")
        return response
    except Exception as e:
        print(f"❌ Errore upload file: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/v1/files/{file_id}/url")
async def get_file_url(file_id: str, expiry: int = 24):
    if file_id not in id_to_filename:
        return JSONResponse(status_code=404, content={"error": "File ID non trovato"})
    filename = id_to_filename[file_id]
    url = f"/v1/files/{filename}"
    print(f"➡️ Restituisco URL per id {file_id}: {url}")
    return {"url": url}

@app.get("/v1/files/{filename}")
async def get_file(filename: str = Path(...)):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        return JSONResponse(status_code=404, content={"error": "File non trovato"})
    return FileResponse(file_path)

from fastapi import Request
from fastapi.responses import JSONResponse

@app.post("/v1/ocr")
async def ocr_endpoint(request: Request):
    try:
        json_data = await request.json()
        document = json_data.get("document")
        document_url = document.get("document_url")

        filename = document_url.split("/")[-1]
        file_id = None
        for fid, fname in id_to_filename.items():
            if fname == filename:
                file_id = fid
                break
        if not file_id:
            return JSONResponse(status_code=404, content={"error": "File non caricato"})

        file_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(file_path):
            return JSONResponse(status_code=404, content={"error": "File non trovato"})

        ext = os.path.splitext(filename)[1].lower()

        pages_result = []
        doc_size_bytes = os.path.getsize(file_path)
        model_name = json_data.get("model", "mistral-ocr-latest")

        if ext == ".pdf":
            images = convert_from_path(file_path)
            for idx, image in enumerate(images):
                text = pytesseract.image_to_string(image).strip()
                width, height = image.size
                dpi = image.info.get('dpi', (72, 72))[0]

                # Image bbox full page
                image_id = str(uuid.uuid4())
                image_data = {
                    "id": image_id,
                    "top_left_x": 0,
                    "top_left_y": 0,
                    "bottom_right_x": width,
                    "bottom_right_y": height,
                    "image_base64": "",  # se vuoi puoi codificare l'immagine qui
                    "image_annotation": ""
                }

                page_obj = {
                    "index": idx,
                    "markdown": text,
                    "images": [image_data],
                    "dimensions": {
                        "dpi": dpi,
                        "height": height,
                        "width": width
                    }
                }
                pages_result.append(page_obj)

        else:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image).strip()
            width, height = image.size
            dpi = image.info.get('dpi', (72, 72))[0]

            image_id = str(uuid.uuid4())
            image_data = {
                "id": image_id,
                "top_left_x": 0,
                "top_left_y": 0,
                "bottom_right_x": width,
                "bottom_right_y": height,
                "image_base64": "",
                "image_annotation": ""
            }
            page_obj = {
                "index": 0,
                "markdown": text,
                "images": [image_data],
                "dimensions": {
                    "dpi": dpi,
                    "height": height,
                    "width": width
                }
            }
            pages_result.append(page_obj)

        response = {
            "pages": pages_result,
            "model": model_name,
            "document_annotation": "",
            "usage_info": {
                "pages_processed": len(pages_result),
                "doc_size_bytes": doc_size_bytes
            }
        }
        return response

    except Exception as e:
        print(f"❌ Errore OCR generico: {e}")
        return JSONResponse(status_code=500, content={"error": f"Errore OCR: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9090)

