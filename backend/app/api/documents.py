from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.services.rag.rag_pipeline import rag_store_async

ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_MIME_TYPES = {"application/pdf"}

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    # Check extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid file extension")
    
    contents = await file.read()
    
    try:
        # Pass to RAG ingestion, embedding and storing service
        result = await rag_store_async(contents, file, ext)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Catch all other exceptions and return 500
        raise HTTPException(status_code=500, detail=str(e))

    return result
