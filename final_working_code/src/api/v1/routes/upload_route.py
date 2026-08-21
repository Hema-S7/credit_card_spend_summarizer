from fastapi import APIRouter, UploadFile, File, HTTPException

from src.api.v1.services.upload_service import handle_file_upload

upload_router = APIRouter(prefix="/api/v1", tags=["Upload"])


@upload_router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = await file.read()
        return handle_file_upload(file.filename, file_bytes)

    except Exception as ex:
        raise HTTPException(status_code=500, detail=str(ex))
