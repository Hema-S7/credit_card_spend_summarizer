from pathlib import Path
from src.ingestion.ingestion import run_ingestion


def handle_file_upload(filename, file_bytes):
    destination = f"data/{filename}"

    with open(destination, "wb") as f:
        f.write(file_bytes)

    run_ingestion(str(destination))

    return {
        "status": "success",
        "message": "Document uploaded successfully.",
    }
