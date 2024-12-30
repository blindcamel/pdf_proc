from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
import asyncio
import fitz
import pytesseract
import numpy as np
import uuid
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingStatus(Enum):
    DETECTED = "detected"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# Configuration
class Settings:
    # Directory for uploaded files through API
    UPLOAD_DIR = Path("uploads")
    # Directory to watch for files to process
    INPUT_DIR = Path("filein")
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_MIME_TYPES = {"application/pdf"}

class PDFHandler(FileSystemEventHandler):
    def __init__(self, app):
        self.app = app
        self.processing_status = {}
        
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.pdf'):
            filename = Path(event.src_path).name
            self.processing_status[filename] = {
                "status": ProcessingStatus.DETECTED,
                "timestamp": datetime.now(),
                "path": event.src_path
            }
            logger.info(f"New PDF detected: {filename} - Status: {ProcessingStatus.DETECTED}")
            asyncio.create_task(self._process_and_track(Path(event.src_path)))
            
    async def _process_and_track(self, file_path: Path):
        filename = file_path.name
        try:
            self.processing_status[filename]["status"] = ProcessingStatus.PROCESSING
            result = await process_pdf(file_path)
            self.processing_status[filename].update({
                "status": ProcessingStatus.COMPLETED,
                "result": result,
                "completed_at": datetime.now()
            })
            logger.info(f"Completed processing {filename} - Method: {result['source']}")
        except Exception as e:
            self.processing_status[filename].update({
                "status": ProcessingStatus.FAILED,
                "error": str(e),
                "failed_at": datetime.now()
            })
            logger.error(f"Failed processing {filename}: {str(e)}")

settings = Settings()
settings.UPLOAD_DIR.mkdir(exist_ok=True)
settings.INPUT_DIR.mkdir(exist_ok=True)

async def process_pdf(file_path: Path) -> dict:
    """Process PDF file and extract text using OCR if necessary."""
    logger.info(f"Processing file: {file_path}")
    try:
        doc = fitz.open(str(file_path))
        page_count = len(doc)
        
        # Try normal text extraction first
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"
        
        if text.strip():
            doc.close()
            return {
                "text": text.strip(),
                "source": "direct",
                "page_count": page_count
            }
        
        # If no text found, use OCR
        logger.info(f"No text found in PDF {file_path}, falling back to OCR")
        text = ""
        for page in doc:
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            text += pytesseract.image_to_string(img) + "\n"
        
        doc.close()
        return {
            "text": text.strip(),
            "source": "ocr",
            "page_count": page_count
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF file: {str(e)}"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize the file watcher
    event_handler = PDFHandler(app)
    app.state.event_handler = event_handler  # Store reference for access
    observer = Observer()
    observer.schedule(event_handler, str(settings.INPUT_DIR), recursive=False)
    observer.start()
    logger.info("File watcher started for 'filein' directory")
    yield
    # Cleanup on shutdown
    observer.stop()
    observer.join()
    logger.info("File watcher stopped")

app = FastAPI(lifespan=lifespan)

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF upload through API endpoint."""
    file_path = None
    try:
        # Generate unique filename
        file_path = settings.UPLOAD_DIR / f"{uuid.uuid4()}.pdf"
        logger.info(f"Saving uploaded file to: {file_path}")
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the PDF
        return await process_pdf(file_path)
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    finally:
        # Cleanup
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")

@app.post("/process-file/")
async def process_existing_file(filename: str):
    """Process a file that already exists in the input directory."""
    file_path = settings.INPUT_DIR / filename
    logger.info(f"Request to process existing file: {file_path}")
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {filename}"
        )
    
    try:
        return await process_pdf(file_path)
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/processing-status/{filename}")
async def get_processing_status(filename: str):
    """Get processing status for a specific file."""
    event_handler = app.state.event_handler
    if filename not in event_handler.processing_status:
        raise HTTPException(status_code=404, detail="File not found in processing history")
    return event_handler.processing_status[filename]

@app.get("/list-files/")
async def list_input_files():
    """List all files in the input directory."""
    try:
        files = [f for f in os.listdir(settings.INPUT_DIR) if f.endswith('.pdf')]
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/")
async def root():
    return {
        "message": "PDF Processing API",
        "endpoints": {
            "POST /upload": "Upload and process a new PDF file",
            "POST /process-file": "Process an existing file from the input directory",
            "GET /list-files": "List all PDF files in the input directory",
            "GET /": "This information"
        }
    }