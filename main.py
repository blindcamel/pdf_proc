# Standard library imports
import asyncio
import logging
import os
import uuid
import shutil
from contextlib import asynccontextmanager
from datetime import datetime
from dotenv import load_dotenv
from enum import Enum
from pathlib import Path

# Third-party imports
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Local imports
from invoice_data_extractor import InvoiceDataExtractor
from pdf_renamer import PDFRenamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Enum for tracking PDF processing status"""

    DETECTED = "detected"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Settings:
    """Application configuration settings"""

    BASE_DIR = Path(
        __file__
    ).parent.absolute()  # Get the directory where script is running

    # UPLOAD_DIR = Path("uploads")  # Directory for API uploaded files
    UPLOAD_DIR = BASE_DIR / "uploads"  # Directory for API uploaded files

    # INPUT_DIR = Path("filein")  # Directory to watch for new files
    # PROCESSED_DIR = Path("processed")  # Base processed directory
    # PROCESSED_OCR_DIR = Path("processed/OCR")  # OCR-specific directory
    INPUT_DIR = BASE_DIR / "filein"  # Directory to watch for new files
    PROCESSED_DIR = BASE_DIR / "processed"  # Base processed directory
    PROCESSED_OCR_DIR = BASE_DIR / "processed/OCR"  # OCR-specific directory
    
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
    ALLOWED_MIME_TYPES = {"application/pdf"}
    
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class PDFHandler(FileSystemEventHandler):
    """Handles file system events for PDF processing"""

    def __init__(self, app):
        self.app = app
        self.processing_status = {}
        self.queue = asyncio.Queue()
        self.invoice_extractor = InvoiceDataExtractor(settings.OPENAI_API_KEY)
        self.pdf_renamer = PDFRenamer()

    def on_created(self, event):
        """Triggered when a new file is created in the watched directory"""
        if event.is_directory or not event.src_path.endswith(".pdf"):
            return

        filename = Path(event.src_path).name
        self.processing_status[filename] = {
            "status": ProcessingStatus.DETECTED,
            "timestamp": datetime.now(),
            "path": event.src_path,
        }
        logger.info(
            f"New PDF detected: {filename} - Status: {ProcessingStatus.DETECTED}"
        )
        asyncio.run_coroutine_threadsafe(
            self.queue.put(event.src_path), self.app.state.loop
        )

    async def _process_and_track(self, file_path: Path):
        """Process PDF and track its status"""
        filename = file_path.name
        try:
            self.processing_status[filename]["status"] = ProcessingStatus.PROCESSING
            result = await process_pdf(file_path)
            self.processing_status[filename].update(
                {
                    "status": ProcessingStatus.COMPLETED,
                    "result": result,
                    "completed_at": datetime.now(),
                }
            )
            logger.info(f"Completed processing {filename} - Method: {result['source']}")
            # logger.info(f"Extracted text from {filename}:\n{result['text']}")   #prints entire output

            # Extract invoice data
            extracted_data = await self.invoice_extractor.extract_data(result["text"])
            if extracted_data:
                new_path = await self.pdf_renamer.rename_file(file_path, extracted_data)
                if new_path:
                    file_path = new_path  # Update path for file movement
                    filename = new_path.name  # Update filename for status tracking
                    self.processing_status[filename].update(
                        {"extracted_data": extracted_data, "renamed": True}
                    )
                    logger.info(f"Renamed file to: {filename}")
                else:
                    logger.warning(f"Failed to rename {filename}")
                    self.processing_status[filename].update({"rename_failed": True})
            else:
                logger.warning(f"Failed to extract data from {filename}")
                self.processing_status[filename].update({"extraction_failed": True})

            try:
                if result["source"] == "ocr":
                    settings.PROCESSED_OCR_DIR.mkdir(parents=True, exist_ok=True)
                    target_dir = settings.PROCESSED_OCR_DIR
                else:
                    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                    target_dir = settings.PROCESSED_DIR

                target_path = target_dir / Path(file_path).name
                shutil.move(str(file_path), str(target_path))

                self.processing_status[filename].update(
                    {"file_moved": True, "final_location": str(target_path)}
                )
                logger.info(f"Moved {filename} to {target_path}")

            except PermissionError as e:
                logger.error(f"Permission error moving {filename}: {str(e)}")
                self.processing_status[filename].update(
                    {"file_moved": False, "move_error": "Permission denied"}
                )
            except OSError as e:
                logger.error(f"OS error moving {filename}: {str(e)}")
                self.processing_status[filename].update(
                    {"file_moved": False, "move_error": str(e)}
                )

        except Exception as e:
            self.processing_status[filename].update(
                {
                    "status": ProcessingStatus.FAILED,
                    "error": str(e),
                    "failed_at": datetime.now(),
                }
            )
            logger.error(f"Failed processing {filename}: {str(e)}")


# Initialize settings
settings = Settings()
settings.UPLOAD_DIR.mkdir(exist_ok=True)
settings.INPUT_DIR.mkdir(exist_ok=True)


async def process_pdf(file_path: Path) -> dict:
    """
    Process PDF file and extract text using OCR if necessary.
    Returns a dictionary containing extracted text and metadata.
    """
    logger.info(f"Processing file: {file_path}")
    try:
        doc = fitz.open(str(file_path))
        page_count = len(doc)

        # Try direct text extraction first
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"

        if text.strip():
            doc.close()
            return {"text": text.strip(), "source": "direct", "page_count": page_count}

        # Fall back to OCR if no text found
        logger.info(f"No text found in PDF {file_path}, falling back to OCR")
        text = ""
        for page in doc:
            pix = page.get_pixmap()
            print(
                f"Pixmap dims: {pix.width}x{pix.height}, samples per pixel: {pix.n}"
            )  # Debug

            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            print(f"Numpy array shape: {img.shape}")  # Debug

            text += pytesseract.image_to_string(img) + "\n"
            print(f"OCR output length: {len(text)}")  # Debug

        doc.close()
        return {"text": text.strip(), "source": "ocr", "page_count": page_count}

    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error processing PDF file: {str(e)}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks"""
    # Store the event loop
    app.state.loop = asyncio.get_running_loop()

    # Initialize file watcher
    event_handler = PDFHandler(app)
    app.state.event_handler = event_handler

    async def process_queue():
        """Background task to process PDFs from queue"""
        while True:
            file_path = await event_handler.queue.get()
            await event_handler._process_and_track(Path(file_path))
            event_handler.queue.task_done()

    process_task = asyncio.create_task(process_queue())
    app.state.process_task = process_task

    # Start file system observer
    observer = Observer()
    observer.schedule(event_handler, str(settings.INPUT_DIR), recursive=False)
    observer.start()
    logger.info("File watcher started for 'filein' directory")
    yield

    # Cleanup on shutdown
    observer.stop()
    observer.join()
    logger.info("File watcher stopped")


# Initialize FastAPI application
app = FastAPI(lifespan=lifespan)


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF upload through API endpoint"""
    file_path = None
    try:
        file_path = settings.UPLOAD_DIR / f"{uuid.uuid4()}.pdf"
        logger.info(f"Saving uploaded file to: {file_path}")

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return await process_pdf(file_path)

    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")


@app.post("/process-file/")
async def process_existing_file(filename: str):
    """Process a file that already exists in the input directory"""
    file_path = settings.INPUT_DIR / filename
    logger.info(f"Request to process existing file: {file_path}")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        return await process_pdf(file_path)
    except Exception as e:
        logger.error(f"Error processing file {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/processing-status/{filename}")
async def get_processing_status(filename: str):
    """Get processing status for a specific file"""
    event_handler = app.state.event_handler
    if filename not in event_handler.processing_status:
        raise HTTPException(
            status_code=404, detail="File not found in processing history"
        )
    return event_handler.processing_status[filename]


@app.get("/list-files/")
async def list_input_files():
    """List all PDF files in the input directory"""
    try:
        files = [f for f in os.listdir(settings.INPUT_DIR) if f.endswith(".pdf")]
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-all/")
async def process_all_files():
    """Process all PDF files currently in the input directory"""
    event_handler = app.state.event_handler
    input_files = [f for f in settings.INPUT_DIR.glob("*.pdf")]

    if not input_files:
        return {"message": "No PDF files found in input directory"}

    # Queue all files for processing
    for file_path in input_files:
        if file_path.name not in event_handler.processing_status:
            event_handler.processing_status[file_path.name] = {
                "status": ProcessingStatus.DETECTED,
                "timestamp": datetime.now(),
                "path": str(file_path),
            }
            await event_handler.queue.put(str(file_path))
            logger.info(f"Queued {file_path.name} for processing")

    return {
        "message": f"Queued {len(input_files)} files for processing",
        "files": [f.name for f in input_files],
    }


@app.get("/debug-paths/")
async def debug_paths():
    """Debug endpoint to show directory paths"""
    return {
        "base_dir": str(settings.BASE_DIR),
        "input_dir": str(settings.INPUT_DIR),
        "input_dir_exists": settings.INPUT_DIR.exists(),
        "input_dir_is_dir": settings.INPUT_DIR.is_dir(),
        "files_in_input_dir": [
            f.name for f in settings.INPUT_DIR.glob("*") if f.is_file()
        ],
        "pdf_files_in_input_dir": [f.name for f in settings.INPUT_DIR.glob("*.pdf")],
    }


@app.get("/")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "PDF Processing API",
        "endpoints": {
            "POST /upload": "Upload and process a new PDF file",
            "POST /process-file": "Process an existing file from the input directory",
            "GET /list-files": "List all PDF files in the input directory",
            "GET /": "This information",
        },
    }
