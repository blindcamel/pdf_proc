# Standard library imports
import asyncio
import logging
import os
import uuid
import shutil
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path

# Third-party imports
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Local imports
# from invoice_data_extractor import InvoiceDataExtractor
from invoice_data_extractor_assistant import InvoiceDataExtractor
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
    PROCESSED_FAILED_DIR = BASE_DIR / "processed/failed"  # Directory for failed files

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
        self.pdf_splitter = PDFSplitter()
        # Flag to track if the process_queue task is running
        self.is_running = False

    def on_created(self, event):
        """Triggered when a new file is created in the watched directory"""
        if event.is_directory or not event.src_path.endswith(".pdf"):
            return

        file_path = Path(event.src_path)
        filename = file_path.name

        # Only add to queue if not already being processed
        if filename not in self.processing_status:
            self.processing_status[filename] = {
                "status": ProcessingStatus.DETECTED,
                "timestamp": datetime.now(),
                "path": str(file_path),
            }
            logger.info(
                f"New PDF detected: {filename} - Status: {ProcessingStatus.DETECTED}"
            )

            # Use the event loop to add the file to the queue
            if hasattr(self.app.state, "loop"):
                asyncio.run_coroutine_threadsafe(
                    self.queue.put(str(file_path)), self.app.state.loop
                )
            else:
                logger.error("Event loop not found in app state")
        else:
            logger.info(f"File {filename} already in processing queue")

    async def start_processing(self):
        """Start the background task to process PDFs from queue"""
        if not self.is_running:
            self.is_running = True
            asyncio.create_task(self.process_queue())
            logger.info("Started PDF processing background task")

    async def process_queue(self):
        """Background task to process PDFs from queue"""
        logger.info("PDF processing queue started")
        try:
            while True:
                # Get the next file path from the queue
                file_path = await self.queue.get()
                logger.info(f"Processing next file from queue: {file_path}")

                try:
                    # Process the file
                    await self._process_and_track(Path(file_path))
                except Exception as e:
                    # Catch any exceptions to prevent the loop from breaking
                    logger.error(f"Error in process_queue for {file_path}: {str(e)}")

                # Mark the task as done
                self.queue.task_done()
                logger.info(f"Completed processing file: {file_path}")
        except Exception as e:
            # Catch any exceptions to log them
            logger.error(f"Error in process_queue main loop: {str(e)}")
            self.is_running = False
            # Restart the task
            asyncio.create_task(self.start_processing())

    async def _process_and_track(self, file_path: Path):
        """Process PDF and track its status"""
        file_path = Path(file_path)  # Ensure file_path is a Path object
        filename = file_path.name
        original_filename = filename  # Keep track of the original filename

        logger.info(f"Starting to process file: {filename}")

        try:
            # Update status to processing
            if filename in self.processing_status:
                self.processing_status[filename]["status"] = ProcessingStatus.PROCESSING
                logger.info(f"Updated status of {filename} to PROCESSING")
            else:
                # Create an entry if it doesn't exist
                self.processing_status[filename] = {
                    "status": ProcessingStatus.PROCESSING,
                    "timestamp": datetime.now(),
                    "path": str(file_path),
                }
                logger.info(f"Created new status entry for {filename} as PROCESSING")

            # Process the PDF
            result = await process_pdf(file_path)
            logger.info(f"PDF processing result for {filename}: {result['source']}")

            # Extract and rename if data available
            renamed = False
            new_filename = filename

            if result.get("text"):
                extracted_data = await self.invoice_extractor.extract_data(
                    result["text"]
                )

                if extracted_data:
                    logger.info(f"Extracted data from {filename}: {extracted_data}")
                    new_path = await self.pdf_renamer.rename_file(
                        file_path, extracted_data
                    )

                    if new_path:
                        # Update the file path and filename after renaming
                        renamed = True
                        file_path = new_path
                        new_filename = new_path.name

                        # Create an entry for the new filename if it doesn't exist
                        if new_filename not in self.processing_status:
                            self.processing_status[new_filename] = (
                                self.processing_status[filename].copy()
                            )

                        # Update the entry with extraction and rename info
                        self.processing_status[new_filename].update(
                            {
                                "extracted_data": extracted_data,
                                "renamed": True,
                                "original_filename": original_filename,
                                "path": str(
                                    file_path
                                ),  # Update the path to the new location
                            }
                        )
                        logger.info(f"Renamed file from {filename} to {new_filename}")
                    else:
                        logger.warning(f"Failed to rename {filename}")
                        self.processing_status[filename].update({"rename_failed": True})
                else:
                    logger.warning(f"Failed to extract data from {filename}")
                    self.processing_status[filename].update({"extraction_failed": True})

            # Determine which status entry to update
            status_key = new_filename if renamed else filename

            # Update status to completed with results
            self.processing_status[status_key].update(
                {
                    "status": ProcessingStatus.COMPLETED,
                    "result": result,
                    "completed_at": datetime.now(),
                }
            )
            logger.info(f"Updated status of {status_key} to COMPLETED")

            # Move the file to the appropriate processed directory
            try:
                if result["source"] == "ocr":
                    settings.PROCESSED_OCR_DIR.mkdir(parents=True, exist_ok=True)
                    target_dir = settings.PROCESSED_OCR_DIR
                else:
                    settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
                    target_dir = settings.PROCESSED_DIR

                # Ensure file still exists before moving
                if file_path.exists():
                    target_path = target_dir / file_path.name
                    logger.info(f"Moving {file_path} to {target_path}")
                    shutil.move(str(file_path), str(target_path))

                    self.processing_status[status_key].update(
                        {"file_moved": True, "final_location": str(target_path)}
                    )
                    logger.info(f"Successfully moved {file_path.name} to {target_path}")
                else:
                    logger.error(f"File {file_path} does not exist, cannot move")
                    self.processing_status[status_key].update(
                        {"file_moved": False, "move_error": "File does not exist"}
                    )
            except PermissionError as e:
                logger.error(f"Permission error moving {file_path.name}: {str(e)}")
                self.processing_status[status_key].update(
                    {"file_moved": False, "move_error": f"Permission denied: {str(e)}"}
                )
            except OSError as e:
                logger.error(f"OS error moving {file_path.name}: {str(e)}")
                self.processing_status[status_key].update(
                    {"file_moved": False, "move_error": str(e)}
                )

        except Exception as e:
            # Handle any exceptions during processing
            error_msg = str(e)
            logger.error(f"Error processing {filename}: {error_msg}")
            traceback_info = traceback.format_exc()
            logger.error(f"Traceback: {traceback_info}")

            # Update the status to failed
            if filename in self.processing_status:
                self.processing_status[filename].update(
                    {
                        "status": ProcessingStatus.FAILED,
                        "error": error_msg,
                        "traceback": traceback_info,
                        "failed_at": datetime.now(),
                    }
                )
            else:
                self.processing_status[filename] = {
                    "status": ProcessingStatus.FAILED,
                    "error": error_msg,
                    "traceback": traceback_info,
                    "failed_at": datetime.now(),
                    "path": str(file_path),
                }
                # Try to move the failed file to the failed directory
            try:
                # Create failed directory if it doesn't exist
                settings.PROCESSED_FAILED_DIR.mkdir(parents=True, exist_ok=True)

                # Ensure file still exists before moving
                if file_path.exists():
                    target_path = settings.PROCESSED_FAILED_DIR / file_path.name
                    logger.info(f"Moving failed file {file_path} to {target_path}")
                    shutil.move(str(file_path), str(target_path))

                    # Update status with move information
                    if filename in self.processing_status:
                        self.processing_status[filename].update(
                            {"file_moved": True, "final_location": str(target_path)}
                        )
                    logger.info(
                        f"Successfully moved failed file {file_path.name} to {target_path}"
                    )
                else:
                    logger.error(f"Failed file {file_path} does not exist, cannot move")
                    if filename in self.processing_status:
                        self.processing_status[filename].update(
                            {"file_moved": False, "move_error": "File does not exist"}
                        )
            except PermissionError as e:
                logger.error(
                    f"Permission error moving failed file {file_path.name}: {str(e)}"
                )
                if filename in self.processing_status:
                    self.processing_status[filename].update(
                        {
                            "file_moved": False,
                            "move_error": f"Permission denied: {str(e)}",
                        }
                    )
            except OSError as e:
                logger.error(f"OS error moving failed file {file_path.name}: {str(e)}")
                if filename in self.processing_status:
                    self.processing_status[filename].update(
                        {"file_moved": False, "move_error": str(e)}
                    )


class PDFSplitter:
    """Handles PDF document splitting based on page mapping data"""

    def __init__(self, temp_dir=None):
        """Initialize the PDF splitter

        Args:
            temp_dir: Directory to use for temporary files during processing
        """
        self.temp_dir = temp_dir or Path("temp_splits")
        self.temp_dir.mkdir(exist_ok=True, parents=True)

    async def split_document(self, file_path, page_mapping):
        """Split a PDF document based on document_id groupings in page mapping

        Args:
            file_path: Path to the PDF file to split
            page_mapping: Dictionary with (document_id, page_number) keys and metadata values

        Returns:
            List of paths to the split PDF files
        """
        logger.info(f"Splitting document: {file_path}")

        # Group pages by document_id
        document_groups = self._group_by_document_id(page_mapping)

        # Create a list to store paths of split documents
        split_paths = []

        try:
            # Open the source PDF
            with fitz.open(str(file_path)) as doc:
                # Process each document group
                for doc_id, page_info in document_groups.items():
                    # Sort pages by page_number
                    sorted_pages = sorted(page_info, key=lambda x: x[0])

                    # Get metadata from first page in group
                    metadata = page_mapping[(doc_id, sorted_pages[0][0])]
                    company, po_num, inv_num = metadata

                    # Create output filename
                    output_filename = f"{company}_PO{po_num}_INV{inv_num}.pdf"
                    if len(document_groups) > 1:
                        output_filename = (
                            f"{company}_PO{po_num}_INV{inv_num}_doc{doc_id}.pdf"
                        )

                    output_path = self.temp_dir / output_filename

                    # Create new PDF for this document group
                    await self._create_pdf_subset(doc, sorted_pages, output_path)

                    split_paths.append(output_path)
                    logger.info(f"Created split document: {output_path}")

        except Exception as e:
            logger.error(f"Error splitting PDF {file_path}: {str(e)}")
            raise

        return split_paths

    def _group_by_document_id(self, page_mapping):
        """Group pages by document_id from the page mapping

        Args:
            page_mapping: Dictionary with (document_id, page_number) keys and metadata values

        Returns:
            Dictionary with document_id keys and lists of (page_number, metadata) values
        """
        document_groups = {}

        for (doc_id, page_num), metadata in page_mapping.items():
            if doc_id not in document_groups:
                document_groups[doc_id] = []

            document_groups[doc_id].append((page_num, metadata))

        return document_groups

    async def _create_pdf_subset(self, src_doc, pages, output_path):
        """Create a new PDF with selected pages from the source document

        Args:
            src_doc: Source PyMuPDF document
            pages: List of (page_number, metadata) tuples
            output_path: Path where the new PDF should be saved
        """
        # Create a new PDF document
        new_doc = fitz.open()

        try:
            # Add each page to the new document
            for page_num, _ in pages:
                # PyMuPDF pages are 0-indexed
                new_doc.insert_pdf(
                    src_doc, from_page=page_num - 1, to_page=page_num - 1
                )

            # Save the new document
            new_doc.save(str(output_path))

        finally:
            # Clean up
            new_doc.close()

    async def cleanup(self):
        """Remove temporary files and directories"""
        import shutil

        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {str(e)}")


async def _process_and_track(self, file_path: Path):
    """Process PDF and track its status"""
    file_path = Path(file_path)  # Ensure file_path is a Path object
    filename = file_path.name
    # original_filename = filename

    logger.info(f"Starting to process file: {filename}")

    try:
        # Update status to processing
        if filename in self.processing_status:
            self.processing_status[filename]["status"] = ProcessingStatus.PROCESSING
            logger.info(f"Updated status of {filename} to PROCESSING")
        else:
            # Create an entry if it doesn't exist
            self.processing_status[filename] = {
                "status": ProcessingStatus.PROCESSING,
                "timestamp": datetime.now(),
                "path": str(file_path),
            }
            logger.info(f"Created new status entry for {filename} as PROCESSING")

        # Process the PDF
        result = await process_pdf(file_path)
        logger.info(f"PDF processing result for {filename}: {result['source']}")

        # Extract document information including page mapping
        # extracted_data = None
        page_mapping = None
        if result.get("text"):
            # extracted_data = await self.invoice_extractor.extract_data(result["text"])

            # Here we would call a method to get page mapping data
            page_mapping = await self.invoice_extractor.extract_page_mapping(
                result["text"]
            )

            if page_mapping:
                logger.info(
                    f"Extracted page mapping from {filename} with {len(page_mapping)} pages"
                )
                self.processing_status[filename].update({"page_mapping": page_mapping})

                # Split the document if needed
                if len(set(doc_id for doc_id, _ in page_mapping.keys())) > 1:
                    logger.info(
                        f"Document {filename} contains multiple documents, splitting..."
                    )

                    # Split the document based on page mapping
                    split_paths = await self.pdf_splitter.split_document(
                        file_path, page_mapping
                    )

                    # Update the status with split information
                    self.processing_status[filename].update(
                        {
                            "split": True,
                            "split_count": len(split_paths),
                            "split_paths": [str(p) for p in split_paths],
                        }
                    )

                    # Process each split document
                    for split_path in split_paths:
                        split_filename = split_path.name

                        # Create an entry for the split file
                        self.processing_status[split_filename] = {
                            "status": ProcessingStatus.PROCESSING,
                            "timestamp": datetime.now(),
                            "path": str(split_path),
                            "parent_file": filename,
                            "split_from": str(file_path),
                        }

                        # Extract document ID from filename
                        doc_id = None
                        for d_id in set(doc_id for doc_id, _ in page_mapping.keys()):
                            if f"_doc{d_id}" in split_filename:
                                doc_id = d_id
                                break

                        # Get specific metadata for this document
                        if doc_id:
                            # Find first page for this doc_id
                            first_page = min(
                                page_num
                                for d_id, page_num in page_mapping.keys()
                                if d_id == doc_id
                            )
                            doc_metadata = page_mapping.get((doc_id, first_page))

                            if doc_metadata:
                                # Move the split file to the appropriate processed directory
                                await self._move_processed_file(
                                    split_path, result["source"]
                                )

                                # Update status for this split file
                                self.processing_status[split_filename].update(
                                    {
                                        "status": ProcessingStatus.COMPLETED,
                                        "metadata": doc_metadata,
                                        "completed_at": datetime.now(),
                                    }
                                )

                    # Original file processing is completed
                    self.processing_status[filename].update(
                        {
                            "status": ProcessingStatus.COMPLETED,
                            "result": result,
                            "completed_at": datetime.now(),
                        }
                    )

                    return  # Early return since we've handled all split files
            else:
                logger.warning(f"Failed to extract page mapping from {filename}")
                self.processing_status[filename].update({"page_mapping_failed": True})

        # Continue with normal processing for single documents
        # (Existing renaming code would go here)

        # Rest of your existing _process_and_track method...

    except Exception as e:
        # Handle any exceptions during processing
        error_msg = str(e)
        logger.error(f"Error processing {filename}: {error_msg}")
        traceback_info = traceback.format_exc()
        logger.error(f"Traceback: {traceback_info}")

        # Update the status to failed
        if filename in self.processing_status:
            self.processing_status[filename].update(
                {
                    "status": ProcessingStatus.FAILED,
                    "error": error_msg,
                    "traceback": traceback_info,
                    "failed_at": datetime.now(),
                }
            )
        else:
            self.processing_status[filename] = {
                "status": ProcessingStatus.FAILED,
                "error": error_msg,
                "traceback": traceback_info,
                "failed_at": datetime.now(),
                "path": str(file_path),
            }


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

    # Start the background processing task
    await event_handler.start_processing()

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


@app.get("/processing-status/")
@app.get("/processing-status/{filename}")
async def get_processing_status(filename: str = None):
    """Get processing status for a specific file or all files"""
    event_handler = app.state.event_handler
    
    # If no filename is provided, return all statuses
    if filename is None:
        return {
            "total_files": len(event_handler.processing_status),
            "statuses": event_handler.processing_status
        }
    
    # If filename is provided, return status for that file
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
            "POST /process-all": "Process all files in /uploads",
            "GET /list-files": "List all PDF files in the input directory",
            "GET /debug-paths": "debug paths",
            "GET /processing-status": "List all",
            "GET /processing-status/{filename}": "List one",
            "GET /": "This information",
        },
    }
