# Standard library imports
import asyncio
import logging
import os
# import uuid
import shutil
import tempfile
import traceback
import zipfile
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from pathlib import Path

# Third-party imports
import pymupdf as fitz  # PyMuPDF
# import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Local imports
# from invoice_data_extractor import InvoiceDataExtractor
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

    UPLOAD_DIR = BASE_DIR / "upload"  # Directory for API uploaded files

    # INPUT_DIR = Path("filein")  # Directory to watch for new files
    # PROCESSED_DIR = Path("processed")  # Base processed directory
    # PROCESSED_OCR_DIR = Path("processed/OCR")  # OCR-specific directory
    INPUT_DIR = BASE_DIR / "filein"  # Directory to watch for new files
    PROCESSED_DIR = BASE_DIR / "processed"  # Base processed directory
    PROCESSED_OCR_DIR = BASE_DIR / "processed/OCR"  # OCR-specific directory
    PROCESSED_FAILED_DIR = BASE_DIR / "processed/failed"  # Directory for failed files
    PROCESSED_ORIGINALS_DIR = (
        BASE_DIR / "processed/processed_originals"
    )  # Directory for original files

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit
    ALLOWED_MIME_TYPES = {"application/pdf"}

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def ensure_directories():
    """Ensure all required directories exist"""
    for directory in [
        settings.UPLOAD_DIR,
        settings.INPUT_DIR,
        settings.PROCESSED_DIR,
        settings.PROCESSED_OCR_DIR,
        settings.PROCESSED_FAILED_DIR,
        settings.PROCESSED_ORIGINALS_DIR,
        Path("temp_splits"),
    ]:
        directory.mkdir(parents=True, exist_ok=True)


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
        if event.is_directory or not (event.src_path.lower().endswith(".pdf")):
            return

        file_path = Path(event.src_path)

        # Normalize extension to lowercase (.PDF -> .pdf)
        if file_path.suffix != file_path.suffix.lower():
            normalized_path = file_path.with_suffix(file_path.suffix.lower())
            try:
                file_path.rename(normalized_path)
                logger.info(f"Normalized filename extension: {file_path.name} -> {normalized_path.name}")
                file_path = normalized_path
            except Exception as e:
                logger.warning(f"Could not normalize extension for {file_path.name}: {e}")

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
        original_path = file_path  # Keep track of the original path

        logger.info(f"Starting to process file: {filename}")

        try:
            # Update status to processing
            if filename in self.processing_status:
                self.processing_status[filename]["status"] = ProcessingStatus.PROCESSING
                self.processing_status[filename]["original_path"] = str(original_path)
                logger.info(f"Updated status of {filename} to PROCESSING")
            else:
                # Create an entry if it doesn't exist
                self.processing_status[filename] = {
                    "status": ProcessingStatus.PROCESSING,
                    "timestamp": datetime.now(),
                    "path": str(file_path),
                    "original_path": str(original_path),
                }
                logger.info(f"Created new status entry for {filename} as PROCESSING")

            # Process the PDF


            # Extract and rename if data available
            renamed = False
            new_filename = filename

            # Call extract_data directly with file path
            (
                extracted_data,
                sent_text,
                api_response,
            ) = await self.invoice_extractor.extract_data(file_path)

            # Store API request and response data
            self.processing_status[filename].update(
                {"api_request_text": sent_text, "api_response": api_response}
            )

            if extracted_data:
                logger.info(f"Extracted data from {filename}: {extracted_data}")

                # Check if the PDF contains multiple documents
                doc_ids = set(key[0] for key in extracted_data.keys())
                if len(doc_ids) > 1:
                    logger.info(
                        f"Multiple documents detected in {filename}, initiating document splitting"
                    )
                    self.processing_status[filename].update(
                        {"multiple_documents": True}
                    )

                    # Split the document based on extracted data
                    split_paths = await self.pdf_splitter.split_document(
                        file_path, extracted_data
                    )

                    if split_paths:
                        logger.info(
                            f"Successfully split {filename} into {len(split_paths)} documents"
                        )
                        self.processing_status[filename].update(
                            {
                                "split": True,
                                "split_count": len(split_paths),
                                "split_paths": [str(p) for p in split_paths],
                            }
                        )
                else:
                    # Handle single document case
                    # Get the value from the first page
                    first_page_key = min(extracted_data.keys(), key=lambda k: k[1])
                    representative_value = extracted_data[first_page_key]

                    # Rename file using the representative value
                    new_path = await self.pdf_renamer.rename_file(
                        file_path, representative_value
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
                                "original_path": str(original_path),
                                "path": str(file_path),
                            }
                        )
                        logger.info(
                            f"Renamed file from {filename} to {new_filename}"
                        )
                    else:
                        logger.warning(f"Failed to rename {filename}")
                        self.processing_status[filename].update(
                            {"rename_failed": True}
                        )
            else:
                logger.warning(f"Failed to extract data from {filename}")
                self.processing_status[filename].update({"extraction_failed": True})

            # Determine which status entry to update
            status_key = new_filename if renamed else filename

            # Update status to completed with results
            self.processing_status[status_key].update(
                {
                    "status": ProcessingStatus.COMPLETED,
                    "result": api_response,
                    "completed_at": datetime.now(),
                }
            )
            logger.info(f"Updated status of {status_key} to COMPLETED")

            # Create necessary directories
            settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
            settings.PROCESSED_ORIGINALS_DIR.mkdir(parents=True, exist_ok=True)

            # Move files to appropriate locations
            try:
                # Handle multiple documents case
                if self.processing_status[status_key].get("multiple_documents", False):
                    # Move original file to processed_originals
                    if Path(original_path).exists():
                        original_target = (
                            settings.PROCESSED_ORIGINALS_DIR / original_path.name
                        )
                        logger.info(
                            f"Moving original file {original_path} to {original_target}"
                        )
                        shutil.move(str(original_path), str(original_target))
                        self.processing_status[status_key].update(
                            {
                                "original_file_moved": True,
                                "original_final_location": str(original_target),
                            }
                        )

                    # Move split files from temp_splits to processed
                    split_paths = [
                        Path(p)
                        for p in self.processing_status[status_key].get(
                            "split_paths", []
                        )
                    ]
                    for split_path in split_paths:
                        if split_path.exists():
                            target_path = settings.PROCESSED_DIR / split_path.name
                            logger.info(
                                f"Moving split file {split_path} to {target_path}"
                            )
                            shutil.move(str(split_path), str(target_path))
                            logger.info(
                                f"Successfully moved split file {split_path.name} to {target_path}"
                            )

                # Handle single document case
                else:
                    # Move original (pre-rename) file to processed_originals if renamed
                    if renamed:
                        original_path_obj = Path(original_path)
                        if original_path_obj.exists():
                            original_target = (
                                settings.PROCESSED_ORIGINALS_DIR / original_filename
                            )
                            logger.info(
                                f"Moving original file {original_path_obj} to {original_target}"
                            )
                            shutil.move(str(original_path_obj), str(original_target))
                            self.processing_status[status_key].update(
                                {
                                    "original_file_moved": True,
                                    "original_final_location": str(original_target),
                                }
                            )

                    # Move renamed file to processed
                    actual_path = Path(
                        self.processing_status[status_key].get("path", str(file_path))
                    )
                    if actual_path.exists():
                        target_path = settings.PROCESSED_DIR / actual_path.name
                        logger.info(
                            f"Moving renamed file {actual_path} to {target_path}"
                        )
                        shutil.move(str(actual_path), str(target_path))
                        self.processing_status[status_key].update(
                            {"file_moved": True, "final_location": str(target_path)}
                        )
                        logger.info(
                            f"Successfully moved {actual_path.name} to {target_path}"
                        )
                    else:
                        logger.error(f"File {actual_path} does not exist, cannot move")
                        self.processing_status[status_key].update(
                            {"file_moved": False, "move_error": "File does not exist"}
                        )

            except Exception as e:
                logger.error(f"Error moving files: {str(e)}")
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
        """Split a PDF document based on document_id groupings in page mapping"""
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
                    # Sort pages by absolute index to maintain proper document order
                    sorted_pages = sorted(page_info, key=lambda x: x[0])

                    # We need to use the metadata from the first page in this group
                    # The metadata is already included in each tuple in page_info
                    _, metadata = sorted_pages[0]
                    company, po_num, inv_num = metadata

                    # Create output filename
                    output_filename = f"{company} {po_num} {inv_num}.pdf"
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
            Dictionary with document_id keys and lists of (real_pdf_page_index, metadata) values
        """
        document_groups = {}

        # For each document, get the real page indices in the PDF
        sorted_keys = sorted(page_mapping.keys())  # Sort by (doc_id, page_num)

        # Map each key to its real page index in the PDF (0-indexed)
        for i, key in enumerate(sorted_keys):
            doc_id, _ = key
            metadata = page_mapping[key]

            if doc_id not in document_groups:
                document_groups[doc_id] = []

            # The real page index in the PDF is i (0-indexed)
            document_groups[doc_id].append((i, metadata))

        return document_groups

    async def _create_pdf_subset(self, src_doc, pages, output_path):
        """Create a new PDF with selected pages from the source document

        Args:
            src_doc: Source PyMuPDF document
            pages: List of (absolute_page_index, metadata) tuples
            output_path: Path where the new PDF should be saved
        """
        # Create a new PDF document
        new_doc = fitz.open()

        try:
            # Sort pages by absolute index to maintain proper document order
            sorted_pages = sorted(pages, key=lambda x: x[0])

            # Add each page to the new document using the absolute indices
            for absolute_index, _ in sorted_pages:
                # PyMuPDF uses 0-indexed pages
                new_doc.insert_pdf(
                    src_doc, from_page=absolute_index, to_page=absolute_index
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


# Initialize settings
settings = Settings()
settings.UPLOAD_DIR.mkdir(exist_ok=True)
settings.INPUT_DIR.mkdir(exist_ok=True)
settings.PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
settings.PROCESSED_ORIGINALS_DIR.mkdir(exist_ok=True, parents=True)

ensure_directories()


async def process_pdf(file_path: Path) -> dict:
    """
    Process PDF file and extract text.
    Returns a dictionary containing extracted text and metadata.
    """
    logger.info(f"Processing file: {file_path}")
    try:
        doc = fitz.open(str(file_path))
        page_count = len(doc)

        # Extract text
        text = ""
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text + "\n"

        doc.close()
        return {"text": text.strip(), "source": "direct", "page_count": page_count}

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

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html at root
@app.get("/", include_in_schema=False)
async def frontend():
    return FileResponse("static/index.html")


@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle Zip upload through API endpoint"""
    file_path = None
    try:
        # Use the original filename instead of generating a UUID
        file_path = settings.UPLOAD_DIR / file.filename
        logger.info(f"Saving uploaded file to: {file_path}")

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Return success message without processing
        return {
            "status": "success",
            "message": "File uploaded successfully",
            "filename": file.filename,
            "file_path": str(file_path),
        }
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/{filename}")
async def process_zip_file(filename: str):
    """
    Extract the contents of a zip file to the input directory and process
    """
    # Verify the filename has .zip extension
    if not filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400, detail="File must be a ZIP archive with .zip extension"
        )

    # Construct file paths
    zip_path = settings.UPLOAD_DIR / filename

    # Check if the file exists
    if not zip_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    try:
        # Extract the zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get list of PDF files in the zip
            pdf_files = [f for f in zip_ref.namelist() if f.lower().endswith(".pdf")]

            if not pdf_files:
                return {
                    "status": "error",
                    "message": "No PDF files found in the ZIP archive",
                }

            # Extract PDF files to the filein directory
            for pdf_file in pdf_files:
                zip_ref.extract(pdf_file, settings.INPUT_DIR)

            return {
                "status": "success",
                "message": f"Extracted {len(pdf_files)} PDF files to processing queue",
                "extracted_files": pdf_files,
            }

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file format")
    except Exception as e:
        logger.error(f"Error extracting ZIP file {filename}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error extracting ZIP file: {str(e)}"
        )


@app.get("/download")
async def download_processed_files(background_tasks: BackgroundTasks):
    """Download processed files as a zip file"""
    try:
        # Create a temporary file for the zip using mkstemp
        fd, temp_name = tempfile.mkstemp(suffix=".zip")
        os.close(fd)  # Close the file descriptor
        temp_path = Path(temp_name)

        # Create a zip file containing processed PDFs
        with zipfile.ZipFile(temp_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add all PDF files from the processed directory (case-insensitive match)
            files_added = 0
            for pdf_file in settings.PROCESSED_DIR.iterdir():
                if pdf_file.is_file() and pdf_file.suffix.lower() == ".pdf":
                    zipf.write(pdf_file, arcname=pdf_file.name)
                    files_added += 1

            if files_added == 0:
                return {"message": "No files found to download", "status": "empty"}

        # Ensure the file is fully written before serving
        if temp_path.exists() and temp_path.stat().st_size > 0:
            # Add cleanup task to background tasks
            background_tasks.add_task(lambda p=temp_path: p.unlink(missing_ok=True))

            # Return the zip file as a download
            return FileResponse(
                path=str(temp_path),  # Convert Path to string explicitly
                filename="processed.zip",
                media_type="application/zip",
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create zip file")

    except Exception as e:
        logger.error(f"Error creating download: {str(e)}")
        # Clean up the temp file if it exists
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500, detail=f"Error creating download: {str(e)}"
        )


@app.post("/cleanup")
async def cleanup_files():
    """Remove all files in the upload, processed, and filein directories while preserving subdirectories"""
    removed_count = 0
    skipped_count = 0

    # Define directories to clean
    directories = [settings.UPLOAD_DIR, settings.PROCESSED_DIR, settings.INPUT_DIR]

    try:
        # Process each directory
        for directory in directories:
            if not directory.exists() or not directory.is_dir():
                logger.warning(
                    f"Directory does not exist or is not a directory: {directory}"
                )
                continue

            # Remove only files in the root of each directory
            for file_path in directory.glob("*"):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        removed_count += 1
                        logger.info(f"Removed file: {file_path}")
                    except Exception as e:
                        logger.error(f"Error removing file {file_path}: {str(e)}")
                        skipped_count += 1

        return {
            "status": "success",
            "message": f"Cleanup completed. Removed {removed_count} files, skipped {skipped_count} files.",
            "removed_count": removed_count,
            "skipped_count": skipped_count,
        }

    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")


@app.get("/processing-status/")
@app.get("/processing-status/{filename}")
async def get_processing_status(filename: str = None):
    """Get processing status for a specific file or all files"""
    event_handler = app.state.event_handler

    # If no filename is provided, return all statuses
    if filename is None:
        # Convert the processing_status dictionary to a serializable format
        serializable_statuses = {}
        for key, value in event_handler.processing_status.items():
            # Ensure key is converted to string if it's not already
            str_key = str(key)
            serializable_statuses[str_key] = value

        return {
            "total_files": len(serializable_statuses),
            "statuses": serializable_statuses,
        }

    # If filename is provided, convert to string to ensure it's hashable
    str_filename = str(filename)

    # Check if the file exists in processing status
    if str_filename not in event_handler.processing_status:
        raise HTTPException(
            status_code=404, detail="File not found in processing history"
        )

    return event_handler.processing_status[str_filename]


@app.delete("/processing-status/")
@app.delete("/processing-status/{filename}")
async def clear_processing_status(filename: str = None):
    """Clear processing status for a specific file or all files"""
    event_handler = app.state.event_handler

    # If filename is provided, clear only that file's status
    if filename is not None:
        if filename not in event_handler.processing_status:
            raise HTTPException(
                status_code=404, detail="File not found in processing history"
            )

        # Remove the status entry for this file
        del event_handler.processing_status[filename]
        return {
            "message": f"Processing status cleared for {filename}",
            "cleared_files": 1,
        }

    # If no filename is provided, clear all statuses
    total_cleared = len(event_handler.processing_status)
    event_handler.processing_status.clear()

    return {
        "message": "All processing statuses cleared",
        "cleared_files": total_cleared,
    }


@app.get("/list-upload/")
async def list_upload():
    """List all Zip files in the upload directory"""
    try:
        files = [
            f for f in os.listdir(settings.UPLOAD_DIR) if f.lower().endswith(".zip")
        ]
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-input")
async def list_input():
    """List all PDF files in the input directory"""
    try:
        files = [
            f for f in os.listdir(settings.INPUT_DIR) if f.lower().endswith(".pdf")
        ]
        return {"files": files}
    except Exception as e:
        logger.error(f"Error listing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process-all/")
async def process_all_files():
    """Process all PDF files currently in the input directory"""
    event_handler = app.state.event_handler
    input_files = [f for f in settings.INPUT_DIR.glob("*.[pP][dD][fF]")]

    if not input_files:
        return {"message": "No PDF files found in input directory"}

    # Queue all files for processing
    for file_path in input_files:
        # Normalize extension to lowercase (.PDF -> .pdf)
        if file_path.suffix != file_path.suffix.lower():
            normalized_path = file_path.with_suffix(file_path.suffix.lower())
            try:
                file_path.rename(normalized_path)
                logger.info(f"Normalized filename extension: {file_path.name} -> {normalized_path.name}")
                file_path = normalized_path
            except Exception as e:
                logger.warning(f"Could not normalize extension for {file_path.name}: {e}")

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
        "files_in_upload_dir": [
            f.name for f in settings.UPLOAD_DIR.glob("*") if f.is_file()
        ],
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.get("/root")
async def root():
    """Root endpoint providing API information"""
    return {
        "message": "PDF Processing API",
        "endpoints": {
            "POST /upload": "Upload a Zip file",
            "POST /process": "Process a Zip file in /upload/",
            "GET /download": "Download processed files as a zip file",
            "POST /cleanup": "Remove all files in /upload/, /processed/ and /filein/ directories. Subdirectories remain untouched.",
            "GET /list-upload": "List all Zip files in the upload directory",
            "GET /list-input": "List all PDF files in the input directory",
            "GET /debug-paths": "debug paths",
            "GET /processing-status": "List all",
            "GET /processing-status/{filename}": "List one",
            "DELETE /processing-status": "Clear all",
            "DELETE /processing-status/{filename}": "Clear one",
            "GET /": "This information",
            "GET /health": "Health check endpoint",
        },
    }


# listening port
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="info")
