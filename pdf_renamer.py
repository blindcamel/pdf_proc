import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

class PDFRenamer:
    """Handles renaming of PDF files based on extracted invoice data"""
    
    def __init__(self):
        self.valid_chars = '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    def sanitize_filename(self, filename: str) -> str:
        """Remove invalid characters from filename"""
        return ''.join(c for c in filename if c in self.valid_chars)

    def create_new_filename(self, data: List[str]) -> Optional[str]:
        """Create new filename from extracted data"""
        try:
            if len(data) != 3 or any(not item for item in data) or 'error' in data:
                logger.error(f"Invalid data for filename creation: {data}")
                return None
            
            # Create filename: CompanyName PO# Invoice#.pdf
            filename = f"{data[0]} {data[1]} {data[2]}.pdf"
            return self.sanitize_filename(filename)
            
        except Exception as e:
            logger.error(f"Error creating filename: {str(e)}")
            return None

    async def rename_file(self, file_path: Path, data: List[str]) -> Optional[Path]:
        """
        Rename PDF file based on extracted data
        Returns: New Path if successful, None if failed
        """
        try:
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return None

            new_filename = self.create_new_filename(data)
            if not new_filename:
                return None

            new_path = file_path.parent / new_filename
            file_path.rename(new_path)
            logger.info(f"Successfully renamed {file_path} to {new_path}")
            return new_path

        except Exception as e:
            logger.error(f"Error renaming file {file_path}: {str(e)}")
            return None