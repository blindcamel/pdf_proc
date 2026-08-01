import ast
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class InvoiceDataExtractor:
    """Handles extraction of invoice data using OpenAI's Responses API"""

    def __init__(self, api_key: Optional[str] = None):
        # Load environment variables from .env file
        load_dotenv()

        # Get API key with priority:
        # 1. Explicitly passed api_key parameter
        # 2. Fly.io secrets or other secret stores via _get_secret()
        # 3. Environment variable OPENAI_API_KEY
        api_key = api_key or self._get_secret() or os.getenv("OPENAI_API_KEY")

        if not api_key:
            logger.warning("OpenAI API key not found. Some features may not work.")

        # Initialize the async OpenAI client if credentials are available
        if api_key:
            self.client = AsyncOpenAI(api_key=api_key)
        else:
            self.client = None

        # Load system prompt from file
        prompt_path = Path(__file__).parent / "PDFProc_Prompt.txt"
        try:
            self.system_prompt = prompt_path.read_text(encoding="utf-8")
            logger.info(f"Loaded system prompt from {prompt_path}")
        except FileNotFoundError:
            self.system_prompt = "You are an invoice parser. Extract structured data from invoice documents."
            logger.warning(f"System prompt file not found at {prompt_path}, using default")

    def _get_secret(self):
        """Retrieve API key from environment variables (including Fly.io secrets)"""
        try:
            # Try to get from Fly.io secrets (which are exposed as environment variables)
            fly_api_key = os.getenv("FLY_OPENAI_API_KEY")
            if fly_api_key:
                logger.info("Using API key from Fly.io secrets")
                return fly_api_key

            # If not running in Fly.io environment, check for other potential sources
            # This is a placeholder for any other secret management you might implement
            logger.info("No Fly.io secrets found, checking alternative sources")

            # For now, return None and let the calling code fall back to environment variables
            return None

        except Exception as e:
            logger.error(f"Unexpected error retrieving secret: {str(e)}")
            return None

    async def _upload_file(self, file_content):
        """Upload a file to OpenAI and return the file ID"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name

        try:
            # Upload the file
            with open(temp_path, "rb") as file:
                response = await self.client.files.create(
                    file=file, purpose="user_data"
                )
                return response.id
        finally:
            # Clean up the temporary file
            os.unlink(temp_path)

    async def extract_data(self, file_path):
        """
        Extract invoice data from a PDF file using OpenAI's Responses API.
        Accepts a file path to a PDF.
        Returns: Tuple of (extracted_data, sent_content, api_response)
        """
        try:
            with open(file_path, "rb") as file:
                file_content = file.read()

            # Upload the file and get file_id
            file_id = await self._upload_file(file_content)

            # Create the input with PDF attachment
            input_messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Extract invoice data from this PDF."},
                        {"type": "input_file", "file_id": file_id},
                    ],
                },
            ]

            # Call the Responses API
            # temperature=0 to minimize randomness/"creativity" - we want
            # consistent, deterministic extraction of structured data.
            response = await self.client.responses.create(
                model="gpt-4o",
                input=input_messages,
                max_output_tokens=1000,
                temperature=0,
            )

            response_text = response.output_text
            sent_content = f"PDF file: {file_path}"

            full_response = {"status": "success", "response_text": response_text}

            # Validate and parse response
            try:
                # Remove unnecessary formatting if present
                cleaned_result = re.sub(
                    r"^```(?:json|python)?\n|\n```$", "", response_text.strip()
                )

                # Safely evaluate the string as a Python object
                data_obj = ast.literal_eval(cleaned_result)

                # Format should be a list containing a dictionary
                if (
                    isinstance(data_obj, list)
                    and len(data_obj) > 0
                    and isinstance(data_obj[0], dict)
                ):
                    dict_obj = data_obj[0]  # Extract the dictionary from the list

                    # Validate structure - dictionary should have tuple keys and list values
                    if not all(
                        isinstance(k, tuple) and len(k) == 2 for k in dict_obj.keys()
                    ):
                        raise ValueError(
                            "Invalid key format: Expected (document_id, page_number) tuples"
                        )

                    if not all(
                        isinstance(v, list) and len(v) == 3 for v in dict_obj.values()
                    ):
                        raise ValueError(
                            "Invalid value format: Expected [CompanyName, PO#, Invoice#] lists"
                        )

                    return dict_obj, sent_content, full_response

                raise ValueError(
                    "Invalid response format: Expected a list containing a dictionary with tuple keys."
                )

            except (SyntaxError, ValueError) as e:
                logger.error(f"Failed to parse assistant response: {cleaned_result}")
                logger.error(f"Parsing error: {str(e)}")
                return (
                    None,
                    sent_content,
                    {
                        "status": "parse_error",
                        "error": str(e),
                        "response": response_text,
                    },
                )

        except Exception as e:
            logger.error(f"API extraction error: {str(e)}")
            return (
                None,
                str(file_path),
                {"status": "api_error", "error": str(e), "response": None},
            )
