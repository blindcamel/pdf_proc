import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# JSON schema for structured output from the Responses API
INVOICE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "document_id": {"type": "integer"},
                    "pages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "page_number":    {"type": "integer"},
                                "company":        {"type": "string"},
                                "purchase_order": {"type": "string"},
                                "invoice_number": {"type": "string"},
                            },
                            "required": ["page_number", "company", "purchase_order", "invoice_number"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["document_id", "pages"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["documents"],
    "additionalProperties": False,
}


class InvoiceDataExtractor:
    """Handles extraction of invoice data using OpenAI's Responses API"""

    # Cumulative token counters across all PDFs processed in this session
    _session_input_tokens: int = 0
    _session_cached_tokens: int = 0
    _session_output_tokens: int = 0
    _session_pdf_count: int = 0

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

        extracted_data is a dict with (document_id, page_number) tuple keys
        and [company, purchase_order, invoice_number] list values — same
        shape as before so main.py and pdf_renamer.py need no changes.
        """
        try:
            with open(file_path, "rb") as file:
                file_content = file.read()

            # Upload the file and get file_id
            file_id = await self._upload_file(file_content)

            sent_content = f"PDF file: {file_path}"

            # Call the Responses API.
            # - instructions= is the preferred way to pass a system prompt in
            #   the Responses API; it is weighted more heavily than a system
            #   role message embedded in the input array.
            # - text.format json_schema enforces the exact JSON shape we need,
            #   making "return the whole PDF" failures structurally impossible.
            # - gpt-4.1-mini does not support the temperature parameter;
            #   structured output (json_schema + strict) provides the
            #   determinism we need instead.
            response = await self.client.responses.create(
                model="gpt-4.1-mini",
                instructions=self.system_prompt,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Extract invoice data from this PDF.",
                            },
                            {
                                "type": "input_file",
                                "file_id": file_id,
                            },
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "invoice_extraction",
                        "schema": INVOICE_JSON_SCHEMA,
                        "strict": True,
                    }
                },
                max_output_tokens=2000,
            )

            response_text = response.output_text

            # Log per-request token usage and accumulate session totals
            usage = response.usage
            if usage:
                input_tok   = getattr(usage, "input_tokens", 0) or 0
                output_tok  = getattr(usage, "output_tokens", 0) or 0
                # cached_tokens lives inside input_tokens_details on the Responses API
                details     = getattr(usage, "input_tokens_details", None)
                cached_tok  = getattr(details, "cached_tokens", 0) or 0

                InvoiceDataExtractor._session_pdf_count    += 1
                InvoiceDataExtractor._session_input_tokens  += input_tok
                InvoiceDataExtractor._session_cached_tokens += cached_tok
                InvoiceDataExtractor._session_output_tokens += output_tok

                logger.info(
                    f"[Token usage] {file_path.name if hasattr(file_path, 'name') else file_path} | "
                    f"input={input_tok}  cached={cached_tok}  output={output_tok}  "
                    f"(cache hit rate this call: "
                    f"{cached_tok/input_tok*100:.0f}%)" if input_tok else "(no input tokens reported)"
                )
                logger.info(
                    f"[Session totals — {InvoiceDataExtractor._session_pdf_count} PDFs] "
                    f"input={InvoiceDataExtractor._session_input_tokens}  "
                    f"cached={InvoiceDataExtractor._session_cached_tokens}  "
                    f"output={InvoiceDataExtractor._session_output_tokens}  "
                    f"overall cache rate="
                    f"{InvoiceDataExtractor._session_cached_tokens/InvoiceDataExtractor._session_input_tokens*100:.0f}%"
                    if InvoiceDataExtractor._session_input_tokens else
                    f"[Session totals — {InvoiceDataExtractor._session_pdf_count} PDFs] no token data yet"
                )

            full_response = {"status": "success", "response_text": response_text}

            # Parse and validate the JSON response
            try:
                data_obj = json.loads(response_text)

                documents = data_obj.get("documents", [])
                if not isinstance(documents, list) or len(documents) == 0:
                    raise ValueError("Response contained no documents")

                # Rebuild the tuple-keyed dict that the rest of the app expects:
                # { (document_id, page_number): [company, purchase_order, invoice_number] }
                dict_obj = {}
                for doc in documents:
                    doc_id = doc["document_id"]
                    for page in doc["pages"]:
                        key = (doc_id, page["page_number"])
                        dict_obj[key] = [
                            page["company"],
                            page["purchase_order"],
                            page["invoice_number"],
                        ]

                # Basic sanity checks
                if not all(isinstance(k, tuple) and len(k) == 2 for k in dict_obj.keys()):
                    raise ValueError("Invalid key format after JSON→tuple conversion")

                if not all(isinstance(v, list) and len(v) == 3 for v in dict_obj.values()):
                    raise ValueError("Invalid value format after JSON→list conversion")

                return dict_obj, sent_content, full_response

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Failed to parse API response: {response_text}")
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
