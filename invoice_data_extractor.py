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
    """Handles extraction of invoice data using OpenAI Assistant API"""

    def __init__(
        self, api_key: Optional[str] = None, assistant_id: Optional[str] = None
    ):
        # Load environment variables from .env file
        load_dotenv()

        # Get API key with priority:
        # 1. Explicitly passed api_key parameter
        # 2. Fly.io secrets or other secret stores via _get_secret()
        # 3. Environment variable OPENAI_API_KEY
        api_key = api_key or self._get_secret() or os.getenv("OPENAI_API_KEY")

        # Get Assistant ID from environment or parameter
        assistant_id = (
            assistant_id
            or os.getenv("OPENAI_ASSISTANT_ID")
            or os.getenv("FLY_OPENAI_ASSISTANT_ID")
        )

        if not api_key:
            logger.warning("OpenAI API key not found. Some features may not work.")
        if not assistant_id:
            logger.warning("OpenAI Assistant ID not found. Some features may not work.")

        # Initialize the async OpenAI client if credentials are available
        if api_key:
            self.client = AsyncOpenAI(api_key=api_key)
            self.assistant_id = assistant_id
        else:
            self.client = None
            self.assistant_id = None

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

    async def extract_data(self, file_path_or_text):
        """
        Extract invoice data from a PDF file or text using OpenAI API.
        Accepts either a file path to a PDF or text content.
        Returns: Tuple of (extracted_data, sent_content, api_response)
        """
        try:
            # Determine if input is a file path or text
            is_file_path = not isinstance(file_path_or_text, str) or (
                isinstance(file_path_or_text, str) and os.path.exists(file_path_or_text)
            )

            if is_file_path:
                # Handle PDF file
                file_path = file_path_or_text

                with open(file_path, "rb") as file:
                    file_content = file.read()

                # Upload the file and get file_id
                file_id = await self._upload_file(file_content)

                # Create the message with PDF attachment
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract invoice data from this PDF."},
                            {"type": "file", "file": {"file_id": file_id}},
                        ],
                    },
                ]

                # Call the API
                response = await self.client.chat.completions.create(
                    # model="gpt-5.4-mini", messages=messages, max_completion_tokens=1000
                    model="gpt-4o", messages=messages, max_completion_tokens=1000
                )

                response_text = response.choices[0].message.content
                sent_content = f"PDF file: {file_path}"

                full_response = {"status": "success", "response_text": response_text}

            else:
                # Handle text content (maintaining backward compatibility)
                text = file_path_or_text

                # Create and run a new thread with the assistant
                thread_run = await self.client.beta.threads.create_and_run(
                    assistant_id=self.assistant_id,
                    thread={"messages": [{"role": "user", "content": text}]},
                )

                # Wait for the assistant's response
                run = await self.client.beta.threads.runs.retrieve(
                    thread_id=thread_run.thread_id, run_id=thread_run.id
                )

                while run.status not in ["completed", "failed"]:
                    run = await self.client.beta.threads.runs.retrieve(
                        thread_id=thread_run.thread_id, run_id=thread_run.id
                    )

                if run.status == "failed":
                    logger.error("Assistant processing failed.")
                    return None, text, {"status": "failed", "response": None}

                # Fetch messages from the thread
                messages = await self.client.beta.threads.messages.list(
                    thread_id=thread_run.thread_id
                )
                response_text = messages.data[0].content[0].text.value.strip()
                sent_content = text

                # Store the full response object for debugging
                full_response = {
                    "thread_id": thread_run.thread_id,
                    "run_id": thread_run.id,
                    "status": run.status,
                    "response_text": response_text,
                    "messages": [
                        {
                            "role": msg.role,
                            "content": [
                                c.text.value if hasattr(c, "text") else str(c)
                                for c in msg.content
                            ],
                        }
                        for msg in messages.data
                    ],
                }

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
                file_path_or_text
                if isinstance(file_path_or_text, str)
                else str(file_path_or_text),
                {"status": "api_error", "error": str(e), "response": None},
            )
