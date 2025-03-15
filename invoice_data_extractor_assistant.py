import ast
import logging
import os
import re
from typing import List, Optional
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

        # Get API key and Assistant ID from environment variables or argument
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        assistant_id = assistant_id or os.getenv(
            "OPENAI_ASSISTANT_ID"
        )  # Store your assistant ID

        if not api_key:
            raise ValueError("OpenAI API key is required")
        if not assistant_id:
            raise ValueError("OpenAI Assistant ID is required")

        # Initialize the async OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        self.assistant_id = assistant_id

    async def extract_data(self, text: str) -> Optional[List[str]]:
        """
        Extract invoice data from text using OpenAI Assistant API.
        Returns: List of [CompanyName, PO#, Invoice#] or None if extraction fails.
        """
        try:
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
                return None

            # Fetch messages from the thread
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_run.thread_id
            )
            response_text = messages.data[0].content[0].text.value.strip()

            # Validate and parse response
            try:
                # Remove unnecessary formatting if present
                cleaned_result = re.sub(
                    r"^```(?:json|python)?\n|\n```$", "", response_text.strip()
                )

                # Safely evaluate the string as a Python object
                data_obj = ast.literal_eval(cleaned_result)

                # Check if it's a dictionary with tuple keys
                if isinstance(data_obj, dict) and any(
                    isinstance(k, tuple) for k in data_obj.keys()
                ):
                    # Get the first value, which should be our list
                    for value in data_obj.values():
                        if isinstance(value, list) and len(value) == 3:
                            return value
                    raise ValueError("No valid data list found in dictionary values")

                # Handle direct list format (original expectation)
                elif isinstance(data_obj, list) and len(data_obj) == 3:
                    return data_obj

                raise ValueError(
                    "Invalid response format: Expected a list with 3 items or dict with list values."
                )

            except (SyntaxError, ValueError) as e:
                logger.error(f"Failed to parse assistant response: {cleaned_result}")
                logger.error(f"Parsing error: {str(e)}")
                return None

        except Exception as e:
            logger.error(f"API extraction error: {str(e)}")
            return None
