import ast
import logging
import os
import re
import boto3
from botocore.exceptions import ClientError
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

        # Get API key and Assistant ID from secrets or environment variables or argument
        api_key = api_key or self._get_secret() or os.getenv("OPENAI_API_KEY")
        assistant_id = assistant_id or os.getenv(
            "OPENAI_ASSISTANT_ID"
        )  # Store your assistant ID

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

    def _get_secret(self):
        """Retrieve API key from AWS Secrets Manager"""
        try:
            secret_name = "prod/pdf_proc/openai_api_keys"
            region_name = "us-west-2"

            # Create a Secrets Manager client
            session = boto3.session.Session()
            client = session.client(
                service_name="secretsmanager", region_name=region_name
            )

            # Get the secret value
            get_secret_value_response = client.get_secret_value(SecretId=secret_name)

            # Parse and return the secret
            secret = get_secret_value_response["SecretString"]
            # If secret is in JSON format, you'll need to parse it
            # This assumes the secret contains a key called 'OPENAI_API_KEY'
            import json

            secret_dict = json.loads(secret)
            return secret_dict.get("OPENAI_API_KEY")

        except ClientError as e:
            logger.error(f"Error retrieving secret: {str(e)}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing secret JSON: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error retrieving secret: {str(e)}")
            return None

    async def extract_data(self, text: str):
        """
        Extract invoice data from text using OpenAI Assistant API.
        Returns: List of dictionaries containing page mapping data with document and page identifiers
        as tuple keys and extraction data as values, or None if extraction fails.
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
                return None, text, {"status": "failed", "response": None}

            # Fetch messages from the thread
            messages = await self.client.beta.threads.messages.list(
                thread_id=thread_run.thread_id
            )
            response_text = messages.data[0].content[0].text.value.strip()

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

                # Check if it's a dictionary with tuple keys (expected format)
                if isinstance(data_obj, dict):
                    # Validate structure - dictionary should have tuple keys and list values
                    if not all(
                        isinstance(k, tuple) and len(k) == 2 for k in data_obj.keys()
                    ):
                        raise ValueError(
                            "Invalid key format: Expected (document_id, page_number) tuples"
                        )

                    if not all(
                        isinstance(v, list) and len(v) == 3 for v in data_obj.values()
                    ):
                        raise ValueError(
                            "Invalid value format: Expected [CompanyName, PO#, Invoice#] lists"
                        )

                    return data_obj, text, full_response

                raise ValueError(
                    "Invalid response format: Expected a dictionary with tuple keys."
                )

            except (SyntaxError, ValueError) as e:
                logger.error(f"Failed to parse assistant response: {cleaned_result}")
                logger.error(f"Parsing error: {str(e)}")
                return (
                    None,
                    text,
                    {
                        "status": "parse_error",
                        "error": str(e),
                        "response": full_response,
                    },
                )

        except Exception as e:
            logger.error(f"API extraction error: {str(e)}")
            return (
                None,
                text,
                {"status": "api_error", "error": str(e), "response": None},
            )
