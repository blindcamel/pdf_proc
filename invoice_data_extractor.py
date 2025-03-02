import ast
import logging
import os
import re
from typing import List, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class InvoiceDataExtractor:
    """Handles extraction of invoice data using OpenAI API"""

    def __init__(self, api_key: Optional[str] = None):
        # Load environment variables from .env file
        load_dotenv()

        # Get API key from environment variables or argument
        api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key is required")

        # Initialize the async OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)

    def _create_prompt(self, text: str) -> str:
        return f"""
        Extract the following details from this invoice text:
        1. Company Name (no spaces, retain capitalization)
        2. Purchase Order Number (all lowercase)
        3. Invoice Number
        Return the results in a plain text comma separated list between brackets like ['CompanyName','purchaseorder#','invoice#'], 
        using 'error' if any value is uncertain.
        
        Do NOT format the output inside triple backticks or as a code block.

        Invoice text:
        \"\"\"{text}\"\"\" 
        """

    async def extract_data(self, text: str) -> Optional[List[str]]:
        """
        Extract invoice data from text using OpenAI API.
        Returns: List of [CompanyName, PO#, Invoice#] or None if extraction fails.
        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use chat-based model
                messages=[{"role": "user", "content": self._create_prompt(text)}],
                max_tokens=100,
                temperature=0,
            )

            result = response.choices[0].message.content.strip()

            # Validate and parse response
            try:
                # Remove triple backticks and language hints if present
                cleaned_result = re.sub(r"^```(?:json|python)?\n|\n```$", "", result.strip())

                # Safely evaluate the string as a Python list
                data = ast.literal_eval(cleaned_result)

                # Validate expected structure
                if not isinstance(data, list) or len(data) != 3:
                    raise ValueError("Invalid response format: Expected a list with 3 items.")

                return data

            except (SyntaxError, ValueError):
                logger.error(f"Failed to parse API response: {cleaned_result}")
                return None

        except Exception as e:
            logger.error(f"API extraction error: {str(e)}")
            return None