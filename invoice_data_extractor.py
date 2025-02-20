import json
import logging
from typing import List, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

class InvoiceDataExtractor:
    """Handles extraction of invoice data using OpenAI API"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        
    def _create_prompt(self, text: str) -> str:
        return f"""
        Extract the following details from this invoice text:
        1. Company Name (no spaces, retain capitalization)
        2. Purchase Order Number (all lowercase)
        3. Invoice Number
        Return the results in a Python list format like ['CompanyName','purchaseorder#','invoice#'], 
        using 'error' if any value is uncertain.
        
        Invoice text:
        \"\"\"{text}\"\"\""""

    async def extract_data(self, text: str) -> Optional[List[str]]:
        """
        Extract invoice data from text using OpenAI API
        Returns: List of [CompanyName, PO#, Invoice#] or None if extraction fails
        """
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "user", "content": self._create_prompt(text)}
                ],
                temperature=0
            )
            
            result = response.choices[0].message.content.strip()
            
            # Validate response format
            try:
                data = json.loads(result.replace("'", '"'))
                if not isinstance(data, list) or len(data) != 3:
                    raise ValueError("Invalid response format")
                return data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse API response: {result}")
                return None
                
        except Exception as e:
            logger.error(f"API extraction error: {str(e)}")
            return None