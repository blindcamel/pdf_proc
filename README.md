# PDF Data Extractor

**PDF Data Extractor** extracts three key elements from single-page invoice PDFs:  
- **Invoice Number**  
- **Purchase Order Number**  
- **Company Name**  

## Output Format
Extracted data is returned as a key:value pair.

example 1:

{
(1,1):["CompanyA", "Purchase_Order#", "Invoice_Number"]
}

example 2:

{
    (1, 1):["CompanyA", "456123abc", "INV456"],
    (1, 2):["CompanyA", "456785swz", "INV457"]
}


**Keys**: Tuples `(document_id, page_number)`  
  - `document_id`: Groups pages belonging to the same document.  
  - `page_number`: Page sequence within that document.  

**Values**: Lists of extracted strings in this order:  
  1. **Company Name**  
  2. **Purchase Order Number**  
  3. **Invoice Number**  

If extraction is uncertain, the value is replaced with "error".


## Extraction Rules
### Company Name
- Capitalization is retained.  
- Spaces are removed: **"The Parts Works" → "ThePartsWorks"**  
- Only letters; **no special characters**.  

### Purchase Order Number
- Always **6 digits followed by 2+ lowercase letters**.  
  - Example: `"123456swz"`  
- Converts any letters to **lowercase**.  

### Invoice Number
- Extracted from the text string **closest to the phrase "Invoice Number"**.  
- If extraction is uncertain, `"error"` is returned.  

-------


http://localhost:8000
"endpoints": {
    "POST /upload": "Upload and process a new PDF file",
    "POST /process-file": "Process an existing file from the input directory",
    "POST /process-all": "Process all files in /uploads",
    "GET /list-files": "List all PDF files in the input directory",
    "GET /debug-paths": "debug paths",
    "GET /processing-status": "List all",
    "GET /processing-status/{filename}": "List one",
    "DELETE /processing-status": "Clear all",
    "DELETE /processing-status/{filename}": "Clear one",
    "GET /": "This information",
}

example:
curl -X POST "http://localhost:8000/upload/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@./filein/aaa.pdf"

curl -X GET "http://localhost:8000/list-files/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \


curl -X POST "http://localhost:8000/process-file/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@./filein/TPW98705.pdf"

curl -X POST http://localhost:8000/process-all/

#get processing status
curl -X GET http://localhost:8000/processing-status/TPW98705.pdf