## Page Mapping Structure
* **Keys** are tuples of `(document_id, page_number)`
   * `document_id`: Groups pages that belong to the same document
   * `page_number`: Sequence number within the document
* **Values** are lists of strings containing:
   * [0]: Company name
   * [1]: Purchase order number
   * [2]: Invoice number
   
Pages with the same document_id belong together in sequence.



Application uses a dictionary to track document pages and their metadata:

```python
page_mapping: Dict[Tuple[int, int], List[str]] = {
    (1, 1): ["Company A", "PO123", "INV456"],
    (1, 2): ["Company A", "PO124", "INV457"],
    (2, 1): ["Company B", "PO789", "INV012"],
    (3, 1): ["Company C", "PO345", "INV678"]
}
```


http://localhost:8000
    {
    "message": "PDF Processing API",
    "endpoints": {
        "POST /upload": "Upload and process a new PDF file",
        "POST /process-file": "Process an existing file from the input directory",
        "GET /list-files": "List all PDF files in the input directory",
        "GET /": "This information"
    }
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