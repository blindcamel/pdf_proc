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
     -F "file=@./filein/aaa.pdf"