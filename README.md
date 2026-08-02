# PDF Data Extractor

**PDF Data Extractor** is a FastAPI service that processes invoice PDFs — extracting key billing data, renaming files, and organizing them for downstream use. It accepts ZIP archives of PDFs via a web UI or REST API, sends each PDF to OpenAI for structured data extraction, renames the files based on the extracted data, and makes the results available for download.

## How It Works

1. **Upload** a ZIP file containing one or more invoice PDFs via the web UI or `POST /upload/`.
2. **Trigger processing** by selecting the ZIP and clicking "Process Selected File" (or `POST /process/{filename}`). PDFs are extracted from the ZIP into the `filein/` watch directory.
3. **Automatic processing** begins immediately — the file watcher detects new PDFs and queues them for extraction.
4. Each PDF is uploaded to OpenAI's **Responses API** (`gpt-4.1-mini`) with a structured JSON schema. The model extracts:
   - **Company Name** — normalized and matched against a shortname dictionary
   - **Purchase Order Number** — 6 digits + 2+ lowercase letters (e.g. `123456ab`)
   - **Invoice Number**
5. Files are **renamed** to `CompanyName PO# InvoiceNumber.pdf` and moved to `processed/`.
6. Multi-invoice PDFs (multiple documents in one file) are **split** into separate files.
7. **Download** all processed files as a ZIP via the web UI or `GET /download`.

## OpenAI Integration (branch_5)

This branch migrates from the legacy **Assistants API** to the **Responses API**:

- Uses `client.responses.create` with `model="gpt-4.1-mini"`
- PDF files are uploaded with `purpose="user_data"` and passed as `input_file` content blocks
- System prompt loaded from `PDFProc_Prompt.txt` at runtime via the `instructions=` parameter
- Structured output enforced via `text.format json_schema` with `strict=True` — eliminates free-form response failures
- Token usage (input, cached, output) is logged per-request and as running session totals
- `dev-requirements.txt` updated: `openai` bumped to `1.109.1`; removed stray OS packages (`mupdf`, `poppler-utils`) and invalid PyPI entries

## Case-Sensitivity Fix (branch_5)

Files with uppercase extensions (e.g. `.PDF`) are now normalized to lowercase (`.pdf`) in two places:
- `PDFHandler.on_created()` — when the file watcher detects a new file
- `process_all_files()` — when manually triggering processing of all files in `filein/`

## Company Name Normalization

Company names are normalized and matched against a shortname dictionary (defined in `PDFProc_Prompt.txt`). The process:

1. **Normalize**: Remove all spaces and punctuation, retain letter case.  
   Example: `"The Parts Works"` → `"ThePartsWorks"`
2. **Match**: Case-insensitive substring match against dictionary keys. First match wins.  
   Example: `"IMLSecuritySupply"` → `"IML"`
3. **No match**: Use the normalized name as-is.

## Output Format

### OpenAI response (JSON)

OpenAI returns structured JSON matching the enforced schema:

```json
{
  "documents": [
    {
      "document_id": 1,
      "pages": [
        {"page_number": 1, "company": "IML", "purchase_order": "271181sw", "invoice_number": "9003328979"},
        {"page_number": 2, "company": "IML", "purchase_order": "271181sw", "invoice_number": "9003328979"}
      ]
    }
  ]
}
```

Example — two separate invoices in one PDF:
```json
{
  "documents": [
    {"document_id": 1, "pages": [{"page_number": 1, "company": "Graybar", "purchase_order": "123456ab", "invoice_number": "INV-001"}]},
    {"document_id": 2, "pages": [{"page_number": 1, "company": "Fastenal", "purchase_order": "654321cd", "invoice_number": "INV-002"}]}
  ]
}
```

### Internal representation (after parsing)

`invoice_data_extractor.py` converts the JSON into a tuple-keyed dict that `main.py`, `PDFRenamer`, and `PDFSplitter` all consume:

```
{ (document_id, page_number): ["CompanyName", "PO#", "InvoiceNumber"] }
```

Same examples above, after conversion:
```python
# Single document, two pages
{(1, 1): ["IML", "271181sw", "9003328979"], (1, 2): ["IML", "271181sw", "9003328979"]}

# Two separate invoices
{(1, 1): ["Graybar", "123456ab", "INV-001"], (2, 1): ["Fastenal", "654321cd", "INV-002"]}
```

**Keys**: Tuples `(document_id, page_number)`
  - `document_id`: Groups pages belonging to the same invoice document.
  - `page_number`: Page sequence within that document.

**Values**: Lists in this order:
  1. **Company Name** (normalized/shortname)
  2. **Purchase Order Number**
  3. **Invoice Number**

If a field cannot be extracted, an empty string `""` is used.

## Extraction Rules

### Company Name
- Spaces and punctuation removed; capitalization retained.
- Matched against shortname dictionary (case-insensitive partial match).
- Example: `"The Parts Works"` → `"TPW"`

### Purchase Order Number
- Always **6 digits followed by 2+ lowercase letters**.
- Example: `"123456swz"`
- Any uppercase letters are converted to lowercase.

### Invoice Number
- Extracted from the field nearest to the label "Invoice Number" on the page.
- Empty string `""` if not found.

## Directory Structure

```
filein/                     # Watch directory — drop PDFs here for auto-processing
upload/                     # Uploaded ZIP files
processed/                  # Renamed, processed PDFs
processed/processed_originals/  # Original (pre-rename) copies
processed/failed/           # Files that failed processing
processed/OCR/              # OCR output (reserved)
temp_splits/                # Temporary storage during multi-doc splitting
static/                     # Web UI (index.html)
```

## Running Locally

**Docker Compose (recommended for development):**
```bash
docker compose up
```

**Direct (with Python 3.11+):**
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Set `OPENAI_API_KEY` in a `.env` file or as an environment variable.

---

## API Reference

**Base URL (local):** `http://localhost:8000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI (index.html) |
| `GET` | `/root` | API info / endpoint listing |
| `POST` | `/upload/` | Upload a ZIP file |
| `POST` | `/process/{filename}` | Extract PDFs from a ZIP and queue for processing |
| `POST` | `/process-all/` | Queue all PDFs currently in `filein/` |
| `GET` | `/download` | Download all processed PDFs as a ZIP |
| `POST` | `/cleanup` | Remove all files in `upload/`, `processed/`, and `filein/` |
| `GET` | `/list-upload/` | List ZIP files in the upload directory |
| `GET` | `/list-input` | List PDF files in the input directory |
| `GET` | `/processing-status/` | Get status of all processed files |
| `GET` | `/processing-status/{filename}` | Get status of a specific file |
| `DELETE` | `/processing-status/` | Clear all processing statuses |
| `DELETE` | `/processing-status/{filename}` | Clear status for a specific file |
| `GET` | `/debug-paths/` | Show resolved directory paths |
| `GET` | `/health` | Health check |

### curl Examples

```bash
# Upload a ZIP file
curl -X POST "http://localhost:8000/upload/" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@./041025.zip"

# Process a specific uploaded ZIP
curl -X POST "http://localhost:8000/process/041025.zip"

# Process all PDFs currently in filein/
curl -X POST http://localhost:8000/process-all/

# Download processed files
curl -O -J -L http://localhost:8000/download

# Get processing status for all files
curl -X GET http://localhost:8000/processing-status/

# Get processing status for a specific file
curl -X GET http://localhost:8000/processing-status/TPW98705.pdf

# Clear all processing statuses
curl -X DELETE http://localhost:8000/processing-status/

# List uploaded ZIPs
curl -X GET http://localhost:8000/list-upload/

# List input PDFs
curl -X GET http://localhost:8000/list-input

# Cleanup all files
curl -X POST http://localhost:8000/cleanup

# Health check
curl -X GET http://localhost:8000/health
```
