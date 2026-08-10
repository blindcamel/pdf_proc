# Project Notes for Claude/Cline

## Project Overview

FastAPI application that watches a directory for PDF files, extracts invoice data using OpenAI, renames/splits PDFs, and serves a web frontend.

**Key files:**
- `main.py` — FastAPI app, file watcher, PDF splitting, API endpoints
- `invoice_data_extractor.py` — OpenAI Responses API integration, JSON schema, token tracking
- `pdf_renamer.py` — Sanitizes and renames PDFs based on extracted data
- `PDFProc_Prompt.txt` — System prompt loaded at runtime
- `static/index.html` — Frontend UI
- `compose.yml` — Docker Compose dev setup (port 8000)

## OpenAI Integration

- Uses the **Responses API** (`client.responses.create`), **not** the Assistants API.
- Client: `AsyncOpenAI` from `openai` package.
- File uploads use `purpose="user_data"` (not `"assistants"`).
- System prompt passed via `instructions=` parameter (preferred over system role message).
- Model: `gpt-5.4-mini` (does **not** support `temperature` parameter).
- Structured output enforced via `text.format` with `json_schema` + `strict: True` — eliminates need for `temperature=0`.
- Message content types: `input_text` and `input_file` (with `file_id`).
- Response text accessed via `response.output_text`.
- Token usage: `response.usage.input_tokens`, `output_tokens`, and `input_tokens_details.cached_tokens`.
- Session-level token counters tracked as class variables on `InvoiceDataExtractor`.

## JSON Schema / Data Shape

The API returns a flat list of invoice objects (one per page/invoice):

```json
{
  "invoices": [
    {
      "document_id": 1,
      "page_number": 1,
      "company": "...",
      "purchase_order": "...",
      "invoice_number": "..."
    }
  ]
}
```

Internally converted to a tuple-keyed dict:
`{ (document_id, page_number): [company, purchase_order, invoice_number] }`

## File Processing Flow

1. **Watchdog** monitors `filein/` for new PDFs → queues them.
2. **`/process-all/`** endpoint also queues all PDFs currently in `filein/`.
3. **`/upload/`** accepts ZIP files → **`/process/{filename}`** extracts PDFs into `filein/`.
4. `InvoiceDataExtractor.extract_data()` uploads PDF to OpenAI, calls Responses API.
5. If **multiple documents** detected (multiple `document_id` values): `PDFSplitter` splits into separate PDFs → moved to `processed/`, original → `processed/processed_originals/`.
6. If **single document**: `PDFRenamer` renames to `"Company PO# Invoice#.pdf"` → moved to `processed/`, original → `processed/processed_originals/`.
7. Failed files tracked in `processing_status` dict; can be moved to `processed/failed/`.

## Directory Structure

| Directory | Purpose |
|---|---|
| `filein/` | Input: watched for new PDFs |
| `upload/` | Uploaded ZIP files |
| `processed/` | Final renamed/split PDFs |
| `processed/processed_originals/` | Pre-rename originals |
| `processed/failed/` | Failed processing |
| `processed/OCR/` | OCR output (ocrmypdf sidecar) |
| `temp_splits/` | Temporary split PDFs during processing |
| `static/` | Frontend static files |

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Frontend (index.html) |
| `POST` | `/upload/` | Upload a ZIP file |
| `POST` | `/process/{filename}` | Extract PDFs from ZIP into filein/ |
| `POST` | `/process-all/` | Queue all PDFs in filein/ |
| `GET` | `/download` | Download processed files as ZIP |
| `POST` | `/cleanup` | Remove files from upload/, processed/, filein/ |
| `GET` | `/list-upload/` | List ZIPs in upload/ |
| `GET` | `/list-input` | List PDFs in filein/ |
| `GET` | `/processing-status/` | All processing statuses |
| `GET` | `/processing-status/{filename}` | Status for one file |
| `DELETE` | `/processing-status/` | Clear all statuses |
| `DELETE` | `/processing-status/{filename}` | Clear one status |
| `GET` | `/debug-paths/` | Show directory paths |
| `GET` | `/health` | Health check |

## Infrastructure

- **Docker Compose** (`compose.yml`): `python:3.11-slim-bullseye` + `ocrmypdf-alpine` sidecar, port 8000.
- **Deployment**: Fly.io (see `api/flyio.http`, `api/render.yaml`). API key via `FLY_OPENAI_API_KEY` env var (falls back to `OPENAI_API_KEY`).
- **Dev**: `uvicorn main:app --reload` on port 8000.
- PDF extension normalization: `.PDF` → `.pdf` on ingest.

## Do Not Reintroduce

- `OPENAI_ASSISTANT_ID` env var or assistant ID references.
- `OpenAI-Beta: assistants=v2` headers.
- `purpose="assistants"` file uploads.
- `/v1/assistants/...` API endpoints.
- `temperature` parameter (not supported by current model).
- `client.chat.completions.create` or `client.beta.assistants` calls.
