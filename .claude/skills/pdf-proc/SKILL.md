---
name: pdf-proc
description: FastAPI service that watches a directory for invoice PDFs, extracts
  structured billing data via the OpenAI Responses API (gpt-4.1-mini / gpt-5.4-mini),
  then renames or splits the PDFs and serves a web UI. Use this skill when working
  in the pdf_proc repo — modifying the OpenAI Responses API integration, the JSON
  extraction schema, the filein/ → processed/ pipeline, PDF splitting or renaming
  logic, FastAPI endpoints, the Watchdog file watcher, Docker Compose or Fly.io
  deployment, or the static web UI.
---

## When to use this skill

- Adding or changing FastAPI endpoints in `main.py`
- Modifying the OpenAI Responses API call or JSON schema in `invoice_data_extractor.py`
- Changing how PDFs are renamed (`pdf_renamer.py`) or split (`main.py` PDFSplitter)
- Debugging the `filein/` → `processed/` pipeline or file watcher
- Updating the system prompt in `PDFProc_Prompt.txt`
- Working on the web UI (`static/index.html`)
- Docker Compose or Fly.io deployment changes

---

## Key files

| File | Purpose |
|---|---|
| `main.py` | FastAPI app, Watchdog file watcher, PDF splitting, all API endpoints |
| `invoice_data_extractor.py` | OpenAI Responses API integration, JSON schema, token tracking |
| `pdf_renamer.py` | Sanitizes and renames PDFs based on extracted data |
| `PDFProc_Prompt.txt` | System prompt loaded at runtime; contains company shortname dictionary |
| `static/index.html` | Frontend web UI |
| `compose.yml` | Docker Compose dev setup (port 8000) |

---

## File processing flow

1. **Watchdog** monitors `filein/` for new PDFs → queues them for processing.
2. **`/process-all/`** endpoint also queues all PDFs currently in `filein/`.
3. **`/upload/`** accepts ZIP files → **`/process/{filename}`** extracts PDFs into `filein/`.
4. `InvoiceDataExtractor.extract_data()` uploads the PDF to OpenAI (`purpose="user_data"`), calls the Responses API.
5. If **multiple documents** detected (multiple `document_id` values): `PDFSplitter` splits into separate PDFs → moved to `processed/`, original → `processed/processed_originals/`.
6. If **single document**: `PDFRenamer` renames to `"Company PO# Invoice#.pdf"` → moved to `processed/`, original → `processed/processed_originals/`.
7. Failed files tracked in `processing_status` dict; can be moved to `processed/failed/`.
8. PDF extension normalization: `.PDF` → `.pdf` on ingest (both in `PDFHandler.on_created()` and `process_all_files()`).

---

## OpenAI Responses API conventions

- Client: `AsyncOpenAI` from the `openai` package.
- Call: `client.responses.create` — **not** `client.chat.completions.create` or `client.beta.assistants`.
- Model: `gpt-5.4-mini` (does **not** support `temperature`).
- File uploads: `purpose="user_data"` — **not** `"assistants"`.
- System prompt: passed via `instructions=` parameter (preferred over a system role message).
- Structured output: `text.format` with `json_schema` + `strict: True` — eliminates free-form response failures and the need for `temperature=0`.
- Message content types: `input_text` and `input_file` (with `file_id`).
- Response text: accessed via `response.output_text`.
- Token usage: `response.usage.input_tokens`, `output_tokens`, `input_tokens_details.cached_tokens`.
- Session-level token counters tracked as class variables on `InvoiceDataExtractor`.

---

## JSON schema & internal data shape

### OpenAI response (JSON)

```json
{
  "invoices": [
    {
      "document_id": 1,
      "page_number": 1,
      "company": "IML",
      "purchase_order": "271181sw",
      "invoice_number": "9003328979"
    }
  ]
}
```

### Internal representation (after parsing)

`invoice_data_extractor.py` converts the JSON into a tuple-keyed dict consumed by `main.py`, `PDFRenamer`, and `PDFSplitter`:

```python
{ (document_id, page_number): ["CompanyName", "PO#", "InvoiceNumber"] }

# Single document, two pages
{(1, 1): ["IML", "271181sw", "9003328979"], (1, 2): ["IML", "271181sw", "9003328979"]}

# Two separate invoices in one PDF
{(1, 1): ["Graybar", "123456ab", "INV-001"], (2, 1): ["Fastenal", "654321cd", "INV-002"]}
```

- **Keys**: `(document_id, page_number)` — `document_id` groups pages of the same invoice.
- **Values**: `[company_shortname, purchase_order, invoice_number]` — empty string `""` if a field cannot be extracted.

---

## Extraction rules

### Company name
- Spaces and punctuation removed; capitalization retained.
- Matched case-insensitively against the shortname dictionary in `PDFProc_Prompt.txt`. First match wins.
- Example: `"The Parts Works"` → `"TPW"`

### Purchase order number
- Always **6 digits followed by 2+ lowercase letters** (e.g. `123456swz`).
- Any uppercase letters are converted to lowercase.

### Invoice number
- Extracted from the field nearest to the label "Invoice Number" on the page.
- Empty string `""` if not found.

---

## Directory structure

| Directory | Purpose |
|---|---|
| `filein/` | Input: watched for new PDFs |
| `upload/` | Uploaded ZIP files |
| `processed/` | Final renamed/split PDFs |
| `processed/processed_originals/` | Pre-rename originals |
| `processed/failed/` | Files that failed processing |
| `processed/OCR/` | OCR output (ocrmypdf sidecar) |
| `temp_splits/` | Temporary split PDFs during processing |
| `static/` | Frontend static files |

---

## API endpoints

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

---

## Common tasks / recipes

### Add a new API endpoint
1. Add the route function in `main.py` using FastAPI decorators (`@app.get`, `@app.post`, etc.).
2. Update the endpoint table in `CLAUDE.md` and this skill.
3. Add a curl example to `README.md` and an `.http` file under `api/`.

### Add or change an extracted field
1. Update the JSON schema in `invoice_data_extractor.py` (the `json_schema` dict).
2. Update the system prompt in `PDFProc_Prompt.txt` to instruct the model on the new field.
3. Update the internal tuple-keyed dict parsing logic in `invoice_data_extractor.py`.
4. Update consumers: `pdf_renamer.py`, `PDFSplitter` in `main.py`, and any status/logging code.

### Add a company shortname
Edit `PDFProc_Prompt.txt` — the shortname dictionary is defined there. Format: `"NormalizedKey": "Shortname"`.

### Debug a failed PDF
1. Check `GET /processing-status/{filename}` for the error message.
2. Look at the raw OpenAI response — add a `print(response.output_text)` temporarily in `extract_data()`.
3. Verify the PDF is not corrupted and that OCR ran if the PDF is image-only.
4. Check token usage — very large PDFs may hit context limits.

### Run locally
```bash
# Docker Compose (recommended)
docker compose up

# Direct
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Set `OPENAI_API_KEY` in a `.env` file or as an environment variable.

---

## Infrastructure

- **Docker Compose** (`compose.yml`): `python:3.11-slim-bullseye` + `ocrmypdf-alpine` sidecar, port 8000.
- **Deployment**: Fly.io (see `api/flyio.http`, `api/render.yaml`). API key via `FLY_OPENAI_API_KEY` env var (falls back to `OPENAI_API_KEY`).
- **Dev**: `uvicorn main:app --reload` on port 8000.
