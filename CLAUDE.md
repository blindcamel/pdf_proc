# Project Notes for Claude/Cline

## OpenAI Integration

- Uses the **Responses API** (`client.responses.create`), **not** the Assistants API.
- File uploads use `purpose="user_data"` (not `"assistants"`).
- System prompt is loaded from `PDFProc_Prompt.txt` at runtime.
- Model: `gpt-4o`, `temperature=0` for deterministic extraction.
- Message content types: `input_text` and `input_file` (with `file_id`).
- Response text accessed via `response.output_text`.

## Do Not Reintroduce

- `OPENAI_ASSISTANT_ID` env var or assistant ID references.
- `OpenAI-Beta: assistants=v2` headers.
- `purpose="assistants"` file uploads.
- `/v1/assistants/...` API endpoints.
