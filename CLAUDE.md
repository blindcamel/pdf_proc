# pdf_proc — guardrails

FastAPI + OpenAI Responses API service for invoice PDF extraction, renaming, and splitting.

**Full playbook:** `.claude/skills/pdf-proc/SKILL.md`

---

## Non-negotiables

- Use the **Responses API** (`client.responses.create`) — never the Assistants API.
- Model: `gpt-5.4-mini` — does **not** support `temperature`.
- File uploads: `purpose="user_data"` (not `"assistants"`).
- Structured output via `text.format` + `json_schema` + `strict: True`.
- System prompt passed via `instructions=` parameter.
- Response text accessed via `response.output_text`.

## Do not reintroduce

- `OPENAI_ASSISTANT_ID` env var or any assistant ID references
- `OpenAI-Beta: assistants=v2` headers
- `purpose="assistants"` file uploads
- `/v1/assistants/...` API endpoints
- `temperature` parameter
- `client.chat.completions.create` or `client.beta.assistants` calls
