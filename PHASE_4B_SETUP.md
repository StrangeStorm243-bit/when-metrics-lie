# Phase 4B: Claude-powered Analyst Agent Setup

## Overview
Phase 4B adds an optional AI-powered analyst using Claude API. The deterministic analyst remains the default and fallback.

## Backend Setup

### 1. Environment Variables
Set the following environment variable before starting the backend:

**Linux/macOS:**
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

**Windows (PowerShell):**
```powershell
$env:ANTHROPIC_API_KEY="your-api-key-here"
```

**Windows (CMD):**
```cmd
set ANTHROPIC_API_KEY=your-api-key-here
```

Optional (defaults to `claude-sonnet-4-20250514`):
```bash
export SPECTRA_LLM_MODEL="claude-sonnet-4-20250514"
```

### 2. Dependencies
No new dependencies required. Uses existing `requests` package from `web` optional dependencies.

### 3. Run Backend
```bash
cd web/backend
uvicorn app.main:app --reload --port 8000
```

## Frontend Setup

### 1. No Changes Required
Frontend uses existing dependencies. No new packages needed.

### 2. Run Frontend
```bash
cd web/frontend
npm run dev
```

## Usage

### 1. Without API Key (Deterministic Mode)
- Open `/compare` page
- Select two runs to compare
- "Enhanced Analyst (AI)" toggle is OFF by default
- All analyst responses use deterministic logic

### 2. With API Key (AI Mode)
- Set `ANTHROPIC_API_KEY` environment variable
- Restart backend
- Open `/compare` page
- Toggle "Enhanced Analyst (AI)" ON
- Click analyst prompt chips or click rows to get AI-powered insights
- If API key is missing or request fails, automatically falls back to deterministic mode with a warning message

## API Endpoint

### POST `/llm/compare-explain`
**Request:**
```json
{
  "intent": "overview",
  "focus": {"type": "scenario", "key": "scenario_id"},
  "context": { ... },
  "user_question": "What got worse?"
}
```

**Response (200 OK):**
```json
{
  "title": "AI Analysis Title",
  "body_markdown": "Markdown formatted analysis...",
  "evidence_keys": ["scenario:scenario_id", "component:name"]
}
```

**Response (501 Not Implemented):**
```json
{
  "detail": "LLM features require ANTHROPIC_API_KEY environment variable to be set"
}
```

## Architecture

### Backend Files
- `web/backend/app/routers/llm.py` - LLM API router
- `web/backend/app/llm_contracts.py` - Pydantic request/response models
- `web/backend/app/services/claude_client.py` - Claude API client

### Frontend Files
- `web/frontend/lib/api.ts` - Added `compareExplain()` function
- `web/frontend/app/compare/page.tsx` - Added AI toggle and integration

### Context Bundle
The AI receives a compact context bundle including:
- Experiment labels (names, IDs, run IDs)
- Headline score comparison
- Worst-case scenario
- Top 10 scenario changes
- Top 10 component changes
- Flags summary (added/removed/persisting by severity)
- Top 5 added flags

## Testing

### Test Without API Key
1. Don't set `ANTHROPIC_API_KEY`
2. Start backend
3. Open `/compare`
4. Toggle AI ON
5. Click a prompt chip
6. Should see warning: "AI features require ANTHROPIC_API_KEY. Falling back to deterministic analyst."
7. Deterministic response should appear

### Test With API Key
1. Set `ANTHROPIC_API_KEY`
2. Start backend
3. Open `/compare`
4. Select two runs and click "Compare"
5. Toggle AI ON
6. Click a prompt chip
7. Should see AI-powered response with title "AI Analyst: ..."

## Error Handling
- 501 errors (no API key) → Fallback to deterministic with warning
- 500 errors (API failure) → Fallback to deterministic with error message
- Network errors → Fallback to deterministic with error message
- All errors are non-blocking; deterministic analyst always works

