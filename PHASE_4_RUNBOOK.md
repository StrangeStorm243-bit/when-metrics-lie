# Phase 4 Runbook + Status

## Phase 4 Status

### ✅ Completed in Phase 4

- **`/new`** - Create experiment with dataset selection dropdown
- **`/experiments/[id]`** - View experiment results with export/print/share
- **`/compare`** - Compare two runs side-by-side with deterministic analyst
- **`/assistant`** - Deterministic chat interface for loaded run context
- **Dataset dropdown** - Select from available CSV files in `data/` directory
- **HTTP 400 dataset errors** - Invalid dataset paths return 400 with clear error messages (not 500)

### ⚠️ Known Limits

- **No model upload yet** - Spectra evaluates existing model outputs, not training models
- **Assistant is deterministic** - LLM integration toggle shows "coming soon" unless `ANTHROPIC_API_KEY` is configured
- **Single dataset per experiment** - Each experiment uses one dataset; cross-dataset comparisons require manual interpretation
- **No global search** - Assistant requires context to be loaded for a specific run before answering

### ➡️ Phase 5 Entry Point

- Model upload + evaluation adapters
- True conversational query layer with tool-calling
- Scenario library expansion + metric scoring policies
- Hosted deployment & auth

## Quick Start Commands

### Windows PowerShell

**Backend:**
```powershell
cd web\backend
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```powershell
cd web\frontend
npm run dev
```

### macOS/Linux

**Backend:**
```bash
cd web/backend
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd web/frontend
npm run dev
```

## 5-Step Demo Script

1. **Create & Run Experiment**
   - Open `http://localhost:3000/new`
   - Enter experiment name
   - Select metric (e.g., "accuracy")
   - Select stress suite (e.g., "default")
   - Select dataset from dropdown (e.g., "demo_binary_label_noise.csv")
   - Click "Create & Run"
   - Wait for experiment to complete

2. **View Results**
   - Automatically redirected to `/experiments/{id}`
   - Verify results load: headline score, component scores, scenario results, flags
   - Test export JSON button
   - Test print functionality

3. **Re-run Experiment**
   - Click "Refresh Now" or navigate back to experiment page
   - Verify multiple runs exist in the experiment

4. **Compare Runs**
   - Open `http://localhost:3000/compare`
   - Select experiment from dropdown A
   - Select run from dropdown A
   - Select experiment from dropdown B (can be same or different)
   - Select run from dropdown B
   - Click "Compare"
   - Verify diff view shows: headline score change, component changes, scenario changes, flags diff
   - Test deterministic analyst prompt chips
   - (Optional) Toggle "Enhanced Analyst (LLM · optional)" ON to test AI features (requires ANTHROPIC_API_KEY)

5. **Assistant Chat**
   - Open `http://localhost:3000/assistant`
   - Verify pinned message appears: "I analyze experiment runs by stress-testing metrics..."
   - Select experiment from dropdown
   - Select run from dropdown
   - Click "Load Context"
   - Verify context status shows: "Context loaded: {experiment_name} • {run_id}"
   - Type a question (e.g., "What's the headline score?")
   - Click "Send"
   - Verify deterministic response mentions headline score, worst scenario, flags
   - Click "Reset" - verify pinned message reappears
   - (Optional) Toggle "Enhanced Analyst (LLM · optional)" ON (requires ANTHROPIC_API_KEY)

## Phase 4 Status

### What Exists

- **`/new`** - Create experiment with dataset selection
  - Metric and stress suite presets
  - Dataset dropdown from `/presets/datasets`
  - Automatic experiment execution after creation

- **`/experiments/[id]`** - Results view
  - Headline score and component scores
  - Scenario stress results with severity indicators
  - Flags organized by severity
  - Export JSON functionality
  - Print-friendly styling
  - Share functionality (copy summary/JSON)

- **`/compare`** - Run comparison
  - Side-by-side diff of headline scores, components, scenarios, flags
  - Deterministic analyst with prompt chips
  - Optional LLM-powered analyst (Phase 4B) with graceful fallback

- **`/assistant`** - Deterministic chat
  - Experiment and run selection
  - Context loading for specific runs
  - Deterministic responses based on loaded result data
  - Evidence-based answers mentioning headline score, worst scenario, flags
  - Optional LLM integration toggle (Phase 4B) - not required for demo

- **API Endpoints**
  - `GET /presets/datasets` - Returns relative paths like `data/demo_binary_label_noise.csv`
  - `GET /experiments/{id}/runs` - List runs for an experiment
  - `POST /llm/compare-explain` - LLM explanation (returns 501 if ANTHROPIC_API_KEY missing)

### Known Limitations

- **Assistant is deterministic only** - No full tool-calling or conversational follow-ups yet
- **No model upload** - Spectra evaluates existing model outputs, not training models
- **No multi-dataset comparability** - Each experiment uses a single dataset
- **No global search** - Assistant requires context to be loaded for a specific run
- **No production auth** - Designed for local development
- **LLM features optional** - Requires ANTHROPIC_API_KEY environment variable; gracefully falls back to deterministic

## Error Handling

### Dataset Errors
- Invalid or missing dataset paths return **HTTP 400** with helpful error message
- CSV reading errors are caught and return **HTTP 400** (not 500)
- Error messages include resolved absolute paths for debugging

### LLM Errors
- Missing `ANTHROPIC_API_KEY` returns **HTTP 501** (Not Implemented)
- Frontend shows warning: "ANTHROPIC_API_KEY not set — using deterministic analyst"
- All LLM errors fall back to deterministic analysis without crashing

## Build Verification

**Frontend Build:**
```powershell
cd web\frontend
npm run build
```

Should complete with:
- ✓ Compiled successfully
- ✓ Linting and checking validity of types
- ✓ Generating static pages (7/7)

**Backend Health Check:**
```powershell
curl http://localhost:8000/health
```

Should return: `{"status":"ok"}`

## Troubleshooting

### Dataset Not Found
- Check that CSV files exist in `data/` directory
- Verify dataset path in experiment config is relative (e.g., `data/demo_binary_label_noise.csv`)
- Check backend logs for resolved absolute path

### LLM Features Not Working
- Verify `ANTHROPIC_API_KEY` environment variable is set
- Check backend logs for 501 errors
- Frontend should show warning and fall back to deterministic

### Frontend Build Fails
- Run `npm install` in `web/frontend`
- Check Node.js version (requires Node 18+)
- Verify TypeScript config is correct

### Backend Crashes
- Check Python version (requires Python 3.10+)
- Verify dependencies: `pip install -r web/backend/requirements.txt`
- Check that repo root contains `pyproject.toml`
