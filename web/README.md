# Spectra Web UI

This directory contains the web frontend and backend for Spectra MVP UI.

## Quick Start

### Prerequisites

- Python 3.8+ with `pip`
- Node.js 18+ with `npm`
- Backend dependencies installed: `pip install -e ".[web]"` (from repo root)

### Running the Backend

1. Navigate to the backend directory:
   ```bash
   cd web/backend
   ```

2. Start the FastAPI server:
   ```bash
   python -m uvicorn app.main:app --reload --port 8000
   ```

   The server will be available at `http://localhost:8000`.
   - API docs: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

### Running the Frontend

1. Navigate to the frontend directory:
   ```bash
   cd web/frontend
   ```

2. Install dependencies (first time only):
   ```bash
   npm install
   ```

3. Create `.env.local` (if it doesn't exist):
   ```bash
   echo "NEXT_PUBLIC_SPECTRA_API_BASE=http://127.0.0.1:8000" > .env.local
   ```

4. Start the Next.js development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at `http://localhost:3000`.

### Seeding a Demo Experiment

To quickly create and run a demo experiment:

1. Make sure the backend is running (see above).

2. Run the demo seed script:
   ```bash
   cd web/backend
   python demo_seed.py
   ```

3. The script will:
   - Create an experiment using the first available metric and stress suite preset
   - Run the experiment
   - Print the experiment ID and frontend URL

4. Open the printed URL in your browser to view the results.

## Demo Flow Checklist

1. ✅ **Start Backend**: Run `uvicorn` in `web/backend`
2. ✅ **Start Frontend**: Run `npm run dev` in `web/frontend`
3. ✅ **Seed Demo**: Run `python demo_seed.py` in `web/backend`
4. ✅ **View Results**: Open the printed URL in your browser
5. ✅ **Try Sharing**: Use "Copy Summary" or "Copy JSON" buttons
6. ✅ **Try Printing**: Use "Print" button to see print-friendly layout

## Project Structure

- **`backend/`**: FastAPI backend that bridges to the core Spectra engine
- **`frontend/`**: Next.js frontend (App Router) for the web UI

## Features

- **Create & Run Experiments**: Use `/new` to create experiments with metric and stress suite presets
- **View Results**: See headline scores, component scores, scenario results, and flags
- **Share Results**: Copy human-readable summaries or full JSON
- **Print Support**: Print-friendly layout for reports

## Troubleshooting

- **Backend won't start**: Ensure dependencies are installed with `pip install -e ".[web]"` from repo root
- **Frontend can't connect**: Check that `NEXT_PUBLIC_SPECTRA_API_BASE` in `.env.local` matches your backend URL
- **No dataset found**: Ensure a CSV dataset exists in `data/` or set `config.dataset_path` when creating experiments

## See Also

- `backend/README.md` - Backend architecture and API details
- `frontend/README.md` - Frontend development guide

