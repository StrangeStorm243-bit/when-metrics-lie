# Spectra Frontend

Next.js frontend for Spectra evaluation engine.

## Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Environment Variables

Create `.env.local` with:
```
NEXT_PUBLIC_SPECTRA_API_BASE=http://127.0.0.1:8000
```

Make sure the backend is running on the configured port before using the frontend.

## Project Structure

**Canonical frontend root:** `web/frontend/`

All frontend files should be located directly under `web/frontend/`:
- `web/frontend/components/` - React components
- `web/frontend/lib/` - Utility functions
- `web/frontend/app/` - Next.js app router pages
- `web/frontend/tailwind.config.ts` - Tailwind configuration
- `web/frontend/postcss.config.js` - PostCSS configuration

**Note:** If `web/frontend/web/` exists, it is accidental and should be deleted when not locked.

Check structure: `./scripts/check_structure.ps1`

## How to Run Structure Check

From repo root:
```powershell
./web/frontend/scripts/check_structure.ps1
```

From `web/frontend/`:
```powershell
./scripts/check_structure.ps1
```

