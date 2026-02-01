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

