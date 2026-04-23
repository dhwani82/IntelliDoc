# Document Parser – PDF Chat RAG

Start the full app (backend + frontend) and open it in your browser with one command.

## Quick start

From the **Document Parser** folder:

```bash
npm start
```

This will:

1. Free ports 8000 and 3000 if something is already using them  
2. Start the **backend** (FastAPI) in a new window → http://127.0.0.1:8000  
3. Start the **frontend** (Next.js) in a new window → http://127.0.0.1:3000  
4. Open your browser at **http://127.0.0.1:3000**

Two terminal windows will stay open (backend and frontend). Close them when you’re done.

## Alternative: run the script directly

From the **Document Parser** folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\pdf-chat-rag\start.ps1
```

Or from inside **pdf-chat-rag**:

```powershell
.\start.ps1
```

## Requirements

- **Backend:** Python with a virtual environment at `pdf-chat-rag\backend\.venv` and dependencies installed  
- **Frontend:** Node.js; run `npm install` once in `pdf-chat-rag\frontend` if you haven’t

The app uses **http://127.0.0.1** (not localhost) to avoid connection issues on Windows.

## If the frontend says "Failed to connect to the server"

The backend may not be running. Start it manually:

**Option 1 – double‑click:**  
`pdf-chat-rag\backend\start-backend.bat`

**Option 2 – terminal:**

```powershell
cd "d:\Projects\Document Parser\pdf-chat-rag\backend"
.\.venv\Scripts\Activate.ps1
python main.py
```

Leave that window open. When you see `Uvicorn running on http://0.0.0.0:8000`, refresh the app in your browser.
