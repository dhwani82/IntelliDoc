from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Request, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import uuid
from dotenv import load_dotenv
import uvicorn

from pdf_processor import process_pdf
from vector_store import VectorStore
from chat_service import ChatService

load_dotenv()

# Allow large PDF uploads (500 MB)
MAX_UPLOAD_BYTES = 500 * 1024 * 1024

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload status for background processing: doc_id -> "processing" | "ready" | "failed"
upload_status: dict[str, str] = {}
upload_error: dict[str, str] = {}  # doc_id -> error message when failed

# Initialize services
vector_store = VectorStore()
vector_store.load_existing()  # Load existing FAISS index and metadata if available
chat_service = ChatService()

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("data", exist_ok=True)


class ChatRequest(BaseModel):
    doc_id: str
    question: str


async def _process_pdf_background(doc_id: str, file_path: str):
    """Run PDF processing in background and set upload status."""
    global upload_status, upload_error
    try:
        await process_pdf(file_path, doc_id, vector_store)
        upload_status[doc_id] = "ready"
        if doc_id in upload_error:
            del upload_error[doc_id]
        print(f"[BACKEND] Background processing done for doc_id={doc_id}")
    except Exception as e:
        import traceback
        upload_status[doc_id] = "failed"
        upload_error[doc_id] = str(e) if str(e) else type(e).__name__
        print(f"[BACKEND] Background processing failed: {e}")
        traceback.print_exc()
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass


@app.get("/upload/status/{doc_id}")
async def get_upload_status(doc_id: str):
    """Return processing status for an uploaded document."""
    status = upload_status.get(doc_id, "unknown")
    out = {"doc_id": doc_id, "status": status}
    if status == "failed" and doc_id in upload_error:
        out["error"] = upload_error[doc_id]
    return out


async def get_upload_file(request: Request) -> UploadFile:
    """
    Custom dependency that parses the multipart body with an elevated part-size limit.

    FastAPI's default UploadFile dependency uses Starlette's multipart parser with a
    smaller max_part_size, which rejects large PDFs before our handler runs. Parsing
    via request.form(max_part_size=...) lets uploads up to MAX_UPLOAD_BYTES through.
    """
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"Expected multipart/form-data. Got Content-Type: {content_type[:80]}",
        )
    # Higher than Starlette's default so large PDF parts are accepted.
    form = await request.form(max_part_size=MAX_UPLOAD_BYTES)
    file = form.get("file")
    if file is not None and hasattr(file, "read") and hasattr(file, "filename"):
        return file
    for key in form.keys():
        value = form.get(key)
        if value is not None and hasattr(value, "read") and hasattr(value, "filename"):
            return value
    keys = list(form.keys()) if form else []
    print(f"[BACKEND] Upload failed: form keys={keys}, content_type={content_type[:100]}")
    raise HTTPException(
        status_code=400,
        detail=f"No file in request. Form keys received: {keys}. Send the PDF as form field 'file'.",
    )


@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = Depends(get_upload_file),
):
    """
    Accept a PDF upload, persist it, and kick off indexing asynchronously.

    The file is streamed to disk in fixed-size chunks rather than read entirely
    into memory. That keeps peak RAM low for large PDFs (up to MAX_UPLOAD_BYTES)
    and lets the handler return a doc_id as soon as the bytes are on disk.

    Extract/chunk/embed work is scheduled via BackgroundTasks instead of running
    inline so the HTTP response is not blocked by minutes of indexing. Clients
    poll GET /upload/status/{doc_id} until status is ready or failed.
    """
    if not getattr(file, "filename", None) or not str(file.filename).lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    doc_id = str(uuid.uuid4())
    file_path = f"uploads/{doc_id}.pdf"
    upload_status[doc_id] = "processing"
    if doc_id in upload_error:
        del upload_error[doc_id]

    try:
        # Stream write: never hold the full PDF in a single buffer.
        with open(file_path, "wb") as buffer:
            chunk_size = 1024 * 1024  # 1 MB
            total = 0
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
                total += len(chunk)
        print(f"[BACKEND] Upload saved: {file.filename} -> {file_path} ({total} bytes)")

        # Defer heavy processing so upload returns promptly with doc_id.
        background_tasks.add_task(_process_pdf_background, doc_id, file_path)
        return {"doc_id": doc_id}
    except Exception as e:
        upload_status[doc_id] = "failed"
        upload_error[doc_id] = str(e)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/chat")
async def chat(
    request: ChatRequest,
    backend: str | None = Query(
        default=None,
        description="Generation backend for this request: gemini or local. "
        "Overrides GENERATION_BACKEND when set.",
    ),
):
    """
    Endpoint B: Answer question using RAG with document context (streaming for faster perceived response)
    """
    try:
        if backend is not None:
            from local_llm import resolve_generation_backend

            backend = backend.strip() or None
            if backend:
                try:
                    backend = resolve_generation_backend(backend)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
            else:
                backend = None
        return StreamingResponse(
            chat_service.get_answer_stream(
                doc_id=request.doc_id,
                question=request.question,
                vector_store=vector_store,
                generation_backend=backend,
            ),
            media_type="application/x-ndjson",
        )
    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


@app.get("/")
async def root():
    return {"message": "Document Parser API"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok", "service": "Document Parser API"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

