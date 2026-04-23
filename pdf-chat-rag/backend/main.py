from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Request, Depends
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
    allow_origins=["*"],  # In production, specify your frontend URL
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
    """Dependency: parse multipart form with large size limit and return the uploaded file."""
    content_type = request.headers.get("content-type", "")
    if "multipart/form-data" not in content_type:
        raise HTTPException(
            status_code=400,
            detail=f"Expected multipart/form-data. Got Content-Type: {content_type[:80]}",
        )
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
    Upload PDF: save to disk quickly (streamed), return immediately.
    Processing (extract, chunk, embed) runs in background. Poll GET /upload/status/{doc_id} for readiness.
    """
    if not getattr(file, "filename", None) or not str(file.filename).lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="File must be a PDF")

    doc_id = str(uuid.uuid4())
    file_path = f"uploads/{doc_id}.pdf"
    upload_status[doc_id] = "processing"
    if doc_id in upload_error:
        del upload_error[doc_id]

    try:
        # Stream to disk in chunks so large files don't use huge memory
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

        # Process in background so response returns in seconds
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
async def chat(request: ChatRequest):
    """
    Endpoint B: Answer question using RAG with document context (streaming for faster perceived response)
    """
    try:
        return StreamingResponse(
            chat_service.get_answer_stream(
                doc_id=request.doc_id,
                question=request.question,
                vector_store=vector_store
            ),
            media_type="application/x-ndjson",
        )
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

