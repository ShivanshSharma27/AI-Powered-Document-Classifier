import os
import re
import uuid
import httpx
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from PIL import Image
import pytesseract
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# -------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------
SECRET_KEY = "my-secret-key-change-in-production"
ALGORITHM = "HS256"
STORAGE_DIR = "saved_documents"
os.makedirs(STORAGE_DIR, exist_ok=True)

# URL of the separate Inference API service
INFERENCE_API_URL = "http://127.0.0.1:8001/predict"

if os.name == "nt":
    win_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(win_tesseract):
        pytesseract.pytesseract.tesseract_cmd = win_tesseract

# Database setup
engine = create_engine("sqlite:///./documents.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# -------------------------------------------------------------
# 2. DATABASE TABLE
# -------------------------------------------------------------
class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    saved_path = Column(String(255))
    predicted_category = Column(String(64))
    confidence_score = Column(Float)
    status = Column(String(32))               # "PROCESSED" or "NEEDS_REVIEW"
    extracted_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# -------------------------------------------------------------
# 3. HELPER FUNCTIONS (CALLING INFERENCE API & OCR)
# -------------------------------------------------------------
async def call_inference_api(file_bytes: bytes, filename: str) -> tuple[str, float]:
    """Sends the document image over HTTP to the separate Inference API."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            files = {"file": (filename, file_bytes, "image/jpeg")}
            response = await client.post(INFERENCE_API_URL, files=files)
            if response.status_code == 200:
                data = response.json()
                return data["category"], data["confidence"]
            else:
                print(f"[Inference API Error] Status: {response.status_code}, Body: {response.text}")
        except httpx.RequestError as exc:
            print(f"[Inference API Connection Failed]: {exc}")

    return "Unrecognized", 0.0


def run_ocr_and_redact(image_path: str) -> str:
    """Reads words using Tesseract and masks sensitive identification numbers."""
    try:
        img = Image.open(image_path)
        raw_text = pytesseract.image_to_string(img)
    except Exception:
        raw_text = ""

    # Sensitive ID masking
    redacted = re.sub(r"\b[2-9]{1}\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b", "[Aadhaar Redacted]", raw_text)
    redacted = re.sub(r"\b([A-Z]{5})([0-9]{4})([A-Z]{1})\b", r"\1XXXX\3", redacted)

    return redacted.strip()

# -------------------------------------------------------------
# 4. FASTAPI APP & ENDPOINTS
# -------------------------------------------------------------
app = FastAPI(title="AI Document Management Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username != "admin" or form_data.password != "password":
        raise HTTPException(status_code=400, detail="Invalid username or password")
    
    token = jwt.encode(
        {"sub": form_data.username, "exp": datetime.utcnow() + timedelta(hours=8)},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return {"access_token": token, "token_type": "bearer"}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(STORAGE_DIR, unique_name)
    file_bytes = await file.read()

    # 1. Save uploaded file to disk
    with open(filepath, "wb") as f:
        f.write(file_bytes)

    # 2. Call dedicated Inference API (port 8001)
    predicted_category, confidence = await call_inference_api(file_bytes, file.filename)

    # 3. Perform OCR & Redaction
    safe_ocr_text = run_ocr_and_redact(filepath)

    # 4. Decision Rule: >= 70% threshold
    if confidence >= 0.70 and predicted_category != "Unrecognized":
        doc_status = "PROCESSED"
    else:
        doc_status = "NEEDS_REVIEW"

    # 5. Persist record in SQLite
    record = DocumentRecord(
        filename=file.filename,
        saved_path=filepath,
        predicted_category=predicted_category,
        confidence_score=confidence,
        status=doc_status,
        extracted_text=safe_ocr_text
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "filename": record.filename,
        "category": record.predicted_category,
        "confidence": record.confidence_score,
        "status": record.status,
        "extracted_text": record.extracted_text
    }


@app.get("/documents")
def list_documents(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    return db.query(DocumentRecord).order_by(DocumentRecord.created_at.desc()).all()


class FixRequest(BaseModel):
    correct_category: str

@app.patch("/review/{doc_id}")
def manually_fix_document(
    doc_id: int,
    body: FixRequest,
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    doc = db.query(DocumentRecord).filter(DocumentRecord.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.predicted_category = body.correct_category
    doc.status = "MANUALLY_VERIFIED"
    db.commit()

    return {"message": "Document updated successfully", "new_category": doc.predicted_category}


if __name__ == "__main__":
    import uvicorn
    # Main backend on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)