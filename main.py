import os
import re
import uuid
from datetime import datetime, timedelta
from typing import Optional

import bcrypt
import httpx
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from jose import jwt, JWTError
from pydantic import BaseModel, Field
from PIL import Image
import pytesseract
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# -------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -------------------------------------------------------------
SECRET_KEY = "my-secret-key-change-in-production"
ALGORITHM = "HS256"
STORAGE_DIR = "saved_documents"
os.makedirs(STORAGE_DIR, exist_ok=True)

# Dedicated YOLO Inference Microservice URL (running on port 8001)
INFERENCE_API_URL = "http://127.0.0.1:8001/predict"
INTERNAL_API_KEY = "internal-secret-token"

# Windows Tesseract auto-detection
if os.name == "nt":
    win_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(win_tesseract):
        pytesseract.pytesseract.tesseract_cmd = win_tesseract

# Local YOLO fallback initialization
LOCAL_MODEL_PATH = "best.pt"
if not os.path.exists(LOCAL_MODEL_PATH):
    LOCAL_MODEL_PATH = os.path.join("runs", "classify", "doc_classifier", "weights", "best.pt")

local_yolo = None
try:
    from ultralytics import YOLO
    if os.path.exists(LOCAL_MODEL_PATH):
        local_yolo = YOLO(LOCAL_MODEL_PATH)
except Exception:
    local_yolo = None

# SQLite Database setup
engine = create_engine("sqlite:///./documents.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False)
Base = declarative_base()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

# -------------------------------------------------------------
# 2. DATABASE MODELS & MIGRATION
# -------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255))
    saved_path = Column(String(255))
    predicted_category = Column(String(64))
    confidence_score = Column(Float)
    status = Column(String(32))
    extracted_text = Column(Text)
    uploaded_by = Column(String(64), index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# Auto-migration: Ensure 'uploaded_by' column exists if db was created previously
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE documents ADD COLUMN uploaded_by VARCHAR(64) DEFAULT 'admin'"))
        conn.commit()
    except Exception:
        pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Password utilities using native bcrypt
def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8")[:72],
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False

def get_password_hash(password: str) -> str:
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")

# Default admin seeding
def seed_default_admin():
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            db.add(User(username="admin", hashed_password=get_password_hash("password")))
            db.commit()
    finally:
        db.close()

seed_default_admin()

# Extract and validate username from JWT bearer token
def get_current_user_name(token: str = Depends(oauth2_scheme)) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

# -------------------------------------------------------------
# 3. CLASSIFICATION & OCR PIPELINE
# -------------------------------------------------------------
LABEL_MAP = {
    "aadharcard": "Aadhaar Card",
    "pancard": "PAN Card",
    "marksheet": "Marksheet"
}

async def classify_document_image(file_bytes: bytes, filename: str, filepath: str) -> tuple[str, float]:
    """Queries inference microservice on port 8001 with local fallback."""
    headers = {"X-Inference-Key": INTERNAL_API_KEY}
    files = {"file": (filename, file_bytes, "image/jpeg")}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(INFERENCE_API_URL, files=files, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                return data["category"], data["confidence"]
    except httpx.RequestError:
        pass

    # Local Fallback
    if local_yolo is not None:
        try:
            results = local_yolo(filepath, imgsz=416, verbose=False)
            probs = results[0].probs
            if probs is not None:
                top_idx = int(probs.top1)
                raw_name = results[0].names[top_idx].lower().strip()
                conf = float(probs.top1conf.cpu().item())
                return LABEL_MAP.get(raw_name, "Unrecognized"), round(conf, 2)
        except Exception:
            pass

    return "Unrecognized", 0.0


def run_ocr_and_redact(image_path: str) -> str:
    """Reads text with OCR and redacts sensitive ID sequences."""
    try:
        img = Image.open(image_path)
        raw_text = pytesseract.image_to_string(img)
    except Exception:
        raw_text = ""

    redacted = re.sub(r"\b[2-9]{1}\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b", "[Aadhaar Redacted]", raw_text)
    redacted = re.sub(r"\b([A-Z]{5})([0-9]{4})([A-Z]{1})\b", r"\1XXXX\3", redacted)
    return redacted.strip()

# -------------------------------------------------------------
# 4. FASTAPI APP & ENDPOINTS
# -------------------------------------------------------------
app = FastAPI(title="AI Document Scanner Pipeline", version="2.1.0")

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

class RegisterSchema(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=4, max_length=100)

class FixRequest(BaseModel):
    correct_category: str

@app.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(payload: RegisterSchema, db: Session = Depends(get_db)):
    clean_username = payload.username.strip().lower()
    
    existing = db.query(User).filter(User.username == clean_username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already taken. Please choose another."
        )

    new_user = User(
        username=clean_username,
        hashed_password=get_password_hash(payload.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "User registered successfully", "username": new_user.username}


@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    clean_username = form_data.username.strip().lower()
    user = db.query(User).filter(User.username == clean_username).first()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid username or password"
        )

    token = jwt.encode(
        {"sub": user.username, "exp": datetime.utcnow() + timedelta(hours=8)},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return {"access_token": token, "token_type": "bearer", "username": user.username}


@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    current_user: str = Depends(get_current_user_name),
    db: Session = Depends(get_db)
):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Use JPG, PNG, or WEBP.")

    unique_name = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(STORAGE_DIR, unique_name)
    file_bytes = await file.read()

    # 1. Save document to disk
    with open(filepath, "wb") as f:
        f.write(file_bytes)

    # 2. Classify image layout
    predicted_category, confidence = await classify_document_image(file_bytes, file.filename, filepath)

    # 3. OCR and sensitive data masking
    safe_ocr_text = run_ocr_and_redact(filepath)

    # 4. Routing Rule: 70% threshold
    doc_status = "PROCESSED" if (confidence >= 0.70 and predicted_category != "Unrecognized") else "NEEDS_REVIEW"

    # 5. Persist record scoped to the authenticated user
    record = DocumentRecord(
        filename=file.filename,
        saved_path=filepath,
        predicted_category=predicted_category,
        confidence_score=confidence,
        status=doc_status,
        extracted_text=safe_ocr_text,
        uploaded_by=current_user
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
        "extracted_text": record.extracted_text,
        "uploaded_by": record.uploaded_by
    }


# STRICT USER ISOLATION: Retrieve only the authenticated user's documents
@app.get("/documents")
def list_documents(
    current_user: str = Depends(get_current_user_name),
    db: Session = Depends(get_db)
):
    return (
        db.query(DocumentRecord)
        .filter(DocumentRecord.uploaded_by == current_user)
        .order_by(DocumentRecord.created_at.desc())
        .all()
    )


# STRICT USER ISOLATION: Ensure user owns the document before applying manual fixes
@app.patch("/review/{doc_id}")
def manually_fix_document(
    doc_id: int,
    body: FixRequest,
    current_user: str = Depends(get_current_user_name),
    db: Session = Depends(get_db)
):
    doc = (
        db.query(DocumentRecord)
        .filter(DocumentRecord.id == doc_id, DocumentRecord.uploaded_by == current_user)
        .first()
    )
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or you lack permission to review it."
        )

    doc.predicted_category = body.correct_category
    doc.status = "MANUALLY_VERIFIED"
    db.commit()

    return {"message": "Document updated successfully", "new_category": doc.predicted_category}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
