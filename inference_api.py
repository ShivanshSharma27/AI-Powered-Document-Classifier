import os
import io
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="Document Classification Inference API", version="1.0.0")

# 1. Load trained weights
MODEL_PATH = "best.pt"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join("runs", "classify", "doc_classifier", "weights", "best.pt")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

print(f"[Inference Engine] Loading model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

LABEL_MAP = {
    "aadharcard": "Aadhaar Card",
    "pancard": "PAN Card",
    "marksheet": "Marksheet"
}

# 2. Output Schema
class PredictionResponse(BaseModel):
    category: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health_check():
    return {"status": "online", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """Receives image bytes, performs YOLO inference, and returns probabilities."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty image file received.")

    try:
        # Load image via PIL to validate image integrity
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Run inference
        results = model.predict(source=img, imgsz=416, verbose=False)
        probs = results[0].probs

        if probs is None:
            raise HTTPException(status_code=500, detail="Model returned empty probabilities.")

        top_idx = int(probs.top1)
        raw_name = results[0].names[top_idx].lower().strip()
        confidence = float(probs.top1conf.cpu().item())
        clean_category = LABEL_MAP.get(raw_name, "Unrecognized")

        # Extract all class probability scores
        all_scores = probs.data.cpu().numpy()
        scores_dict = {
            LABEL_MAP.get(results[0].names[i].lower().strip(), results[0].names[i]): round(float(all_scores[i]), 4)
            for i in range(len(all_scores))
        }

        return PredictionResponse(
            category=clean_category,
            confidence=round(confidence, 4),
            probabilities=scores_dict
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Dedicated port 8001 for inference service
    uvicorn.run("inference_api:app", host="127.0.0.1", port=8001, reload=True)