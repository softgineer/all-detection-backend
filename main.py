"""
main.py  –  FastAPI Backend for ALL Detection System
=====================================================
Endpoints:
  GET  /                    → health check
  GET  /api/health          → detailed health + model status
  POST /api/predict         → upload image → prediction JSON
  GET  /api/model/info      → training report + model metadata
  POST /api/train           → (re)train the ensemble (admin use)
  GET  /api/docs            → auto Swagger UI (built-in)
"""

import os, sys, json, time, asyncio
from contextlib import asynccontextmanager
from pathlib    import Path

from fastapi              import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses    import JSONResponse
from pydantic             import BaseModel

sys.path.insert(0, os.path.dirname(__file__))

from utils.predictor import ALLPredictor

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MODELS_DIR = str(BASE_DIR / "models")

# ── Singleton predictor ────────────────────────────────────────────────────
predictor = ALLPredictor(models_dir=MODELS_DIR)

# ── Training state ─────────────────────────────────────────────────────────
training_state = {"status": "idle", "message": "", "metrics": None}

# ── App startup: auto-train if no models exist ─────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_files = ["pca.pkl", "svm.pkl", "rf.pkl", "gb.pkl", "config.pkl"]
    all_exist   = all((BASE_DIR / "models" / f).exists() for f in model_files)
    if not all_exist:
        print("[Startup] No trained models found — training now (handcrafted mode)…")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _run_training, 300, False)
        print("[Startup] Training complete.")
    else:
        print("[Startup] Pre-trained models found — loading on first predict call.")
    yield

def _run_training(n_per_class: int = 300, use_deep: bool = False):
    """Blocking training function — run in executor."""
    from utils.trainer import train
    training_state["status"]  = "running"
    training_state["message"] = "Training in progress…"
    try:
        metrics = train(n_per_class=n_per_class,
                        use_deep=use_deep,
                        models_dir=MODELS_DIR,
                        verbose=True)
        training_state["status"]  = "done"
        training_state["message"] = "Training completed successfully."
        training_state["metrics"] = metrics
        # Reset predictor so it reloads fresh models
        predictor._loaded = False
    except Exception as e:
        training_state["status"]  = "error"
        training_state["message"] = str(e)

# ── FastAPI app ────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "ALL Detection API",
    description = (
        "Multi-Stage Ensemble Learning System for Acute Lymphoblastic "
        "Leukemia (ALL) Detection from Blood Smear Images.\n\n"
        "Built for the BTech final-year project — FUTA, Dept. of Software Engineering.\n"
        "Author: Goodness Ikubuwaje Oluwasegun (SEN/20/5102)\n"
        "Supervisor: Dr. Mrs. O. V. Olatunde"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# ── CORS — allow the Vercel frontend ─────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # replace with specific frontend URL in prod
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ══════════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def root():
    return {
        "system": "ALL Detection API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/api/health", tags=["Health"])
async def health():
    model_files = ["pca.pkl", "svm.pkl", "rf.pkl", "gb.pkl"]
    models_ready = all(
        (Path(MODELS_DIR) / f).exists() for f in model_files)

    report = {}
    report_path = Path(MODELS_DIR) / "training_report.json"
    if report_path.exists():
        with open(report_path) as f:
            report = json.load(f)

    return {
        "status":        "healthy",
        "models_ready":  models_ready,
        "models_dir":    MODELS_DIR,
        "training_report": report,
        "training_state":  training_state,
    }


@app.post("/api/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Upload a blood smear image (PNG / JPG / BMP / TIFF).
    Returns the ensemble classification result with confidence,
    per-classifier votes, diagnostic features, and a processed preview.
    """
    # ── Validate file type ────────────────────────────────────────────
    allowed = {"image/png", "image/jpeg", "image/bmp",
               "image/tiff", "image/tif", "application/octet-stream"}
    if file.content_type and file.content_type not in allowed:
        # also allow by extension
        ext = Path(file.filename or "").suffix.lower()
        if ext not in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}:
            raise HTTPException(
                status_code=415,
                detail="Unsupported file type. Upload PNG, JPG, BMP, or TIFF.")

    # ── Read image bytes ──────────────────────────────────────────────
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(contents) > 20 * 1024 * 1024:   # 20 MB limit
        raise HTTPException(status_code=413, detail="File too large. Max 20 MB.")

    # ── Run prediction ────────────────────────────────────────────────
    try:
        t0     = time.perf_counter()
        result = predictor.predict_bytes(contents)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503,
                            detail=f"Models not ready: {e}. "
                                   f"Call POST /api/train first.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Prediction failed: {str(e)}")

    return JSONResponse({
        "success":      True,
        "filename":     file.filename,
        "inference_ms": elapsed_ms,
        **result,
    })


@app.get("/api/model/info", tags=["Model"])
async def model_info():
    """Return training report and model configuration."""
    report_path = Path(MODELS_DIR) / "training_report.json"
    config_path = Path(MODELS_DIR) / "config.pkl"

    if not report_path.exists():
        raise HTTPException(status_code=404,
                            detail="No training report found. Train the model first.")

    with open(report_path) as f:
        report = json.load(f)

    return {
        "model": "Multi-Stage Ensemble (SVM + Random Forest + Gradient Boosting)",
        "feature_strategy": "Hybrid (Deep ResNet50 + GLCM + Shape + Statistical)",
        "fusion": "L2 Normalisation + PCA (95% variance retained)",
        "voting": "Soft majority vote (average probability)",
        "training_report": report,
    }


class TrainRequest(BaseModel):
    n_per_class: int  = 300
    use_deep:    bool = False


@app.post("/api/train", tags=["Model"])
async def trigger_training(req: TrainRequest, background_tasks: BackgroundTasks):
    """
    (Re)train the ensemble on synthetic data.
    - n_per_class : number of images per class (ALL + Normal)
    - use_deep    : include ResNet50 deep features (slower, more accurate)

    Returns immediately; training runs in the background.
    Poll GET /api/health to check completion.
    """
    if training_state["status"] == "running":
        return JSONResponse({"status": "already_running",
                             "message": "Training already in progress."})

    background_tasks.add_task(_run_training, req.n_per_class, req.use_deep)
    return {
        "status":  "started",
        "message": f"Training started with n_per_class={req.n_per_class}, "
                   f"use_deep={req.use_deep}. Poll /api/health for status.",
    }


@app.get("/api/training-status", tags=["Model"])
async def training_status():
    return training_state


# ── Run directly ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


# ══════════════════════════════════════════════════════════════════════════
# BATCH PREDICT — multiple images in one request
# ══════════════════════════════════════════════════════════════════════════
from typing import List
import asyncio

@app.post("/api/predict/batch", tags=["Prediction"])
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Upload multiple blood smear images at once (up to 20).
    Returns a list of prediction results, one per image.
    Processes all images concurrently for speed.
    """
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded.")
    if len(files) > 20:
        raise HTTPException(status_code=400,
                            detail="Maximum 20 images per batch request.")

    async def process_one(upload: UploadFile) -> dict:
        contents = await upload.read()
        if len(contents) == 0:
            return {"filename": upload.filename, "error": "Empty file"}
        if len(contents) > 20 * 1024 * 1024:
            return {"filename": upload.filename, "error": "File too large (max 20 MB)"}
        try:
            t0     = time.perf_counter()
            result = predictor.predict_bytes(contents)
            ms     = round((time.perf_counter() - t0) * 1000, 1)
            return {"filename": upload.filename, "inference_ms": ms,
                    "success": True, **result}
        except Exception as e:
            return {"filename": upload.filename, "success": False,
                    "error": str(e)}

    results = await asyncio.gather(*[process_one(f) for f in files])

    # Summary stats
    successful = [r for r in results if r.get("success")]
    all_count  = sum(1 for r in successful if r.get("label_id") == 1)
    norm_count = sum(1 for r in successful if r.get("label_id") == 0)

    return {
        "total":      len(files),
        "successful": len(successful),
        "failed":     len(files) - len(successful),
        "summary": {
            "ALL_positive": all_count,
            "Normal":       norm_count,
        },
        "results": list(results),
    }
