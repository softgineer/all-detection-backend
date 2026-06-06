# ALL Detection System — Backend API

**Multi-Stage Ensemble Learning for Acute Lymphoblastic Leukemia Detection**
BTech Final Year Project · FUTA · Dept. of Software Engineering
Author: Goodness Ikubuwaje Oluwasegun (SEN/20/5102)
Supervisor: Dr. Mrs. O. V. Olatunde

---

## What This Does

This is the Python backend that powers the ALL Detection web application at
https://all-detection.vercel.app/

It receives a blood smear image, runs it through the full pipeline, and returns
a classification result with confidence scores, per-classifier votes, and
extracted diagnostic features.

### Full Pipeline

```
Blood Smear Image
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 1: Image Preprocessing       │
│  Bilateral Filter + Otsu Segment    │
└─────────────────┬───────────────────┘
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
┌──────────────┐    ┌──────────────────────┐
│ Deep Feature │    │ Handcrafted Features  │
│ ResNet50     │    │ GLCM + Shape + Stats  │
│ (2048-d)     │    │ (43-d)                │
└──────┬───────┘    └──────────┬───────────┘
       └───────────┬───────────┘
                   ▼
      ┌────────────────────────┐
      │  L2 Norm + PCA Fusion  │
      └────────────┬───────────┘
                   │
      ┌────────────┼────────────┐
      ▼            ▼            ▼
   [ SVM ]    [ Random   ] [ Gradient ]
   [ RBF ]    [ Forest   ] [ Boosting ]
      │            │            │
      └────────────┼────────────┘
                   ▼
          Majority Vote Ensemble
                   │
                   ▼
          ALL-Positive | Normal
```

---

## Project Structure

```
ALL_backend/
├── main.py                   ← FastAPI app (all routes)
├── requirements.txt          ← Python dependencies
├── render.yaml               ← Render.com deployment config
├── Procfile                  ← Heroku/Railway deployment
├── utils/
│   ├── preprocessor.py       ← Bilateral filter + Otsu segmentation
│   ├── feature_extractor.py  ← ResNet50 + GLCM + Shape + Stat features
│   ├── trainer.py            ← Trains ensemble, saves model artefacts
│   └── predictor.py          ← Inference pipeline (loads models + predicts)
├── models/                   ← Auto-created after training
│   ├── pca.pkl
│   ├── svm.pkl
│   ├── rf.pkl
│   ├── gb.pkl
│   ├── config.pkl
│   └── training_report.json
└── static/
    └── frontend.jsx          ← React frontend source
```

---

## Local Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd ALL_backend
pip install -r requirements.txt
```

### 2. Train the model (first time only)

```bash
python utils/trainer.py
```

This generates synthetic blood smear images, extracts features,
trains SVM + Random Forest + Gradient Boosting, and saves model
artefacts to `models/`.

> **Note:** By default, training uses handcrafted features only (fast).
> To include ResNet50 deep features, edit `trainer.py` and set
> `use_deep=True`. This requires more RAM and takes longer.

### 3. Start the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API is now live at: **http://localhost:8000**
Interactive docs at: **http://localhost:8000/docs**

---

## API Endpoints

| Method | Endpoint              | Description                              |
|--------|-----------------------|------------------------------------------|
| GET    | `/`                   | Health check                             |
| GET    | `/api/health`         | Detailed status + training report        |
| POST   | `/api/predict`        | **Upload image → get classification**    |
| GET    | `/api/model/info`     | Model metadata + training metrics        |
| POST   | `/api/train`          | Retrain model (background task)          |
| GET    | `/api/training-status`| Poll training progress                   |
| GET    | `/docs`               | Swagger UI (interactive API explorer)    |

### Example: Predict with cURL

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@your_blood_smear.png"
```

### Example Response

```json
{
  "success": true,
  "filename": "blood_smear.png",
  "inference_ms": 284.5,
  "label": "ALL-Positive",
  "label_id": 1,
  "confidence": 94.63,
  "probabilities": { "ALL": 94.63, "Normal": 5.37 },
  "classifier_probs": { "SVM": 96.2, "RF": 93.8, "GB": 93.9 },
  "votes": { "SVM": 1, "RF": 1, "GB": 1 },
  "color": "#ef4444",
  "features": {
    "nucleus_area_px": 4821.0,
    "circularity": 0.7123,
    "solidity": 0.8901,
    "nc_ratio": 0.0963,
    "eccentricity": 0.4512,
    "glcm_contrast": 12.3401,
    "glcm_energy": 0.000412,
    "glcm_homogeneity": 0.3214,
    "glcm_correlation": 0.9123,
    "mean_intensity_R": 142.3,
    "mean_intensity_G": 110.7,
    "mean_intensity_B": 95.2
  },
  "preview_b64": "<base64 PNG of processed cell>"
}
```

---

## Deployment

### Option A — Render.com (Recommended, Free)

1. Push this folder to a GitHub repository
2. Go to https://render.com → New Web Service
3. Connect your GitHub repo
4. Render will auto-detect `render.yaml` and configure everything
5. After deployment, copy your URL (e.g. `https://all-detection-api.onrender.com`)
6. Update `API_BASE` in `static/frontend.jsx` and redeploy the Vercel frontend

### Option B — Railway.app

1. Push to GitHub
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Railway auto-detects Python and installs requirements.txt

### Option C — Hugging Face Spaces

1. Create a Space at https://huggingface.co/spaces
2. Choose "Docker" or "Gradio" runtime
3. Upload all files
4. Add a `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY . .
   RUN python utils/trainer.py
   CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
   ```

### Connecting to the Vercel Frontend

After deploying, update `static/frontend.jsx`:

```js
const API_BASE = "https://your-actual-backend-url.onrender.com";
```

Then rebuild and redeploy the Vercel project.

---

## Using Real Datasets

Once you have downloaded the actual datasets:

1. Organise images into folders:
```
data/
  ALL/        ← ALL-positive images (*.png, *.jpg, *.bmp)
  Normal/     ← Normal images
```

2. Modify `trainer.py` to use `load_from_directory()` instead of the
   synthetic generator, and call `train()` with the real image paths.

### Dataset Sources
- ALL-IDB1 & ALL-IDB2: https://homes.di.unimi.it/scotti/all/
- ALL-IDB (Kaggle): https://www.kaggle.com/datasets/andrewmvd/leukemia-classification
- C-NMC 2019: https://www.cancerimagingarchive.net/collection/c-nmc-2019/

---

## System Performance (Synthetic Data Benchmark)

| Metric    | SVM   | Random Forest | Grad. Boosting | Ensemble |
|-----------|-------|---------------|----------------|----------|
| Accuracy  | 93.59%| 94.23%        | 93.59%         | **96.15%**|
| Precision | 92.86%| 93.51%        | 93.18%         | **95.89%**|
| Recall    | 94.87%| 95.51%        | 94.23%         | **96.51%**|
| F1-Score  | 93.85%| 94.50%        | 93.70%         | **96.20%**|
| ROC-AUC   | 97.12%| 97.89%        | 97.44%         | **98.73%**|
