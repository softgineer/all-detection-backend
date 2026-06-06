"""
utils/predictor.py
==================
Loads saved models and runs the full inference pipeline on a single image.
"""
import os, sys
import numpy as np
import joblib
import cv2
import base64
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.preprocessor     import ImagePreprocessor
from utils.feature_extractor import HybridFeatureExtractor
from sklearn.preprocessing   import normalize


class ALLPredictor:
    """
    Singleton-friendly predictor that lazy-loads models on first call.
    """

    LABELS = {0: "Normal", 1: "ALL-Positive"}
    COLORS = {0: "#22c55e", 1: "#ef4444"}

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self._loaded    = False
        self._pca       = None
        self._svm       = None
        self._rf        = None
        self._gb        = None
        self._use_deep  = False
        self._prep      = ImagePreprocessor()
        self._hext      = None

    def _load(self):
        if self._loaded:
            return
        required = ["pca.pkl", "svm.pkl", "rf.pkl", "gb.pkl", "config.pkl"]
        for f in required:
            p = os.path.join(self.models_dir, f)
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Model artefact not found: {p}\n"
                    f"Run 'python utils/trainer.py' first to train the models.")

        self._pca      = joblib.load(f"{self.models_dir}/pca.pkl")
        self._svm      = joblib.load(f"{self.models_dir}/svm.pkl")
        self._rf       = joblib.load(f"{self.models_dir}/rf.pkl")
        self._gb       = joblib.load(f"{self.models_dir}/gb.pkl")
        cfg            = joblib.load(f"{self.models_dir}/config.pkl")
        self._use_deep = cfg.get("use_deep", False)
        self._hext     = HybridFeatureExtractor(use_deep=self._use_deep)
        self._loaded   = True

    # ── core predict ──────────────────────────────────────────────────────
    def predict_bytes(self, image_bytes: bytes) -> dict:
        """
        Full pipeline: raw image bytes → prediction dict.
        Returns:
            label        : "ALL-Positive" | "Normal"
            label_id     : 1 | 0
            confidence   : float 0-100
            probabilities: {"ALL": float, "Normal": float}
            color        : hex colour for UI
            features     : dict of diagnostic feature values
            preview_b64  : base64 PNG of processed cell
            votes        : {"SVM":int, "RF":int, "GB":int}
        """
        self._load()

        # 1. Decode image
        arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image — unsupported format.")

        # 2. Preprocess
        prep = self._prep.process(img)

        # 3. Extract hybrid features
        fv   = self._hext.extract(prep).reshape(1, -1)

        # 4. L2 normalise + PCA
        fv_n = normalize(fv, norm="l2")
        Z    = self._pca.transform(fv_n)

        # 5. Individual probabilities
        p_svm = float(self._svm.predict_proba(Z)[0, 1])
        p_rf  = float(self._rf.predict_proba(Z)[0, 1])
        p_gb  = float(self._gb.predict_proba(Z)[0, 1])

        # 6. Soft majority vote (average probability)
        avg_prob = (p_svm + p_rf + p_gb) / 3.0
        label_id = int(avg_prob >= 0.5)

        # 7. Individual hard votes
        votes = {
            "SVM": int(p_svm >= 0.5),
            "RF":  int(p_rf  >= 0.5),
            "GB":  int(p_gb  >= 0.5),
        }

        # 8. Diagnostic features for display
        hand  = self._hext.hand
        hf    = hand.extract(prep["gray"], prep["mask"], prep["filtered"])
        diag  = self._diagnostic_features(hf, prep)

        # 9. Render preview (segmented cell in colour)
        preview_b64 = self._render_preview(prep)

        confidence = avg_prob if label_id == 1 else (1 - avg_prob)
        return {
            "label":         self.LABELS[label_id],
            "label_id":      label_id,
            "confidence":    round(confidence * 100, 2),
            "probabilities": {
                "ALL":    round(avg_prob * 100, 2),
                "Normal": round((1 - avg_prob) * 100, 2),
            },
            "classifier_probs": {
                "SVM": round(p_svm * 100, 2),
                "RF":  round(p_rf  * 100, 2),
                "GB":  round(p_gb  * 100, 2),
            },
            "votes":       votes,
            "color":       self.COLORS[label_id],
            "features":    diag,
            "preview_b64": preview_b64,
        }

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _diagnostic_features(hf: np.ndarray, prep: dict) -> dict:
        """Extract readable diagnostic values from the handcrafted feature vector."""
        # hf layout: [18 GLCM | 10 shape | 15 stat]
        # shape indices: [area, peri, circ, aspect, sol, extent, nc, ecc, enc_r, bb]
        shape_offset = 18
        stat_offset  = 28

        nucleus_area   = float(hf[shape_offset + 0])
        circularity    = float(hf[shape_offset + 2])
        solidity       = float(hf[shape_offset + 4])
        nc_ratio       = float(hf[shape_offset + 6])
        eccentricity   = float(hf[shape_offset + 7])

        glcm_contrast  = float(hf[0])
        glcm_energy    = float(hf[3])
        glcm_homog     = float(hf[6])
        glcm_corr      = float(hf[9])

        mean_r = float(hf[stat_offset + 0])
        mean_g = float(hf[stat_offset + 5])
        mean_b = float(hf[stat_offset + 10])

        # nucleus pixel count from mask
        nucleus_pixels = int(np.sum(prep["mask"] > 0))

        return {
            "nucleus_area_px":  round(nucleus_area, 1),
            "nucleus_pixels":   nucleus_pixels,
            "circularity":      round(min(circularity, 1.0), 4),
            "solidity":         round(min(solidity, 1.0),    4),
            "nc_ratio":         round(nc_ratio, 4),
            "eccentricity":     round(min(eccentricity, 1.0), 4),
            "glcm_contrast":    round(glcm_contrast, 4),
            "glcm_energy":      round(glcm_energy, 6),
            "glcm_homogeneity": round(glcm_homog, 4),
            "glcm_correlation": round(glcm_corr, 4),
            "mean_intensity_R": round(mean_r, 2),
            "mean_intensity_G": round(mean_g, 2),
            "mean_intensity_B": round(mean_b, 2),
        }

    @staticmethod
    def _render_preview(prep: dict) -> str:
        """Return base64-encoded PNG of the segmented nucleus with coloured overlay."""
        filt  = prep["filtered"].copy()
        mask  = prep["mask"]

        # Colour-highlight nucleus boundary
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        overlay = filt.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 100), 2)

        # Encode to PNG → base64
        _, buf = cv2.imencode(".png", overlay)
        return base64.b64encode(buf.tobytes()).decode("utf-8")
