"""
utils/feature_extractor.py
Deep (ResNet50) + Handcrafted (GLCM, Shape, Statistical) hybrid features
"""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore")


class DeepFeatureExtractor:
    def __init__(self):
        self._model = None

    def _load(self):
        if self._model:
            return
        import os, tensorflow as tf
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        self._model = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet",
            pooling="avg", input_shape=(224, 224, 3))
        self._model.trainable = False

    def extract(self, norm_rgb: np.ndarray) -> np.ndarray:
        self._load()
        import tensorflow as tf
        batch = np.expand_dims(norm_rgb.astype(np.float32), 0)
        return self._model(batch, training=False).numpy().flatten()


class HandcraftedFeatureExtractor:
    DISTANCES = [1, 2]
    ANGLES    = [0, np.pi/4, np.pi/2, 3*np.pi/4]

    def extract(self, gray, mask, bgr) -> np.ndarray:
        return np.concatenate([
            self._glcm(gray, mask),
            self._shape(mask, gray),
            self._stats(bgr, mask),
        ])

    def _glcm(self, gray, mask):
        roi = gray.copy(); roi[mask == 0] = 0
        roi_q = (roi // 4).astype(np.uint8)
        glcm  = graycomatrix(roi_q, self.DISTANCES, self.ANGLES,
                              levels=64, symmetric=True, normed=True)
        feats = []
        for prop in ["contrast","energy","homogeneity","correlation",
                     "dissimilarity","ASM"]:
            v = graycoprops(glcm, prop)
            feats.extend([v.mean(), v.std(), v.max()])
        return np.array(feats, np.float32)          # 18

    def _shape(self, mask, gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(10, np.float32)
        cnt  = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt) + 1e-6
        peri = cv2.arcLength(cnt, True) + 1e-6
        circ = 4*np.pi*area / peri**2
        x,y,w,h = cv2.boundingRect(cnt)
        hull     = cv2.convexHull(cnt)
        solidity = area / (cv2.contourArea(hull) + 1e-6)
        _, enc_r = cv2.minEnclosingCircle(cnt)
        M   = cv2.moments(cnt)
        m00 = M["m00"] + 1e-6
        mu20, mu02, mu11 = M["mu20"]/m00, M["mu02"]/m00, M["mu11"]/m00
        lam1 = (mu20+mu02)/2 + np.sqrt(((mu20-mu02)/2)**2+mu11**2)
        lam2 = (mu20+mu02)/2 - np.sqrt(((mu20-mu02)/2)**2+mu11**2)
        ecc  = np.sqrt(max(0, 1-lam2/(lam1+1e-9)))
        return np.array([area, peri, circ, float(w)/(h+1e-6), solidity,
                         area/(w*h+1e-6), area/(gray.shape[0]*gray.shape[1]),
                         ecc, enc_r, float(w*h)], np.float32)  # 10

    @staticmethod
    def _stats(bgr, mask):
        rgb, feats = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), []
        for c in range(3):
            vals = rgb[:,:,c][mask>0].astype(np.float32)
            if len(vals) == 0:
                feats.extend([0.]*5); continue
            feats += [float(np.mean(vals)), float(np.var(vals)),
                      float(np.std(vals)), float(skew(vals)),
                      float(kurtosis(vals))]
        return np.array(feats, np.float32)          # 15


class HybridFeatureExtractor:
    def __init__(self, use_deep=True):
        self.use_deep = use_deep
        self.deep     = DeepFeatureExtractor() if use_deep else None
        self.hand     = HandcraftedFeatureExtractor()

    def extract(self, prep: dict) -> np.ndarray:
        h = self.hand.extract(prep["gray"], prep["mask"], prep["filtered"])
        if not self.use_deep:
            return h
        d = self.deep.extract(prep["normalized"])
        return np.concatenate([d, h])
