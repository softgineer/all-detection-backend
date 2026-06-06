"""
utils/preprocessor.py
Bilateral filter + Otsu WBC nucleus segmentation
"""
import cv2
import numpy as np


class ImagePreprocessor:
    def __init__(self, target_size=(224, 224), sigma_s=15.0, sigma_r=75.0):
        self.target_size = target_size
        self.sigma_s = sigma_s
        self.sigma_r = sigma_r

    def process(self, img: np.ndarray) -> dict:
        resized    = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
        filtered   = self._bilateral(resized)
        normalized = self._normalize(filtered)
        gray       = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        mask       = self._segment(gray)
        segmented  = filtered.copy()
        segmented[mask == 0] = 0
        return dict(original=resized, filtered=filtered,
                    normalized=normalized, gray=gray,
                    mask=mask, segmented=segmented)

    def process_bytes(self, data: bytes) -> dict | None:
        arr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return None if img is None else self.process(img)

    def _bilateral(self, img):
        d = max(5, int(3 * self.sigma_s))
        d = d | 1  # ensure odd
        return cv2.bilateralFilter(img, d, self.sigma_r, self.sigma_s)

    @staticmethod
    def _normalize(img):
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], np.float32)
        std  = np.array([0.229, 0.224, 0.225], np.float32)
        return (rgb - mean) / std

    @staticmethod
    def _segment(gray):
        inv = cv2.bitwise_not(gray)
        _, binary = cv2.threshold(inv, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=2)
        closed = cv2.morphologyEx(opened,  cv2.MORPH_CLOSE, k, iterations=3)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, 8)
        if n <= 1:
            return closed
        biggest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
        return np.where(labels == biggest, 255, 0).astype(np.uint8)
