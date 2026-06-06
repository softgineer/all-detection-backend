"""
utils/trainer.py
================
Trains the full Multi-Stage Ensemble pipeline on synthetic OR real data
and saves all artefacts to the models/ directory:
  models/pca.pkl
  models/svm.pkl
  models/rf.pkl
  models/gb.pkl
  models/config.pkl
  models/training_report.json
"""
import os, sys, json, time, warnings, glob
import numpy as np
import cv2
import joblib
import random

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.preprocessor      import ImagePreprocessor
from utils.feature_extractor  import HybridFeatureExtractor

from sklearn.svm              import SVC
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition    import PCA
from sklearn.preprocessing    import normalize
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (accuracy_score, precision_score,
                                      recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling   import SMOTE


# ─────────────────────────────────────────────
#  Synthetic image generators (demo/fallback)
# ─────────────────────────────────────────────
def _make_all_cell(size=224):
    img = np.ones((size, size, 3), dtype=np.uint8) * 240
    cx, cy = size//2, size//2
    axes   = (random.randint(55, 75), random.randint(45, 65))
    angle  = random.randint(0, 180)
    col    = (random.randint(60,100), random.randint(20,60), random.randint(100,160))
    cv2.ellipse(img, (cx,cy), axes, angle, 0, 360, col, -1)
    for _ in range(8):
        ox, oy = random.randint(-20,20), random.randint(-20,20)
        cv2.circle(img, (cx+ox, cy+oy), random.randint(15,30), col, -1)
    nx, ny = cx+random.randint(-20,20), cy+random.randint(-20,20)
    cv2.circle(img, (nx,ny), random.randint(8,14),
               (random.randint(20,50), random.randint(10,30), random.randint(60,100)), -1)
    cv2.ellipse(img, (cx,cy),
                (axes[0]+random.randint(5,12), axes[1]+random.randint(5,12)),
                angle, 0, 360, (200,185,210), random.randint(3,8))
    for _ in range(random.randint(5,10)):
        rx, ry = random.randint(10,size-10), random.randint(10,size-10)
        if abs(rx-cx)>60 or abs(ry-cy)>60:
            rr = (random.randint(14,20), random.randint(10,15))
            cv2.ellipse(img,(rx,ry),rr,random.randint(0,180),0,360,(80,100,200),-1)
            cv2.ellipse(img,(rx,ry),(rr[0]-5,rr[1]-4),0,0,360,(120,140,220),-1)
    img   = cv2.GaussianBlur(img, (3,3), 0)
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)


def _make_normal_cell(size=224):
    img = np.ones((size, size, 3), dtype=np.uint8) * 245
    cx, cy = size//2, size//2
    r   = random.randint(28, 38)
    col = (random.randint(80,120), random.randint(40,80), random.randint(120,170))
    cv2.circle(img, (cx,cy), r, col, -1)
    cv2.circle(img, (cx,cy), r+random.randint(10,18), (210,195,220), random.randint(4,8))
    for _ in range(random.randint(3,8)):
        px, py = random.randint(0,size), random.randint(0,size)
        cv2.circle(img,(px,py),random.randint(2,5),(140,120,180),-1)
    for _ in range(random.randint(8,14)):
        rx, ry = random.randint(10,size-10), random.randint(10,size-10)
        if abs(rx-cx)>50 or abs(ry-cy)>50:
            rr = (random.randint(14,20), random.randint(10,15))
            cv2.ellipse(img,(rx,ry),rr,random.randint(0,180),0,360,(80,100,200),-1)
            cv2.ellipse(img,(rx,ry),(rr[0]-5,rr[1]-4),0,0,360,(125,145,220),-1)
    img   = cv2.GaussianBlur(img, (3,3), 0)
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
#  Shared: train classifiers + evaluate + save
# ─────────────────────────────────────────────
def _train_and_save(X_train, y_train, X_test, y_test,
                    use_deep, models_dir, t0, verbose):
    """
    Given already-extracted feature matrices, runs SMOTE → PCA →
    trains SVM/RF/GB → evaluates ensemble → saves all artefacts.
    Returns metrics dict.
    """
    os.makedirs(models_dir, exist_ok=True)

    # 1. SMOTE
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    if verbose:
        print(f"[Trainer] After SMOTE : {X_train_sm.shape[0]} training samples")

    # 2. L2 normalise
    X_tr_n = normalize(X_train_sm, norm="l2")
    X_te_n = normalize(X_test,     norm="l2")

    # 3. PCA — first find how many components give 95% variance,
    #    then enforce a minimum of 50 so we don't lose too much information
    pca_probe = PCA(n_components=0.95, random_state=42)
    pca_probe.fit(X_tr_n)
    n_auto = pca_probe.n_components_
    n_final = max(n_auto, min(50, X_tr_n.shape[1]))   # at least 50, at most full dim

    pca     = PCA(n_components=n_final, random_state=42)
    Z_train = pca.fit_transform(X_tr_n)
    Z_test  = pca.transform(X_te_n)
    var_explained = pca.explained_variance_ratio_.sum() * 100
    if verbose:
        print(f"[Trainer] PCA        : {X_tr_n.shape[1]}-d → "
              f"{Z_train.shape[1]}-d  ({var_explained:.1f}% variance retained)")

    # 4. Train classifiers
    if verbose: print("[Trainer] Training SVM (RBF, C=10, class_weight=balanced) …")
    svm = SVC(kernel="rbf", C=10, gamma="auto",
              class_weight="balanced", probability=True, random_state=42)
    svm.fit(Z_train, y_train_sm)

    if verbose: print("[Trainer] Training Random Forest (n=200, class_weight=balanced) …")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                random_state=42, n_jobs=-1)
    rf.fit(Z_train, y_train_sm)

    if verbose: print("[Trainer] Training Gradient Boosting (n=150) …")
    gb = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                    max_depth=4, random_state=42)
    gb.fit(Z_train, y_train_sm)

    # 5. Ensemble evaluation (soft majority vote)
    p_svm  = svm.predict_proba(Z_test)[:, 1]
    p_rf   = rf.predict_proba(Z_test)[:, 1]
    p_gb   = gb.predict_proba(Z_test)[:, 1]
    avg    = (p_svm + p_rf + p_gb) / 3.0
    y_pred = (avg >= 0.5).astype(int)

    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred))  * 100, 2),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)) * 100, 2),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0))    * 100, 2),
        "f1_score":  round(float(f1_score(y_test, y_pred, zero_division=0))        * 100, 2),
        "roc_auc":   round(float(roc_auc_score(y_test, avg))      * 100, 2),
        "n_train":   int(X_train_sm.shape[0]),
        "n_test":    int(X_test.shape[0]),
        "pca_components": int(Z_train.shape[1]),
        "feature_dim":    int(X_tr_n.shape[1]),
        "use_deep":       use_deep,
        "training_time_s": round(time.time() - t0, 1),
        "classifier_accuracy": {
            "SVM": round(float(accuracy_score(y_test, svm.predict(Z_test)))*100, 2),
            "RF":  round(float(accuracy_score(y_test, rf.predict(Z_test))) *100, 2),
            "GB":  round(float(accuracy_score(y_test, gb.predict(Z_test))) *100, 2),
        }
    }

    if verbose:
        print(f"\n[Trainer] ── Ensemble Results ─────────────────")
        print(f"  Accuracy   : {metrics['accuracy']}%")
        print(f"  Precision  : {metrics['precision']}%")
        print(f"  Recall     : {metrics['recall']}%")
        print(f"  F1-Score   : {metrics['f1_score']}%")
        print(f"  ROC-AUC    : {metrics['roc_auc']}%")
        print(f"  Individual : SVM={metrics['classifier_accuracy']['SVM']}%  "
              f"RF={metrics['classifier_accuracy']['RF']}%  "
              f"GB={metrics['classifier_accuracy']['GB']}%")
        print(f"  Train time : {metrics['training_time_s']}s")

    # 6. Save artefacts
    joblib.dump(pca, f"{models_dir}/pca.pkl")
    joblib.dump(svm, f"{models_dir}/svm.pkl")
    joblib.dump(rf,  f"{models_dir}/rf.pkl")
    joblib.dump(gb,  f"{models_dir}/gb.pkl")
    joblib.dump({"use_deep": use_deep}, f"{models_dir}/config.pkl")
    with open(f"{models_dir}/training_report.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    if verbose:
        print(f"[Trainer] Models saved → '{models_dir}/'")

    return metrics


# ─────────────────────────────────────────────
#  OPTION A: Synthetic data trainer
# ─────────────────────────────────────────────
def train(
    n_per_class: int  = 300,
    use_deep:    bool = False,
    models_dir:  str  = "models",
    verbose:     bool = True,
):
    """Train on synthetic blood smear images (demo/fallback mode)."""
    t0   = time.time()
    prep = ImagePreprocessor()
    hext = HybridFeatureExtractor(use_deep=use_deep)

    if verbose:
        print(f"[Trainer] Generating {n_per_class * 2} synthetic images …")

    X, y = [], []
    for i in range(n_per_class):
        if verbose and i % 50 == 0:
            print(f"\r[Trainer]  ALL    {i}/{n_per_class}", end="", flush=True)
        pd_ = prep.process(_make_all_cell())
        X.append(hext.extract(pd_)); y.append(1)
    if verbose: print()

    for i in range(n_per_class):
        if verbose and i % 50 == 0:
            print(f"\r[Trainer]  Normal {i}/{n_per_class}", end="", flush=True)
        pd_ = prep.process(_make_normal_cell())
        X.append(hext.extract(pd_)); y.append(0)
    if verbose: print()

    X = np.array(X, np.float32)
    y = np.array(y, np.int32)

    X_tv,    X_test,  y_tv,    y_test  = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42)
    X_train, X_val,   y_train, y_val   = train_test_split(
        X_tv, y_tv, test_size=0.15/0.85, stratify=y_tv, random_state=42)

    if verbose:
        print(f"[Trainer] Split → train={len(X_train)}, "
              f"val={len(X_val)}, test={len(X_test)}")

    return _train_and_save(X_train, y_train, X_test, y_test,
                           use_deep, models_dir, t0, verbose)


# ─────────────────────────────────────────────
#  OPTION B: Real dataset trainer
# ─────────────────────────────────────────────
def train_from_directory(
    train_dir:  str,
    test_dir:   str  = None,
    use_deep:   bool = True,
    models_dir: str  = "models",
    test_split: float = 0.20,
    verbose:    bool  = True,
):
    """
    Train on REAL blood smear images loaded from disk.

    Expected folder structure:
        train_dir/
            ALL/        ← ALL-positive images (.jpg .png .bmp .tiff)
            Normal/     ← Normal / benign images

        test_dir/       ← optional; if None, 20% of train_dir is used
            ALL/
            Normal/

    For the Kaggle mehradaria dataset the folders are named:
        benign/   → treated as Normal
        malignant/ (with sub-folders early/, pre/, pro/) → treated as ALL
    The function auto-detects both naming conventions.
    """
    t0   = time.time()
    prep = ImagePreprocessor()
    hext = HybridFeatureExtractor(use_deep=use_deep)
    exts = ["*.jpg","*.jpeg","*.png","*.bmp","*.tiff","*.tif",
            "*.JPG","*.JPEG","*.PNG","*.BMP"]

    # ── helper: collect (filepath, label) from a root folder ────────────
    def collect_files(root):
        """
        Supports two naming conventions:
          1. ALL/ and Normal/          (our standard layout)
          2. benign/ and malignant/    (Kaggle mehradaria layout)
             malignant may have sub-folders: early/, pre/, pro/
        """
        records = []  # list of (path, label_int)

        # Convention 1 — ALL / Normal
        for cls, lbl in [("ALL", 1), ("Normal", 0)]:
            folder = os.path.join(root, cls)
            if os.path.isdir(folder):
                files = []
                for ext in exts:
                    files.extend(glob.glob(os.path.join(folder, ext)))
                for fp in files:
                    records.append((fp, lbl))
                if verbose and files:
                    print(f"[Trainer]   {cls:8s} ({lbl}) : {len(files):5d} images  ← {folder}")

        # Convention 2 — benign / malignant
        for cls, lbl in [("benign", 0), ("malignant", 1)]:
            folder = os.path.join(root, cls)
            if os.path.isdir(folder):
                files = []
                # direct images
                for ext in exts:
                    files.extend(glob.glob(os.path.join(folder, ext)))
                # sub-folders (early, pre, pro …)
                for sub in os.listdir(folder):
                    sub_path = os.path.join(folder, sub)
                    if os.path.isdir(sub_path):
                        for ext in exts:
                            files.extend(glob.glob(os.path.join(sub_path, ext)))
                for fp in files:
                    records.append((fp, lbl))
                if verbose and files:
                    label_name = "Normal" if lbl == 0 else "ALL"
                    print(f"[Trainer]   {cls:10s}→{label_name} ({lbl}) : {len(files):5d} images  ← {folder}")

        return records

    # ── helper: extract features from a file list ────────────────────────
    def extract_features(records, split_name):
        X, y = [], []
        n = len(records)
        for i, (fp, lbl) in enumerate(records):
            if verbose and i % 50 == 0:
                print(f"\r[Trainer] {split_name} {i+1}/{n}", end="", flush=True)
            img = cv2.imread(fp)
            if img is None:
                if verbose:
                    print(f"\n[Trainer] WARNING: could not read {fp} — skipping")
                continue
            pd_  = prep.process(img)
            fv   = hext.extract(pd_)
            X.append(fv)
            y.append(lbl)
        if verbose: print()
        return np.array(X, np.float32), np.array(y, np.int32)

    # ── Load training files ──────────────────────────────────────────────
    if verbose: print(f"\n[Trainer] Scanning training folder: {train_dir}")
    train_records = collect_files(train_dir)
    if len(train_records) == 0:
        raise ValueError(
            f"No images found in '{train_dir}'.\n"
            "Make sure the folder contains ALL/ and Normal/ sub-folders "
            "(or benign/ and malignant/) with image files inside.")

    random.shuffle(train_records)

    # ── Load or split test files ─────────────────────────────────────────
    if test_dir and os.path.isdir(test_dir):
        if verbose: print(f"[Trainer] Scanning test folder   : {test_dir}")
        test_records = collect_files(test_dir)
        if verbose:
            print(f"[Trainer] Train files : {len(train_records)}")
            print(f"[Trainer] Test files  : {len(test_records)}")
    else:
        # Stratified split from train_dir
        if verbose:
            print(f"[Trainer] No separate test_dir — using {int(test_split*100)}% of train_dir as test")
        labels        = [r[1] for r in train_records]
        train_idx, test_idx = train_test_split(
            range(len(train_records)), test_size=test_split,
            stratify=labels, random_state=42)
        test_records  = [train_records[i] for i in test_idx]
        train_records = [train_records[i] for i in train_idx]
        if verbose:
            print(f"[Trainer] Train files : {len(train_records)}")
            print(f"[Trainer] Test files  : {len(test_records)}")

    # ── Extract features ─────────────────────────────────────────────────
    if verbose: print("\n[Trainer] Extracting training features …")
    X_train, y_train = extract_features(train_records, "Train")

    if verbose: print("[Trainer] Extracting test features …")
    X_test,  y_test  = extract_features(test_records,  "Test ")

    if verbose:
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"[Trainer] Train class {u} ({'ALL' if u==1 else 'Normal'}) : {c} samples")
        unique, counts = np.unique(y_test, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"[Trainer] Test  class {u} ({'ALL' if u==1 else 'Normal'}) : {c} samples")

    return _train_and_save(X_train, y_train, X_test, y_test,
                           use_deep, models_dir, t0, verbose)


# ─────────────────────────────────────────────
#  Entry point — change here to switch modes
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── REAL DATASET MODE (uncomment and set your path) ──────────────────
    train_from_directory(
        train_dir  = "data/train",   # folder with ALL/ and Normal/ inside
        test_dir   = "data/test",    # set to None to auto-split from train_dir
        use_deep   = True,           # True = ResNet50 + handcrafted features
        models_dir = "models",
        verbose    = True,
    )

    # ── SYNTHETIC MODE (uncomment to use instead) ─────────────────────────
    # train(n_per_class=300, use_deep=True, models_dir="models", verbose=True)
