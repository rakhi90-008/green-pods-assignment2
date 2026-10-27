import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from skimage.feature import local_binary_pattern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
from common import get_class_dirs, image_files, compute_metrics, save_metrics

def extract_features(path, size=128):
    # Color descriptors: mean/std of HSV and CIELAB channels.
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB").resize((size, size)))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    feats = []
    for img in (hsv, lab):
        for ch in cv2.split(img):
            feats.extend([float(np.mean(ch)), float(np.std(ch)),
                          float(np.percentile(ch, 10)), float(np.percentile(ch, 50)),
                          float(np.percentile(ch, 90))])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # LBP texture histogram.
    radius, points = 2, 16
    lbp = local_binary_pattern(gray, points, radius, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, points + 4), range=(0, points + 3), density=True)
    feats.extend(hist.astype(float).tolist())
    return np.asarray(feats, dtype=np.float32)

def load_split(root, split, class_names):
    X, y, paths = [], [], []
    for idx, name in enumerate(class_names):
        folder = Path(root) / split / name
        for p in image_files(folder):
            try:
                X.append(extract_features(p))
                y.append(idx)
                paths.append(p)
            except Exception as e:
                print(f"Skipping unreadable image {p}: {e}")
    if not X:
        raise ValueError(f"No readable images found under {Path(root)/split}")
    return np.vstack(X), np.asarray(y), paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/split")
    ap.add_argument("--output_dir", default="outputs/classical")
    args = ap.parse_args()
    root, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    class_names = [p.name for p in get_class_dirs(root / "train")]
    X_train, y_train, _ = load_split(root, "train", class_names)
    X_val, y_val, _ = load_split(root, "val", class_names)
    X_test, y_test, test_paths = load_split(root, "test", class_names)

    model = Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(C=3.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_metrics = compute_metrics(y_val, val_pred, class_names)
    test_pred = model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_pred, class_names)
    joblib.dump({"model": model, "class_names": class_names}, out / "classical_svm.joblib")
    save_metrics(test_metrics, class_names, out, "test")
    save_metrics(val_metrics, class_names, out, "validation")
    print("Validation:", {k: round(v, 4) for k, v in val_metrics.items() if isinstance(v, float)})
    print("TEST:", {k: round(v, 4) for k, v in test_metrics.items() if isinstance(v, float)})
    print(f"Saved model and metrics to {out.resolve()}")

if __name__ == "__main__":
    main()
