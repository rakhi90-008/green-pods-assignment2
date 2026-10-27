import argparse, time
from pathlib import Path
import joblib
import cv2
import numpy as np
import torch
from PIL import Image
from skimage.feature import local_binary_pattern
from torch import nn
from torchvision import models, transforms
from common import choose_device

def classical_features(path, size=128):
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
    lbp = local_binary_pattern(gray, 16, 2, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 20), range=(0, 19), density=True)
    feats.extend(hist.astype(float).tolist())
    return np.asarray(feats, dtype=np.float32).reshape(1, -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--classical_model", default="outputs/classical/classical_svm.joblib")
    ap.add_argument("--cnn_model", default="outputs/mobilenet/best_mobilenet_v2.pth")
    args = ap.parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    classical_bundle = joblib.load(args.classical_model)
    classical = classical_bundle["model"]
    class_names = classical_bundle["class_names"]
    feat = classical_features(image_path)
    probs = classical.predict_proba(feat)[0]
    idx = int(np.argmax(probs))
    print("\n=== Classical CV + SVM ===")
    print(f"Prediction: {class_names[idx]} (estimated SVM probability: {probs[idx]:.3f})")

    device = choose_device()
    checkpoint = torch.load(args.cnn_model, map_location=device)
    cnn_classes = checkpoint["class_names"]
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(cnn_classes))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    # Standard ImageNet normalization used during training.
    tf = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    with Image.open(image_path) as im:
        x = tf(im.convert("RGB")).unsqueeze(0).to(device)
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    elapsed_ms = (time.perf_counter() - start) * 1000
    idx = int(np.argmax(probs))
    print("\n=== MobileNetV2 ===")
    print(f"Prediction: {cnn_classes[idx]} (confidence: {probs[idx]:.3f})")
    print(f"Single-image model-forward latency: {elapsed_ms:.2f} ms on {device}; excludes image loading/preprocessing.")
    print("\nInterpret scores as model confidence, not a guarantee of correct ripeness.")

if __name__ == "__main__":
    main()
