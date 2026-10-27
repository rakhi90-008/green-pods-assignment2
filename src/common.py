from pathlib import Path
import json
import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def get_class_dirs(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(dirs) < 2:
        raise ValueError(
            f"Expected at least 2 class subfolders directly inside {root}. "
            "Set --data_dir to the directory containing the class folders."
        )
    return dirs

def image_files(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(y_true, y_pred, labels):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(labels))), average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(labels))), target_names=labels,
            output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist()
    }

def save_metrics(metrics, labels, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    report = metrics["classification_report"]
    rows = []
    for label in labels:
        item = report[label]
        rows.append((label, item["precision"], item["recall"], item["f1-score"], item["support"]))
    with open(output_dir / f"{prefix}_per_class_metrics.csv", "w", encoding="utf-8") as f:
        f.write("class,precision,recall,f1_score,support\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    plt.figure(figsize=(7, 6))
    sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{prefix}: confusion matrix")
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}_confusion_matrix.png", dpi=160)
    plt.close()

def choose_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
