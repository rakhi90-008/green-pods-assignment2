import argparse, copy, time, json
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from common import get_class_dirs, choose_device, set_seed, compute_metrics, save_metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/split")
    ap.add_argument("--output_dir", default="outputs/mobilenet")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=4)
    args = ap.parse_args()
    set_seed(42)
    root, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = choose_device()
    print("Device:", device)
    class_names = [p.name for p in get_class_dirs(root / "train")]
    weights = MobileNet_V2_Weights.DEFAULT
    mean, std = weights.transforms().mean, weights.transforms().std
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_ds = datasets.ImageFolder(root / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(root / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(root / "test", transform=eval_tf)
    if train_ds.classes != class_names or val_ds.classes != class_names or test_ds.classes != class_names:
        raise ValueError("Class folders differ between splits; rerun 02_split_dataset.py.")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    try:
        model = models.mobilenet_v2(weights=weights)
    except Exception as e:
        raise RuntimeError("Could not load pretrained MobileNetV2 weights. Check internet access for the first run, or cache weights first. " + str(e))
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    best_loss, best_state, stale = float("inf"), None, 0
    history = []

    for epoch in range(args.epochs):
        start = time.perf_counter()
        model.train()
        train_loss_sum, train_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * y.size(0)
            train_n += y.size(0)

        model.eval()
        val_loss_sum, val_n, val_true, val_pred = 0.0, 0, [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss_sum += loss.item() * y.size(0)
                val_n += y.size(0)
                val_true.extend(y.cpu().tolist())
                val_pred.extend(logits.argmax(1).cpu().tolist())
        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)
        val_acc = sum(int(a == b) for a, b in zip(val_true, val_pred)) / max(len(val_true), 1)
        history.append({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"Epoch {epoch+1}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_acc:.4f} elapsed={time.perf_counter()-start:.1f}s")
        if val_loss < best_loss:
            best_loss, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
            torch.save({"state_dict": best_state, "class_names": class_names}, out / "best_mobilenet_v2.pth")
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping: validation loss did not improve.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    with open(out / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    model.eval()
    test_true, test_pred = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            test_true.extend(y.tolist())
            test_pred.extend(logits.argmax(1).cpu().tolist())
    elapsed = time.perf_counter() - start
    metrics = compute_metrics(test_true, test_pred, class_names)
    metrics["test_inference_seconds_for_all_images_batch_mode"] = elapsed
    metrics["test_image_count"] = len(test_ds)
    metrics["average_batch_mode_ms_per_image"] = 1000 * elapsed / max(len(test_ds), 1)
    metrics["device"] = str(device)
    save_metrics(metrics, class_names, out, "test")
    print("TEST:", {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)})
    print("Inference timing is batch-mode and includes dataloader iteration; measure single-image latency separately for deployment.")
    print(f"Saved checkpoint and metrics to {out.resolve()}")

if __name__ == "__main__":
    main()
