import argparse, shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from common import get_class_dirs, image_files, set_seed

def copy_files(paths, labels, out, split):
    for path, label in zip(paths, labels):
        p = Path(path)
        dest = Path(out) / split / label
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest / p.name)

def main():
    ap = argparse.ArgumentParser(description="Create train/validation/test folders from the supplied Orange Dataset.")
    ap.add_argument("--data_dir", default="data/original_dataset")
    ap.add_argument("--output_dir", default="data/split")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--validation_fraction", type=float, default=0.15)
    args = ap.parse_args()
    set_seed(args.seed)
    src, out = Path(args.data_dir), Path(args.output_dir)

    # Provided dataset already contains official train/ and test/ folders.
    if (src / "train").is_dir() and (src / "test").is_dir():
        train_root, test_root = src / "train", src / "test"
        train_classes = [p.name for p in get_class_dirs(train_root)]
        test_classes = [p.name for p in get_class_dirs(test_root)]
        if set(train_classes) != set(test_classes):
            raise ValueError(f"Train/test classes differ: train={train_classes}, test={test_classes}")
        all_paths, all_labels = [], []
        for label in train_classes:
            files = image_files(train_root / label)
            if len(files) < 2:
                raise ValueError(f"Class {label} needs at least 2 training images.")
            all_paths.extend(str(p) for p in files)
            all_labels.extend([label] * len(files))
        train_paths, val_paths, train_y, val_y = train_test_split(
            all_paths, all_labels, test_size=args.validation_fraction,
            random_state=args.seed, stratify=all_labels
        )
        test_paths, test_y = [], []
        for label in train_classes:
            files = image_files(test_root / label)
            test_paths.extend(str(p) for p in files)
            test_y.extend([label] * len(files))
        if out.exists():
            shutil.rmtree(out)
        copy_files(train_paths, train_y, out, "train")
        copy_files(val_paths, val_y, out, "val")
        copy_files(test_paths, test_y, out, "test")
    else:
        # Fallback for a flat root with one folder per class: 70/15/15 split.
        class_dirs = get_class_dirs(src)
        records = [(p, d.name) for d in class_dirs for p in image_files(d)]
        if any(sum(1 for _, y in records if y == d.name) < 3 for d in class_dirs):
            raise ValueError("Each class needs at least 3 images for a stratified train/val/test split.")
        paths = [str(p) for p, _ in records]
        labels = [y for _, y in records]
        train_paths, temp_paths, train_y, temp_y = train_test_split(
            paths, labels, test_size=0.30, random_state=args.seed, stratify=labels
        )
        val_paths, test_paths, val_y, test_y = train_test_split(
            temp_paths, temp_y, test_size=0.50, random_state=args.seed, stratify=temp_y
        )
        if out.exists():
            shutil.rmtree(out)
        copy_files(train_paths, train_y, out, "train")
        copy_files(val_paths, val_y, out, "val")
        copy_files(test_paths, test_y, out, "test")

    print(f"Split created at {out.resolve()}")
    for split in ("train", "val", "test"):
        print(split, {p.name: len(image_files(p)) for p in sorted((out / split).iterdir()) if p.is_dir()})
    print("When source data has official train/test folders, the test set is preserved and validation is sampled only from train.")

if __name__ == "__main__":
    main()
