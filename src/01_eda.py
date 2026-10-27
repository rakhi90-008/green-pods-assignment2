import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from common import get_class_dirs, image_files

def discover_groups(root):
    root = Path(root)
    groups = []
    # Original supplied structure: root/train/class and root/test/class.
    if (root / "train").is_dir() and (root / "test").is_dir():
        for split in ("train", "test"):
            for cdir in get_class_dirs(root / split):
                groups.append((split, cdir.name, cdir))
    else:
        for cdir in get_class_dirs(root):
            groups.append(("all", cdir.name, cdir))
    return groups

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/original_dataset")
    ap.add_argument("--output_dir", default="outputs/eda")
    args = ap.parse_args()
    root, out = Path(args.data_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = discover_groups(root)
    rows, bad = [], []
    for split, class_name, folder in groups:
        for p in image_files(folder):
            try:
                with Image.open(p) as im:
                    im = ImageOps.exif_transpose(im)
                    w, h = im.size
                    im.verify()
                rows.append({"split": split, "class": class_name, "path": str(p),
                             "width": w, "height": h, "pixels": w*h,
                             "aspect_ratio": round(w/max(h, 1), 4)})
            except Exception as e:
                bad.append({"split": split, "class": class_name, "path": str(p), "error": str(e)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No readable images found under {root}. Check the --data_dir path.")
    df.to_csv(out / "image_metadata.csv", index=False)
    pd.DataFrame(bad, columns=["split","class","path","error"]).to_csv(out / "unreadable_images.csv", index=False)
    counts = df.groupby(["split", "class"]).size().rename("image_count").reset_index()
    counts.to_csv(out / "class_counts.csv", index=False)
    counts.pivot(index="class", columns="split", values="image_count").fillna(0).plot(kind="bar", figsize=(9,5))
    plt.title("Image counts by class and provided split")
    plt.ylabel("Images")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out / "class_distribution.png", dpi=160)
    plt.close()
    df.groupby(["split","class"])[["width","height","pixels","aspect_ratio"]].describe().to_csv(out / "image_size_summary.csv")

    # Small contact sheet for manual visual review
    class_names = sorted(df["class"].unique())
    fig, axes = plt.subplots(len(class_names), 5, figsize=(15, max(3, 3*len(class_names))))
    if len(class_names) == 1:
        axes = [axes]
    for i, cls in enumerate(class_names):
        candidates = df[df["class"] == cls]["path"].head(5).tolist()
        for j in range(5):
            ax = axes[i][j] if len(class_names) > 1 else axes[0][j]
            ax.axis("off")
            if j < len(candidates):
                try:
                    with Image.open(candidates[j]) as im:
                        ax.imshow(ImageOps.exif_transpose(im).convert("RGB"))
                    ax.set_title(cls, fontsize=9)
                except Exception:
                    pass
    plt.tight_layout()
    plt.savefig(out / "sample_contact_sheet.png", dpi=140)
    plt.close(fig)
    print("Readable images:", len(df))
    print("Unreadable images:", len(bad))
    print("Class counts:")
    print(counts.to_string(index=False))
    print("Outputs saved to:", out.resolve())
    print("Review contact sheet manually for label errors, lighting, background, blur and unusual examples.")

if __name__ == "__main__":
    main()
