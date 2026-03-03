"""
train.py — Fine-tune YOLOv8 on a custom traffic dataset (YOLO format).

Usage:
    python train.py --epochs 50 --batch 16 --imgsz 640

Dataset expected at:
    dataset/
    ├── images/
    │   ├── train/   *.jpg / *.png
    │   └── val/     *.jpg / *.png
    └── labels/
        ├── train/   *.txt  (YOLO format)
        └── val/     *.txt

YOLO label format per line:
    <class_id> <x_centre> <y_centre> <width> <height>   (all normalised 0-1)

Class IDs (defined in dataset.yaml):
    0: Car
    1: Bus
    2: Truck
    3: Motorcycle
    4: Ambulance
    5: Fire Brigade
"""

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLOv8 on custom traffic dataset")
    p.add_argument("--model",   default="yolov8n.pt",  help="Base YOLOv8 weights")
    p.add_argument("--data",    default="dataset.yaml", help="Path to dataset YAML")
    p.add_argument("--epochs",  type=int, default=50,   help="Number of training epochs")
    p.add_argument("--batch",   type=int, default=8,    help="Batch size (use 4-8 for CPU)")
    p.add_argument("--imgsz",   type=int, default=640,  help="Input image size")
    p.add_argument("--project", default="models",       help="Save directory")
    p.add_argument("--name",    default="traffic_v1",   help="Run name")
    p.add_argument("--device",  default="cpu",          help="cpu or 0 (GPU id)")
    return p.parse_args()


def check_dataset(data_yaml: str):
    """Verify dataset.yaml exists and images/labels dirs are present."""
    if not os.path.exists(data_yaml):
        print(f"✗  dataset.yaml not found at: {data_yaml}")
        print("   Create it or run:  python train.py --help")
        sys.exit(1)

    for split in ("train", "val"):
        img_dir = os.path.join("dataset", "images", split)
        lbl_dir = os.path.join("dataset", "labels", split)
        if not os.path.isdir(img_dir):
            print(f"⚠  Missing images directory: {img_dir}")
        if not os.path.isdir(lbl_dir):
            print(f"⚠  Missing labels directory: {lbl_dir}")

    n_train = len(os.listdir(os.path.join("dataset", "images", "train"))) if os.path.isdir("dataset/images/train") else 0
    n_val   = len(os.listdir(os.path.join("dataset", "images", "val")))   if os.path.isdir("dataset/images/val")   else 0
    print(f"  Dataset: {n_train} train images, {n_val} val images")


def train(args):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("✗  ultralytics not installed. Run:  pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("  NEXUS Traffic AI — YOLOv8 Custom Training")
    print("=" * 60)
    print(f"  Model   : {args.model}")
    print(f"  Dataset : {args.data}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  ImgSz   : {args.imgsz}")
    print(f"  Device  : {args.device}")
    print("=" * 60)

    check_dataset(args.data)

    model = YOLO(args.model)

    results = model.train(
        data       = args.data,
        epochs     = args.epochs,
        batch      = args.batch,
        imgsz      = args.imgsz,
        project    = args.project,
        name       = args.name,
        device     = args.device,
        workers    = 2,                 # safe for CPU
        patience   = 15,               # early stopping
        save       = True,
        plots      = True,
        verbose    = True,
        optimizer  = "AdamW",
        lr0        = 0.001,
        lrf        = 0.01,
        augment    = True,
        mosaic     = 1.0,
        mixup      = 0.1,
    )

    best_path = os.path.join(args.project, args.name, "weights", "best.pt")
    print("\n✅  Training complete!")
    print(f"   Best weights: {best_path}")
    print(f"   Copy to models/ and update detect.py model_path to use them.")
    return results


def validate(args):
    """Quick validation run on trained weights."""
    from ultralytics import YOLO
    best = os.path.join(args.project, args.name, "weights", "best.pt")
    if not os.path.exists(best):
        print(f"✗  No weights found at {best}. Train first.")
        return
    model = YOLO(best)
    metrics = model.val(data=args.data, device=args.device)
    print(f"\nValidation mAP50: {metrics.box.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
    validate(args)
