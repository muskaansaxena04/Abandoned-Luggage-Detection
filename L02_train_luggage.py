"""
Phase 1 / step 2: fine-tune YOLO26 on the luggage dataset (classes: backpack, bag, trolley).

The dataset already has offline augmentation (3x on train, from the Roboflow export), and Ultralytics adds its
usual online augmentation on top; defaults are kept. Early stopping uses the validation split.

  python L02_train_luggage.py --models yolo26s.pt --epochs 3 --fraction 0.05 --name-suffix _pilot   # pilot: check speed
  python L02_train_luggage.py                                                                         # yolo26s + yolo26m
  python L02_train_luggage.py --models yolo26m.pt yolo26l.pt --epochs 100 --patience 20

Delete pilot runs (runs/lug_*_pilot) before the real test step.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["yolo26s.pt", "yolo26m.pt"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--data", default="data_luggage/data_fixed.yaml")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--fraction", type=float, default=1.0, help="use only this fraction of train (for pilots)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--name-suffix", default="")
    a = p.parse_args()

    project = str(Path("runs").resolve())
    for w in a.models:
        for seed in a.seeds:
            tag = f"lug_{Path(w).stem}_s{seed}{a.name_suffix}"
            print(f"\n===== training {tag} =====")
            YOLO(w).train(
                data=a.data, epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch,
                fraction=a.fraction, seed=seed, deterministic=True, workers=a.workers,
                project=project, name=tag, exist_ok=False,
            )


if __name__ == "__main__":
    main()
