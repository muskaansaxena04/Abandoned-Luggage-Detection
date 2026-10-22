"""
Step 2: fine-tune YOLOv8m and YOLOv11m on CCTV-Dusseldorf.

Paper setup: COCO-pretrained, 2 classes, 640x640, batch 16, momentum 0.937, up to 150 epochs,
best.pt (best validation fitness) used for testing.

NOTE: optimizer is left at Ultralytics' default ('auto'). For small datasets like this one
'auto' selects AdamW and overrides `momentum`. The paper lists 0.937 (the default value) and
does not name an optimizer, so the defaults are the most faithful choice.

Windows: everything must run under `if __name__ == "__main__":` because the data-loader
workers re-import this file.

  python 02_train_yolo.py                       # yolov8m + yolo11m, seeds 0 1 2
  python 02_train_yolo.py --seeds 0 --models yolov8m.pt
"""
import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["yolov8m.pt", "yolo11m.pt"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--data", default="data_yolo/data_fixed.yaml")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    project = str(Path("runs").resolve())   # absolute path avoids Ultralytics nesting it under runs/detect
    for w in a.models:
        for seed in a.seeds:
            tag = f"dus_{Path(w).stem}_s{seed}"
            print(f"\n===== training {tag} =====")
            YOLO(w).train(
                data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                momentum=0.937, optimizer=a.optimizer,
                seed=seed, deterministic=True, workers=a.workers,
                project=project, name=tag, exist_ok=False,
            )
            # outputs: runs/<tag>/weights/best.pt, results.csv, results.png, PR_curve.png, confusion_matrix.png


if __name__ == "__main__":
    main()