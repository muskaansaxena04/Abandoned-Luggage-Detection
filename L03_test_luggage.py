"""
Phase 1 / step 3: evaluate the fine-tuned luggage models.

Reports, per model:
  - ALL classes (backpack / bag / trolley): P, R, mAP@0.5, mAP@0.5:0.95
  - each class separately
  - MERGED "luggage" (single class): what the abandonment pipeline actually needs
  - inference speed (ms/image)

Choose the model on --split val; evaluate --split test ONCE for the number you report.

  python L03_test_luggage.py --split val
  python L03_test_luggage.py --split test
"""
import argparse
import csv
import glob
from pathlib import Path

from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", nargs="+", default=None)
    p.add_argument("--data", default="data_luggage/data_fixed.yaml")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results_luggage/luggage_results.csv")
    a = p.parse_args()

    weights = a.weights or sorted(glob.glob("runs/lug_*/weights/best.pt"))
    if not weights:
        raise SystemExit("No weights found (runs/lug_*/weights/best.pt).")
    project = str(Path("runs/test").resolve())
    rows = []

    for w in weights:
        tag = Path(w).parents[1].name
        print(f"\n===== {tag} on {a.split} =====")
        m = YOLO(w)
        r = m.val(data=a.data, split=a.split, imgsz=a.imgsz, batch=16, conf=0.001, plots=True,
                  workers=a.workers, project=project, name=f"{tag}_{a.split}", exist_ok=True)
        ms = r.speed.get("inference", float("nan"))
        rows.append([tag, a.split, "ALL", r.box.mp, r.box.mr, r.box.map50, r.box.map, ms])
        for i, c in enumerate(r.box.ap_class_index):
            rows.append([tag, a.split, r.names[int(c)], r.box.p[i], r.box.r[i], r.box.ap50[i], r.box.ap[i], ms])

        try:   # class-merged evaluation: every luggage type counts as one class
            r2 = m.val(data=a.data, split=a.split, imgsz=a.imgsz, batch=16, conf=0.001, plots=False,
                       single_cls=True, workers=a.workers, project=project,
                       name=f"{tag}_{a.split}_merged", exist_ok=True)
            rows.append([tag, a.split, "MERGED-luggage", r2.box.mp, r2.box.mr, r2.box.map50, r2.box.map, ms])
        except Exception as e:
            print("[warn] merged evaluation failed:", e)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    new = not Path(a.out).exists()
    with open(a.out, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["tag", "split", "scope", "P", "R", "mAP50", "mAP50-95", "infer_ms"])
        wr.writerows(rows)

    print("\n%-28s %-5s %-15s %6s %6s %7s %9s %7s" % ("tag", "split", "scope", "P", "R", "mAP50", "mAP50-95", "ms"))
    for t, s, sc, P, R, m50, m95, ms in rows:
        print("%-28s %-5s %-15s %6.3f %6.3f %7.3f %9.3f %7.1f" % (t, s, sc, P, R, m50, m95, ms))
    print(f"\nappended to {a.out}")


if __name__ == "__main__":
    main()