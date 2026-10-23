"""
Phase 1 / step 1: get the Roboflow LuggageDataset (v6) into data_luggage/ (separate from the Dusseldorf data).

  A) Roboflow API (use a NEW key, never paste it anywhere):
        PowerShell:  $env:ROBOFLOW_API_KEY="your_key"
        python L01_download_luggage_data.py
  B) ZIP downloaded from the Universe page (Download Dataset -> YOLO format):
        python L01_download_luggage_data.py --zip path\\to\\LuggageDataset.v6.yolov8.zip

Prints split sizes, instances per class and the small/medium/large mix, so you can compare with the repo README
(29,053 images; train 25,302 / valid 2,954 / test 797; 130,475 instances; classes backpack, bag, trolley).
"""
import argparse
import os
import zipfile
from pathlib import Path

import yaml

p = argparse.ArgumentParser()
p.add_argument("--zip", default=None)
p.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"))
p.add_argument("--workspace", default="guns-detection-cvwjs")
p.add_argument("--project", default="luggagedataset-24pgo")
p.add_argument("--version", type=int, default=6)
p.add_argument("--out", default="data_luggage")
a = p.parse_args()

out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

if a.zip:
    with zipfile.ZipFile(a.zip) as z:
        z.extractall(out)
else:
    if not a.api_key:
        raise SystemExit("Set ROBOFLOW_API_KEY (or pass --api-key), or use --zip.")
    from roboflow import Roboflow
    ver = Roboflow(api_key=a.api_key).workspace(a.workspace).project(a.project).version(a.version)
    ver.download("yolov8", location=str(out))

yamls = list(out.rglob("data.yaml"))
if not yamls:
    raise SystemExit("data.yaml not found after download/extract.")
root = yamls[0].parent
y = yaml.safe_load(open(yamls[0]))
names = y["names"]
names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def split_dir(*cands):
    for c in cands:
        if (root / c / "images").is_dir():
            return c
    return None


sd = {"train": split_dir("train"), "val": split_dir("valid", "val", "validation"), "test": split_dir("test")}
fixed = {"path": str(root.resolve()), "names": names}
for k, v in sd.items():
    if v:
        fixed[k] = f"{v}/images"
yaml.safe_dump(fixed, open(root / "data_fixed.yaml", "w"), sort_keys=False)
print("classes:", names)
print("data yaml ->", root / "data_fixed.yaml")

SMALL, MED = 0.0014, 0.0225      # normalised-area thresholds used in the repo README (~24x24 and ~96x96 at 640)
print("\nsplit        images  instances   per class                        small / medium / large")
tot_i = tot_n = 0
for k, v in sd.items():
    if not v:
        print(f"{k:6s} missing")
        continue
    n_img = len(list((root / v / "images").iterdir()))
    cnt = {n: 0 for n in names}
    s = m = l = 0
    for lf in (root / v / "labels").glob("*.txt"):
        for line in lf.read_text().splitlines():
            t = line.split()
            if len(t) < 5:
                continue
            c = int(float(t[0]))
            if len(t) > 5:
                xs, ys = [float(q) for q in t[1::2]], [float(q) for q in t[2::2]]
                w, h = max(xs) - min(xs), max(ys) - min(ys)
            else:
                w, h = float(t[3]), float(t[4])
            cnt[names[c]] += 1
            area = w * h
            s += area < SMALL
            m += SMALL <= area < MED
            l += area >= MED
    n_ins = sum(cnt.values())
    tot_i += n_img
    tot_n += n_ins
    print(f"{k:6s} {n_img:10d} {n_ins:10d}   {cnt}   {s} / {m} / {l}")
print(f"TOTAL  {tot_i:10d} {tot_n:10d}")
print("\nNext: python L02_train_luggage.py --models yolo26s.pt --epochs 3 --fraction 0.05 --name-suffix _pilot   (quick pilot)")
