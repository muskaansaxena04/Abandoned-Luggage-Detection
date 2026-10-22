"""
Step 1 (alternative to 01_download_data.py): build the dataset from a downloaded ZIP.

  python 01b_prepare_from_zip.py --zip path/to/dataset.zip

Accepts a zip in YOLO format (train/images, train/labels, data.yaml)
or COCO format (train/_annotations.coco.json + images). Produces BOTH:

  data_yolo/{train,valid,test}/{images,labels}  + data_yolo/data_fixed.yaml   (YOLOv8 / YOLOv11 training + val)
  data_coco/{train,valid,test}/<images> + _annotations.coco.json              (DETR + COCO / size metrics)

Then it prints class names and counts so you can compare with the paper's CCTV-Dusseldorf numbers
(350 images, 7751 annotations = 5115 person + 2636 luggage; 75/10/15 split).
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

import yaml
from PIL import Image

SPLITS = {"train": ["train"], "valid": ["valid", "val", "validation"], "test": ["test"]}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

p = argparse.ArgumentParser()
p.add_argument("--zip", required=True)
p.add_argument("--raw", default="data_raw")
p.add_argument("--out-yolo", default="data_yolo")
p.add_argument("--out-coco", default="data_coco")
a = p.parse_args()

raw, oy, oc = Path(a.raw), Path(a.out_yolo), Path(a.out_coco)
for d in (raw, oy, oc):
    if d.exists():
        shutil.rmtree(d)           # these folders are generated; start clean
raw.mkdir(parents=True)
with zipfile.ZipFile(a.zip) as z:
    z.extractall(raw)


def find_dir(root, split, kind):
    for sv in SPLITS[split]:
        for cand in (root / sv / kind, root / kind / sv):
            if cand.is_dir():
                return cand
    return None


def list_images(d):
    return sorted(x for x in d.iterdir() if x.suffix.lower() in IMG_EXT)


def find_coco_json(root, split):
    for sv in SPLITS[split]:
        for cand in (root / sv / "_annotations.coco.json", root / sv / "annotations.json"):
            if cand.exists():
                return cand
    return None


yaml_files = list(raw.rglob("data.yaml"))
coco_any = any(find_coco_json(r.parent, "train") for r in raw.rglob("_annotations.coco.json")) \
    or bool(list(raw.rglob("_annotations.coco.json")))

# =================== YOLO zip ===================
if yaml_files:
    root = yaml_files[0].parent
    y = yaml.safe_load(open(yaml_files[0]))
    names = y["names"]
    names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    print("YOLO-format zip detected. classes:", names)

    for s in SPLITS:
        idir, ldir = find_dir(root, s, "images"), find_dir(root, s, "labels")
        if idir is None:
            print(f"  [warn] no '{s}' split found")
            continue
        (oy / s / "images").mkdir(parents=True)
        (oy / s / "labels").mkdir(parents=True)
        (oc / s).mkdir(parents=True)
        imgs, anns, aid = [], [], 1
        for iid, ip in enumerate(list_images(idir), 1):
            shutil.copy(ip, oy / s / "images" / ip.name)
            shutil.copy(ip, oc / s / ip.name)
            w, h = Image.open(ip).size
            imgs.append({"id": iid, "file_name": ip.name, "width": w, "height": h})
            lf = (ldir / (ip.stem + ".txt")) if ldir else None
            if lf and lf.exists():
                shutil.copy(lf, oy / s / "labels" / lf.name)
                for line in lf.read_text().splitlines():
                    t = line.split()
                    if len(t) < 5:
                        continue
                    c = int(float(t[0]))
                    if len(t) > 5:   # polygon -> bounding box
                        xs = [float(v) * w for v in t[1::2]]
                        ys = [float(v) * h for v in t[2::2]]
                        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                    else:
                        xc, yc, bw, bh = [float(v) for v in t[1:5]]
                        x0, x1 = (xc - bw / 2) * w, (xc + bw / 2) * w
                        y0, y1 = (yc - bh / 2) * h, (yc + bh / 2) * h
                    anns.append({"id": aid, "image_id": iid, "category_id": c + 1,
                                 "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                                 "iscrowd": 0})
                    aid += 1
        coco = {"images": imgs, "annotations": anns,
                "categories": [{"id": i + 1, "name": n} for i, n in enumerate(names)]}
        json.dump(coco, open(oc / s / "_annotations.coco.json", "w"))

# =================== COCO zip ===================
elif coco_any:
    root = next(r.parent.parent if r.parent.name in sum(SPLITS.values(), []) else r.parent
                for r in raw.rglob("_annotations.coco.json"))
    print("COCO-format zip detected.")
    data = {}
    for s in SPLITS:
        jp = find_coco_json(root, s)
        if jp:
            data[s] = (jp, json.load(open(jp)))
        else:
            print(f"  [warn] no '{s}' split found")
    used = sorted({an["category_id"] for _, j in data.values() for an in j["annotations"]})
    catname = {c["id"]: c["name"] for _, j in data.values() for c in j["categories"]}
    names = [catname[i] for i in used]
    idx = {cid: i for i, cid in enumerate(used)}
    print("classes (from annotations):", names)

    for s, (jp, j) in data.items():
        (oy / s / "images").mkdir(parents=True)
        (oy / s / "labels").mkdir(parents=True)
        (oc / s).mkdir(parents=True)
        by_img = {}
        for an in j["annotations"]:
            by_img.setdefault(an["image_id"], []).append(an)
        for im in j["images"]:
            src = jp.parent / im["file_name"]
            shutil.copy(src, oy / s / "images" / src.name)
            shutil.copy(src, oc / s / src.name)
            W, H = im["width"], im["height"]
            lines = []
            for an in by_img.get(im["id"], []):
                x, y_, w, h = an["bbox"]
                lines.append(f"{idx[an['category_id']]} {(x + w / 2) / W:.6f} {(y_ + h / 2) / H:.6f} "
                             f"{w / W:.6f} {h / H:.6f}")
            (oy / s / "labels" / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        shutil.copy(jp, oc / s / "_annotations.coco.json")
else:
    raise SystemExit("Could not find data.yaml (YOLO) or _annotations.coco.json (COCO) in the zip. "
                     "Send me the zip's folder listing and I will adapt this script.")

# data yaml with absolute paths (valid/val/test folder names normalised to train/valid/test)
fixed = {"path": str(oy.resolve()), "train": "train/images", "val": "valid/images",
         "test": "test/images", "names": names}
yaml.safe_dump(fixed, open(oy / "data_fixed.yaml", "w"), sort_keys=False)

# ---------- sanity report ----------
print("\nSplit report (paper: 350 imgs total, 7751 annotations = 5115 person + 2636 luggage; 75/10/15):")
tot_img, tot_ann, per_cls, size_ok = 0, 0, {}, True
for s in SPLITS:
    jp = oc / s / "_annotations.coco.json"
    if not jp.exists():
        continue
    j = json.load(open(jp))
    cn = {c["id"]: c["name"] for c in j["categories"]}
    small = sum(1 for an in j["annotations"] if an["area"] < 32 ** 2)
    med = sum(1 for an in j["annotations"] if 32 ** 2 <= an["area"] < 96 ** 2)
    print(f"  {s:6s} images={len(j['images']):4d} annotations={len(j['annotations']):5d} "
          f"small={small} medium={med}")
    tot_img += len(j["images"]); tot_ann += len(j["annotations"])
    for an in j["annotations"]:
        per_cls[cn[an["category_id"]]] = per_cls.get(cn[an["category_id"]], 0) + 1
    size_ok &= all((im["width"], im["height"]) == (640, 640) for im in j["images"])
print(f"  TOTAL  images={tot_img} annotations={tot_ann} per class={per_cls}")

if not size_ok:
    print("\n[warn] some images are not 640x640. The paper resized to 640x640 (stretched); AP_small/AP_medium "
          "depend on that pixel scale. Resize before running if you want comparable numbers.")
if "test" not in [s for s in SPLITS if (oc / s).exists()]:
    print("\n[warn] no test split: the paper's protocol needs a held-out test set (15%).")
low = {n.lower() for n in names}
if "person" not in low or not (low & {"luggage", "suitcase", "baggage"}):
    print(f"\n[warn] class names {names}: eval scripts expect 'person' and one of luggage/suitcase/baggage. "
          "Add your luggage class name to LUGGAGE in eval_utils.py.")
print("\nNext: python 02_train_yolo.py")
