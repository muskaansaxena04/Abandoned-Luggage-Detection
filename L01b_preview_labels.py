import random
from pathlib import Path
import cv2, yaml

yml = next(Path("data_luggage").rglob("data_fixed.yaml"))
root = yml.parent
names = yaml.safe_load(open(yml))["names"]
imgs = list((root / "train" / "images").glob("*.*"))
random.seed(0)
out = Path("results_luggage/preview"); out.mkdir(parents=True, exist_ok=True)
for p in random.sample(imgs, 6):
    im = cv2.imread(str(p)); h, w = im.shape[:2]
    lf = root / "train" / "labels" / (p.stem + ".txt")
    for line in (lf.read_text().splitlines() if lf.exists() else []):
        t = line.split(); c = int(float(t[0])); v = [float(x) for x in t[1:]]
        if len(v) == 4:
            cx, cy, bw, bh = v
            x1, y1, x2, y2 = (cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h
        else:
            xs, ys = v[0::2], v[1::2]
            x1, y1, x2, y2 = min(xs)*w, min(ys)*h, max(xs)*w, max(ys)*h
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(im, str(names[c]), (int(x1), max(12, int(y1)-4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(str(out / p.name), im)
print("saved to", out)