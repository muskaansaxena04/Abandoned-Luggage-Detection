"""
Phase 2: abandoned-luggage pipeline (two detectors + two trackers + radius/time logic + alarm).

  person model  : stock COCO YOLO26 (class 0 = person only), tracker: --person-tracker
  luggage model : our fine-tuned model (backpack/bag/trolley, all treated as "luggage"), tracker: --luggage-tracker

Logic (follows the base paper's Algorithm 1, with video-time instead of wall-clock):
  - frames are sampled every --sample seconds (paper: 0.25 s)
  - a person owns a bag if they were near it at first sight, or stayed near it for >= --dwell seconds
  - near = centre distance <= r_factor * person box height (paper: 40 px ~ one person height) or --radius-px
  - a bag that moves is ignored; ownership is re-assigned to the people near it when it moves
  - a stationary bag with no owner near it for >= --threshold seconds triggers ONE alarm per bag id

  python P2_abandoned_pipeline.py --video path\\to\\clip.mp4 --max-seconds 30
"""
import argparse
import csv
import math
from collections import defaultdict, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

BLUE, GREEN, ORANGE, RED, WHITE = (255, 0, 0), (0, 200, 0), (0, 165, 255), (0, 0, 255), (255, 255, 255)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--luggage-weights", default="runs/lug_yolo26s_s0/weights/best.pt")
    p.add_argument("--person-weights", default="yolo26s.pt")
    p.add_argument("--person-tracker", default="botsort.yaml")
    p.add_argument("--luggage-tracker", default="bytetrack.yaml")
    p.add_argument("--person-conf", type=float, default=0.25)
    p.add_argument("--luggage-conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--sample", type=float, default=0.25, help="seconds between processed frames")
    p.add_argument("--threshold", type=float, default=10.0, help="unattended seconds before alarm")
    p.add_argument("--r-factor", type=float, default=1.0, help="radius = r_factor * person box height")
    p.add_argument("--radius-px", type=float, default=0.0, help="if > 0, fixed radius in pixels instead")
    p.add_argument("--move-px", type=float, default=6.0, help="displacement that counts as movement")
    p.add_argument("--move-window", type=float, default=2.0, help="seconds of history for movement test")
    p.add_argument("--dwell", type=float, default=2.0, help="seconds near a bag to become its owner")
    p.add_argument("--grace", type=float, default=1.0, help="keep a lost person's last position this long")
    p.add_argument("--bag-grace", type=float, default=5.0, help="keep a hidden bag's state this long")
    p.add_argument("--min-size", type=float, default=20.0, help="ignore boxes shorter than this (px)")
    p.add_argument("--out", default="results_pipeline")
    p.add_argument("--max-seconds", type=float, default=0.0, help="stop after this much video (0 = all)")
    p.add_argument("--show", action="store_true", help="show a live window (needs working cv2.imshow)")
    return p.parse_args()


def run_tracker(model, frame, tracker, conf, imgsz, classes=None, agnostic=False):
    r = model.track(frame, persist=True, tracker=tracker, conf=conf, imgsz=imgsz,
                    classes=classes, agnostic_nms=agnostic, verbose=False)[0]
    out = []
    if r.boxes is None or r.boxes.id is None:
        return out
    xyxy = r.boxes.xyxy.cpu().numpy()
    ids = r.boxes.id.int().cpu().tolist()
    cls = r.boxes.cls.int().cpu().tolist()
    cf = r.boxes.conf.cpu().tolist()
    for b, i, c, s in zip(xyxy, ids, cls, cf):
        out.append((int(i), int(c), float(s), b))
    return out


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


class Bag:
    def __init__(self, bid, cname):
        self.id, self.cname = bid, cname
        self.hist = deque()                 # (t, (cx, cy))
        self.owners = set()
        self.near_time = defaultdict(float)
        self.unatt_since = None
        self.initialised = False
        self.moving = False
        self.last_seen = 0.0
        self.box = None


def main():
    a = get_args()
    out_dir = Path(a.out)
    (out_dir / "alarms").mkdir(parents=True, exist_ok=True)

    person_model = YOLO(a.person_weights)
    luggage_model = YOLO(a.luggage_weights)
    print("person model classes :", {0: person_model.names.get(0)})
    print("luggage model classes:", luggage_model.names)
    if person_model.names.get(0) != "person":
        raise SystemExit("Class 0 of the person model is not 'person' - check --person-weights.")

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps * a.sample))
    dt = step / fps
    print(f"video {W}x{H} @ {fps:.2f} fps, processing every {step} frames (= {dt:.2f} s)")

    stem = Path(a.video).stem
    writer = cv2.VideoWriter(str(out_dir / f"{stem}_annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (W, H))
    log_path = out_dir / f"{stem}_alarms.csv"
    log = open(log_path, "w", newline="")
    log_w = csv.writer(log)
    log_w.writerow(["video_time_s", "frame", "bag_id", "bag_class", "x1", "y1", "x2", "y2", "unattended_s"])

    pmem = {}            # person id -> dict(c, h, box, t)
    bags = {}            # bag id -> Bag
    alarmed = set()
    idx = -1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        t = idx / fps
        if a.max_seconds and t > a.max_seconds:
            break

        # ---- detect + track ----
        persons = run_tracker(person_model, frame, a.person_tracker, a.person_conf, a.imgsz, classes=[0])
        luggage = run_tracker(luggage_model, frame, a.luggage_tracker, a.luggage_conf, a.imgsz, agnostic=True)

        # ---- people memory (short grace so a missed detection does not look like "left") ----
        visible_people = set()
        for pid, _, _, b in persons:
            h = float(b[3] - b[1])
            if h < a.min_size:
                continue
            pmem[pid] = dict(c=center(b), h=h, box=b, t=t)
            visible_people.add(pid)
        for pid in [k for k, v in pmem.items() if t - v["t"] > 30.0]:
            del pmem[pid]
        active = {k: v for k, v in pmem.items() if t - v["t"] <= a.grace}

        # ---- per-bag logic ----
        seen_bags = set()
        for bid, c, _, b in luggage:
            if float(b[3] - b[1]) < a.min_size:
                continue
            seen_bags.add(bid)
            bag = bags.get(bid) or bags.setdefault(bid, Bag(bid, luggage_model.names.get(c, str(c))))
            bag.last_seen, bag.box = t, b
            bc = center(b)

            bag.hist.append((t, bc))
            while bag.hist and t - bag.hist[0][0] > a.move_window:
                bag.hist.popleft()
            span = bag.hist[-1][0] - bag.hist[0][0]
            disp = math.hypot(bc[0] - bag.hist[0][1][0], bc[1] - bag.hist[0][1][1])
            bag.moving = span >= 1.0 and disp > a.move_px

            near, dists = set(), {}
            for pid, p in active.items():
                d = math.hypot(p["c"][0] - bc[0], p["c"][1] - bc[1])
                r = a.radius_px if a.radius_px > 0 else a.r_factor * p["h"]
                if d <= r:
                    near.add(pid)
                    dists[pid] = d

            if not bag.initialised:
                bag.owners, bag.initialised = set(near), True
            if bag.moving:
                bag.owners = set(near)               # whoever handles the moving bag owns it
                bag.unatt_since = None
                bag.near_time.clear()
                bag.near = dists
                continue

            for pid in near:
                bag.near_time[pid] += dt
                if bag.near_time[pid] >= a.dwell:
                    bag.owners.add(pid)
            for pid in list(bag.near_time):
                if pid not in near:
                    bag.near_time[pid] = 0.0

            attended = bool(bag.owners & near)
            if attended:
                bag.unatt_since = None
            elif bag.unatt_since is None:
                bag.unatt_since = t
            bag.near = {pid: d for pid, d in dists.items() if pid in bag.owners}

            if bag.unatt_since is not None and t - bag.unatt_since >= a.threshold and bid not in alarmed:
                alarmed.add(bid)
                el = t - bag.unatt_since
                x1, y1, x2, y2 = [int(v) for v in b]
                log_w.writerow([f"{t:.2f}", idx, bid, bag.cname, x1, y1, x2, y2, f"{el:.1f}"])
                log.flush()
                print(f"[ALARM] t={t:.1f}s bag #{bid} ({bag.cname}) unattended {el:.1f}s")
                snap = frame.copy()
                cv2.rectangle(snap, (x1, y1), (x2, y2), RED, 3)
                cv2.imwrite(str(out_dir / "alarms" / f"{stem}_bag{bid}_t{t:.0f}s.jpg"), snap)

        for bid in [k for k, v in bags.items() if t - v.last_seen > a.bag_grace]:
            del bags[bid]

        # ---- draw ----
        for pid in visible_people:
            b = pmem[pid]["box"]
            cv2.rectangle(frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), BLUE, 2)
            cv2.putText(frame, f"person {pid}", (int(b[0]), max(12, int(b[1]) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 1)
        for bid in seen_bags:
            bag = bags[bid]
            b = bag.box
            x1, y1, x2, y2 = [int(v) for v in b]
            if bid in alarmed:
                col, state = RED, "ABANDONED"
            elif bag.moving:
                col, state = GREEN, "moving"
            elif bag.unatt_since is not None:
                col, state = ORANGE, f"unattended {t - bag.unatt_since:.1f}s"
            else:
                col, state = GREEN, "attended"
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 3 if bid in alarmed else 2)
            cv2.putText(frame, f"{bag.cname} {bid} {state}", (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            bc = center(b)
            for pid, d in getattr(bag, "near", {}).items():
                if pid in pmem:
                    pc = pmem[pid]["c"]
                    cv2.line(frame, (int(bc[0]), int(bc[1])), (int(pc[0]), int(pc[1])), (255, 0, 255), 1)
                    cv2.putText(frame, f"{d:.0f}px", (int(pc[0]), int(pc[1])),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
        cv2.putText(frame, f"t={t:.1f}s  alarms={len(alarmed)}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)

        writer.write(frame)
        if a.show:
            cv2.imshow("pipeline", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    log.close()
    if a.show:
        cv2.destroyAllWindows()
    print(f"\nDone. {len(alarmed)} alarm(s): {sorted(alarmed)}")
    print("annotated video ->", out_dir / f"{stem}_annotated.mp4")
    print("alarm log       ->", log_path)


if __name__ == "__main__":
    main()