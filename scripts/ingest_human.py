"""Prepare the raw phone clips: crop to a 256 px square, mirror to match LIBERO's layout, find the task segment.

Writes the processed clips, segments.csv, and one contact sheet per task to check the segments by eye.
"""
import argparse
import csv
import glob
import os
from multiprocessing import Pool

import av
import cv2
import numpy as np

SIZE = 256
CROP_CX = 0.52  # workspace centre, as a fraction of the frame width
ROI_Y0 = 0.25  # ignore the top of the frame, where the demonstrator's body moves
LAG = 3  # compare frames 3 apart, so slow hand motion stands out from noise
ACT_RES = (384, 216)
THR_X = 2  # motion threshold, as a multiple of the noise in the still parts; tuned by eye


def crop_square(frame, cx=CROP_CX):
    h, w = frame.shape[:2]
    x0 = int(np.clip(round(cx * w - h / 2), 0, w - h))
    # mirrored: the home scene is a left-right mirror of LIBERO's
    return cv2.flip(cv2.resize(frame[:, x0:x0 + h], (SIZE, SIZE), interpolation=cv2.INTER_AREA), 1)


def decode(src, dst_mp4):
    """Read the raw clip once: write the 256 px crop and return (fps, motion per frame)."""
    inp = av.open(src)
    vs = inp.streams.video[0]
    vs.thread_type = "AUTO"
    fps = vs.average_rate
    out = av.open(dst_mp4, "w")
    os_ = out.add_stream("libx264", rate=fps)
    os_.width = os_.height = SIZE
    os_.pix_fmt = "yuv420p"
    os_.options = {"crf": "17"}
    grays = []
    for f in inp.decode(vs):
        rgb = f.to_ndarray(format="rgb24")
        out.mux(os_.encode(av.VideoFrame.from_ndarray(crop_square(rgb), format="rgb24")))
        small = cv2.resize(rgb, ACT_RES, interpolation=cv2.INTER_AREA)
        grays.append(cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), (5, 5), 0)[int(ROI_Y0 * ACT_RES[1]):])
    out.mux(os_.encode())
    out.close()
    inp.close()
    g = np.stack(grays).astype(np.float32)
    act = np.zeros(len(g), np.float32)
    # 99th percentile, not mean: the hand is a small part of the frame
    act[LAG:] = np.percentile(np.abs(g[LAG:] - g[:-LAG]).reshape(len(g) - LAG, -1), 99, axis=1)
    return float(fps), act


def runs(mask):
    """(start, end) of each run of True values, end exclusive."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def segment(act, fps, thr, min_s=0.25, edge_s=0.5):
    """Task segment (t0, t1) in frames, from reach to retract, or None.

    Each clip goes: the demonstrator's arm leaving the laptop, still, task, still, the arm going back to the laptop.
    Motion touching either end of the clip is the laptop arm. The task runs from the first to the last remaining
    burst of motion, so pauses during the task don't split it. Bursts shorter than min_s are noise.
    """
    n = len(act)
    cand = [(a, b) for a, b in runs(act > thr) if b - a >= min_s * fps and a > edge_s * fps and b < n - edge_s * fps]
    return (cand[0][0], cand[-1][1]) if cand else None


def selftest():
    fps, a = 24, np.full(400, 1.0)
    a[:20] = 10  # laptop arm at the start
    a[100:150] = a[200:250] = 6  # task with a 2 s pause
    a[380:] = 10  # laptop arm at the end
    a[300:304] = 6  # too short, ignored
    assert segment(a, fps, thr=3) == (100, 250), segment(a, fps, thr=3)
    assert segment(np.ones(100), fps, thr=3) is None
    print("selftest ok")


def load_overrides(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["file"]: r for r in csv.DictReader(f)}


def contact_row(mp4, act, fps, thr, seg, label, w=160):
    frames = [f.to_ndarray(format="bgr24") for f in av.open(mp4).decode(video=0)]
    n = len(frames)
    t0, t1 = seg if seg else (0, n)
    picks = [("start", max(t0 - 1, 0)), ("task-start", t0), ("mid", (t0 + t1) // 2), ("task-end", t1 - 1), ("goal", min(t1, n - 1))]
    tiles = []
    for name, i in picks:
        im = cv2.resize(frames[i], (w, w))
        cv2.putText(im, f"{name} {i / fps:.1f}s", (3, 12), 0, 0.38, (0, 255, 255), 1)
        tiles.append(im)
    plot = np.full((w, 2 * w, 3), 255, np.uint8)
    xs = np.linspace(0, 2 * w - 1, n).astype(int)
    top = max(act.max(), thr * 2)
    if seg:
        plot[:, xs[t0]:xs[t1 - 1] + 1] = (210, 240, 210)
    for x, v in zip(xs, act):
        cv2.line(plot, (x, w - 1), (x, w - 1 - int(v / top * (w - 20))), (0, 0, 200), 1)
    ty = w - 1 - int(thr / top * (w - 20))
    cv2.line(plot, (0, ty), (2 * w, ty), (200, 0, 0), 1)
    cv2.putText(plot, label, (3, 12), 0, 0.38, (0, 0, 0), 1)
    return np.hstack(tiles + [plot])


def _decode_job(args):
    src, dst = args
    fps, act = decode(src, dst)
    np.save(dst[:-4] + ".npy", np.r_[fps, act])
    return src


def main(root, workers):
    clips = sorted(c for c in glob.glob(f"{root}/raw/*/*") if c.lower().endswith((".mov", ".mp4")) and "/static/" not in c)
    jobs = []
    for c in clips:
        dst = f"{root}/processed/{c.split('/')[-2]}/{os.path.splitext(os.path.basename(c))[0]}.mp4"
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst[:-4] + ".npy"):
            jobs.append((c, dst))
    with Pool(workers) as p:
        for done in p.imap_unordered(_decode_job, jobs):
            print("decoded", done)

    acts = {}
    for c in clips:
        a = np.load(f"{root}/processed/{c.split('/')[-2]}/{os.path.splitext(os.path.basename(c))[0]}.npy")
        acts[c] = float(a[0]), a[1:]
    # Noise level: the 20th percentile of each clip falls in a still part.
    # Clips this threshold gets wrong are fixed by hand in metadata.csv.
    thr = THR_X * float(np.median([np.percentile(a[LAG:], 20) for _, a in acts.values()]))
    overrides = load_overrides(f"{root}/metadata.csv")
    rows, sheets = [], {}
    for c in clips:
        fps, act = acts[c]
        rel = os.path.relpath(c, f"{root}/raw")
        task, clip = rel.split("/")[0], os.path.splitext(os.path.basename(c))[0]
        o = overrides.get(rel, {})
        seg, source = segment(act, fps, thr), "auto"
        if o.get("t_start_s") or o.get("t_end_s"):  # start and end can be overridden separately
            t0 = round(float(o["t_start_s"]) * fps) if o.get("t_start_s") else seg[0]
            t1 = round(float(o["t_end_s"]) * fps) if o.get("t_end_s") else seg[1]
            seg, source = (t0, t1), "override"
        ok = int(o.get("ok") or 1) if seg else 0
        rows.append(dict(file=rel, task=task, take=int(clip.rsplit("_", 1)[1]), ok=ok, source=source if seg else "failed",
                         fps=fps, n_frames=len(act), f_start=seg[0] if seg else "", f_end=seg[1] if seg else "",
                         t_start_s=f"{seg[0] / fps:.2f}" if seg else "", t_end_s=f"{seg[1] / fps:.2f}" if seg else "",
                         notes=o.get("notes", "")))
        mp4 = f"{root}/processed/{task}/{clip}.mp4"
        label = f"{clip[-2:]} {source}" + ("" if ok else " EXCLUDED")
        sheets.setdefault(task, []).append(contact_row(mp4, act, fps, thr, seg, label))
    with open(f"{root}/segments.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    for task, rs in sheets.items():
        cv2.imwrite(f"{root}/contact_{task}.jpg", np.vstack(rs), [cv2.IMWRITE_JPEG_QUALITY, 85])
    n_ok = sum(r["ok"] for r in rows)
    print(f"threshold {thr:.2f} | {n_ok}/{len(rows)} clips usable | failed: {[r['file'] for r in rows if r['source'] == 'failed']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/human")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    selftest() if a.selftest else main(a.root, a.workers)
