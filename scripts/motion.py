"""Gripper and hand tracks for the diagnostics: where is the robot gripper or human hand in the image, and how open
is it? One value per V-JEPA feature step.

  uv  (T,2)  position in the image, (x, y) in [0,1]: the gripper projected into the camera, or the human palm centre
  ap  (T,)   how open: robot finger gap, or the human thumb-index distance divided by palm size
  ok  (T,)   found (always for the robot; for the human, the hand was detected or a gap of up to 5 steps was filled in)
"""
import argparse
import csv
import multiprocessing as mp
import os

import av
import cv2
import h5py
import numpy as np
import torch

import features
from features import HZ, render_demo, speed_ratio
from splits import SUITE, libero90_tasks

OUT = "data/motion"
HAND_MODEL = "data/models/hand_landmarker.task"
PALM = [0, 5, 9, 13, 17]  # wrist and finger bases: they stay put when grasping, unlike the fingertips


def robot_job(job):
    """Render one demo, project the gripper into the image, and read the finger gap."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera

    name, key, frames = render_demo(job)
    sim = features._envs[job[1]].sim
    M = get_camera_transform_matrix(sim, "agentview", 256, 256)
    with h5py.File(f"data/demos/{job[0]}/{name}_demo.hdf5") as f:
        rs = f["data"][key]["robot_states"][:]  # fingers (2), gripper position (3), orientation (4)
    rc = project_points_from_world_to_camera(rs[:, 2:5], M, 256, 256)  # (row, col), already upright
    uv = np.stack([rc[:, 1], rc[:, 0]], 1).astype(np.float32) / 255
    return name, key, frames, uv, (rs[:, 0] - rs[:, 1]).astype(np.float32)


def save(path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **kw)


def attach(eps, k, dev):
    """For each LAM window of each episode, set e["kin"] (n,4): image motion (2), change in grip opening, grip
    opening at t+k; and e["kok"] (n,): position known at both ends.
    Motion is scaled per embodiment, since a hand and the robot move by different amounts in the image. Grip is
    scaled to [0,1]. The scales come from the non-TARGET robot demos and all human clips."""
    from lam import windows
    from splits import TARGET

    goal = lambda e: "libero_90" not in e.get("h5", "")
    raw = {}
    for e in eps:
        m = np.load(f"{OUT}/human/{e['clip']}.npz" if e["dom"] == "human" else
                    f"{OUT}/{'robot' if goal(e) else 'robot90'}/{e['task']}/{e['key']}.npz")
        assert len(m["uv"]) == len(e["x"]), (e.get("clip", e.get("key")), len(m["uv"]), len(e["x"]))
        ts = np.array(windows(e, k))
        raw[id(e)] = m["uv"][ts + k] - m["uv"][ts], m["ap"][ts], m["ap"][ts + k], m["ok"][ts] & m["ok"][ts + k]
    for dom in ("robot", "human"):
        d = [raw[id(e)] for e in eps if e["dom"] == dom and goal(e) and not (dom == "robot" and e["task"] in TARGET)]
        if not d:
            continue
        ok = np.concatenate([r[3] for r in d])
        us = np.concatenate([r[0] for r in d])[ok].std()
        lo, hi = np.nanpercentile(np.concatenate([r[2] for r in d])[ok], [5, 95])
        for e in eps:
            if e["dom"] == dom:
                du, a0, a1, kok = raw[id(e)]
                a0, a1 = (a0 - lo) / (hi - lo), (a1 - lo) / (hi - lo)
                e["kin"] = torch.tensor(np.nan_to_num(np.c_[du / us, a1 - a0, a1]), dtype=torch.float32, device=dev)
                e["kok"] = torch.tensor(kok, device=dev)


def robot(limit, workers, suite=SUITE, shard="0/1"):
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    out = f"{OUT}/robot" if suite == SUITE else f"{OUT}/robot90"
    names = libero_task_map[suite] if suite == SUITE else libero90_tasks()  # LIBERO-90: the filtered tasks, as in features.py
    i, n = map(int, shard.split("/"))
    jobs = []
    for name in names[i::n]:
        with h5py.File(f"data/demos/{suite}/{name}_demo.hdf5") as f:
            keys = sorted(f["data"], key=lambda s: int(s.split("_")[1]))[:limit]
        jobs += [(suite, libero_task_map[suite].index(name), name, k) for k in keys if not os.path.exists(f"{out}/{name}/{k}.npz")]
    print(f"robot: {len(jobs)} demos", flush=True)
    with mp.get_context("spawn").Pool(workers, initializer=torch.set_num_threads, initargs=(1,)) as p:
        for name, key, frames, uv, ap in p.imap_unordered(robot_job, jobs):
            save(f"{out}/{name}/{key}.npz", uv=uv, ap=ap, ok=np.ones(len(uv), bool))
            if key == "demo_0":
                check_image(f"{out}/check_{name}.jpg", frames, uv)
            print("robot", name, key, len(frames), flush=True)


def hands(frames):
    """Run MediaPipe on every frame: palm centre (T,2), grip opening (T,), found (T,).
    Uses the hand that moves most; the other one usually rests on the table."""
    import mediapipe as mpipe
    from mediapipe.tasks.python import BaseOptions, vision

    det = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL), running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3))
    per = {}  # landmarks (T,21,2) per hand label, NaN where missing
    for t, f in enumerate(frames):
        r = det.detect(mpipe.Image(image_format=mpipe.ImageFormat.SRGB, data=np.ascontiguousarray(f)))
        for lm, hd in zip(r.hand_landmarks, r.handedness):
            arr = per.setdefault(hd[0].category_name, np.full((len(frames), 21, 2), np.nan, np.float32))
            arr[t] = [(p.x, p.y) for p in lm]
    det.close()
    if not per:
        return np.full((len(frames), 2), np.nan, np.float32), np.full(len(frames), np.nan, np.float32), np.zeros(len(frames), bool)
    moved = lambda a: np.nansum(np.linalg.norm(np.diff(np.nanmean(a[:, PALM], 1), axis=0), axis=1))
    lm = max(per.values(), key=moved)
    uv = np.nanmean(lm[:, PALM], 1)
    ap = np.linalg.norm(lm[:, 4] - lm[:, 8], axis=1) / np.linalg.norm(lm[:, 0] - lm[:, 9], axis=1)
    return uv, ap, ~np.isnan(uv[:, 0])


def fill(v, hit, gap=5):
    """Fill gaps of up to `gap` steps by linear interpolation; return the values and the new found-mask."""
    v, ok, idx = v.copy(), hit.copy(), np.flatnonzero(hit)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= gap + 1:
            w = (np.arange(a + 1, b) - a) / (b - a)
            v[a + 1:b] = v[a] * (1 - w[:, None] if v.ndim > 1 else 1 - w) + v[b] * (w[:, None] if v.ndim > 1 else w)
            ok[a + 1:b] = True
    return v, ok


def human():
    r = speed_ratio()
    for s in csv.DictReader(open("data/human/segments.csv")):
        fps, clip = float(s["fps"]), os.path.splitext(s["file"])[0]
        if os.path.exists(f"{OUT}/human/{clip}.npz"):
            continue
        frames = np.stack([f.to_ndarray(format="rgb24") for f in av.open(f"data/human/processed/{clip}.mp4").decode(video=0)])
        src = np.round(np.arange(0, len(frames) / fps, r / HZ) * fps).astype(int).clip(max=len(frames) - 1)  # same resampling as features.py
        uv, ap, hit = hands(frames[src])
        uv, ok = fill(uv, hit)
        ap, _ = fill(ap, hit)
        save(f"{OUT}/human/{clip}.npz", uv=uv, ap=ap, ok=ok)
        check_image(f"{OUT}/check_human_{os.path.basename(clip)}.jpg", frames[src], uv)
        print("human", clip, len(src), f"hand found {hit.mean():.2f}, after filling {ok.mean():.2f}", flush=True)


def check_image(path, frames, uv):
    """Six frames with the tracked point drawn, to check by eye that it sits on the gripper or hand."""
    tiles = []
    for t in np.linspace(0, len(frames) - 1, 6).astype(int):
        im = frames[t].copy()
        if not np.isnan(uv[t, 0]):
            cv2.circle(im, (int(uv[t, 0] * 255), int(uv[t, 1] * 255)), 5, (255, 0, 0), 2)
        tiles.append(im)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))


def selftest():
    v = np.array([0, np.nan, np.nan, 3, np.nan] + [np.nan] * 6 + [9], np.float32)
    f, ok = fill(v, ~np.isnan(v))
    assert np.allclose(f[:4], [0, 1, 2, 3]) and ok[:4].all() and not ok[4:11].any(), (f, ok)
    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["robot", "human"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--suite", default=SUITE)
    p.add_argument("--shard", default="0/1")
    a = p.parse_args()
    selftest()
    if not a.selftest:
        if a.only != "human":
            robot(a.limit, a.workers, a.suite, a.shard)
        if a.only != "robot":
            human()
