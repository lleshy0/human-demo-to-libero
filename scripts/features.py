"""Cache frozen V-JEPA 2 features for the robot demos and human clips.

For each step t, the 16 frames ending at t go through V-JEPA 2. The output for the newest frames is pooled to a
4x4 grid of 1024 numbers per cell and saved as fp16.

Robot: frames re-rendered at 256 px from the recorded sim states, 20 Hz.
Human: the processed 256 px clips, resampled so a human task takes about as many steps as a robot one.
"""
import argparse
import csv
import glob
import os
import time
import multiprocessing as mp

import av
import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F

from splits import ALIGN, SUITE, libero90_tasks

MODEL = "facebook/vjepa2-vitl-fpc64-256"
WIN = 16  # frames per window (0.8 s at 20 Hz)
HZ = 20  # LIBERO control rate
DEMOS = "data/demos/libero_goal"
OUT = "data/features"
# normalise only: the HF processor would resize and crop off the edges of our 256 px frames
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


@torch.inference_mode()
def encode(model, frames, bs=16, ts=None):
    """uint8 frames (T,256,256,3) -> features (T,4,4,1024), one for the window ending at each step.
    ts: only encode the windows ending at these steps (rollouts only need the newest)."""
    x = torch.from_numpy(frames).cuda().permute(0, 3, 1, 2)
    idx = (torch.arange(len(x))[:, None] + torch.arange(1 - WIN, 1)[None]).clamp(min=0)
    idx = idx if ts is None else idx[ts]
    out = []
    for i in range(0, len(idx), bs):
        v = (x[idx[i:i + bs]].float() / 255 - MEAN.cuda()) / STD.cuda()
        h = model(pixel_values_videos=v.to(torch.bfloat16), skip_predictor=True).last_hidden_state
        h = h.view(len(v), WIN // 2, 16, 16, -1)[:, -1].permute(0, 3, 1, 2)  # output for the newest frames
        out.append(F.avg_pool2d(h.float(), 4).permute(0, 2, 3, 1).half().cpu())
    return torch.cat(out).numpy()


_envs = {}


def render_demo(job):
    """Re-render one demo at 256 px by replaying its recorded sim states."""
    suite, task_id, task_name, key = job
    if task_id not in _envs:
        for env in _envs.values():  # one renderer per worker; more ran the node out of memory
            env.close()
        _envs.clear()
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
        _envs[task_id] = OffScreenRenderEnv(
            bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
            camera_heights=256, camera_widths=256, camera_names=["agentview"])
        _envs[task_id].reset()
    env = _envs[task_id]
    with h5py.File(f"data/demos/{suite}/{task_name}_demo.hdf5") as f:
        states = f["data"][key]["states"][:]
    frames = np.stack([env.regenerate_obs_from_state(s)["agentview_image"][::-1] for s in states])  # renders upside down
    return task_name, key, np.ascontiguousarray(frames)


def robot_dir(suite):
    return f"{OUT}/robot" if suite == SUITE else f"{OUT}/robot90"


def robot_jobs(limit, suite, shard):
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    names = libero_task_map[suite] if suite == SUITE else libero90_tasks()  # LIBERO-90: only the filtered tasks
    i, n = map(int, shard.split("/"))
    jobs = []
    for name in names[i::n]:
        with h5py.File(f"data/demos/{suite}/{name}_demo.hdf5") as f:
            keys = sorted(f["data"], key=lambda s: int(s.split("_")[1]))[:limit]
        tid = libero_task_map[suite].index(name)
        jobs += [(suite, tid, name, k) for k in keys if not os.path.exists(f"{robot_dir(suite)}/{name}/{k}.npy")]
    return jobs


def speed_ratio():
    """How much slower the human demos are than the robot's: median human task time / median demo time, averaged over ALIGN tasks."""
    seg = [r for r in csv.DictReader(open("data/human/segments.csv")) if r["ok"] == "1"]
    ratios = {}
    for t in ALIGN:
        h = np.median([(int(r["f_end"]) - int(r["f_start"])) / float(r["fps"]) for r in seg if r["task"] == t])
        with h5py.File(f"{DEMOS}/{t}_demo.hdf5") as f:
            rb = np.median([f["data"][k]["actions"].shape[0] / HZ for k in f["data"]])
        ratios[t] = h / rb
        print(f"speed: {t} human {h:.2f}s / robot {rb:.2f}s = {ratios[t]:.2f}")
    return float(np.mean(list(ratios.values())))


def human(model):
    r = speed_ratio()
    print(f"human/robot speed ratio {r:.2f}: one robot step = {r / HZ * 1000:.0f} ms of human video")
    rows = []
    for s in csv.DictReader(open("data/human/segments.csv")):
        fps, clip = float(s["fps"]), os.path.splitext(s["file"])[0]
        frames = np.stack([f.to_ndarray(format="rgb24") for f in av.open(f"data/human/processed/{clip}.mp4").decode(video=0)])
        src = np.round(np.arange(0, len(frames) / fps, r / HZ) * fps).astype(int).clip(max=len(frames) - 1)
        dst = f"{OUT}/human/{clip}.npy"
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.save(dst, encode(model, frames[src]))
        # task segment, in resampled steps
        k0, k1 = int(np.searchsorted(src, int(s["f_start"]))), int(np.searchsorted(src, int(s["f_end"])))
        rows.append(dict(file=s["file"], task=s["task"], ok=s["ok"], n_steps=len(src), k_start=k0, k_end=k1, ratio=f"{r:.4f}"))
        print("human", clip, len(src), "steps")
    with open(f"{OUT}/human_index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def robot(model, limit, workers, suite=SUITE, shard="0/1"):
    jobs = robot_jobs(limit, suite, shard)
    out = robot_dir(suite)
    print(f"robot: {len(jobs)} demos to encode")
    first, t0, n = {}, time.time(), 0
    with mp.get_context("spawn").Pool(workers) as p:  # spawn, not fork: CUDA is already initialised
        for name, key, frames in p.imap_unordered(render_demo, jobs):
            os.makedirs(f"{out}/{name}", exist_ok=True)
            np.save(f"{out}/{name}/{key}.npy", encode(model, frames))
            first.setdefault(name, frames[0])
            n += len(frames)
            print(f"robot {name}/{key} {len(frames)} steps | {n / (time.time() - t0):.0f} steps/s", flush=True)
    if first:  # one start frame per task, to check it is oriented like the human view
        h = next(iter(av.open(sorted(glob.glob("data/human/processed/*/*.mp4"))[0]).decode(video=0)))
        tiles = [cv2.resize(f, (192, 192)) for f in list(first.values())[:9]] + [cv2.resize(h.to_ndarray(format="rgb24"), (192, 192))]
        cv2.imwrite(f"{out}/orientation_{shard.replace('/', 'of')}.jpg" if suite != SUITE else f"{OUT}/orientation.jpg", cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["robot", "human"])
    p.add_argument("--limit", type=int, default=None, help="demos per task (smoke test)")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--suite", default=SUITE, help="libero_90: extra robot data, filtered by splits.libero90_tasks()")
    p.add_argument("--shard", default="0/1", help="i/n: this job's share of the task list")
    a = p.parse_args()
    from transformers import VJEPA2Model

    m = VJEPA2Model.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    if a.only != "human":
        robot(m, a.limit, a.workers, a.suite, a.shard)
    if a.only != "robot":
        human(m)
