"""TCC phase embedding: V-JEPA features -> 128 numbers encoding progress through the task, trained with
cycle-consistency (cycle-back regression, as in XIRL). Scored on held-out clips (robot demos 40-49, human takes 11-12).
"""
import argparse
import csv
import glob
import json
import os
import random
import subprocess

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import kendalltau

from splits import ALIGN, TARGET, check_robot, libero90_tasks, robot_tasks

FEAT = "data/features"
TEMP = 0.1


class G(nn.Module):
    """V-JEPA grid (...,16,1024) -> 128 numbers. Each grid cell is projected separately, so the layout is kept."""

    def __init__(self, d=128, drop=0.1):
        super().__init__()
        self.tok = nn.Sequential(nn.Dropout(drop), nn.LayerNorm(1024), nn.Linear(1024, 128), nn.GELU())  # dropout on the features stands in for image augmentation
        self.head = nn.Sequential(nn.Linear(16 * 128, 512), nn.GELU(), nn.Linear(512, d))

    def forward(self, x):
        return self.head(self.tok(x.float()).flatten(-2))


def load(device, diagnostic_eval=True, libero90=False):
    """Load all episodes as dicts: task, dom (robot/human), x (T,16,1024) features, t0/t1 (task segment),
    val (held out), plus key/h5 (robot) or clip (human). libero90 adds the filtered LIBERO-90 demos, kept on CPU."""
    eps = []
    # TARGET robot demos are only used for evaluation, never training
    for t in robot_tasks(diagnostic=diagnostic_eval):
        with h5py.File(f"data/demos/libero_goal/{t}_demo.hdf5") as f:
            for k in f["data"]:
                x = torch.from_numpy(np.load(f"{FEAT}/robot/{t}/{k}.npy")).flatten(1, 2).to(device)
                i = int(k.split("_")[1])
                eps.append(dict(task=t, dom="robot", key=k, h5=f.filename, x=x, t0=0, t1=len(x), val=i >= 40 or t in TARGET))  # demos have no still parts to trim
    if libero90:
        for t in libero90_tasks():
            for p in sorted(glob.glob(f"{FEAT}/robot90/{t}/*.npy")):
                k = os.path.basename(p)[:-4]
                x = torch.from_numpy(np.load(p)).flatten(1, 2)
                eps.append(dict(task=t, dom="robot", key=k, h5=f"data/demos/libero_90/{t}_demo.hdf5", x=x, t0=0,
                                t1=len(x), val=int(k.split("_")[1]) >= 40))
    for r in csv.DictReader(open(f"{FEAT}/human_index.csv")):
        if r["ok"] != "1":
            continue
        x = torch.from_numpy(np.load(f"{FEAT}/human/{os.path.splitext(r['file'])[0]}.npy")).flatten(1, 2).to(device)
        take = int(os.path.splitext(r["file"])[0].rsplit("_", 1)[1])
        eps.append(dict(task=r["task"], dom="human", clip=os.path.splitext(r["file"])[0], x=x, t0=int(r["k_start"]),
                        t1=int(r["k_end"]), val=take > 10))
    return eps


def pair_pool(eps, pairs):
    """Training episodes to draw pairs from, per pair type and task."""
    tr = [e for e in eps if not e["val"]]
    pools = {}
    for t in {e["task"] for e in tr}:
        r = [e for e in tr if e["task"] == t and e["dom"] == "robot"]
        h = [e for e in tr if e["task"] == t and e["dom"] == "human"]
        if r:
            pools[("rr", t)] = (r, r)
        if pairs == "all" and h:
            pools[("hh", t)] = (h, h)
        if pairs == "all" and h and r and t in ALIGN:
            pools[("hr", t)] = (h, r)
    check_robot([t for (_, t), (a, b) in pools.items() if any(e["dom"] == "robot" for e in a + b)])
    return pools


def sample(e, n, trim):
    t0, t1 = (e["t0"], e["t1"]) if trim else (0, len(e["x"]))
    pos = (torch.arange(n) + torch.rand(n)) / n  # evenly spread, with random jitter, over the trimmed clip
    idx = (t0 + pos * (t1 - t0)).long().clamp(max=t1 - 1)
    return e["x"][idx], (idx - t0).float() / max(t1 - t0 - 1, 1)


def cycle_loss(u, v, tu):
    """Cycle-back regression: each frame of u finds its soft match in v, which finds its soft match back in u.
    The loss is how far that lands from the starting time."""
    d = u.shape[-1]
    alpha = torch.softmax(-torch.cdist(u, v) ** 2 / d / TEMP, -1)
    beta = torch.softmax(-torch.cdist(alpha @ v, u) ** 2 / d / TEMP, -1)
    mu = (beta @ tu.unsqueeze(-1)).squeeze(-1)
    return F.mse_loss(mu, tu)


@torch.no_grad()
def embed(g, e, trim=True):
    t0, t1 = (e["t0"], e["t1"]) if trim else (0, len(e["x"]))
    return g(e["x"][t0:t1]) if g else e["x"][t0:t1].float().mean(1)  # g=None: raw V-JEPA features, averaged over the grid


def nn_idx(a, b):
    return torch.cdist(a, b).argmin(1).cpu().numpy()


@torch.no_grad()
def evaluate(g, eps, trim=True):
    """Kendall's tau between held-out clips of the same task, and the human-to-robot phase error per task."""
    val = [e for e in eps if e["val"]]
    emb = {id(e): embed(g, e, trim) for e in val}
    out = {}
    for t in sorted({e["task"] for e in val}):
        for dom in ("robot", "human"):
            es = [e for e in val if e["task"] == t and e["dom"] == dom][:10]
            taus = [kendalltau(np.arange(len(emb[id(a)])), nn_idx(emb[id(a)], emb[id(b)]))[0] for a in es for b in es if a is not b]
            if taus:
                out[f"tau/{dom}/{t}"] = float(np.nanmean(taus))
        hs = [e for e in val if e["task"] == t and e["dom"] == "human"]
        rs = [e for e in val if e["task"] == t and e["dom"] == "robot"][:10]
        errs = []
        for h in hs:
            for r in rs:
                j = nn_idx(emb[id(h)], emb[id(r)])
                errs.append(np.abs(np.linspace(0, 1, len(j)) - j / max(len(emb[id(r)]) - 1, 1)).mean())
        if errs:  # TARGET: robot demos used for this check only
            out[f"retrieval_err/{t}" + (" (diagnostic)" if t in TARGET else "")] = float(np.mean(errs))
    return out


def main(a):
    torch.manual_seed(a.seed), random.seed(a.seed), np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = load(dev)
    pools = pair_pool(eps, a.pairs)
    print({k: (len(x), len(y)) for k, (x, y) in pools.items()})
    g = G().to(dev)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-4, weight_decay=1e-2)
    keys = list(pools)
    for step in range(1, a.steps + 1):
        loss = 0
        for _ in range(a.batch):
            pa, pb = pools[random.choice(keys)]
            ea = random.choice(pa)
            eb = random.choice([e for e in pb if e is not ea])
            (xa, ta), (xb, tb) = sample(ea, a.frames, not a.no_trim), sample(eb, a.frames, not a.no_trim)
            ua, ub = g(xa), g(xb)
            loss = loss + cycle_loss(ua, ub, ta.to(dev)) + cycle_loss(ub, ua, tb.to(dev))
        opt.zero_grad()
        (loss / a.batch).backward()
        opt.step()
        if step % 500 == 0:
            print(f"step {step} loss {loss.item() / a.batch:.4f}", flush=True)
    g.eval()
    m = {"trained": evaluate(g, eps), "raw_vjepa": evaluate(None, eps)}  # always scored on trimmed clips, so runs compare
    for k in sorted(m["trained"]):
        print(f"{k:70s} trained {m['trained'][k]:.3f}  raw {m['raw_vjepa'].get(k, float('nan')):.3f}")
    out = f"runs/tcc_{a.name}"
    os.makedirs(out, exist_ok=True)
    torch.save(g.state_dict(), f"{out}/model.pt")
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    json.dump({"config": vars(a), "commit": commit, "metrics": m}, open(f"{out}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--pairs", choices=["rr", "all"], default="all")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--no-trim", action="store_true")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=8, help="video pairs per step")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
