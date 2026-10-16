"""Latent action model. An encoder turns frames t and t+k into a small code, and a forward model must predict
frame t+k from frame t and the code. A contrastive loss gives a human transition and the robot transition at the
same task phase (matched by TCC, ALIGN tasks only) the same code.
"""
import argparse
import json
import os
import random
import subprocess

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from splits import ALIGN, TARGET
from tcc import G, load

NOISE = 0.1  # noise on the code during training, so it can't carry the whole next frame


class LAM(nn.Module):
    def __init__(self, d=16):
        super().__init__()
        self.enc_tok = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 128), nn.GELU())
        self.enc = nn.Sequential(nn.Linear(2 * 16 * 128, 512), nn.GELU(), nn.Linear(512, d))
        self.fwd_tok = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 256), nn.GELU())
        self.pos = nn.Parameter(torch.zeros(16, 256))
        self.fwd = nn.Sequential(nn.Linear(256 + 256 + d, 512), nn.GELU(), nn.Linear(512, 1024))

    def encode(self, f0, f1):
        return torch.tanh(self.enc(torch.cat([self.enc_tok(f0.float()), self.enc_tok(f1.float())], -2).flatten(-2)))

    def predict(self, f0, a):
        """Each grid cell predicts its own change from itself, the scene average and the code."""
        h = self.fwd_tok(f0.float()) + self.pos
        ctx = h.mean(-2, keepdim=True).expand_as(h)
        return f0.float() + self.fwd(torch.cat([h, ctx, a.unsqueeze(-2).expand(*h.shape[:-1], a.shape[-1])], -1))


def windows(e, k):
    """Start steps t of all windows that fit in the clip."""
    return range(e["t0"], e["t1"] - k)


def batch(eps, k, n):
    es = random.choices(eps, k=n)
    ts = [random.choice(windows(e, k)) for e in es]
    return torch.stack([e["x"][t] for e, t in zip(es, ts)]), torch.stack([e["x"][t + k] for e, t in zip(es, ts)])


def nce_batch(hum, rob, z, k, n):
    """Pairs of a human ALIGN transition and the robot transition at the same phase in a random demo of the task."""
    f0h, f1h, f0r, f1r = [], [], [], []
    for _ in range(n):
        h = random.choice(hum)
        r = random.choice([e for e in rob if e["task"] == h["task"]])
        th = random.choice(windows(h, k))
        tr = int(torch.cdist(z[id(h)][th:th + 1], z[id(r)][: len(r["x"]) - k]).argmin())
        f0h.append(h["x"][th]), f1h.append(h["x"][th + k]), f0r.append(r["x"][tr]), f1r.append(r["x"][tr + k])
    return [torch.stack(v) for v in (f0h, f1h, f0r, f1r)]


def info_nce(ah, ar, temp=0.1):
    logits = F.normalize(ah, dim=-1) @ F.normalize(ar, dim=-1).T / temp
    y = torch.arange(len(ah), device=ah.device)
    return (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y)) / 2


def robot_chunks(e, k):
    """Robot action over each window: summed position and rotation change (6) and the final gripper command (1)."""
    with h5py.File(e["h5"]) as f:
        a = f["data"][e["key"]]["actions"][:]
    return np.stack([np.r_[a[t:t + k, :6].sum(0), a[t + k - 1, 6]] for t in windows(e, k)])


def fit_probe(xtr, ytr, xte, yte, classify=False, steps=500):
    """Train a small probe on the full batch; return balanced accuracy (classify) or R²."""
    m = nn.Sequential(nn.Linear(xtr.shape[1], 64), nn.GELU(), nn.Linear(64, 2 if classify else ytr.shape[1])).to(xtr.device)
    opt = torch.optim.Adam(m.parameters(), 1e-2)
    mu, sd = (0, 1) if classify else (ytr.mean(0), ytr.std(0) + 1e-6)
    for _ in range(steps):
        loss = F.cross_entropy(m(xtr), ytr) if classify else F.mse_loss(m(xtr), (ytr - mu) / sd)
        opt.zero_grad(), loss.backward(), opt.step()
    with torch.no_grad():
        if classify:
            p = m(xte).argmax(1)
            return float(np.mean([(p[yte == c] == c).float().mean().item() for c in (0, 1)]))
        r2 = 1 - ((m(xte) * sd + mu - yte) ** 2).sum(0) / ((yte - yte.mean(0)) ** 2).sum(0)
        return float(r2.mean())


@torch.no_grad()
def codes(lam, e, k):
    ts = list(windows(e, k))
    x = e["x"].to(next(lam.parameters()).device)  # LIBERO-90 episodes are kept on CPU
    return lam.encode(x[ts], x[[t + k for t in ts]])


@torch.no_grad()
def evaluate(lam, eps, k):
    tr = [e for e in eps if not e["val"]]
    va = [e for e in eps if e["val"] and not (e["dom"] == "robot" and e["task"] in TARGET)]
    m = {}
    l1, l1_copy = [], []
    for e in va:
        ts = list(windows(e, k))
        f0, f1 = e["x"][ts], e["x"][[t + k for t in ts]]
        l1.append((lam.predict(f0, lam.encode(f0, f1)) - f1.float()).abs().mean().item())
        l1_copy.append((f0.float() - f1.float()).abs().mean().item())
    m["fwd_l1/val"], m["fwd_l1/copy_baseline"] = float(np.mean(l1)), float(np.mean(l1_copy))
    rob_tr, rob_va = [e for e in tr if e["dom"] == "robot"], [e for e in va if e["dom"] == "robot"]
    dev = eps[0]["x"].device
    xa = lambda es: torch.cat([codes(lam, e, k) for e in es])
    ya = lambda es: torch.tensor(np.concatenate([robot_chunks(e, k) for e in es]), dtype=torch.float32, device=dev)
    with torch.enable_grad():
        m["action_probe_r2/val"] = fit_probe(xa(rob_tr), ya(rob_tr), xa(rob_va), ya(rob_va))
        hum_tr, hum_va = [e for e in tr if e["dom"] == "human"], [e for e in va if e["dom"] == "human"]
        lab = lambda es: torch.tensor(sum([[int(e["dom"] == "human")] * len(windows(e, k)) for e in es], []), device=dev)
        m["domain_probe_acc/val"] = fit_probe(xa(rob_tr + hum_tr), lab(rob_tr + hum_tr), xa(rob_va + hum_va), lab(rob_va + hum_va), classify=True)
    # distance from each held-out code to the closest robot training code
    bank = xa(rob_tr)
    nn_d = lambda q: torch.cdist(q, bank).min(1).values.median().item()
    m["nn_dist_to_robot_codes/robot_val"] = nn_d(xa(rob_va))
    for t in sorted({e["task"] for e in va if e["dom"] == "human"}):
        m[f"nn_dist_to_robot_codes/human_val/{t}"] = nn_d(xa([e for e in va if e["dom"] == "human" and e["task"] == t]))
    return m


def main(a):
    torch.manual_seed(a.seed), random.seed(a.seed), np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = load(dev)
    # no TARGET robot data in training
    train = [e for e in eps if not e["val"] and not (e["dom"] == "robot" and e["task"] in TARGET)]
    hum = [e for e in train if e["dom"] == "human" and e["task"] in ALIGN]
    rob = [e for e in train if e["dom"] == "robot" and e["task"] in ALIGN]
    g = G().to(dev)
    g.load_state_dict(torch.load(f"runs/tcc_{a.tcc}/model.pt", map_location=dev))
    g.eval()
    with torch.no_grad():
        z = {id(e): g(e["x"]) for e in hum + rob}
    lam = LAM(a.d).to(dev)
    opt = torch.optim.AdamW(lam.parameters(), lr=3e-4, weight_decay=1e-2)
    for step in range(1, a.steps + 1):
        f0, f1 = batch(train, a.k, 256)
        act = lam.encode(f0, f1)
        rec = (lam.predict(f0, act + NOISE * torch.randn_like(act)) - f1.float()).abs().mean()
        nce = torch.zeros((), device=dev)
        if a.nce > 0:
            f0h, f1h, f0r, f1r = nce_batch(hum, rob, z, a.k, 64)
            nce = info_nce(lam.encode(f0h, f1h), lam.encode(f0r, f1r))
        loss = rec + a.nce * nce
        opt.zero_grad(), loss.backward(), opt.step()
        if step % 1000 == 0:
            print(f"step {step} rec {rec.item():.4f} nce {nce.item():.3f}", flush=True)
    lam.eval()
    m = evaluate(lam, eps, a.k)
    for k_, v in m.items():
        print(f"{k_:70s} {v:.4f}")
    out = f"runs/lam_{a.name}"
    os.makedirs(out, exist_ok=True)
    torch.save({"state": lam.state_dict(), "d": a.d, "k": a.k}, f"{out}/model.pt")
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    json.dump({"config": vars(a), "commit": commit, "metrics": m}, open(f"{out}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--tcc", default="main", help="runs/tcc_<name> used to pair human/robot phases")
    p.add_argument("--nce", type=float, default=0.1, help="InfoNCE weight λ (0 = no cross-embodiment loss)")
    p.add_argument("--d", type=int, default=16)
    p.add_argument("--k", type=int, default=10, help="steps between f_t and f_t+k; = decoder chunk length")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
