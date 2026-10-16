"""Decoder: latent action + robot state -> the next k robot actions. Trained on robot demos of the non-TARGET tasks.
Also saves every training code with its actions, for the nearest-neighbour baseline.
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

from lam import LAM, codes, windows
from splits import check_robot
from tcc import load


class Decoder(nn.Module):
    def __init__(self, d, h, prop_mu, prop_sd):
        super().__init__()
        self.h = h
        self.register_buffer("mu", prop_mu)
        self.register_buffer("sd", prop_sd)
        self.net = nn.Sequential(nn.Linear(d + 9, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, h * 7))

    def forward(self, a, prop):
        """prop: finger positions (2), gripper position (3), gripper orientation (4)."""
        return self.net(torch.cat([a, (prop - self.mu) / self.sd], -1)).view(-1, self.h, 7)


def load_lam(name, dev):
    ck = torch.load(f"runs/lam_{name}/model.pt", map_location=dev)
    lam = LAM(ck["d"]).to(dev)
    lam.load_state_dict(ck["state"])
    return lam.eval(), ck["k"]


def robot_set(eps, lam, k, val):
    """Codes, robot state at t and the next k actions, for every window of the robot demos."""
    es = [e for e in eps if e["dom"] == "robot" and e["val"] == val]
    check_robot({e["task"] for e in es})
    A, P, Y = [], [], []
    for e in es:
        with h5py.File(e["h5"]) as f:
            act, prop = f["data"][e["key"]]["actions"][:], f["data"][e["key"]]["robot_states"][:]
        ts = list(windows(e, k))
        A.append(codes(lam, e, k)), P.append(prop[ts]), Y.append(np.stack([act[t:t + k] for t in ts]))
    A = torch.cat(A)
    t = lambda x: torch.tensor(np.concatenate(x), dtype=torch.float32, device=A.device)
    return A, t(P), t(Y)


def main(a):
    torch.manual_seed(a.seed), random.seed(a.seed), np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    eps = load(dev, diagnostic_eval=False)  # never loads TARGET robot data
    lam, k = load_lam(a.lam, dev)
    A, P, Y = robot_set(eps, lam, k, val=False)
    Av, Pv, Yv = robot_set(eps, lam, k, val=True)
    dec = Decoder(A.shape[1], k, P.mean(0), P.std(0) + 1e-6).to(dev)
    opt = torch.optim.AdamW(dec.parameters(), lr=3e-4, weight_decay=1e-2)
    for step in range(1, a.steps + 1):
        i = torch.randint(len(A), (512,), device=dev)
        loss = ((dec(A[i], P[i]) - Y[i]) ** 2).mean()
        opt.zero_grad(), loss.backward(), opt.step()
        if step % 2000 == 0:
            print(f"step {step} mse {loss.item():.4f}", flush=True)
    dec.eval()
    with torch.no_grad():
        pred = dec(Av, Pv)
        m = {"val_mse/learned": ((pred - Yv) ** 2).mean().item(),
             "val_mse/nearest_neighbour": ((Y[torch.cdist(Av, A).argmin(1)] - Yv) ** 2).mean().item(),
             "val_mse/mean_chunk_baseline": ((Y.mean(0) - Yv) ** 2).mean().item(),
             "val_gripper_sign_acc/learned": (pred[..., 6].sign() == Yv[..., 6].sign()).float().mean().item()}
    for k_, v in m.items():
        print(f"{k_:45s} {v:.4f}")
    out = f"runs/dec_{a.name}"
    os.makedirs(out, exist_ok=True)
    torch.save({"state": dec.state_dict(), "lam": a.lam, "d": A.shape[1], "h": k, "nn_codes": A, "nn_chunks": Y}, f"{out}/model.pt")
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    json.dump({"config": vars(a), "commit": commit, "metrics": m}, open(f"{out}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--lam", default="main")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
