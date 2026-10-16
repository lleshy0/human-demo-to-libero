"""Policy: TCC phase embedding + task ID -> latent action. Trained on videos only, with the LAM's codes as targets."""
import argparse
import json
import os
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn
from libero.libero.benchmark.libero_suite_task_map import libero_task_map

from decoder import load_lam
from lam import codes, windows
from splits import ALIGN, HUMAN, SUITE, TARGET
from tcc import G, load

TASKS = libero_task_map[SUITE]


class Policy(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(128 + len(TASKS), 512), nn.GELU(),
                                 nn.Linear(512, 512), nn.GELU(), nn.Linear(512, d), nn.Tanh())  # codes are in [-1, 1]

    def forward(self, z, task_id):
        return self.net(torch.cat([z, nn.functional.one_hot(task_id, len(TASKS)).float()], -1))


def dataset(eps, g, lam, k, val):
    Z, T, A = [], [], []
    for e in eps:
        if e["val"] != val:
            continue
        ts = list(windows(e, k))
        Z.append(g(e["x"][ts])), A.append(codes(lam, e, k))
        T.append(torch.full((len(ts),), TASKS.index(e["task"]), device=e["x"].device))
    return torch.cat(Z), torch.cat(T), torch.cat(A)


def main(a):
    torch.manual_seed(a.seed), random.seed(a.seed), np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    diagnostic = a.source == "robot" and a.tasks == "target"
    tasks = {"align": ALIGN, "target": TARGET, "human": HUMAN}[a.tasks]
    # TARGET robot data only loads for the diagnostic D3 run
    eps = [e for e in load(dev, diagnostic_eval=diagnostic) if e["dom"] == a.source and e["task"] in tasks]
    for e in eps:  # load() holds out all TARGET robot demos; D3 trains on demos 0-39 like other tasks
        if e["dom"] == "robot":
            e["val"] = int(e["key"].split("_")[1]) >= 40
    assert eps, "no episodes for this source/task selection"
    g = G().to(dev)
    g.load_state_dict(torch.load(f"runs/tcc_{a.tcc}/model.pt", map_location=dev))
    g.eval()
    lam, k = load_lam(a.lam, dev)
    with torch.no_grad():
        Z, T, A = dataset(eps, g, lam, k, val=False)
        Zv, Tv, Av = dataset(eps, g, lam, k, val=True)
    pi = Policy(A.shape[1]).to(dev)
    opt = torch.optim.AdamW(pi.parameters(), lr=3e-4, weight_decay=1e-2)
    for step in range(1, a.steps + 1):
        i = torch.randint(len(Z), (256,), device=dev)
        loss = ((pi(Z[i], T[i]) - A[i]) ** 2).mean()
        opt.zero_grad(), loss.backward(), opt.step()
        if step % 2000 == 0:
            print(f"step {step} mse {loss.item():.4f}", flush=True)
    pi.eval()
    with torch.no_grad():
        m = {"val_mse": ((pi(Zv, Tv) - Av) ** 2).mean().item(), "val_mse/mean_code_baseline": ((A.mean(0) - Av) ** 2).mean().item(),
             "n_train": len(Z), "diagnostic": diagnostic}
    print(m)
    out = f"runs/pi_{a.name}"
    os.makedirs(out, exist_ok=True)
    torch.save({"state": pi.state_dict(), "d": A.shape[1], "tcc": a.tcc, "lam": a.lam},f"{out}/model.pt")
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    json.dump({"config": vars(a), "commit": commit, "metrics": m}, open(f"{out}/metrics.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--source", choices=["robot", "human"], required=True)
    p.add_argument("--tasks", choices=["align", "target", "human"], required=True)
    p.add_argument("--lam", default="main")
    p.add_argument("--tcc", default="main")
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
