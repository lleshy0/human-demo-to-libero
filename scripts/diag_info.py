"""Why can't the decoder act out TARGET tasks? Trains small decoders offline and scores them on held-out demos
40-49 of the TARGET tasks and of the training tasks. Uses TARGET robot data, for diagnosis only.

Decoder inputs:
  code     latent action of frames t and t+k
  raw      V-JEPA features of frames t and t+k
  kin      true gripper motion in the image + grip opening (from motion.py)
  kin+img  kin + features of frame t
Trained on:
  other    robot demos 0-39 of the 7 non-TARGET tasks (plus LIBERO-90 with --libero90)
  target   TARGET robot demos 0-39, as a reference
"""
import argparse
import json

import h5py
import numpy as np
import torch
import torch.nn as nn

import motion
from decoder import load_lam
from lam import windows
from splits import TARGET
from tcc import load


class Head(nn.Module):
    """Optional feature grids + optional vector + robot state -> the next k actions."""

    def __init__(self, n_grids, d, h, mu, sd):
        super().__init__()
        self.h = h
        self.register_buffer("mu", mu)
        self.register_buffer("sd", sd)
        self.tok = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 64), nn.GELU())
        self.net = nn.Sequential(nn.Linear(n_grids * 16 * 64 + d + 9, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, h * 7))

    def forward(self, grids, vec, prop):
        x = [self.tok(g.float()).flatten(-2) for g in grids] + [vec, (prop - self.mu) / self.sd]
        return self.net(torch.cat(x, -1)).view(-1, self.h, 7)


@torch.no_grad()
def build(eps, lam, k):
    """All windows of all episodes as flat tensors: frame index, code, robot state, actions, task, demo number."""
    X = torch.cat([e["x"].to(lam.pos.device) for e in eps])  # LIBERO-90 episodes are kept on CPU
    G, A, P, Y, T, I = [], [], [], [], [], []
    off = 0
    for e in eps:
        with h5py.File(e["h5"]) as f:
            act, prop = f["data"][e["key"]]["actions"][:], f["data"][e["key"]]["robot_states"][:]
        ts = list(windows(e, k))
        g = torch.tensor(ts, device=X.device) + off
        G.append(g), A.append(lam.encode(X[g], X[g + k])), P.append(prop[ts])
        Y.append(np.stack([act[t:t + k] for t in ts]))
        T += [e["task"]] * len(ts)
        I += [int(e["key"].split("_")[1])] * len(ts)
        off += len(e["x"])
    t = lambda v: torch.tensor(np.concatenate(v), dtype=torch.float32, device=X.device)
    return X, torch.cat(G), torch.cat(A), t(P), t(Y), np.array(T), np.array(I)


def metrics(pred, y):
    se = (pred - y) ** 2
    return dict(trans_mse=se[..., :3].mean().item(), rot_mse=se[..., 3:6].mean().item(),
                grip_err=(pred[..., 6].sign() != y[..., 6].sign()).float().mean().item())


def main(a):
    torch.manual_seed(0), np.random.seed(0)
    dev = "cuda"
    eps = [e for e in load(dev, diagnostic_eval=True, libero90=a.libero90) if e["dom"] == "robot"]
    lam, k = load_lam(a.lam, dev)
    X, G, A, P, Y, T, I = build(eps, lam, k)
    motion.attach(eps, k, dev)  # same episodes and windows as build()
    K = torch.cat([e["kin"] for e in eps])
    none = torch.zeros(len(A), 0, device=dev)
    inputs = {"code": lambda j: ([], A[j]), "raw": lambda j: ([X[G[j]], X[G[j] + k]], none[j]),
              "kin": lambda j: ([], K[j]), "kin+img": lambda j: ([X[G[j]]], K[j])}
    is_t = torch.tensor(np.isin(T, TARGET), device=dev)
    tr_idx = torch.tensor(I < 40, device=dev)
    is_goal = torch.tensor(np.concatenate([["libero_90" not in e["h5"]] * len(windows(e, k)) for e in eps]), device=dev)
    sets = {"other": ~is_t & tr_idx, "target": is_t & tr_idx & is_goal}
    evals = {"TARGET 40-49": is_t & ~tr_idx, "train tasks 40-49": ~is_t & ~tr_idx & is_goal}  # scored on LIBERO-Goal demos only
    out = {}
    for inp, get in inputs.items():
        for name, m in sets.items():
            idx = m.nonzero().squeeze(1)
            grids, vec = get(idx[:1])
            net = Head(len(grids), vec.shape[1], k, P[idx].mean(0), P[idx].std(0) + 1e-6).to(dev)
            fwd = lambda j: net(*get(j), P[j])
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
            for _ in range(a.steps):
                j = idx[torch.randint(len(idx), (512,), device=dev)]
                loss = ((fwd(j) - Y[j]) ** 2).mean()
                opt.zero_grad(), loss.backward(), opt.step()
            net.eval()
            with torch.no_grad():
                for ev, em in evals.items():
                    j = em.nonzero().squeeze(1)
                    r = metrics(torch.cat([fwd(j[i:i + 4096]) for i in range(0, len(j), 4096)]), Y[j])
                    out[f"{inp} | train {name} | eval {ev}"] = r
                    print(f"{inp:7s} | trained on {name:6s} | eval {ev:18s} | " + " ".join(f"{kk} {v:.4f}" for kk, v in r.items()), flush=True)
    yt = Y[evals["TARGET 40-49"]]
    out["mean chunk | train other | eval TARGET 40-49"] = metrics(Y[sets["other"]].mean(0).expand_as(yt), yt)  # baseline: always predict the average actions
    json.dump({"lam": a.lam, "steps": a.steps, "results": out}, open(f"runs/diag_info{'_libero90' if a.libero90 else ''}.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lam", default="main")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--libero90", action="store_true", help="add the filtered LIBERO-90 robot demos to 'other'")
    main(p.parse_args())
