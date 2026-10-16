"""At the same task phase, does the human hand move like the robot gripper?

For each window of the held-out human ALIGN clips, find the window at the same phase (by TCC) in every robot demo
of the task. Then check whether hand and gripper move in the same image direction and agree on open/closed.
Robot vs robot pairs give the reference.
"""
import argparse
import json

import torch
import torch.nn.functional as F

from lam import windows
from motion import attach
from splits import ALIGN
from tcc import G, load

K = 10


@torch.no_grad()
def main(a):
    dev = "cuda"
    eps = load(dev, diagnostic_eval=False)
    attach(eps, K, dev)
    g = G().to(dev)
    g.load_state_dict(torch.load(f"runs/tcc_{a.tcc}/model.pt", map_location=dev))
    g.eval()
    out = {}
    for task in ALIGN:
        rob = [e for e in eps if e["dom"] == "robot" and e["task"] == task]
        hum = [e for e in eps if e["dom"] == "human" and e["task"] == task]
        zr = [g(e["x"][list(windows(e, K))]) for e in rob]
        for name, queries in (("human_vs_robot", hum), ("robot_vs_robot", rob[:10])):
            cos, grip, n = [], [], 0
            for q in queries:
                zq = g(q["x"][list(windows(q, K))])
                for e, z in zip(rob, zr):
                    if e is q:
                        continue
                    j = torch.cdist(zq, z).argmin(1)  # closest robot window to each human window
                    ok = q["kok"]
                    uq, ur = q["kin"][ok, :2], e["kin"][j[ok], :2]
                    moving = (uq.norm(dim=1) > 0.5) & (ur.norm(dim=1) > 0.5)  # skip near-still windows, their direction is noise
                    cos.append(F.cosine_similarity(uq[moving], ur[moving], dim=1))
                    grip.append(((q["kin"][ok, 3] > 0.5) == (e["kin"][j[ok], 3] > 0.5)).float())
                    n += int(ok.sum())
            c, gr = torch.cat(cos), torch.cat(grip)
            out[f"{task}/{name}"] = dict(motion_cosine_mean=c.mean().item(), same_direction_frac=(c > 0.5).float().mean().item(),
                                         grip_agree_frac=gr.mean().item(), n_window_pairs=n)
            print(task, name, {k: round(v, 3) for k, v in out[f"{task}/{name}"].items()}, flush=True)
    json.dump(out, open("runs/diag_handover.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tcc", default="main")
    main(p.parse_args())
