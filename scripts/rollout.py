"""Run the pipeline in LIBERO.

d1 mode: feeds the decoder the codes of a held-out robot demo every k steps, from that demo's first state (open loop).
policy mode: camera -> V-JEPA -> TCC -> policy -> decoder -> k actions, repeated (closed loop).
"""
import argparse
import json
import os
import subprocess
import multiprocessing as mp

import av
import h5py
import numpy as np
import torch

from splits import SUITE, TARGET, robot_tasks

MAX_STEPS = 300  # LIBERO-Goal step limit, as in LeRobot
WAIT = 10  # do-nothing steps so objects settle, as in LeRobot/OpenVLA
DUMMY = [0, 0, 0, 0, 0, 0, -1]
_w = {}  # models, loaded once per worker


def make_env(task_id, render):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = benchmark.get_benchmark_dict()[SUITE]().get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    kw = dict(camera_heights=256, camera_widths=256, camera_names=["agentview"]) if render else \
        dict(has_offscreen_renderer=False, use_camera_obs=False)
    return OffScreenRenderEnv(bddl_file_name=bddl, **kw)


def proprio(env):
    """Robot state in the same layout as the demos: finger positions (2), gripper position (3), orientation (4)."""
    o = env.env._get_observations()
    return np.concatenate([o["robot0_gripper_qpos"], o["robot0_eef_pos"], o["robot0_eef_quat"]])


def models(a, dev):
    if "dec" in _w:
        return _w
    torch.set_num_threads(1)  # many workers per node: one thread each
    from decoder import Decoder, load_lam
    from policy import Policy
    from tcc import G

    ck = torch.load(f"runs/dec_{a.dec}/model.pt", map_location=dev)
    dec = Decoder(ck["d"], ck["h"], torch.zeros(9), torch.ones(9)).to(dev)
    dec.load_state_dict(ck["state"])
    _w.update(dec=dec.eval(), nn_codes=ck["nn_codes"], nn_chunks=ck["nn_chunks"], k=ck["h"], mean_code=ck["nn_codes"].mean(0))
    _w["lam"], _ = load_lam(ck["lam"], dev)
    if a.mode == "policy":
        from transformers import VJEPA2Model
        from features import MODEL

        _w["vjepa"] = VJEPA2Model.from_pretrained(MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
        if a.pi:
            pk = torch.load(f"runs/pi_{a.pi}/model.pt", map_location=dev)
            pi = Policy(pk["d"]).to(dev)
            pi.load_state_dict(pk["state"])
            g = G().to(dev)
            g.load_state_dict(torch.load(f"runs/tcc_{pk['tcc']}/model.pt", map_location=dev))
            _w.update(pi=pi.eval(), g=g.eval())
    return _w


@torch.no_grad()
def decode(w, code, prop, how, dev):
    if how == "nn":
        return w["nn_chunks"][torch.cdist(code[None], w["nn_codes"]).argmin()].cpu().numpy()
    return w["dec"](code[None], torch.tensor(prop, dtype=torch.float32, device=dev)[None])[0].cpu().numpy()


def run_chunk(env, chunk, frames=None, run=None):
    for act in chunk:
        act = np.clip(act, -1, 1)
        act[6] = 1.0 if act[6] > 0 else -1.0  # gripper is binary in LIBERO demos
        obs, _, done, _ = env.step(act)
        if frames is not None:
            frames.append(obs["agentview_image"][::-1])
        if run is not None:
            run["actions"].append(act.tolist()), run["states"].append(env.sim.get_state().flatten().tolist())
        if done:
            return True
    return False


@torch.no_grad()
def d1_episode(job):
    a, task_id, task_name, key = job
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = models(a, dev)
    k = w["k"]
    env = make_env(task_id, render=False)
    with h5py.File(f"data/demos/libero_goal/{task_name}_demo.hdf5") as f:
        states, rs = f["data"][key]["states"][:], f["data"][key]["robot_states"][:]
    x = torch.from_numpy(np.load(f"data/features/robot/{task_name}/{key}.npy")).flatten(1, 2).to(dev)
    env.reset()
    env.set_init_state(states[0])
    prop_err = float(np.abs(proprio(env) - rs[0]).max())  # check: the env's robot state must match the demos' layout
    ok, steps = False, 0
    for t in range(0, len(x) - k, k):
        code = w["lam"].encode(x[t:t + 1], x[t + k:t + k + 1])[0]
        ok = run_chunk(env, decode(w, code, proprio(env), a.decoder, dev))
        steps = t + k
        if ok:
            break
    env.close()
    return dict(task=task_name, episode=key, success=ok, steps=steps, demo_len=len(x), proprio_err=prop_err)


@torch.no_grad()
def policy_episode(job):
    from features import encode
    from policy import TASKS

    a, task_id, task_name, ep = job
    dev = "cuda"
    w = models(a, dev)
    env = make_env(task_id, render=True)
    from libero.libero import benchmark

    task = benchmark.get_benchmark_dict()[SUITE]()
    init = task.get_task_init_states(task_id)[ep]
    env.reset()
    obs = env.set_init_state(init)
    frames = [obs["agentview_image"][::-1]]
    for _ in range(WAIT):
        obs, _, _, _ = env.step(DUMMY)
        frames.append(obs["agentview_image"][::-1])
    # saved so preview.py can replay it
    run = dict(suite=SUITE, task_id=task_id, size=256, language=task.get_task(task_id).language, actions=[],
               states=[env.sim.get_state().flatten().tolist()])
    tid = torch.tensor([TASKS.index(task_name)], device=dev)
    rng = np.random.default_rng(ep)
    ok, t = False, 0
    while t < MAX_STEPS and not ok:
        if a.baseline == "mean":
            code = w["mean_code"]
        elif a.baseline == "random":
            code = torch.tensor(rng.uniform(-1, 1, w["mean_code"].shape), dtype=torch.float32, device=dev)
        else:
            f = torch.from_numpy(encode(w["vjepa"], np.stack(frames[-16:]), ts=[min(len(frames), 16) - 1])).to(dev).flatten(1, 2)
            code = w["pi"](w["g"](f), tid)[0]
        chunk = decode(w, code, proprio(env), a.decoder, dev)
        ok = run_chunk(env, chunk, frames, run)
        t += len(chunk)
    env.close()
    if ep < a.videos:
        os.makedirs(f"runs/eval_{a.name}/videos", exist_ok=True)
        path = f"runs/eval_{a.name}/videos/{task_name}_{ep:02d}_{'ok' if ok else 'fail'}"
        save_mp4(f"{path}.mp4", frames)
        json.dump({**run, "success": bool(ok)}, open(f"{path}.json", "w"))
    return dict(task=task_name, episode=ep, success=ok, steps=t)


def save_mp4(path, frames, fps=20):
    out = av.open(path, "w")
    s = out.add_stream("libx264", rate=fps)
    s.width = s.height = frames[0].shape[0]
    s.pix_fmt = "yuv420p"
    for fr in frames:
        out.mux(s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(fr), format="rgb24")))
    out.mux(s.encode())
    out.close()


def main(a):
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    names = libero_task_map[SUITE]
    if a.mode == "d1":
        tasks = list(robot_tasks(diagnostic=a.diagnostic))
        jobs = []
        for t in tasks:
            with h5py.File(f"data/demos/libero_goal/{t}_demo.hdf5") as f:
                held_out = range(50) if t in TARGET else range(40, 50)  # no TARGET demo was ever trained on
                jobs += [(a, names.index(t), t, f"demo_{i}") for i in held_out if f"demo_{i}" in f["data"]]
        fn = d1_episode
    else:
        tasks = a.tasks or list(robot_tasks())
        jobs = [(a, names.index(t), t, ep) for t in tasks for ep in range(a.episodes)]
        fn = policy_episode
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as p:
        for r in p.imap_unordered(fn, jobs):
            rows.append(r)
            print(f"{len(rows)}/{len(jobs)} {r['task']} {r['episode']} success={r['success']}", flush=True)
    summary = {t: sum(r["success"] for r in rows if r["task"] == t) / max(1, sum(r["task"] == t for r in rows)) for t in tasks}
    for t, v in summary.items():
        print(f"{t:45s} {v:.2f}" + ("  (TARGET, diagnostic)" if t in TARGET and a.mode == "d1" else ""))
    os.makedirs(f"runs/eval_{a.name}", exist_ok=True)
    commit = subprocess.run(["git", "describe", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
    json.dump({"config": vars(a), "commit": commit, "summary": summary, "episodes": rows}, open(f"runs/eval_{a.name}/results.json", "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=["d1", "policy"], required=True)
    p.add_argument("--dec", default="main", help="runs/dec_<name>")
    p.add_argument("--decoder", choices=["learned", "nn"], default="learned")
    p.add_argument("--pi", help="runs/pi_<name> (policy mode)")
    p.add_argument("--baseline", choices=["mean", "random"], help="replace π")
    p.add_argument("--tasks", nargs="+", help="task names (policy mode); default: the non-TARGET tasks")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--videos", type=int, default=2, help="save videos of the first N episodes per task")
    p.add_argument("--diagnostic", action="store_true", help="D1: include TARGET demos")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    assert a.mode != "policy" or a.pi or a.baseline, "policy mode needs --pi or --baseline"
    main(a)
