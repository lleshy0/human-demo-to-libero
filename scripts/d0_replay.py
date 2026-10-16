"""Simulator setup check: replay each demo's recorded actions from its first state."""
import argparse
import json
import os
from multiprocessing import Pool

import h5py

from splits import SUITE

DEMOS = "data/demos/libero_goal"
_envs = {}


def replay_demo(job):
    task_id, task_name, key = job
    if task_id not in _envs:  # hard_reset=False skips the slow sim rebuild
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        task = benchmark.get_benchmark_dict()[SUITE]().get_task(task_id)
        _envs[task_id] = OffScreenRenderEnv(
            bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file),
            has_offscreen_renderer=False, use_camera_obs=False, hard_reset=False)
    env = _envs[task_id]
    with h5py.File(f"{DEMOS}/{task_name}_demo.hdf5") as f:
        states, actions = f["data"][key]["states"][:], f["data"][key]["actions"][:]
    env.reset()
    env.set_init_state(states[0])
    done, n = False, 0
    for n, a in enumerate(actions, 1):
        _, _, done, _ = env.step(a)
        if done:
            break
    return dict(task=task_name, demo=key, success=bool(done), steps=n, demo_len=len(actions))


if __name__ == "__main__":
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default="results/d0.json")
    a = p.parse_args()
    jobs = []
    for task_id, name in enumerate(libero_task_map[SUITE]):
        with h5py.File(f"{DEMOS}/{name}_demo.hdf5") as f:
            jobs += [(task_id, name, k) for k in sorted(f["data"], key=lambda s: int(s.split("_")[1]))]
    rows = []
    # new process per task: LIBERO envs leak memory
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(replay_demo, jobs, chunksize=10):
            rows.append(r)
            print(f"{len(rows)}/{len(jobs)} {r['task']}/{r['demo']} success={r['success']}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=1)
    for t in libero_task_map[SUITE]:
        rs = [r for r in rows if r["task"] == t]
        print(f"{t:45s} replay success {sum(r['success'] for r in rs)}/{len(rs)}")
    print(f"overall {sum(r['success'] for r in rows)}/{len(rows)}")
