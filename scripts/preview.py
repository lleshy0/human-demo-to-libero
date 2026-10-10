"""Replay a saved LIBERO run. Usage: python preview.py --run results/<run>.json"""
import argparse
import json
import os

import cv2
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")


def preview(run, fps=30, cameras=CAMERAS):
    """Show `run` (the dict run_policy.py saves) by restoring each recorded sim state and rendering it."""
    task = benchmark.get_benchmark_dict()[run["suite"]]().get_task(run["task_id"])
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=run["size"], camera_widths=run["size"])
    env.reset()
    print(f"{run['language']} | {'success' if run['success'] else 'not solved'} in {len(run['actions'])} steps (Esc quits)")
    try:
        for state in run["states"]:
            obs = env.set_init_state(np.array(state))
            frame = np.hstack([obs[c] for c in cameras])[::-1]  # LIBERO renders upside down
            cv2.imshow("libero", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1000 // fps) == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="results/<run>.json saved by run_policy.py")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()
    with open(args.run) as f:
        preview(json.load(f), fps=args.fps)
