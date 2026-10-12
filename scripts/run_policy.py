"""Run a pretrained LeRobot policy on a LIBERO task and save the rollout to results/."""
import argparse
import json
import os
import time

import torch
from libero.libero import benchmark
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.factory import make_env_pre_post_processors
from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, LiberoEnv
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import ACTION

from preview import preview

p = argparse.ArgumentParser()
p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero", help="HF Hub id or local checkpoint dir")
p.add_argument("--suite", default="libero_spatial")
p.add_argument("--task-id", type=int, default=0)
p.add_argument("--init-state", type=int, default=0, help="index into the task's saved init states")
p.add_argument("--steps", type=int, default=None, help="defaults to the suite's eval limit")
p.add_argument("--size", type=int, default=256, help="camera resolution, match the training data")
p.add_argument("--preview", action="store_true", help="replay the rollout afterwards")
p.add_argument("--fps", type=int, default=30, help="preview fps")
p.add_argument("--n-action-steps", type=int, default=1, help="actions per inference call, up to chunk_size")
p.add_argument("--seed", type=int, default=0, help="sim and policy seed")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="avoid mps, SmolVLA fails every LIBERO episode on it")
args = p.parse_args()
torch.manual_seed(args.seed)

# LeRobot's wrapper, so obs match lerobot-eval for any policy
env = LiberoEnv(benchmark.get_benchmark_dict()[args.suite](), args.task_id, args.suite, obs_type="pixels_agent_pos",
                observation_width=args.size, observation_height=args.size, episode_index=args.init_state)
print(env.task_description)

cfg = PreTrainedConfig.from_pretrained(args.policy)
cfg.device = args.device
if args.n_action_steps:
    cfg.n_action_steps = args.n_action_steps
policy = get_policy_class(cfg.type).from_pretrained(args.policy, config=cfg)
policy.reset()
pre, post = make_pre_post_processors(cfg, args.policy, preprocessor_overrides={"device_processor": {"device": cfg.device}})
env_pre, env_post = make_env_pre_post_processors(LiberoConfig(task=args.suite), cfg)


def batched(x):  # lerobot-eval uses a vector env, so obs arrive batched
    return {k: batched(v) for k, v in x.items()} if isinstance(x, dict) else x[None]


def act(obs):
    batch = preprocess_observation(batched(obs))
    batch["task"] = [env.task_description]
    with torch.inference_mode():
        action = post(policy.select_action(pre(env_pre(batch))))
    return env_post({ACTION: action})[ACTION][0].cpu().numpy()


def sim_state():
    return env._env.sim.get_state().flatten().tolist()  # don't cache the sim, LIBERO rebuilds it on reset


obs, _ = env.reset(seed=args.seed)
max_steps = args.steps or TASK_SUITE_MAX_STEPS[args.suite]
actions, states, success = [], [sim_state()], False
try:
    for i in range(max_steps):
        action = act(obs)
        obs, _, done, _, info = env.step(action)
        actions.append(action.tolist())
        print(f"\rstep {i + 1}/{max_steps}", end="", flush=True)
        if done:  # env auto-resets here, so the final state is already gone
            success = info["is_success"]
            break
        states.append(sim_state())
except KeyboardInterrupt:
    print("\nstopped early", end="")
finally:
    env.close()
print(f"\n{'success' if success else 'not solved'} after {len(actions)} steps")

run = {**vars(args), "language": env.task_description, "success": bool(success), "actions": actions, "states": states}
os.makedirs("results", exist_ok=True)
path = f"results/{time.strftime('%Y%m%d-%H%M%S')}_{args.suite}_task{args.task_id}_init{args.init_state}.json"
with open(path, "w") as f:
    json.dump(run, f)
print(f"saved {path}")

if args.preview:
    preview(run, fps=args.fps)
