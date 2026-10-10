#!/bin/bash
# Baseline SmolVLA success rate on the two LIBERO-Goal tasks we use (1: bowl on stove, 8: bowl on plate), 50 episodes each.
# Cost estimate: ~1 h on 1 A100 (1/4 node) = ~128 SBU.
#SBATCH --job-name=eval_baseline
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate human-demo-to-libero
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${MUJOCO_GL:-egl}  # fallback: MUJOCO_GL=osmesa sbatch ...
lerobot-eval --policy.path=HuggingFaceVLA/smolvla_libero --policy.device=cuda \
  --env.type=libero --env.task=libero_goal --env.task_ids='[1,8]' \
  --eval.batch_size=10 --eval.n_episodes=50 --seed=0 \
  --output_dir=outputs/eval_baseline_goal_${SLURM_JOB_ID}
