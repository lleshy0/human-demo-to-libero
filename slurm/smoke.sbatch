#!/bin/bash
# Phase 0 smoke test: headless LIBERO render + 1 SmolVLA episode + V-JEPA 2 embed self-check. ~10 min = ~25 SBU.
#SBATCH --job-name=smoke
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/%x_%j.out
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate human-demo-to-libero
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${MUJOCO_GL:-egl}
nvidia-smi --query-gpu=name,memory.total --format=csv
python src/embed.py
python scripts/run_policy.py --suite libero_goal --task-id 8 --n-action-steps 10
