# Project: Human-video rewards from a JEPA world model for a LIBERO policy

## Context
I'm applying for a robot-learning internship at Humanoid. The challenge: **use manipulation data I recorded myself (phone video) to drive a robot arm in simulation (LIBERO)**. Deadline: **Fri 16 Oct 2026, 23:59 BST**. Deliverable: a public GitHub repo with a README covering how to run it, example outputs, design choices, and what did and didn't work.

What they judge on: (1) creativity, with my data playing a real role; (2) policy performance in sim and/or world-model quality; (3) a simple implementation and clear, honest results ("no AI slop").

About me: strong in ML, RL and diffusion, comfortable with PyTorch. New to robotics and VLAs, so briefly explain any robotics-specific choices.

Compute: a SLURM cluster with A100s (~30k SBU budget, roughly 230 GPU-hours) plus Colab. Estimate the cost of every job before launching it.

## The idea
Use a **frozen V-JEPA 2 video encoder as a reward model specified by my human demo video**. A robot rollout scores well when its trajectory in V-JEPA feature space moves toward, or follows, my demo. Then use that score to improve a bootstrapped policy with an RL-lite loop, **without using the simulator's success signal for training**. The success signal is used only for evaluation.

Core claim to test: *my demo defines the task.* Swap the demo and the policy's behaviour should change.

## Phases (stop and report to me after each phase)

### Phase 0: Setup
- Install LIBERO and get headless rendering working on the cluster (`MUJOCO_GL=egl`, with osmesa as a fallback). Watch for the known LIBERO quirk where camera images come out upside down.
- Get a baseline policy. First search the Hugging Face Hub for an existing SmolVLA (or similar small policy) checkpoint already fine-tuned on LIBERO, and LeRobot-format LIBERO datasets. Only fine-tune it yourself if nothing usable exists. Verify that model and dataset names actually exist; don't guess them.
- Load V-JEPA 2 (e.g. the ViT-L checkpoint in HF `transformers`; verify the exact name) and write `embed(video) -> per-timestep features`, using a sliding window of ~16 frames ending at t with mean-pooled tokens. Cache all embeddings to disk.
- Pick 1–2 tasks I can physically reproduce at home. Prefer the **LIBERO-Goal** suite: it has several tasks in the same scene, which enables the demo-swap test. Example: "put the bowl on the plate" vs "put the bowl on the stove".
- **Early on, give me a short recording spec**: which tasks, a camera angle matching LIBERO's `agentview` camera (front, slightly above), phone on a stand, resolution/fps, how many takes (~5–10 per task), and that I **pull my hand out of frame at the end**, so the goal frame shows the scene and not my hand.

### Phase 1: Go/no-go — is the score meaningful?
- Run the baseline with action noise / different seeds to get ~50–100 rollouts per task with a mix of successes and failures. Record frames plus the ground-truth success flag.
- Implement two scores:
  1. **Goal distance:** −cosine distance between the rollout embedding at t and the embedding of my demo's final window.
  2. **Demo-progress:** align the rollout to my demo trajectory (nearest neighbour / DTW in feature space) → progress ∈ [0, 1].
- Metrics:
  - Spearman correlation of score with time along successful rollouts.
  - AUROC for separating successes from failures.
  - **Demo-swap test:** does demo A rank task-A successes above task-B successes in the same scene?
- Baselines for the encoder: DINOv2 and/or CLIP image features with the same scoring. This tells us whether a JEPA *video* encoder actually helps.
- Also try simple fixes for the human→robot viewpoint gap if needed (cropping to the workspace, masking the arm/hand region). Report what each one changes.
- Report numbers and plots to me. **I'll decide go/no-go.** If the scores don't work, the fallback is to make the gap between human and robot video itself the result: a fidelity analysis of where JEPA features transfer and where they don't.

### Phase 2: Improve the policy with the score
- Main method: **filtered behaviour cloning / reward-weighted regression** (ReST-style). Roll out the policy, keep the top-k% by JEPA score, fine-tune on those, repeat 2–3 rounds.
- Evaluate success rate with the ground-truth checker after each round, against: baseline, random-k filtering (control), and oracle filtering by true success (upper bound).
- Optional if time allows: best-of-N test-time reranking of rollouts by score.
- Look for and report reward hacking (high score, failed task).

### Phase 3: Write-up
- README: one-paragraph idea, a figure of my demo frames next to the sim camera, score curves, AUROC table (JEPA vs DINOv2/CLIP), success rate per round, the demo-swap result, failure cases, honest limitations, and exact run commands.
- Short and concrete. No marketing language.

## Working rules
- Keep the repo small and flat: `scripts/`, `src/`, `slurm/`, `configs/`, `README.md`. Plain functions, no unnecessary abstractions. Fix seeds.
- Never fabricate or "expect" results. Every number in the README must come from a logged run.
- Prefer existing tools (LeRobot, LIBERO's eval code) over reimplementing.
- Before writing code, give me a short plan with a time and SBU estimate per phase, then start Phase 0.