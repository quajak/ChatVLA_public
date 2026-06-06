#!/bin/bash
#SBATCH --job-name=chatvla_libero
#SBATCH --gres=gpu:l40s:4             # 4 L40S on one node; ZeRO-2 shards the optimizer
#SBATCH --nodes=1
#SBATCH --ntasks=1                    # deepspeed launches the 4 ranks itself
#SBATCH --cpus-per-task=24            # 4 ranks x (4 dataloader workers + main proc)
#SBATCH --mem=300GB
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

set -euo pipefail

# Run from the repo root regardless of where sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Toolchain: deepspeed JIT-builds cpu_adam (cpu offload in zero2.json) and needs nvcc.
module load cuda/13.2

# Activate the project's uv-managed environment so `deepspeed`/`python` resolve.
source .venv/bin/activate

bash scripts/train_chatvla_libero.sh
