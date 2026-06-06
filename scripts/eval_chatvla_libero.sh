#!/bin/bash
# Evaluate a trained ChatVLA checkpoint in the LIBERO simulator.
# Adjust TASK, CKPT, and NUM_TRIALS before running.

TASK=libero_spatial                  # libero_spatial | libero_object | libero_goal | libero_10 | libero
CKPT=/scratch/gerigkja/ChatVLA_public/eval/weights/ChatVLA_libero_${TASK}/checkpoint-15000
NUM_TRIALS=20

# LIBERO renders off-screen with EGL; force the headless GL backend.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python evaluate/evaluate_libero.py \
  --model_path "${CKPT}" \
  --task_name "${TASK}" \
  --action_head scale_dp_policy \
  --query_frequency 16 \
  --num_trials ${NUM_TRIALS} \
  --output_json "$(dirname "${CKPT}")/libero_eval_${TASK}.json" \
  --verbose
