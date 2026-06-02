#!/bin/bash
# Quick smoke-test for train_chatvla_libero.sh setup.
# Runs 5 steps on 1 GPU, no wandb, no checkpoint copies, no save.

TASK=libero_spatial
ACTION_HEAD="scale_dp_policy"
MNOP=/root/ChatVLA_public/weights
OUTPUT=/tmp/chatvla_debug

mkdir -p $OUTPUT

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --master_port 29608 --num_gpus=1 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2_debug.json \
  --use_lerobot True \
  --task_name ${TASK} \
  --use_reasoning False \
  --action_dim 7 \
  --state_dim 8 \
  --chunk_size 16 \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "ScaleDP_L" \
  --resume_from_checkpoint False \
  --with_llm_head True \
  --pretrain_image_size 320 \
  --model_name_or_path ${MNOP} \
  --freeze_vision_tower False \
  --freeze_backbone False \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --save_strategy "no" \
  --max_steps 5 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --min_lr 0 \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --dataloader_num_workers 0 \
  --output_dir $OUTPUT \
  --report_to none \
  --lora_enable False \
  --using_moe False
