#!/bin/bash
# Train ChatVLA on LIBERO via LeRobot library.
# Adjust TASK, MNOP, OUTPUT, and --num_gpus before running.

TASK=libero_spatial                        # one of: libero_spatial, libero_object, libero_goal, libero_10, libero
ACTION_HEAD="scale_dp_policy"              # or unet_diffusion_policy
MNOP=/root/ChatVLA_public/weights # official Qwen2-VL-2B weights (or Stage-1 checkpoint)
OUTPUT=/root/ChatVLA_public/eval/weights/ChatVLA_libero_${TASK}

mkdir -p $OUTPUT/src
cp -r ./scripts $OUTPUT/
cp -r ./data_utils $OUTPUT/src/
cp -r ./qwen2_vla $OUTPUT/src/
cp -r ./policy_heads $OUTPUT/src/

deepspeed --master_port 29607 --num_gpus=2 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
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
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 2 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --max_steps 5000 \
  --save_total_limit 10 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --min_lr 0 \
  --logging_steps 4 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --output_dir $OUTPUT \
  --report_to wandb \
  --logging_dir $OUTPUT/log \
  --lora_enable False \
  --using_moe False | tee $OUTPUT/log.log

for dir in "$OUTPUT"/*/; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${MNOP}/preprocessor_config.json $dir
        cp ${MNOP}/chat_template.json $dir
    fi
done
echo $OUTPUT
