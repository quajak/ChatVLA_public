#!/bin/bash
# Train ChatVLA on LIBERO via LeRobot library.
# Adjust TASK, MNOP, OUTPUT, and --num_gpus before running.

TASK=libero_spatial                        # one of: libero_spatial, libero_object, libero_goal, libero_10, libero
ACTION_HEAD="scale_dp_policy"              # or unet_diffusion_policy
MNOP=/scratch/gerigkja/weights/Qwen2-VL-2B-Instruct # official Qwen2-VL-2B weights (or Stage-1 checkpoint)
OUTPUT=/scratch/gerigkja/ChatVLA_public/eval/weights/ChatVLA_libero_${TASK}

mkdir -p $OUTPUT/src
cp -r ./scripts $OUTPUT/
cp -r ./data_utils $OUTPUT/src/
cp -r ./qwen2_vla $OUTPUT/src/
cp -r ./policy_heads $OUTPUT/src/

# 4x L40S, single node. The previous single-GPU config (scripts/zero2.json)
# offloaded the optimizer to CPU because the ~30GB of fp32 Adam state didn't fit
# alongside activations on one 46GB card. CPU Adam + the per-step grad/param PCIe
# copies stalled the GPU (util swinging 0<->100%, ~50% power, heavy "memory
# access" time). With 4 GPUs, ZeRO-2 SHARDS the optimizer state (~30GB -> ~7.5GB
# each), so we drop CPU offload (scripts/zero2_no_offload.json) and keep full
# fp32 Adam entirely on-GPU -> no stall + ~3-4x data-parallel throughput.
#
# Effective batch = num_gpus(4) * per_device(4) * accum(1) = 16, matching the old
# single-GPU 8x2, so the LR stays valid. per_device=4 keeps per-GPU activations
# (~14GB) well within 46GB; bump it (and retune LR) if you want larger batches.

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --master_port 29607 --num_gpus=4 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2_no_offload.json \
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
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 1000 \
  --max_steps 15000 \
  --save_total_limit 10 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --min_lr 0 \
  --logging_steps 4 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 4 \
  --output_dir $OUTPUT \
  --report_to wandb \
  --logging_dir $OUTPUT/log \
  --lora_enable False \
  --using_moe False \
  --torch_compile_model False | tee $OUTPUT/log.log

# Copy the image-processor config into the top-level output dir and every
# checkpoint subdir so AutoProcessor rebuilds the full Qwen2-VL processor at eval
# time (without it, it degrades to a bare tokenizer and silently drops images).
cp ${MNOP}/preprocessor_config.json "$OUTPUT"/
cp ${MNOP}/chat_template.json "$OUTPUT"/
for dir in "$OUTPUT"/*/; do
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${MNOP}/preprocessor_config.json $dir
        cp ${MNOP}/chat_template.json $dir
    fi
done
echo $OUTPUT
