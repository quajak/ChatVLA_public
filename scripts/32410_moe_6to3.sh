#!/bin/bash
TASK=0413_mix_math
LLM=qwen2_vl
#LLM_MODEL_SIZE=7B
LLM_MODEL_SIZE=2B
# lora only vit and tune adapter
ACTION_HEAD="dit_diffusion_policy" #act #transformer_diffusion
echo 'QWEN2 VLA-DiVLA version'
mnop=/home/jovyan/tzb/zhouzy/model_param/Qwen2-VL-${LLM_MODEL_SIZE}-Instruct
OUTPUT=/home/jovyan/tzb/zhouzy/model_param/qwen2_vla_param_${ACTION_HEAD}_results/ckpts/${TASK}/${LLM}_${LLM_MODEL_SIZE}/0423_Qwen2_lora_no_filmlayer_w_reasoningaction_route6_top_3
DIT=/home/share/ljm/model_param/scaledp/zzy/resnet50_with_film_subreason/use_constant_0_mobile_franka_reasoning_w_vl_data_selected_DiT-L_320_240_128_1e-4_numsteps_25000_sub_1_use_task_norm_0_2025_01_21_05_31_07/policy_step_25000_2025-01-22_00-17-18.ckpt

if [ -d "$OUTPUT" ]; then
   echo 'output exists'
else
   echo '!!output not exists!!'
   mkdir -p $OUTPUT
fi

# echo "waiting 3h"
# sleep 3h

mkdir -p $OUTPUT/src
cp -r ./scripts $OUTPUT/
cp -r ./data_utils $OUTPUT/src/
cp -r ./qwen2_vla $OUTPUT/src/
cp -r ./policy_heads $OUTPUT/src/

deepspeed --master_port 29607 --num_gpus=8 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
  --use_reasoning True \
  --using_film False\
  --resume_from_checkpoint False \
  --with_llm_head True \
  --pretrain_image_size 320 \
  --task_name ${TASK} \
  --model_name_or_path $mnop \
  --version v0 \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 25000 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --save_strategy "steps" \
  --save_steps 5000 \
  --save_total_limit 100 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "cosine_with_min_lr"\
  --min_lr 2e-6\
  --logging_steps 4 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --policy_class $ACTION_HEAD \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log \
  --group_by_modality_length False \
  --lora_enable True \
  --non_lora_lr 2e-5 \
  --pretrain_dit_path $DIT \
  --vl_ratio -1 \
  --using_moe True \
  --using_static_expert False \
  --using_shared_routed_expert True \
  --shared_expert_num 0 \
  --routed_expert_num 6 \
  --lora_module "vitllm" \
  --reasoning_layer_id 12 \
  --output_expert_traffic False


#  --load_pretrain False
#  --llm_loss_weight 0.1

mv ./60030.log $OUTPUT
echo $OUTPUT
