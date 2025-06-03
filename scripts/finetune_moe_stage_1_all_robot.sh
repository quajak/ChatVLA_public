#!/bin/bash
TASK=mobile_franka_reasoning_w_vl_data_selected
LLM=qwen2_vl
#LLM_MODEL_SIZE=7B
LLM_MODEL_SIZE=2B
# lora only vit and tune adapter
ACTION_HEAD="dit_diffusion_policy" #act #transformer_diffusion
echo 'QWEN2 VLA-DiVLA version'
mnop=/home/jovyan/tzb/wjj/model_param/Qwen2-VL-${LLM_MODEL_SIZE}-Instruct
OUTPUT=/home/jovyan/tzb/zhouzy/model_param/qwen2_vla_param_${ACTION_HEAD}_results/ckpts/${TASK}/${LLM}_${LLM_MODEL_SIZE}/Qwen2_${LLM_MODEL_SIZE}_v0_subreasoning_dit_w_llava_finetune_moe_2parts_w_lora_step_1_constant_1e_4
#DIT=/home/share/ljm/model_param/scaledp/zzy/resnet50_with_film_subreason/use_constant_0_mobile_franka_reasoning_w_vl_data_selected_DiT-L_320_240_128_1e-4_numsteps_25000_sub_0_use_task_norm_0_2025_01_19_17_14_06/policy_step_25000_2025-01-20_04-35-32.ckpt
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

deepspeed --master_port 29604 --num_gpus=8 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
  --use_reasoning True \
  --using_film True \
  --resume_from_checkpoint False \
  --with_llm_head True \
  --pretrain_image_size 320 \
  --task_name ${TASK} \
  --model_name_or_path $mnop \
  --version v0 \
  --freeze_vision_tower True \
  --freeze_backbone False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 30000 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 2000 \
  --save_total_limit 100 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "constant" \
  --logging_steps 4 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --policy_class $ACTION_HEAD \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log \
  --group_by_modality_length False \
  --using_moe True \
  --freeze_vl_expert True \
  --lora_enable True \
  --non_lora_lr 1e-4 \
  --pretrain_dit_path $DIT

#  --load_pretrain False
#  --llm_loss_weight 0.1

mv ./60030.log $OUTPUT
echo $OUTPUT
