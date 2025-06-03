#!/bin/bash
TASK=mobile_franka_reasoning_w_vl_data_selected
LLM=qwen2_vl
#LLM_MODEL_SIZE=7B
LLM_MODEL_SIZE=2B
# lora only vit and tune adapter
ACTION_HEAD="dit_diffusion_policy" #act #transformer_diffusion
echo 'QWEN2 VLA-DiVLA version'
mnop=/home/jovyan/tzb/wjj/model_param/Qwen2-VL-${LLM_MODEL_SIZE}-Instruct
#OUTPUT=/home/jovyan/tzb/zhumj/model_param/qwen2_vla_param_${ACTION_HEAD}_results/ckpts/${TASK}/${LLM}_${LLM_MODEL_SIZE}/Qwen2_${LLM_MODEL_SIZE}_v0_w_llava_finetune_moe_2parts_w_lora_step_2_cosine_2e_5
OUTPUT=/home/jovyan/tzb/zhouzy/model_param/qwen2_vla_param_${ACTION_HEAD}_results/ckpts/${TASK}/${LLM}_${LLM_MODEL_SIZE}/Qwen2_${LLM_MODEL_SIZE}_v0_w_llava_finetune_moe_2parts_w_lora_step_2_cosine_2e_5_s1_20k
#DIT=/home/share/ljm/model_param/scaledp/11_1_reasoning_all_tasks_DiT-L_320_240_32_1e-4_numsteps_100000_scaledp/policy_step_90000.ckpt
#PRETRAIN=/home/jovyan/tzb/zhumj/model_param/qwen2_vla_param_${ACTION_HEAD}_results/ckpts/all_reasoning_and_llava_finetune/${LLM}_${LLM_MODEL_SIZE}/Qwen2_${LLM_MODEL_SIZE}_v0_w_llava_finetune_moe_2parts_w_lora_step_1_constant_1e_4/checkpoint-28000
PRETRAIN=/home/jovyan/tzb/zhouzy/model_param/qwen2_vla_param_${ACTION_HEAD}_results/ckpts/${TASK}/${LLM}_${LLM_MODEL_SIZE}/Qwen2_${LLM_MODEL_SIZE}_v0_w_llava_finetune_moe_2parts_w_lora_step_1_constant_1e_4/checkpoint-20000
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
  --max_steps 15000 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 5000 \
  --save_total_limit 100 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "cosine" \
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
  --using_vision_moe False \
  --load_pretrain True \
  --model_pretrain $PRETRAIN \
  --lora_enable True \
  --non_lora_lr 2e-5 \
#  --pretrain_dit_path $DIT \

#  --load_pretrain False
#  --llm_loss_weight 0.1

mv ./60030.log $OUTPUT
echo $OUTPUT
