#!/bin/bash
TASK=0501_aloha
LLM=qwen2_vl
#LLM_MODEL_SIZE=7B
LLM_MODEL_SIZE=2B
# lora only vit and tune adapter
ACTION_HEAD="dit_diffusion_policy" #act #transformer_diffusion
echo 'QWEN2 VLA-DiVLA version'
mnop=/home/jovyan/tzb/zhouzy/model_param/Qwen2-VL-${LLM_MODEL_SIZE}-Instruct
OUTPUT=/home/jovyan/tzb/zhouzy/model_param/qwen2_vla_param_${ACTION_HEAD}_results/ckpts/${TASK}/${LLM}_${LLM_MODEL_SIZE}/0501_Qwen2_no_filmlayer_w_reasoningaction_1share_route2_top_1
#DIT=/home/share/ljm/model_param/scaledp/zzy/resnet50_with_film_subreason/use_constant_0_mobile_franka_reasoning_w_vl_data_selected_DiT-L_320_240_128_1e-4_numsteps_25000_sub_1_use_task_norm_0_2025_01_21_05_31_07/policy_step_25000_2025-01-22_00-17-18.ckpt
DIT=/home/share/ljm/model_param/scaledp/wjj/resnet50_with_film_subreason/cosin/chunk_size_50_use_constant_0_3_cameras_all_data_1_17_DiT-H_32_1e-4_numsteps_60000_sub_1_use_task_norm_0_2025_01_17_14_14_27/policy_step_60000_2025-01-18_23-25-30.ckpt
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
  --action_dim 14 \
  --state_dim 14 \
  --chunk_size 50 \
  --policy_class $ACTION_HEAD \
  --policy_head_size "DiT_H" \
  --using_film False\
  --resume_from_checkpoint False \
  --with_llm_head True \
  --pretrain_image_size 320 \
  --task_name ${TASK} \
  --model_name_or_path $mnop \
  --version v0 \
  --freeze_vision_tower False \
  --freeze_backbone False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 50000 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 5000 \
  --save_total_limit 100 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "cosine_with_min_lr"\
  --min_lr 0\
  --logging_steps 4 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log \
  --group_by_modality_length False \
  --lora_enable False \
  --pretrain_dit_path $DIT \
  --vl_ratio 0.33 \
  --using_moe True \
  --init_moe True \
  --using_static_expert False \
  --using_shared_routed_expert True \
  --shared_expert_num 1 \
  --routed_expert_num 2 \
  --routed_top_k 1 \
  --reasoning_layer_id 16 \
  --output_expert_traffic False | tee $OUTPUT/log.log

#  --freeze_action_head True \
#  --close_head_forward True


#  --load_pretrain False
#  --llm_loss_weight 0.1
for dir in "$OUTPUT"/*/ ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${mnop}/preprocessor_config.json $dir
        cp ${mnop}/chat_template.json $dir
        # cp $OUTPUT/non_lora_trainables.bin $dir
    fi
done
mv ./60030.log $OUTPUT
echo $OUTPUT
