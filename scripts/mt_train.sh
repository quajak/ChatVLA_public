#!/bin/bash
LLM=qwen2_vl
LLM_MODEL_SIZE=7B
# LLM_MODEL_SIZE=2_8B
# lora only vit and tune adapter
ACTION_HEAD=unet_diffusion_policy  #act #transformer_diffusion

echo 'Pythia Pythia Pythia Pythia'
PRETRAIN=/gpfs/private/tzb/wjj/model_param/multi_head2/${ACTION_HEAD}_results/checkpoint_all/${LLM}_${LLM_MODEL_SIZE}/vanilla_qwen2_vla_pt_f_vit/Qwen2_vla-v0-robot-action-38k_droid_pretrain_lora_all_wo_film/checkpoint-40000
mnop=/gpfs/private/tzb/wjj/model_param/Qwen2-VL-${LLM_MODEL_SIZE}-Instruct
OUTPUT=/gpfs/private/tzb/wjj/model_param/multi_head2/${ACTION_HEAD}_results/checkpoint_all/${LLM}_${LLM_MODEL_SIZE}/vanilla_qwen2_vla_pt_f_vit/Qwen2_vla-v0-robot-action-11_1_reasoning_all_tasks_lora_all_film_residual
if [ -d "$OUTPUT" ]; then
   echo 'output exists'
else
   echo '!!output not exists!!'
   mkdir -p $OUTPUT
fi

mkdir -p $OUTPUT/src
cp -r ./scripts $OUTPUT/
cp -r ./data_utils $OUTPUT/src/
cp -r ./qwen2_vla $OUTPUT/src/
cp -r ./policy_heads $OUTPUT/src/

deepspeed --master_port 29604 --num_gpus=8 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
  --use_reasoning True \
  --lora_enable True \
  --resume_from_checkpoint False \
  --select_seg_token_mask False \
  --lora_module "vit llm" \
  --load_pretrain False \
  --model_pretrain $PRETRAIN\
  --with_external_vit False \
  --using_film True \
  --using_xattn False \
  --policy_head_type $ACTION_HEAD \
  --with_llm_head True \
  --with_text_fcs False \
  --image_size_wrist 224 \
  --only_using_input_embeddings False \
  --pretrain_image_size 320 \
  --lora_r 64 \
  --lora_alpha 256 \
  --episode_first False \
  --non_lora_lr 3e-5 \
  --task_name "11_1_reasoning_all_tasks" \
  --model_name_or_path $mnop \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower True \
  --freeze_backbone True \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 45000 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 10000 \
  --save_total_limit 50 \
  --learning_rate 2e-4 \
  --weight_decay 0. \
  --warmup_ratio 0.0002 \
  --lr_scheduler_type "cosine" \
  --logging_steps 100 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --policy_class $ACTION_HEAD \
  --concat "token_cat" \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log > $OUTPUT/output.log

#/data/private/data/llava_data/franka_kitchen_finetune/left_cap2/left_cap2_50k.json
#/data/team/zhumj/data/finetune/data/llava_v1_5_mix665k.json
# cp openai/clip-vit-large-patch14-336/preprocessor_config.json $OUTPUT
for dir in "$OUTPUT"/*/ ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp /data/private/weights/openai/clip-vit-large-patch14-336/preprocessor_config.json $dir
        # cp $OUTPUT/non_lora_trainables.bin $dir
    fi
done
mv ./60030.log $OUTPUT
echo $OUTPUT
