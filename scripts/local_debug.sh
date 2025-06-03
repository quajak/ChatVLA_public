#!/bin/bash

deepspeed --master_port 29604 --num_gpus=2 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
  --model_name_or_path /home/jz08/zhouzy/model_Param/Qwen2-VL-2B-Instruct \
  --output_dir /home/jz08/zhouzy/debug_pretrain_nodeepspeed\
  --task_name "local_debug_data_zzy"\
  --version v0\
  --freeze_vision_tower True\
  --freeze_backbone True\
  --bf16 True\
  --num_train_epochs 3\
  --max_steps 2 \
  --per_device_train_batch_size 2\
  --gradient_accumulation_steps 1\
  --save_strategy "steps"\
  --save_steps 1\
  --do_eval False\
  --evaluation_strategy "no"\
  --save_total_limit 15\
  --learning_rate 2e-4\
  --non_lora_lr 2e-5\
  --weight_decay 0.\
  --warmup_ratio 0.005\
  --lr_scheduler_type "constant"\
  --logging_steps 1\
  --tf32 True\
  --gradient_checkpointing True\
  --dataloader_num_workers 0\
  --logging_dir ./loglog\
  --report_to tensorboard\
  --with_llm_head True\
  --use_reasoning True\
  --resume_from_checkpoint False\
  --using_film False\
  --action_dim 10\
  --state_dim 7\
  --delta_control False\
  --group_by_modality_length False\
  --using_moe True\
  --policy_class dit_diffusion_policy\
  --lora_enable True\
  --with_flash_attention False\
  --vl_ratio -1\
  --is_local_debug False\
  --output_expert_traffic False\
  --using_deepspeed_moe False\
  --routed_expert_num 2 \
  --lora_module "vitllm"



mv ./60030.log $OUTPUT
echo $OUTPUT
