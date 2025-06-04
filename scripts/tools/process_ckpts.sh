#!/bin/bash

LLM=qwen2_vl
LLM_MODEL_SIZE=2B

# 源文件夹路径
source_dir=/home/jovyan/tzb/zhouzy/model_param/qwen2_vla_param_dit_diffusion_policy_results/ckpts/0426_mix_no_ocr_math/qwen2_vl_2B/0426_Qwen2_no_filmlayer_w_reasoningaction_route4_top_3
target_dir=/home/jovyan/tzb/zhouzy/model_param/qwen2_vla_param_dit_diffusion_policy_results/ckpts/0426_mix_no_ocr_math/qwen2_vl_2B_pure/0426_Qwen2_no_filmlayer_w_reasoningaction_route4_top_3
source_dir=/home/jovyan/tzb/zhouzy/model_param/qwen2_vla_param_dit_diffusion_policy_results/ckpts/0426_mix_no_ocr_math/qwen2_vl_2B/0426_Qwen2_no_filmlayer_w_reasoningaction_1share_route4_top_2
target_dir=/home/jovyan/tzb/zhouzy/model_param/qwen2_vla_param_dit_diffusion_policy_results/ckpts/0426_mix_no_ocr_math/qwen2_vl_2B_pure/0426_Qwen2_no_filmlayer_w_reasoningaction_1share_route4_top_2
mkdir -p $target_dir
# 目标文件夹路径


# 要排除的文件夹名的通配符
exclude_pattern="global_step*"

# 递归复制文件夹，并排除匹配指定通配符的文件夹
echo "copying checkpoints from >>" $source_dir "<< to >>"$target_dir "<<"
#rsync -av --exclude="$exclude_pattern" --exclude="$exclude_pattern/**" "$source_dir/" "$target_dir/"

for file in "$source_dir"/*; do
    # 检查文件名是否不匹配指定的模式
    if [[ "$(basename "$file")" != checkpoint-* ]] && [[ "$(basename "$file")" != non_lora_* ]]; then
        # 复制文件到目标目录
        cp -r "$file" "$target_dir"
    fi
done

echo 'tranfer checkpoints to non_lora_trainables.bin'
for dir in "$source_dir"/* ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint-25000"* || "$(basename "$dir")" == *"checkpoint-10" ]]; then
      if ! find "$dir" -mindepth 1 -type f -name "non_lora_trainables.bin" | grep -q .; then
        cd "$dir" || exit
	      mkdir -p ${target_dir}/$(basename "$dir")
#        python /home/jovyan/tzb/wjj/projects/dvla_mh_qwen2_vla/evaluate/zero_to_fp32.py ./ ${target_dir}/$(basename "$dir")/non_lora_trainables.bin
        python /home/jovyan/tzb/wjj/projects/dvla_mh_qwen2_vla/evaluate/zero_to_fp32.py ./ ./non_lora_trainables.bin
      fi
      for ff in "$dir"/* ; do
		    cp "$ff" ${target_dir}/$(basename "$dir")/
      done
      cd "/home/jovyan/tzb/zhouzy/model_param/Qwen2-VL-2B-Instruct/" || exit
      cp chat_template.json "$target_dir/$(basename "$dir")"
      cp preprocessor_config.json "$target_dir/$(basename "$dir")"
    fi
done


# 进入目标目录
# cd "/data/junjiewen/droid_results/checkpoint_all" || exit

# 压缩目录并指定相对路径
#tar -czvf "pythia_${LLM_MODEL_SIZE}.tar.gz" "pythia_${LLM_MODEL_SIZE}_pure"
#echo "compress checkpoints to /data/junjiewen/droid_results/checkpoint_all/pythia_${LLM_MODEL_SIZE}.tar.gz"
#
# 删除临时文件夹
#rm -r "pythia_${LLM_MODEL_SIZE}_pure"
#mv $target_dir /data/private/data/model_param/multi_head/${ACTION_HEAD}_results/checkpoint_all/pythia_${LLM_MODEL_SIZE}_pure/vanilla_pythia_pt_f_vit
