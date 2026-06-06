#!/bin/bash
#SBATCH --job-name=chatvla_val4
#SBATCH --gres=gpu:l40s:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200GB
#SBATCH --time=00:40:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.out

# Short validation: confirm the 4x L40S ZeRO-2 (no CPU offload) config fits in
# memory and measure step time. 30 steps, no checkpoints, no wandb.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load cuda/13.2
source .venv/bin/activate

TASK=libero_spatial
ACTION_HEAD="scale_dp_policy"
MNOP=/scratch/gerigkja/weights/Qwen2-VL-2B-Instruct
OUTPUT=/scratch/gerigkja/ChatVLA_public/val4
mkdir -p $OUTPUT
export XDG_CACHE_HOME=/scratch/gerigkja/lerobotcache

# Sample all 4 GPUs once per second for the duration of the run.
( for i in $(seq 1 600); do
    nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used --format=csv,noheader,nounits | paste -sd';'
    sleep 1
  done > $OUTPUT/gpu_samples.csv ) &
MONPID=$!

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
deepspeed --master_port 29608 --num_gpus=4 --num_nodes=1 ./train_vla.py \
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
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --save_strategy "no" \
  --max_steps 30 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.0 \
  --lr_scheduler_type "cosine_with_min_lr" \
  --min_lr 0 \
  --logging_steps 1 \
  --gradient_checkpointing True \
  --dataloader_num_workers 4 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 4 \
  --output_dir $OUTPUT \
  --report_to none \
  --logging_dir $OUTPUT/log \
  --lora_enable False \
  --using_moe False \
  --torch_compile_model False

kill $MONPID 2>/dev/null

echo "=========================================================="
echo "PER-GPU peak memory (MiB) and util/power summary:"
python3 - <<'PY'
import csv, collections
path="/scratch/gerigkja/ChatVLA_public/val4/gpu_samples.csv"
gpu=collections.defaultdict(lambda: {"util":[], "pow":[], "mem":[]})
for line in open(path):
    for rec in line.strip().split(';'):
        f=[x.strip() for x in rec.split(',')]
        if len(f)!=4: continue
        i,u,p,m=f
        try: i=int(i); u=float(u); p=float(p); m=float(m)
        except: continue
        if m<6000: continue          # skip pre-load / idle samples
        gpu[i]["util"].append(u); gpu[i]["pow"].append(p); gpu[i]["mem"].append(m)
for i in sorted(gpu):
    d=gpu[i]; n=len(d["util"]) or 1
    print(f"GPU{i}: mem_peak={max(d['mem']):.0f}MiB  util_mean={sum(d['util'])/n:4.1f}%  "
          f"util@0%={sum(1 for x in d['util'] if x==0)/n*100:4.1f}%  power_mean={sum(d['pow'])/n:5.1f}W")
PY
echo $OUTPUT
