import time
import numpy as np
from PIL import Image
import torch

from eval_act_vlm_real_franka import vla_paligemma_policy

action_head = 'droid_diffusion'
model_size = '70M'
policy_config = {
    "model_path": f"/media/rl/HDD/data/{action_head}_results/pythia_{model_size}/vanilla_pythia_pt_f_vit/llavaPythia-v0-robot-action-5mt_tennis_mug_box_drawer_new_cube_lora_all_20000/checkpoint-20000",
    # "model_path": "/media/eai/WJJ1T/droid/results/pythia_410M/vanilla_pythia_37k_pretrain_lora/checkpoint-10000",
    "model_base": f"/media/rl/HDD/data/weights/pythia/pythia_{model_size}/vanilla_pythia_pt_f_vit/llavaPythia-v0-finetune",
    "enable_lora": True,
    "conv_mode": "pythia",
    "action_head": action_head,
}

policy = vla_paligemma_policy(policy_config)


def measure_inference_time(policy, num_iters=100):
    # Grab image input & format prompt
    # image: Image.Image = get_from_camera(...)
    image = torch.from_numpy(np.random.randint(0, 255, size=(1, 2, 3, 336, 336)).astype(np.uint8)).to(dtype=torch.float16)
    robot_state = torch.from_numpy(np.random.randint(0, 5, size=(1, 7))).to(dtype=torch.float16)
    prompt = "In: What action should the robot take to {<INSTRUCTION>}?\nOut:"
    start = time.time()
    # batch = policy.process_batch_to_llava(image, robot_state, prompt)
    for _ in range(num_iters):
        batch = policy.process_batch_to_llava(image, robot_state, prompt)

        policy.policy(**batch, eval=True)
    end = time.time()
    total_time = end - start
    return num_iters / total_time

# 3.5608751283539193 * 16(8) with multiple batch process

print(measure_inference_time(policy, num_iters=100))

