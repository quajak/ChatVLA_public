import os
from qwen2_vla.model_load_utils import load_model_for_eval
import json
import pickle

import torch
from torchvision import transforms
import cv2

import numpy as np
import time

from aloha_scripts.constants import FPS

from data_utils.utils import compute_dict_mean, set_seed, detach_dict, calibrate_linear_vel, \
    postprocess_base_action  # helper functions
from PIL import Image
from qwen_vl_utils import fetch_image
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel, AutoConfig, AutoModelForMaskedLM
from einops import rearrange
import torch_utils as TorchUtils
# import matplotlib.pyplot as plt
import sys
from policy_heads import *
# from cv2 import aruco
from qwen2_vla.utils.image_processing_qwen2_vla import *
from qwen2_vla.utils.processing_qwen2_vla import *
# ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

import copy


def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp


def get_obs(deplot_env_obs, stats):
    cur_right_rgb = deplot_env_obs['image']['21729895_left']  # camera_extrinsics image
    cur_left_rgb = deplot_env_obs['image']['29392465_left']  # camera_extrinsics image
    cur_wrist_rgb = deplot_env_obs['image']['12035220_left']  # camera_extrinsics image
    cur_wrist_rgb = cv2.resize(cur_wrist_rgb, (480, 270))

    w, h = 480, 270
    center = (w // 2, h // 2)
    angle = 180
    scale = 1.0
    M = cv2.getRotationMatrix2D(center, angle, scale)
    cur_wrist_rgb = cv2.warpAffine(cur_wrist_rgb, M, (w, h))

    cur_right_rgb = cv2.cvtColor(cur_right_rgb, cv2.COLOR_BGRA2BGR)
    cur_left_rgb = cv2.cvtColor(cur_left_rgb, cv2.COLOR_BGRA2BGR)
    cur_wrist_rgb = cv2.cvtColor(cur_wrist_rgb, cv2.COLOR_BGRA2BGR)

    cur_cartesian_position = np.array(deplot_env_obs['robot_state']['cartesian_position'])
    cur_gripper_position = np.expand_dims(np.array(deplot_env_obs['robot_state']['gripper_position']), axis=0)
    cur_state_np_raw = np.concatenate((cur_cartesian_position, cur_gripper_position))
    cur_state_np = pre_process(cur_state_np_raw, 'qpos', stats)

    # [128, 128, 3] np array
    right_rgb_img = cur_right_rgb  # deplot_env_obs['front']
    left_rgb_img = cur_left_rgb  # deplot_env_obs['wrist_1']
    wrist_rgb_img = cur_wrist_rgb

    cur_state = cur_state_np  # deplot_env_obs['state']
    cur_state = np.expand_dims(cur_state, axis=0)

    traj_rgb_np = np.array([left_rgb_img, right_rgb_img, wrist_rgb_img])

    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))

    traj_rgb_np = np.array([[cv2.cvtColor(np.transpose(img, (1, 2, 0)), cv2.COLOR_BGR2RGB) for img in traj_rgb_np[0]]])

    if im_size == 320:  # resize to 320
        traj_rgb_np = np.array([[cv2.resize(img, (320, 240)) for img in traj_rgb_np[0]]])

    traj_rgb_np = np.transpose(traj_rgb_np, (0, 1, 4, 2, 3))
    return cur_state_np_raw, cur_state, traj_rgb_np

def get_observation_from_file(stats, raw_state_name, img_name):
    cur_state_np_raw = np.load(raw_state_name)
    cur_state = pre_process(cur_state_np_raw, 'qpos', stats)
    cur_state = np.expand_dims(cur_state, axis=0)

    img = np.array(Image.open(img_name).convert("RGB"))
    img = cv2.resize(img, (320, 240))
    # Convert to shape (1, 1, 3, height, width)
    traj_rgb_np = np.transpose(img, (2, 0, 1))  # Convert to (3, height, width)
    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=0)  # Add batch dim: (1, 3, height, width)
    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=0)  # Add view dim: (1, 1, 3, height, width)
    return cur_state, traj_rgb_np



def convert_actions(pred_action):
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    # print(f'4. after convert pred_action: {pred_action}')

    return pred_action


class qwen2_vla_policy:
    def __init__(self, policy_config, data_args=None):
        super(qwen2_vla_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_path = policy_config["model_path"]
        self.tokenizer, self.policy, self.multimodal_processor, self.context_len = load_model_for_eval(
            model_path=model_path,
            model_base=model_base, policy_config=policy_config)

        self.tokenizer.add_special_tokens({'additional_special_tokens': ["[SOA]"]})

        self.config = AutoConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)

    def datastruct_droid2qwen2vla(self, raw_lang, views):
        messages = [
            {
                "role": "user",
                "content": [
                ],
            },
        ]
        for i in range(len(views)):
            messages[0]['content'].append(
                {
                    "type": "image",
                    "image": None,
                }
            )
        messages[0]['content'].append({"type": "text", "text": f""})
        messages[0]['content'][-1]['text'] = raw_lang
        # messages[1]['content'] = sample['reasoning'] + "Next action:"
        # print(sample['obs']['raw_language'].decode('utf-8'))
        return messages

    def process_batch_to_qwen2_vla(self, curr_image, robo_state, raw_lang, views=[0,1,2]):

        if len(curr_image.shape) == 5:  # 1,2,3,270,480
            curr_image = curr_image.squeeze(0)

        messages = self.datastruct_droid2qwen2vla(raw_lang, views)
        image_data = torch.chunk(curr_image, curr_image.shape[0], dim=0)
        image_list = []
        for i, each in enumerate(image_data):
            ele = {
            }
            each = Image.fromarray(each.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
            ele['image'] = each
            if i == 2 and len(views) > 1:
                ele['resized_height'] = 56
                ele['resized_width'] = 56
            else:
                ele['resized_height'] = 240
                ele['resized_width'] = 320
            each = fetch_image(ele)
            if i in views:
                image_list.append(torch.from_numpy(np.array(each)))

        image_data = image_list
        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        video_inputs = None
        model_inputs = self.multimodal_processor(
            text=text,
            images=image_data,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        data_dict = dict(states=robo_state)
        for k, v in model_inputs.items():
            data_dict[k] = v
        return data_dict


def eval_bc(policy, policy_config, raw_lang=None, views=[0], eval_in_vqa=False, img_dir=None, output_dir=None):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    assert output_dir is not None, "output_dir is None!!!!!!"
    set_seed(0)

    rand_crop_resize = False
    model_config = policy.config.policy_head_config

    policy.policy.eval()

    import pickle
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # Extract model name from path
    model_name = policy_config['model_path'].split('/')[-2]

    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'diffusion' in policy_config["action_head"] or 'vqbet' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    num_queries = 16
    img_list = os.listdir(img_dir)
    img_list.sort()
    
    # Create a list to store results
    results = []
    
    for name in img_list[:10]:
        with torch.inference_mode():
            robot_state, traj_rgb_np = get_observation_from_file(stats, raw_state_name='raw_state.npy',img_name=os.path.join(img_dir, name))

            robot_state = torch.from_numpy(robot_state).float().cuda()
            curr_image = torch.from_numpy(traj_rgb_np).float().cuda()

            if rand_crop_resize:
                print('rand crop resize is used!')
                original_size = curr_image.shape[-2:]
                print('original size', original_size)
                ratio = 0.95
                curr_image = curr_image[...,
                             int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                             int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                curr_image = curr_image.squeeze(0)
                resize_transform = transforms.Resize(original_size, antialias=True)
                curr_image = resize_transform(curr_image)
                curr_image = curr_image.unsqueeze(0)

            batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang, views)
            all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True,
                                                              tokenizer=policy.tokenizer)
            action_chunks = torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:num_queries]
            
            # Process all actions
            processed_actions = []
            for raw_action in action_chunks:
                # print(f"raw action size: {raw_action.size()}")
                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                action = post_process(raw_action)
                # print(f"after post_process action size: {action.shape}")
                action = convert_actions(action.squeeze())
                # print(f'pred action: {outputs}{action}')
                processed_actions.append(action.tolist())
            
            # Store result for this image
            result = {
                'image_name': name,
                'actions': processed_actions,  # List of all processed actions
                'raw_lang': raw_lang,
                'model_output': outputs
            }
            results.append(result)
    
    # Save all results to a JSON file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'prediction_results_{model_name}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    return


if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    action_head = 'scale_dp_policy'  # 'droid_diffusion'
    model_size = '2B'
    policy_config = {
        # "model_path": "/media/eai/ADDS-4/zhouzy/model_Param/franka_math_vl/0416_Qwen2_lora_no_filmlayer_w_reasoningaction_share1_route2_top_1/checkpoint-25000",
        # "model_base": "/media/eai/ADDS-4/zhouzy/model_Param/Qwen2-VL-2B-Instruct",
        "model_path": "/media/jz08/ADDS-4/zhouzy/model_Param/franka_math_vl/0416_Qwen2_lora_no_filmlayer_w_reasoningaction_share1_route4_top_2/checkpoint-10000",
        "model_base": "/media/jz08/ADDS-4/zhouzy/model_Param/Qwen2-VL-2B-Instruct",
        "pretrain_path": None,
        "enable_lora": True,
        "temp_agg": False,
        "action_head": action_head,
        'model_size': model_size,
        'save_model': False,
        "tinyvla": False,
        "tinyvla_with_output": False

    }
    global im_size
    im_size = 320  # default 480
    select_one = False  # select one embedding or using all
    eval_in_vqa = False
    views = [0] ### for single image, this can not be modified!!
    #################### math
    raw_lang = 'Solve the equation and move to the correct answer.'
    raw_lang = 'Solve the equation and point out the correct answer.'
    # raw_lang = 'Move any objects on the right panel to the left box.'

    policy = qwen2_vla_policy(policy_config)
    img_dir = '/home/jz08/LMUData/images/MMRO_mini_0408_wrist'
    output_dir = '/home/jz08/zhouzy/data/check_ckpt'
    eval_bc(policy, policy_config, raw_lang=raw_lang, views=views, eval_in_vqa=eval_in_vqa, img_dir=img_dir, output_dir=output_dir)

    print()
    exit()
