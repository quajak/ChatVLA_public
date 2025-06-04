import os
from qwen2_vla.model_load_utils import load_model_for_eval

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


def time_ms():
    return time.time_ns() // 1_000_000


def convert_actions(pred_action):
    cur_xyz = pred_action[:3]
    cur_rot6d = pred_action[3:9]
    cur_gripper = np.expand_dims(pred_action[-1], axis=0)

    cur_rot6d = torch.from_numpy(cur_rot6d).unsqueeze(0)
    cur_euler = TorchUtils.rot_6d_to_euler_angles(rot_6d=cur_rot6d, convention="XYZ").squeeze().numpy()
    pred_action = np.concatenate((cur_xyz, cur_euler, cur_gripper))
    print(f'4. after convert pred_action: {pred_action}')

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
        image_data = torch.chunk(curr_image, curr_image.shape[0], dim=0)  # left, right ,wrist
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


def eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=None, select_one=False, views=[0,1,2], eval_in_vqa=False):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)

    rand_crop_resize = False
    model_config = policy.config.policy_head_config

    temporal_agg = policy_config['temp_agg']
    action_dim = getattr(model_config, 'input_dim', 10)
    state_dim = getattr(model_config, 'state_dim', 7)

    policy.policy.eval()

    import pickle
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    if policy_config["action_head"].lower() == 'act':
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    elif 'diffusion' in policy_config["action_head"] or 'vqbet' in policy_config["action_head"]:
        post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    env = deploy_env

    query_frequency = 16
    if temporal_agg:
        query_frequency = 1
        num_queries = int(query_frequency)
    else:
        query_frequency = int(query_frequency / 2)
        num_queries = query_frequency
        from collections import deque
        action_queue = deque(maxlen=num_queries)

    max_timesteps = int(1000 * 10)  # may increase for real-world tasks

    for rollout_id in range(1000):
        rollout_id += 0
        env.reset(randomize=False)
        print(f"env has reset!")

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + num_queries, action_dim],
                                           dtype=torch.bfloat16).cuda()

        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            for t in range(max_timesteps):
                if t % 50 == 1:
                    a = input("q means next eval:")
                    if a == 'q':
                        env.reset(randomize=False)
                        lang_in = input("Input the raw_lang(q and enter mean using default):")
                        if lang_in != 'q' or lang_in != '':
                            raw_lang = lang_in
                            print(raw_lang)
                        action_queue = deque(maxlen=num_queries)
                        break
                    if a == 'c':
                        env.reset(randomize=False)
                        input("Input any key for continue:")
                        action_queue = deque(maxlen=num_queries)
                        break

                obs = deploy_env.get_observation()

                cur_state_np_raw, robot_state, traj_rgb_np = get_obs(obs, stats)
                print("curent robot state!!!!!!!!!!!!!!1", obs['robot_state']['cartesian_position'])

                robot_state = torch.from_numpy(robot_state).float().cuda()

                # todo add resize&crop to wrist camera
                if t % query_frequency == 0:
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
                if t == 0:
                    # warm up
                    for _ in range(2):
                        batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang, views)
                        if policy_config['tinyvla'] and policy_config["tinyvla_with_output"]:
                            policy.policy.evaluate_tinyvla_with_output(**batch, is_eval=True,
                                                                       tokenizer=policy.tokenizer)
                        elif policy_config['tinyvla']:
                            policy.policy.evaluate_tinyvla(**batch, is_eval=True,
                                                           tokenizer=policy.tokenizer)
                        else:
                            all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True,
                                                                          tokenizer=policy.tokenizer, eval_in_vqa=eval_in_vqa)
                            print("*" * 50)
                            print(outputs)

                    print('network warm up done')

                if t % query_frequency == 0:
                    batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang, views)
                    if policy_config['tinyvla'] and policy_config["tinyvla_with_output"]:
                        all_actions, outputs = policy.policy.evaluate_tinyvla_with_output(**batch, is_eval=True,
                                                                                          tokenizer=policy.tokenizer)
                    elif policy_config['tinyvla']:
                        all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True,
                                                                              tokenizer=policy.tokenizer)
                    else:
                        all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True,
                                                                      tokenizer=policy.tokenizer, eval_in_vqa=eval_in_vqa)
                    if not temporal_agg:
                        action_queue.extend(
                            torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:num_queries])

                if temporal_agg:
                    print(f"all_actions: {all_actions.size()}")
                    print(f"all_time_actions: {all_time_actions.size()}")
                    print(f"t: {t}, num_queries:{num_queries}")
                    all_time_actions[[t], t:t + num_queries] = all_actions[:, :num_queries, :]
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = action_queue.popleft()

                print(f"raw action size: {raw_action.size()}")
                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                action = post_process(raw_action)
                print(f"after post_process action size: {action.shape}")
                action = convert_actions(action.squeeze())
                print(f'step {t}, pred action: {outputs}{action}')
                _ = deploy_env.step(action)
                time.sleep(0.05)

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')

    return


if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    sys.path.insert(0, "/home/eai/Dev-Code/droid")
    from droid.robot_env import RobotEnv

    action_head = 'scale_dp_policy'
    model_size = '2B'
    policy_config = {
        ### qwen2 vla 2b with vl data
        # # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< chatvla >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # "model_path": "/media/eai/ADDS-4/zhouzy/model_Param/mobile_franka_reasoning_w_vl_data/mobile_franka_data_selected/Qwen2_2B_v0_w_llava_finetune_moe_2parts_w_lora_step_2_cosine_2e_5_s1_8k_ro3_vl1/checkpoint-10000",
        # "model_base": "/media/eai/ADDS-4/zhouzy/model_Param/Qwen2-VL-2B-Instruct",
        #
        # "pretrain_path": "/media/eai/ADDS-4/zhouzy/model_Param/mobile_franka_reasoning_w_vl_data/mobile_franka_data_selected/Qwen2_2B_v0_w_llava_finetune_moe_2parts_w_lora_step_1_constant_1e_4/checkpoint-8000",
        #
        # "enable_lora": True,
        # "temp_agg": False,
        # "action_head": action_head,
        # 'model_size': model_size,
        # 'save_model': False,
        # "tinyvla": False,
        # "tinyvla_with_output": False
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< tinyvla >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # "model_path": "/media/eai/ADDS-4/zhouzy/model_Param/Qwen2_2B_tinyvla_lora_unet/checkpoint-50000",
        # "model_base": "/media/eai/ADDS-4/zhouzy/model_Param/Qwen2-VL-2B-Instruct",
        # "pretrain_path": None,
        # "enable_lora": True,
        # "temp_agg": False,
        # "action_head": 'unet_diffusion_policy',
        # 'model_size': model_size,
        # 'save_model': False,
        # "tinyvla": True,
        # "tinyvla_with_output": False
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< divla mix >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # "model_path": "/media/eai/MyPassport/model_Param/Qwen2_2B_divla_same_reasoning_lora/checkpoint-50000",
        # "model_base": "/media/eai/MyPassport/model_Param/Qwen2-VL-2B-Instruct",
        # "pretrain_path": None,
        # "enable_lora": True,
        # "temp_agg": False,
        # "action_head": action_head,
        # 'model_size': model_size,
        # 'save_model': False,
        # "tinyvla": False,
        # "tinyvla_with_output": False
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< divla no mix >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # "model_path":"/media/eai/ADDS-4/zhouzy/model_Param/Qwen2_2B_divla_lora_unet/checkpoint-50000",
        # "model_base": "/media/eai/ADDS-4/zhouzy/model_Param/Qwen2-VL-2B-Instruct",
        # "pretrain_path": None,
        # "enable_lora": True,
        # "temp_agg": False,
        # "action_head": 'unet_diffusion_policy',
        # 'model_size': model_size,
        # 'save_model': False,
        # "tinyvla": False,
        # "tinyvla_with_output": False
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< cyf >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        # "model_path": "/media/eai/MyPassport/model/Qwen2_2B_v0_w_llava_finetune_moe_2parts_w_lora_step_2_cosine_2e_5_s1_8k_ro3_vl1/checkpoint-15000",
        # "model_base": "/media/eai/ADDS-4/zhouzy/model_Param/Qwen2-VL-2B-Instruct",
        #
        # "pretrain_path": "/media/eai/ADDS-4/zhouzy/model_Param/mobile_franka_reasoning_w_vl_data/mobile_franka_data_selected/Qwen2_2B_v0_w_llava_finetune_moe_2parts_w_lora_step_1_constant_1e_4/checkpoint-8000",
        #
        # "enable_lora": True,
        # "temp_agg": False,
        # "action_head": action_head,
        # 'model_size': model_size,
        # 'save_model': False,
        # "tinyvla": False,
        # "tinyvla_with_output": False
        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< math >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        ### no moe  + reason_as_cond
        # "model_path": "/media/eai/ADDS-4/zhouzy/model_Param/franka_math_vl/Qwen2_2B_lora_reasoning_as_cond_no_moe_w_vl_nofilmlayer_loraattn/checkpoint-10000",
        # "model_base": "/media/eai/ADDS-4/zhouzy/model_Param/Qwen2-VL-2B-Instruct",
        #
        # "pretrain_path": None,
        # "enable_lora": True,
        # "temp_agg": False,
        # "action_head": action_head,
        # 'model_size': model_size,
        # 'save_model': False,
        # "tinyvla": False,
        # "tinyvla_with_output": False


        "model_path": "/media/eai/ADDS-4/zhouzy/model_Param/franka_math_vl/0416_Qwen2_lora_no_filmlayer_w_reasoningaction_share1_route2_top_1/checkpoint-25000",
        "model_base": "/media/eai/ADDS-4/zhouzy/model_Param/Qwen2-VL-2B-Instruct",
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
    views = [2] # wrist only
    eval_in_vqa=False
    #################### math
    raw_lang = 'Solve the equation and move to the correct answer.'
    raw_lang = 'Solve the equation and point out the correct answer.'
    # raw_lang = 'Move any objects on the right panel to the left box.'
    ########## no scene
    # raw_lang = 'Move the bread to the empty plate.'
    # raw_lang = 'Hang on the cup.'
    # raw_lang = 'Move the tennis ball to the tennis can.'
    # raw_lang = 'Stack the green cube onto the pink cube.'
    # raw_lang = 'Take away the lid of the box and place it on the table.'

    # ############ toy scence
    # raw_lang = 'Move the orange building-block to the basket.'
    # raw_lang = 'Open the drawer.'
    # raw_lang = 'Put the toy into the drawer.'
    # raw_lang = 'Close the drawer.'
    # raw_lang = 'Move the semicircle building-block to the basket.'
    # raw_lang = 'Move the rectangle building-block to the basket.'
    # raw_lang = 'Move all the building-blocks to the basket.'
    # raw_lang = 'Stack the building-blocks sequentially.'
    # raw_lang = 'Put the spider man into the drawer.'
    # raw_lang = 'Move all the toy animals to the left box and all the building-blocks to the right basket.'
    ############## ood
    # raw_lang = 'Put the yellow pikachu into the drawer.'
    # raw_lang = 'Put the spider man into the drawer. '


    ################ bath room
    # raw_lang = 'Put the soap to the soap box.'

    # raw_lang = 'Pick up the tooth-paste and put it on the table.'
    # raw_lang = 'Remove the towel from the shelf.'
    # raw_lang = 'Pick up the cup and hang it on the shelf.'


    ############## kitchen
    # raw_lang = 'Pick up the bread in the refrigerator.'
    # raw_lang = 'Move the bread from the pot to the plate.'

    ############## breakfast
    # raw_lang = 'Move the banana onto the plate.'
    # raw_lang = 'Flip the cup and put it on the tablecloth.'
    # raw_lang = 'Get the plate from the shelf and place it on the tablecloth.'
    # raw_lang = 'Move the bread to the empty plate.'


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    # policy = None
    policy = qwen2_vla_policy(policy_config)

    policy_timestep_filtering_kwargs = {'action_space': 'cartesian_position', 'gripper_action_space': 'position',
                                        'robot_state_keys': ['cartesian_position', 'gripper_position',
                                                             'joint_positions']}
    # resolution (w, h)
    # todo H W or W H?

    policy_camera_kwargs = {
        'hand_camera': {'image': True, 'concatenate_images': False, 'resolution': (480, 270), 'resize_func': 'cv2'},
        'varied_camera': {'image': True, 'concatenate_images': False, 'resolution': (480, 270), 'resize_func': 'cv2'}}

    deploy_env = RobotEnv(
        action_space=policy_timestep_filtering_kwargs["action_space"],
        gripper_action_space=policy_timestep_filtering_kwargs["gripper_action_space"],
        camera_kwargs=policy_camera_kwargs
    )

    deploy_env._robot.establish_connection()
    deploy_env.camera_reader.set_trajectory_mode()
    deploy_env.reset(randomize=False)
    eval_bc(policy, deploy_env, policy_config, save_episode=True, num_rollouts=1, raw_lang=raw_lang,
            select_one=select_one, views=views, eval_in_vqa=eval_in_vqa)

    print()
    exit()

# [0.5553438067436218, 0.0022895748261362314, 0.6198290586471558, -3.119706407105779, -0.006210746497147035, -0.025821790776125078]