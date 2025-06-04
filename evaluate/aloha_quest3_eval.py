import os.path

import numpy as np
import torch
from scipy.interpolate import interp1d

torch.set_default_tensor_type(torch.cuda.HalfTensor)
from torchvision import transforms
from aloha_scripts.utils import *
import time
from data_utils.utils import set_seed
from einops import rearrange
# from kalman_filter import kalman_filter_actions, smooth_actions
import sys
from policy_heads import *
from qwen2_vla.utils.image_processing_qwen2_vla import *
# from paligemma_vla.utils.processing_paligemma_vla import *
from qwen2_vla.utils.processing_qwen2_vla import *
from vla_policy import qwen2_vla_policy
import cv2

def get_image(ts, camera_names, rand_crop_resize=False):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)

    if rand_crop_resize:
        print('rand crop resize is used!')
        original_size = curr_image.shape[-2:]
        ratio = 0.95
        curr_image = curr_image[..., int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
        curr_image = curr_image.squeeze(0)
        resize_transform = transforms.Resize(original_size, antialias=True)
        curr_image = resize_transform(curr_image)
        curr_image = curr_image.unsqueeze(0)
    return curr_image


def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp

# def interpolate_action_batch(batch_actions: np.ndarray, target_len: int, kind='linear'):
#     B, T, D = batch_actions.shape
#     x_old = np.linspace(0, 1, T)
#     x_new = np.linspace(0, 1, target_len)
#     output = np.zeros((B, target_len, D))
#
#     for b in range(B):
#         for d in range(D):
#             f = interp1d(x_old, batch_actions[b, :, d], kind=kind)
#             output[b, :, d] = f(x_new)
#
#     return output
# import torch

import torch

def interpolate_action(batch_actions: torch.Tensor, target_len: int):
    """
    对 batch 动作序列 (B, T, D) 进行线性插值，输出 (B, target_len, D)
    """
    B, T, D = batch_actions.shape
    device = batch_actions.device
    dtype = batch_actions.dtype

    x_old = torch.linspace(0, 1, T, device=device, dtype=dtype)
    x_new = torch.linspace(0, 1, target_len, device=device, dtype=dtype)

    idx = torch.searchsorted(x_old, x_new, right=True).clamp(min=1, max=T - 1)  # (target_len,)

    x0 = x_old[idx - 1]  # (target_len,)
    x1 = x_old[idx]      # (target_len,)

    t = ((x_new - x0) / (x1 - x0 + 1e-8)).unsqueeze(0).unsqueeze(-1)

    b_idx = torch.arange(B, device=device).unsqueeze(1)  # (B, 1)

    y0 = batch_actions[b_idx, idx.unsqueeze(0) - 1]  # gather left
    y1 = batch_actions[b_idx, idx.unsqueeze(0)]      # gather right

    output = (1 - t) * y0 + t * y1
    return output  # (B, target_len, D)



def get_obs(deplot_env_obs, stats, time=0, camera_views=[0,1,2]):
    # cur_bottom_rgb = deplot_env_obs['images']['cam_bottom']
    cur_top_rgb = deplot_env_obs['images']['cam_top']
    cur_left_rgb = deplot_env_obs['images']['cam_left_wrist']
    cur_right_rgb = deplot_env_obs['images']['cam_right_wrist']

    # cur_bottom_rgb = cv2.cvtColor(cur_bottom_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]
    cur_top_rgb = cv2.cvtColor(cur_top_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]
    cur_left_rgb = cv2.cvtColor(cur_left_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]
    cur_right_rgb = cv2.cvtColor(cur_right_rgb, cv2.COLOR_BGRA2BGR)[:, :, ::-1]

    cur_joint_positions = deplot_env_obs['joints']

    cur_state_np = pre_process(cur_joint_positions, 'qpos', stats)

    cur_state = cur_state_np  # deplot_env_obs['state']
    cur_state = np.expand_dims(cur_state, axis=0)

    # [2, 1, 128, 128, 3]
    # [2, 480, 480, 3]
    if len(camera_views) == 1:
        traj_rgb_np = np.array([cur_top_rgb])
    else:
        traj_rgb_np = np.array([cur_top_rgb, cur_left_rgb, cur_right_rgb])
    temp = []
    for each in traj_rgb_np:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        result, encoded_image = cv2.imencode('.jpg', each, encode_param)
        decompressed_image = cv2.imdecode(encoded_image, 1)
        temp.append(decompressed_image)
    traj_rgb_np = np.array(temp)

    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))

    print("#" * 50)
    print(traj_rgb_np.shape)

    return cur_joint_positions, cur_state, traj_rgb_np


def eval_bc(policy, deploy_env, policy_config, raw_lang=None, query_frequency=25, use_interpolate=False):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)
    action_list = []
    rand_crop_resize = True
    model_config = policy.config.policy_head_config

    state_dim = model_config['state_dim']

    policy.policy.eval()

    import pickle
    paths = policy_config['model_path'].split('/')[:-1]
    if 'checkpoint' in paths[-1]:
        paths = paths[:-1]
    stats_path = os.path.join("/".join(paths), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    # if 'fold_shirt' in stats.keys():
    #     if 'fold' in raw_lang.lower():
    #         stats = stats['fold_shirt']
    #     elif 'tablewares' in raw_lang.lower():
    #         stats = stats['clean_table']
    #     else:
    #         stats = stats['other']

    # post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    post_process = lambda a: ((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min']

    action_queue = deque(maxlen=query_frequency)

    max_timesteps = int(1000 * 10)  # may increase for real-world tasks
    time_cur = -1
    time_pre = -1
    for rollout_id in range(1000):

        rollout_id += 0

        print(f"env has reset!")
        robot_state_history = np.zeros((max_timesteps, state_dim))
        image_list = []  # for visualization
        pre_chunk_action = None
        with torch.inference_mode():
            time0 = time.time()
            for t in range(max_timesteps):
                if t % query_frequency == 0:
                    process_time1 = time.time()
                    # obs = deploy_env.get_obs()
                    # cur_state_np_raw, robot_state, traj_rgb_np = get_obs(obs, stats, time=t, camera_views=policy_config['camera_views'])
                    obs = deploy_env.get_obs(['cam_left_wrist', 'cam_right_wrist', 'cam_top', 'joints'])
                    # 'cam_left_wrist', 'cam_right_wrist', 'cam_top', 'joints', 'base'
                    cur_state_np_raw, robot_state, traj_rgb_np = get_obs(obs, stats, time=t,
                                                                         camera_views=policy_config['camera_views'])
                    robot_state_history[t] = cur_state_np_raw
                    robot_state = torch.from_numpy(robot_state).float().cuda()
                    curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
                    if rand_crop_resize:
                        print('rand crop resize is used!')
                        original_size = curr_image.shape[-2:]
                        ratio = 0.95
                        curr_image = curr_image[...,
                                     int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                     int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                        curr_image = curr_image.squeeze(0)
                        resize_transform = transforms.Resize((240, 320), antialias=True)
                        curr_image = resize_transform(curr_image)
                        curr_image = curr_image.unsqueeze(0)

                    image_list.append(curr_image)

                    batch = policy.process_batch_to_vla(curr_image, robot_state, raw_lang)

                    if policy_config['tinyvla']:
                        all_actions, outputs = policy.policy.evaluate_tinyvla(**batch, is_eval=True)
                    else:
                        all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, tokenizer=policy.tokenizer)

                    # all_actions = smooth_actions(all_actions)
                    # all_actions = kalman_filter_actions(all_actions.to(torch.float32)[0].cpu().numpy())
                    # all_actions = torch.from_numpy(all_actions).unsqueeze(0)
                    if pre_chunk_action == None:
                        pre_chunk_action = all_actions
                    else:
                        pass
                    if use_interpolate:
                        all_actions = interpolate_action(all_actions, 2*all_actions.shape[1])


                    while len(action_queue) > 0:
                        action_queue.popleft()
                    action_queue.extend(
                            torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[:query_frequency])

                    process_time2 = time.time()
                    process_t = process_time2 - process_time1
                    time_cur = time.time()
                    print(
                        f"{RED} Execute >>{query_frequency}<< action costs {time_cur - time_pre - process_t}s. Model forward takes {process_t}s {RESET}")
                    time_pre = time_cur
                else:
                    t1 = time.time()
                    deploy_env.get_obs()
                    print(f">>>>>>>>>>>Communication time (Recieving state from robot): {time.time() - t1}s<<<<<<<<<<<<<<")
                raw_action = action_queue.popleft()

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().to(dtype=torch.float32).numpy()
                action = post_process(raw_action)
                print(f"after post_process action size: {action.shape}")

                print(f'step {t}, pred action: {outputs}{action}')
                if len(action.shape) == 2:
                    action = action[0]
                # action_info = deploy_env.step(action.tolist(), mode=policy_config['control_mode'])
                action_info = deploy_env.step(action.tolist())
                action_list.append(action)
                if t == 350:
                    save_path = os.path.join(policy_config['model_path'], 'actions.pkl')
                    print(f"saving actions...to {save_path}")
                    import pickle
                    with open(save_path, 'wb') as file:
                        pickle.dump(action_list, file)

            # print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            # plt.close()

    return


if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    sys.path.insert(0, "/home/eai/Dev-Code/mirocs")
    from run.agilex_robot_env import AgilexRobot

    action_head = 'scale_dp_policy'  # 'unet_diffusion_policy'
    model_size = '2B'
    policy_config = {

        "model_path":"/media/eai/ADDS4_plus/zzy/model_param/franka_math/0429_modified_cosine20_Qwen2_no_filmlayer_w_reasoningaction_1share_route4_top_2/checkpoint-25000/",
        "pretrain_path": None,
        "enable_lora": False,
        "temp_agg": False,
        "action_head": action_head,
        'model_size': model_size,
        'save_model': False,
        "tinyvla": False,
        "tinyvla_with_output": False,
        "camera_views": [2] # top only
    }
    if "qwen" in policy_config['model_path'] and not os.path.exists(os.path.join(policy_config['model_path'], "chat_template.json")):
        raise "Checkpoint must have chat_template.json and preprocessor.json"
    query_frequency = 25
    global im_size
    im_size = 320  # default 480
    select_one = False  # select one embedding or using all
    # views = [2] # wrist only

    eval_in_vqa=False
    use_interpolate=False
    raw_lang = 'I am hungry, is there anything I can eat?'
    raw_lang = 'I want to paste a poster, can you help me?'
    raw_lang = 'I want a container to put water in, can you help me?'

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    policy = None
    agilex_bot = ArxRobot()
    print('Already connected!!!!!!')

    print(f">>>>>>>>>>>>>qwen2vl<<<<<<<<<<<<<<<")
    if 'lora' in policy_config['model_path'].lower():
        policy_config["model_base"] = f"/home/eai/Documents/wjj/Qwen2-VL-{model_size}-Instruct"

    policy = qwen2_vla_policy(policy_config)

    print(policy.policy)

    eval_bc(policy, agilex_bot, policy_config, raw_lang=raw_lang,
            query_frequency=query_frequency, use_interpolate=use_interpolate)

    print()
    exit()