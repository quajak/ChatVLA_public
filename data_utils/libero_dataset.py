import gc
import json
import os
import pickle
import random

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def _compute_norm_stats_from_tensors(states: torch.Tensor, actions: torch.Tensor) -> dict:
    """Compute per-dim mean/std/min/max from pre-loaded (N, dim) tensors."""
    eps = 1e-4
    action_mean = actions.mean(dim=0).float()
    action_std  = actions.std(dim=0).float().clamp(min=1e-2)
    action_min  = actions.min(dim=0).values.float() - eps
    action_max  = actions.max(dim=0).values.float() + eps
    qpos_mean   = states.mean(dim=0).float()
    qpos_std    = states.std(dim=0).float().clamp(min=1e-2)
    return {
        "action_mean":  action_mean.numpy(),
        "action_std":   action_std.numpy(),
        "action_min":   action_min.numpy(),
        "action_max":   action_max.numpy(),
        "qpos_mean":    qpos_mean.numpy(),
        "qpos_std":     qpos_std.numpy(),
        "example_qpos": states[:1].numpy(),
    }


class LiberoLeRobotDataset(Dataset):
    """
    Drop-in replacement for EpisodicDataset that reads from a LeRobot LIBERO
    dataset instead of HDF5 files.  The __getitem__ output is identical to
    EpisodicDataset (after forward_process), so the rest of the training
    pipeline (DataCollator, Trainer) is completely unchanged.
    """

    def __init__(
        self,
        repo_id: str,
        camera_names: list,
        chunk_size: int,
        policy_class: str,
        llava_pythia_process,
        data_args,
        vl_file=None,
        vl_image_dir=None,
        vl_ratio: float = 0,
        is_local_debug: bool = False,
        norm_stats: dict = None,
    ):
        super().__init__()
        self.camera_names         = camera_names
        self.chunk_size           = chunk_size
        self.policy_class         = policy_class
        self.llava_pythia_process = llava_pythia_process
        self.data_args            = data_args
        self.vl_image_dir         = vl_image_dir
        self.is_local_debug       = is_local_debug
        # cv2.resize takes (W, H)
        self.imsize = (320, 240)

        self.augment_images = (
            'diffusion' in policy_class.lower() or 'scale_dp' in policy_class.lower()
        )
        self.transformations = None  # lazy-initialized on first robot sample

        if is_local_debug:
            self._setup_debug_stub()
        else:
            self._load_lerobot(repo_id, norm_stats)

        # ------------------------------------------------------------------
        # VL data mixing (mirrors EpisodicDataset lines 71-113 exactly)
        # ------------------------------------------------------------------
        self.episode_ids = np.arange(self.num_episodes).astype(int)

        if vl_file is not None:
            random.seed(42)
            vl_data_list_lens = []
            if isinstance(vl_file, list):
                vl_data_list = []
                for f in vl_file:
                    tmp = json.load(open(f, 'r'))
                    vl_data_list_lens.append(len(tmp))
                    vl_data_list.extend(tmp)
            else:
                vl_data_list_lens.append(0)
                vl_data_list = json.load(open(vl_file, 'r'))
                vl_data_list_lens.append(len(vl_data_list))

            x = int(self.cumulative_len[-1])
            if vl_ratio != -1 and x > 0:
                target_len = int(x * vl_ratio)
                raw = vl_data_list
                len_vl = len(raw)
                if target_len < len_vl:
                    indices = random.sample(
                        range(vl_data_list_lens[0], len_vl),
                        target_len - vl_data_list_lens[0],
                    )
                    indices.extend(range(vl_data_list_lens[0]))
                    vl_data_list = [raw[i] for i in indices]
                else:
                    vl_data_list = [raw[i % len_vl] for i in range(target_len)]

            self.episode_len.append(len(vl_data_list))
            self.cumulative_len = np.append(
                self.cumulative_len, x + len(vl_data_list)
            ).astype(int)
            self.episode_ids = np.append(
                self.episode_ids, self.episode_ids[-1] + 1
            ).astype(int)
        else:
            vl_data_list = None

        self.vl_data_list = vl_data_list

        # Warm-up: triggers lazy transformation init and validates shapes
        _ = self.__getitem__(0)

    # ------------------------------------------------------------------
    # Internal setup helpers
    # ------------------------------------------------------------------

    def _load_lerobot(self, repo_id: str, norm_stats):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.lerobot_dataset = LeRobotDataset(repo_id)

        meta_eps = self.lerobot_dataset.meta.episodes
        ep_from = torch.tensor(meta_eps['dataset_from_index'], dtype=torch.long)
        ep_to   = torch.tensor(meta_eps['dataset_to_index'], dtype=torch.long)
        self.ep_from      = ep_from
        self.ep_to        = ep_to
        self.num_episodes = int(len(ep_from))
        self.episode_len  = (ep_to - ep_from).tolist()
        self.cumulative_len = np.cumsum(self.episode_len).astype(int)
        self.max_episode_len = max(self.episode_len)

        num_frames = self.lerobot_dataset.num_frames
        first      = self.lerobot_dataset[0]
        action_dim = first['action'].shape[0]
        state_dim  = first['observation.state'].shape[0]

        print(f"[LiberoLeRobotDataset] Preloading {num_frames} frames "
              f"(action_dim={action_dim}, state_dim={state_dim})…")
        self._all_actions = torch.zeros(num_frames, action_dim, dtype=torch.float32)
        self._all_states  = torch.zeros(num_frames, state_dim,  dtype=torch.float32)
        for i in range(num_frames):
            frame = self.lerobot_dataset[i]
            self._all_actions[i] = frame['action'].float()
            self._all_states[i]  = frame['observation.state'].float()
        print("[LiberoLeRobotDataset] Preload complete.")

        # Build int task_index → task string mapping (meta.tasks is now a DataFrame)
        tasks_df = self.lerobot_dataset.meta.tasks
        self._tasks = {int(row['task_index']): task_text for task_text, row in tasks_df.iterrows()}

        if norm_stats is None:
            norm_stats = _compute_norm_stats_from_tensors(
                self._all_states, self._all_actions
            )
        self.norm_stats = norm_stats

    def _setup_debug_stub(self):
        """Minimal in-memory stub used when is_local_debug=True (no download)."""
        self.lerobot_dataset  = None
        self.num_episodes     = 2
        self.episode_len      = [10, 10]
        self.cumulative_len   = np.array([10, 20], dtype=int)
        self.max_episode_len  = 10
        self.ep_from          = torch.tensor([0, 10])
        self.ep_to            = torch.tensor([10, 20])
        self._all_actions     = torch.zeros(20, 7,  dtype=torch.float32)
        self._all_states      = torch.zeros(20, 9,  dtype=torch.float32)
        self._tasks           = {0: "debug task"}
        self.norm_stats       = _compute_norm_stats_from_tensors(
            self._all_states, self._all_actions
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.cumulative_len[-1])

    @property
    def modality_lengths(self) -> list:
        if self.vl_data_list is not None:
            robot_count = int(self.cumulative_len[-2])
        else:
            robot_count = int(self.cumulative_len[-1])
        length_list = [300] * robot_count
        if self.vl_data_list is not None:
            for sample in self.vl_data_list:
                cur_len = sum(len(c['value'].split()) for c in sample['conversations'])
                cur_len += 100
                length_list.append(-cur_len)
        return length_list

    def _locate_transition(self, index: int):
        assert index < self.cumulative_len[-1]
        episode_index = int(np.argmax(self.cumulative_len > index))
        start_ts      = index - (int(self.cumulative_len[episode_index])
                                 - self.episode_len[episode_index])
        episode_id    = int(self.episode_ids[episode_index])
        return episode_index, episode_id, start_ts

    def __getitem__(self, index: int):
        episode_index, episode_id, start_ts = self._locate_transition(index)

        vl_data_only = (
            self.vl_data_list is not None
            and len(self.episode_ids) > 1
            and episode_id == int(self.episode_ids[-1])
        )
        text_data_only = False

        # ------------------------------------------------------------------
        # VL-only path (identical to EpisodicDataset lines 159-196)
        # ------------------------------------------------------------------
        if vl_data_only:
            source = self.vl_data_list[start_ts]
            raw_lang = ""
            try:
                img_list = source["image"]
            except KeyError:
                img_list = None
                text_data_only = True

            image_data = []
            if img_list is None or img_list == "":
                text_data_only = True
            else:
                if not isinstance(img_list, list):
                    img_list = [img_list]
                for img_name in img_list:
                    if self.is_local_debug:
                        img = np.zeros((240, 320, 3), dtype=np.uint8)
                    else:
                        img = np.array(
                            Image.open(
                                os.path.join(self.vl_image_dir, img_name)
                            ).convert("RGB")
                        )
                        img = cv2.resize(img, self.imsize)
                    image_data.append(img)
                image_data = np.array(image_data)
                image_data = torch.from_numpy(image_data)
                assert len(image_data.shape) == 4
                image_data = torch.einsum('k h w c -> k c h w', image_data)

            action_dim = self._all_actions.shape[1]
            state_dim  = self._all_states.shape[1]
            qpos_data   = torch.ones((state_dim,),                     dtype=torch.float32)
            action_data = torch.ones((self.chunk_size, action_dim),    dtype=torch.float32)
            is_pad      = torch.ones((self.chunk_size,),               dtype=torch.bool)

        # ------------------------------------------------------------------
        # Robot path
        # ------------------------------------------------------------------
        else:
            ep_start = int(self.ep_from[episode_index])
            ep_end   = int(self.ep_to[episode_index])

            global_idx = ep_start + start_ts

            if self.is_local_debug:
                # Dummy images
                dummy = np.zeros((240, 320, 3), dtype=np.uint8)
                all_cam_images = np.stack([dummy] * len(self.camera_names), axis=0)
                raw_lang = "debug task"
            else:
                frame = self.lerobot_dataset[global_idx]
                task_index = int(frame['task_index'])
                raw_lang = self._tasks[task_index]

                all_cam_images = []
                for cam_name in self.camera_names:
                    img_t = frame[f'observation.images.{cam_name}']
                    # LeRobot returns float32 [0, 1] with shape (C, H, W)
                    img_np = (img_t.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    img_np = cv2.resize(img_np, self.imsize)
                    all_cam_images.append(img_np)
                all_cam_images = np.stack(all_cam_images, axis=0)  # (num_cams, H, W, 3)

            source = {"conversations": [{"from": "human", "value": raw_lang}]}

            image_data = torch.from_numpy(all_cam_images)
            image_data = torch.einsum('k h w c -> k c h w', image_data)

            # Action chunk from preloaded tensors
            action_start = ep_start + start_ts
            action_end   = min(action_start + self.chunk_size, ep_end)
            actual_len   = action_end - action_start

            padded_action = torch.zeros(
                (self.chunk_size, self._all_actions.shape[1]), dtype=torch.float32
            )
            padded_action[:actual_len] = self._all_actions[action_start:action_end]

            is_pad = torch.zeros(self.chunk_size, dtype=torch.bool)
            is_pad[actual_len:] = True

            qpos_data = self._all_states[global_idx].float()

            # Normalize (matches EpisodicDataset lines 278-286)
            if 'diffusion' in self.policy_class or 'scale_dp' in self.policy_class:
                act_min = torch.from_numpy(self.norm_stats["action_min"]).float()
                act_max = torch.from_numpy(self.norm_stats["action_max"]).float()
                action_data = ((padded_action - act_min) / (act_max - act_min)) * 2 - 1
            else:
                act_mean = torch.from_numpy(self.norm_stats["action_mean"]).float()
                act_std  = torch.from_numpy(self.norm_stats["action_std"]).float()
                action_data = (padded_action - act_mean) / act_std

            qpos_mean = torch.from_numpy(self.norm_stats["qpos_mean"]).float()
            qpos_std  = torch.from_numpy(self.norm_stats["qpos_std"]).float()
            qpos_data = (qpos_data - qpos_mean) / qpos_std

            # Image augmentation (lazy-initialized, same as EpisodicDataset)
            if self.transformations is None and not self.is_local_debug:
                orig_size = image_data.shape[2:]
                ratio = 0.95
                self.transformations = [
                    transforms.RandomCrop(
                        size=[int(orig_size[0] * ratio), int(orig_size[1] * ratio)]
                    ),
                    transforms.Resize(orig_size, antialias=True),
                    transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
                    transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
                ]
            if self.augment_images and self.transformations is not None:
                for t in self.transformations:
                    image_data = t(image_data)

        sample = {
            'image':        image_data,
            'state':        qpos_data,
            'action':       action_data,
            'is_pad':       is_pad,
            'conversation': source,
        }

        del image_data, qpos_data, action_data, is_pad
        gc.collect()
        torch.cuda.empty_cache()

        return self.llava_pythia_process.forward_process(
            sample,
            use_reasoning=self.data_args.use_reasoning,
            vl_data_only=vl_data_only,
            text_data_only=text_data_only,
        )


def load_data_libero(
    repo_id: str,
    camera_names: list,
    chunk_size: int,
    config: dict,
    llava_pythia_process,
    vl_file=None,
    vl_image_dir=None,
    vl_ratio: float = 0,
    is_local_debug: bool = False,
):
    """
    Drop-in replacement for load_data() when --use_lerobot True.
    Returns (train_dataset, None, norm_stats) matching load_data's signature.
    """
    cache_dir = os.path.expanduser(
        f"~/.cache/chatvla/lerobot/{repo_id.replace('/', '__')}"
    )
    os.makedirs(cache_dir, exist_ok=True)
    norm_stats_path = os.path.join(cache_dir, "norm_stats.pkl")

    precomputed = None
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'rb') as f:
            precomputed = pickle.load(f)
        print(f"[load_data_libero] Loaded cached norm_stats from {norm_stats_path}")

    train_dataset = LiberoLeRobotDataset(
        repo_id=repo_id,
        camera_names=camera_names,
        chunk_size=chunk_size,
        policy_class=config['action_head_args'].policy_head_type,
        llava_pythia_process=llava_pythia_process,
        data_args=config['data_args'],
        vl_file=vl_file,
        vl_image_dir=vl_image_dir,
        vl_ratio=vl_ratio,
        is_local_debug=is_local_debug,
        norm_stats=precomputed,
    )

    if precomputed is None:
        with open(norm_stats_path, 'wb') as f:
            pickle.dump(train_dataset.norm_stats, f)
        print(f"[load_data_libero] Saved norm_stats to {norm_stats_path}")

    return train_dataset, None, train_dataset.norm_stats
