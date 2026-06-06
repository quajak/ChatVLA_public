"""Evaluate a trained ChatVLA checkpoint in the LIBERO simulator.

This rolls the policy out in the LIBERO benchmark suites (libero_spatial,
libero_object, libero_goal, libero_10) and reports per-task and average
success rates.

The simulated environment is created with LeRobot's LIBERO support
(`lerobot.envs.libero.LiberoEnv`), so observations come back in the same
camera/robot-state convention that produced the `lerobot/libero_*_image`
training datasets used by `data_utils/libero_dataset.py`.

Conventions matched to training (see data_utils/libero_dataset.py and
lerobot/src/lerobot/processor/env_processor.py):

* State (8-dim) = [eef_pos(3), quat->axis_angle(3), gripper_qpos(2)].
* The main "agentview" image is rotated 180 deg (flip H and W) to match the
  HuggingFaceVLA/LIBERO dataset orientation; the wrist image is left as-is.
  (This mirrors lerobot's xVLA libero processor.)
* Per-camera resize for the Qwen2-VL processor follows training's
  forward_process: a camera whose name contains "wrist" (when there is more
  than one camera) is fed at 56x56, otherwise 240x320.
* Actions are de-normalized with the ScaleDP min/max rule and sent straight to
  the env (7-dim relative end-effector delta + gripper); no ALOHA 6D->euler
  conversion is applied.

Example:
    python evaluate/evaluate_libero.py \
        --model_path /scratch/.../ChatVLA_libero_libero_spatial/checkpoint-5000 \
        --task_name libero_spatial \
        --num_trials 20
"""

import argparse
import json
import os
import pickle
from collections import deque

import numpy as np
import torch
from PIL import Image

# Make the repo root importable when run as `python evaluate/evaluate_libero.py`.
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qwen_vl_utils import fetch_image

from aloha_scripts.constants import LIBERO_CONFIGS
from data_utils.utils import set_seed
from qwen2_vla.model_load_utils import load_model_for_eval
from policy_heads import *  # noqa: F401,F403  (registers policy-head configs)


# ---------------------------------------------------------------------------
# Policy wrapper
# ---------------------------------------------------------------------------
class LiberoPolicy:
    """Thin wrapper around the ChatVLA model for simulated evaluation."""

    def __init__(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config["enable_lora"] else None
        (
            self.tokenizer,
            self.policy,
            self.multimodal_processor,
            self.context_len,
        ) = load_model_for_eval(
            model_path=policy_config["model_path"],
            model_base=model_base,
            policy_config=policy_config,
        )
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[SOA]"]})
        self.policy.eval()

    def build_batch(self, images, robot_state, raw_lang, camera_names):
        """Mirror data_utils/utils.py::forward_process for the inference path.

        images: list of np.uint8 (H, W, 3) arrays, aligned with `camera_names`.
        robot_state: torch.FloatTensor of shape (1, state_dim) on cuda.
        """
        messages = [{"role": "user", "content": []}]
        for _ in images:
            messages[0]["content"].append({"type": "image", "image": None})
        messages[0]["content"].append({"type": "text", "text": raw_lang})

        image_list = []
        for i, img in enumerate(images):
            ele = {"image": Image.fromarray(img.astype(np.uint8))}
            # Same per-camera resize as training's forward_process.
            if i < len(camera_names) and "wrist" in camera_names[i] and len(camera_names) > 1:
                ele["resized_height"] = 56
                ele["resized_width"] = 56
            else:
                ele["resized_height"] = 240
                ele["resized_width"] = 320
            image_list.append(torch.from_numpy(np.array(fetch_image(ele))))

        text = self.multimodal_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.multimodal_processor(
            text=text,
            images=image_list,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        batch = dict(states=robot_state)
        for k, v in model_inputs.items():
            batch[k] = v
        return batch


# ---------------------------------------------------------------------------
# Observation processing (matches the LeRobot LIBERO dataset convention)
# ---------------------------------------------------------------------------
def quat2axisangle(quat):
    """Convert an (x, y, z, w) quaternion to a 3-D axis-angle vector.

    Reimplements lerobot LiberoProcessorStep._quat2axisangle in numpy.
    """
    quat = np.asarray(quat, dtype=np.float32)
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = float(np.sqrt(max(1.0 - w * w, 0.0)))
    if den < 1e-10:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arccos(w)
    axis = quat[:3] / den
    return (axis * angle).astype(np.float32)


def process_obs(obs, camera_names, qpos_mean, qpos_std, flip_agentview=True):
    """Turn a LiberoEnv observation into (images, normalized_state).

    images is a list aligned with `camera_names`: [agentview, wrist].
    """
    pixels = obs["pixels"]
    # LiberoEnv maps agentview->"image", eye-in-hand->"image2".
    agent = np.asarray(pixels["image"])
    wrist_key = "image2" if "image2" in pixels else "wrist_image"
    wrist = np.asarray(pixels[wrist_key])

    if flip_agentview:
        # 180 deg rotation (flip both H and W) to match the dataset orientation.
        agent = agent[::-1, ::-1]
    images = [np.ascontiguousarray(agent), np.ascontiguousarray(wrist)]

    rs = obs["robot_state"]
    eef_pos = np.asarray(rs["eef"]["pos"], dtype=np.float32).reshape(-1)[:3]
    eef_quat = np.asarray(rs["eef"]["quat"], dtype=np.float32).reshape(-1)[:4]
    gripper = np.asarray(rs["gripper"]["qpos"], dtype=np.float32).reshape(-1)[:2]
    axisangle = quat2axisangle(eef_quat)

    state = np.concatenate([eef_pos, axisangle, gripper]).astype(np.float32)  # (8,)
    state = (state - qpos_mean) / qpos_std
    state = torch.from_numpy(state).float().unsqueeze(0).cuda()
    return images, state


# ---------------------------------------------------------------------------
# Stats loading
# ---------------------------------------------------------------------------
def load_stats(model_path, task_name):
    """Load the normalization stats used at training time.

    train_vla.py saves `dataset_stats.pkl` in the output_dir (the parent of the
    checkpoint dir). As a fallback we look at the per-repo cache written by
    data_utils/libero_dataset.py::load_data_libero.
    """
    candidates = [
        os.path.join("/".join(model_path.rstrip("/").split("/")[:-1]), "dataset_stats.pkl"),
        os.path.join(model_path, "dataset_stats.pkl"),
    ]
    repo_id = LIBERO_CONFIGS[task_name]["repo_id"]
    candidates.append(
        os.path.expanduser(
            f"~/.cache/chatvla/lerobot/{repo_id.replace('/', '__')}/norm_stats.pkl"
        )
    )
    for path in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                stats = pickle.load(f)
            print(f"[evaluate_libero] Loaded norm stats from {path}")
            return stats
    raise FileNotFoundError(
        "Could not find dataset_stats.pkl / norm_stats.pkl. Looked in:\n  "
        + "\n  ".join(candidates)
    )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------
def _grab_frame(obs, flip_agentview=True):
    """Return the agentview render as an (H, W, 3) uint8 array for video logging.

    The LIBERO sim renders the agentview upside down (this is why training flips
    it 180 deg in process_obs); apply the same flip so the logged video matches
    what the policy sees and is the right way up.
    """
    frame = np.asarray(obs["pixels"]["image"], dtype=np.uint8)
    if flip_agentview:
        frame = frame[::-1, ::-1]
    return np.ascontiguousarray(frame)


def run_task(policy, env, camera_names, post_process, qpos_mean, qpos_std,
             num_trials, max_steps, query_frequency, flip_agentview, verbose,
             use_reasoning=False, record_video_trials=0):
    """Roll the policy out over `num_trials` and return (success_rate, videos).

    `videos` is a list of dicts {"frames": (T, H, W, 3) uint8, "success": bool}
    for the first `record_video_trials` trials (empty if recording is disabled).
    """
    successes = 0
    videos = []
    for trial in range(num_trials):
        obs, info = env.reset()
        raw_lang = env.task_description
        action_queue = deque(maxlen=query_frequency)
        success = False
        record = trial < record_video_trials
        frames = [_grab_frame(obs, flip_agentview)] if record else None

        with torch.inference_mode():
            for t in range(max_steps):
                if t % query_frequency == 0:
                    images, robot_state = process_obs(
                        obs, camera_names, qpos_mean, qpos_std, flip_agentview
                    )
                    batch = policy.build_batch(images, robot_state, raw_lang, camera_names)
                    all_actions, _text = policy.policy.evaluate(
                        **batch, is_eval=True, tokenizer=policy.tokenizer,
                        use_reasoning=use_reasoning,
                    )
                    action_queue = deque(
                        torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[
                            :query_frequency
                        ]
                    )

                raw_action = action_queue.popleft().squeeze(0).cpu().to(torch.float32).numpy()
                action = post_process(raw_action).squeeze().astype(np.float32)  # (7,)

                obs, _reward, terminated, _truncated, info = env.step(action)
                if record:
                    frames.append(_grab_frame(obs, flip_agentview))
                if terminated:
                    success = bool(info.get("is_success", False))
                    break

        successes += int(success)
        if record:
            videos.append({"frames": np.stack(frames), "success": success})
        if verbose:
            print(f"    trial {trial + 1}/{num_trials}: "
                  f"{'SUCCESS' if success else 'fail'} "
                  f"(running {successes}/{trial + 1})")

    return successes / num_trials, videos


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChatVLA in LIBERO.")
    parser.add_argument("--model_path", required=True,
                        help="Path to the trained checkpoint dir (e.g. .../checkpoint-5000).")
    parser.add_argument("--task_name", default="libero_spatial",
                        choices=list(LIBERO_CONFIGS.keys()),
                        help="LIBERO suite to evaluate.")
    parser.add_argument("--model_base", default=None,
                        help="Base model path (only needed for LoRA checkpoints).")
    parser.add_argument("--processor_path", default=None,
                        help="Path/id of a Qwen2-VL processor (with preprocessor_config.json) "
                             "to use when the checkpoint dir only saved tokenizer files. "
                             "Defaults to a sibling checkpoint-* dir or --model_base.")
    parser.add_argument("--enable_lora", action="store_true")
    parser.add_argument("--action_head", default="scale_dp_policy",
                        help="scale_dp_policy or unet_diffusion_policy (must match training).")
    parser.add_argument("--num_trials", type=int, default=20,
                        help="Rollouts per task (each uses a different init state).")
    parser.add_argument("--task_ids", type=int, nargs="*", default=None,
                        help="Restrict to these task ids within the suite (default: all).")
    parser.add_argument("--query_frequency", type=int, default=16,
                        help="Action chunk replay length (should match training chunk_size).")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Max env steps per rollout (default: LeRobot per-suite cap).")
    parser.add_argument("--observation_size", type=int, default=256,
                        help="Sim render resolution before processor resize.")
    parser.add_argument("--no_flip_agentview", action="store_true",
                        help="Disable the 180-deg agentview flip (use if your dataset "
                             "stored unflipped images).")
    parser.add_argument("--num_steps_wait", type=int, default=10,
                        help="No-op steps after reset to let the scene settle.")
    parser.add_argument("--use_reasoning", action="store_true",
                        help="Set this ONLY if the checkpoint was trained with "
                             "--use_reasoning True. When training had reasoning off "
                             "(the default for the LIBERO script), leave this unset so "
                             "the policy-head conditioning is built from the prompt "
                             "hidden states exactly as at training time (the generated "
                             "text is unsupervised and must not condition the head).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json", default=None,
                        help="Optional path to write a results summary.")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log per-task success rates, an average, and rollout "
                             "videos to Weights & Biases.")
    parser.add_argument("--wandb_project", default="chatvla-libero-eval",
                        help="wandb project name (used with --use_wandb).")
    parser.add_argument("--wandb_entity", default=None,
                        help="wandb entity/team (used with --use_wandb).")
    parser.add_argument("--wandb_run_name", default=None,
                        help="wandb run name (defaults to task_name + checkpoint).")
    parser.add_argument("--wandb_video_trials", type=int, default=3,
                        help="Number of rollouts per task to record as wandb videos "
                             "(0 disables video logging).")
    parser.add_argument("--wandb_video_fps", type=int, default=20,
                        help="Frame rate for logged rollout videos.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)

    wandb_run = None
    if args.use_wandb:
        import wandb
        run_name = args.wandb_run_name or (
            f"{args.task_name}-{os.path.basename(args.model_path.rstrip('/'))}"
        )
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=vars(args),
        )

    # Imported here so the heavy LIBERO/robosuite/EGL stack only loads at runtime.
    from libero.libero import benchmark
    from lerobot.envs.libero import LiberoEnv, TASK_SUITE_MAX_STEPS

    cfg = LIBERO_CONFIGS[args.task_name]
    camera_names = cfg["camera_names"]
    max_steps = args.max_steps or TASK_SUITE_MAX_STEPS.get(args.task_name, cfg["episode_len"])

    policy_config = {
        "model_path": args.model_path,
        "model_base": args.model_base,
        "processor_path": args.processor_path,
        "pretrain_path": None,
        "enable_lora": args.enable_lora,
        "action_head": args.action_head,
        "save_model": False,
    }
    print(f"[evaluate_libero] Loading policy from {args.model_path} ...")
    policy = LiberoPolicy(policy_config)

    stats = load_stats(args.model_path, args.task_name)
    action_min = np.asarray(stats["action_min"], dtype=np.float32)
    action_max = np.asarray(stats["action_max"], dtype=np.float32)
    qpos_mean = np.asarray(stats["qpos_mean"], dtype=np.float32)
    qpos_std = np.asarray(stats["qpos_std"], dtype=np.float32)

    if args.action_head.lower() == "act":
        action_mean = np.asarray(stats["action_mean"], dtype=np.float32)
        action_std = np.asarray(stats["action_std"], dtype=np.float32)
        post_process = lambda a: a * action_std + action_mean
    else:  # scale_dp / diffusion: actions trained in [-1, 1]
        post_process = lambda a: ((a + 1) / 2) * (action_max - action_min) + action_min

    suite = benchmark.get_benchmark_dict()[args.task_name]()
    num_tasks = len(suite.tasks)
    task_ids = args.task_ids if args.task_ids is not None else list(range(num_tasks))

    print(f"[evaluate_libero] Suite '{args.task_name}': {num_tasks} tasks, "
          f"evaluating ids {task_ids}, {args.num_trials} trials each, "
          f"max_steps={max_steps}, cameras={camera_names}")

    results = {}
    for tid in task_ids:
        env = LiberoEnv(
            task_suite=suite,
            task_id=tid,
            task_suite_name=args.task_name,
            obs_type="pixels_agent_pos",
            observation_height=args.observation_size,
            observation_width=args.observation_size,
            init_states=True,
            episode_index=0,
            n_envs=1,
            control_mode="relative",
            num_steps_wait=args.num_steps_wait,
        )
        print(f"\n[Task {tid}] {env.task_description}")
        task_desc = env.task_description
        try:
            rate, videos = run_task(
                policy, env, camera_names, post_process, qpos_mean, qpos_std,
                num_trials=args.num_trials, max_steps=max_steps,
                query_frequency=args.query_frequency,
                flip_agentview=not args.no_flip_agentview,
                verbose=args.verbose,
                use_reasoning=args.use_reasoning,
                record_video_trials=args.wandb_video_trials if wandb_run else 0,
            )
        finally:
            env.close()
        results[tid] = {"task": task_desc, "success_rate": rate}
        print(f"[Task {tid}] success rate: {rate:.3f}")

        if wandb_run is not None:
            log = {f"task_{tid}/success_rate": rate}
            for i, vid in enumerate(videos):
                # wandb.Video wants (T, C, H, W).
                frames = np.transpose(vid["frames"], (0, 3, 1, 2))
                tag = "success" if vid["success"] else "fail"
                log[f"task_{tid}/rollout_{i}_{tag}"] = wandb.Video(
                    frames, fps=args.wandb_video_fps, format="mp4",
                    caption=f"{task_desc} [{tag}]",
                )
            wandb_run.log(log)

    avg = float(np.mean([v["success_rate"] for v in results.values()])) if results else 0.0
    print("\n==================== LIBERO eval summary ====================")
    print(f"Suite: {args.task_name}  |  checkpoint: {args.model_path}")
    for tid, v in results.items():
        print(f"  task {tid:>2}: {v['success_rate']:.3f}  {v['task']}")
    print(f"  ----------------------------------------------------------")
    print(f"  AVERAGE success rate: {avg:.3f}")
    print("=============================================================")

    if wandb_run is not None:
        table = wandb.Table(columns=["task_id", "task", "success_rate"])
        for tid, v in results.items():
            table.add_data(tid, v["task"], v["success_rate"])
        wandb_run.log({
            "average_success_rate": avg,
            "results_table": table,
        })
        wandb_run.summary["average_success_rate"] = avg
        wandb_run.finish()

    if args.output_json:
        summary = {
            "task_name": args.task_name,
            "model_path": args.model_path,
            "num_trials": args.num_trials,
            "average_success_rate": avg,
            "per_task": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[evaluate_libero] Wrote summary to {args.output_json}")


if __name__ == "__main__":
    main()
