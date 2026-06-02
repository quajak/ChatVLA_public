
## Project Overview

ChatVLA is a Vision-Language-Action (VLA) model for robot control and multimodal understanding, built on Qwen2-VL-2B. It uses a two-stage training process with an optional Mixture-of-Experts (MoE) layer and diffusion-based action prediction heads (ScaleDP or UNet).

## Environment Setup

This project uses uv. Use uv for all commands.

## Training Commands

Training uses `deepspeed` with `train_vla.py` as the entry point. See `scripts/` for full argument examples.

**Stage 1** — robot data only, initializes MoE from pretrained dense weights:
```bash
bash scripts/train_chatvla_stage_1.sh
```

**Stage 2** — co-trains with vision-language data:
```bash
bash scripts/train_chatvla_stage_2.sh
```

The DeepSpeed config is at `scripts/zero2.json` (ZeRO-2 with bf16).

## Evaluation

```bash
# Robot policy evaluation (real robot)
python evaluate/evaluate_robot.py

# VQA benchmarks (MMMU, MMStar, etc.) via VLMEvalKit
bash evaluate/evaluate_vqa.sh
```

There is no automated test suite.

## Architecture

### Training Pipeline (`train_vla.py`)
Entry point. Parses four argument groups (`ModelArguments`, `DataArguments`, `TrainingArguments`, `ActionHeadArguments`), loads model and processor, constructs datasets, and calls `train_bc()`.

### VLA Model (`qwen2_vla/`)
- `models/modeling_qwen2_vla.py` — Qwen2-VL backbone extended with MoE routing and action head integration.
- `models/configuration_qwen2_vla.py` — `Qwen2VLAConfig` and `Qwen2VLAVisionConfig`.
- `utils/expert_modules.py` — `StaticMoE` implementation: routes token representations to robot-specific or language-specific expert FFN layers.
- `utils/fusion_modules.py` — Vision-language feature fusion.
- `utils/processing_qwen2_vla.py` / `utils/image_processing_qwen2_vla.py` — Qwen2-style processor adapted for robot observations.
- `train/qwen2_vla_trainer.py` — Custom HF `Trainer` subclass handling DeepSpeed integration, episode-level sampling, and the combined robot+VL loss.
- `model_load_utils.py` — Loads pretrained weights, applies LoRA if requested, sets up MoE from dense FFN weights (`save_mlp_weights_for_init_moe.py` assists with this).

### Policy Heads (`policy_heads/`)
Action prediction is decoupled into installable policy heads:
- `models/transformer_diffusion/modeling_scaledp.py` — **ScaleDP**: DiT (Diffusion Transformer) blocks that denoise predicted robot actions over multiple timesteps.
- `models/unet_diffusion/modeling_unet_diffusion.py` — **UNet diffusion**: alternative convolutional diffusion policy.

Both heads receive latent representations from the VLA backbone and output 10-dim action vectors (6 joints + gripper per arm for ALOHA).

### Data Pipeline (`data_utils/`)
- `utils.py` — `EpisodicDataset`: loads robot trajectories from HDF5 files and optionally mixes in vision-language data from JSON files (LLaVA conversation format). The `vl_ratio` argument controls mixing proportion.
- `data_collator.py` — `DataCollatorForSupervisedDataset`: pads and batches images, input tokens, actions, proprioceptive states, and `is_vl_data` flags so the trainer can apply separate loss terms.

### Robot Configuration (`aloha_scripts/`)
- `constants.py` — Task name lists for Stage 1 and Stage 2, HDF5 dataset paths, gripper calibration, and per-joint normalization (`normalize_action`, `unnormalize_action`).
- `utils.py` — ALOHA-specific utilities.

### Data Flow Summary
1. HDF5 trajectories and/or LLaVA-format JSON → `EpisodicDataset`
2. Images normalized by Qwen2 processor; actions/states normalized per `constants.py`
3. Qwen2-VL encoder → (optional) MoE routing → policy head (ScaleDP/UNet) → 10-dim action
4. Behavioral cloning loss on robot data; language modeling loss on VL data

## Key Arguments (`train_vla.py`)

| Argument | Meaning |
|---|---|
| `--using_moe` | Enable MoE layers in the VLA backbone |
| `--init_moe` | Initialize MoE from pretrained dense FFN weights (Stage 1 only) |
| `--freeze_vl_expert` | Freeze the VL expert pathway (Stage 1) |
| `--vl_ratio` | Fraction of VL data in each batch (0 for Stage 1, ~0.33 for Stage 2) |
| `--policy_head` | `scaledp` (default) or `unet` |
| `--lora` | Enable LoRA fine-tuning on the backbone |
| `--task_name` | Comma-separated task names defined in `aloha_scripts/constants.py` |
