import os
import random

import torch
import torch.nn as nn

from torch.utils.data import Sampler, DataLoader, BatchSampler, Dataset
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.trainer import *

import math
import sys
from transformers import Trainer
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    logger,
)
from typing import List, Optional, Dict, Union
from transformers.utils import is_torch_xla_available
from transformers.trainer_pt_utils import get_dataloader_sampler, nested_gather
mega_batch_mult = 1
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, sample_weights=None, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    robo_indices, robo_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    vl_indices, vl_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(robo_indices) > 0, "Should have at least one robotics sample."
    assert len(vl_indices) > 0, "Should have at least one visual language sample."
    print(f"There are {len(robo_indices)} robotics data and {len(vl_indices)} visual language data.")
    robo_shuffle = [robo_indices[i] for i in get_length_grouped_indices(robo_lengths, batch_size, world_size, generator=None)]
    vl_shuffle = [vl_indices[i] for i in get_length_grouped_indices(vl_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size * mega_batch_mult
    robo_megabatches = [robo_shuffle[i: i + megabatch_size] for i in range(0, len(robo_shuffle), megabatch_size)]
    vl_megabatches = [vl_shuffle[i: i + megabatch_size] for i in range(0, len(vl_shuffle), megabatch_size)]

    last_robo = robo_megabatches[-1]
    last_vl = vl_megabatches[-1]
    additional_batch = last_robo + last_vl
    if sample_weights is not None:
        assert len(sample_weights) == 2, "Sample weights should have exactly two elements. One for robot data, another for vl data."
        print("Using custom sample weights: 1:1")
        robo_weight, vl_weight = sample_weights

        # we now only support robo:vl = 1:1
        repeat_nums = len(vl_megabatches) // len(robo_megabatches)
        robo_len = len(robo_megabatches)
        robo_megabatches = robo_megabatches * repeat_nums
        robo_random = random.sample(range(robo_len - 1), len(vl_megabatches) - len(robo_megabatches))
        temp = [robo_megabatches[i] for i in robo_random]
        robo_megabatches = robo_megabatches + temp

        # construct megabatches according to sample weights 1:1
        megabatches = []
        for x,y in zip(robo_megabatches, vl_megabatches):
            tmp = []
            for i in range(world_size):
                tmp += x[i * batch_size: i * batch_size + batch_size // 2] + y[i * batch_size: i * batch_size + batch_size // 2]
                tmp += x[i * batch_size + batch_size // 2: i * batch_size + batch_size] + y[i * batch_size + batch_size // 2: i * batch_size + batch_size]
                megabatches.append(tmp)
    else:
        megabatches = robo_megabatches[:-1] + vl_megabatches[:-1]
    # shuffle
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # throw the last batch of both dataset
    # if len(additional_batch) >= megabatch_size:
    #     megabatches = [additional_batch[:megabatch_size]] + megabatches
    #     additional_batch = additional_batch[megabatch_size:]
    #
    # if len(additional_batch) > 0:
    #     megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size * mega_batch_mult
    megabatches = [indices[i: i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in 1>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        # if version.parse(importlib.metadata.version("1")) >= version.parse("0.7.0"):
        #     from peft import PeftMixedModel
        #
        #     classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

class QWen2VLATrainer(Trainer):

    def __init__(self, sampler_params, prefetch_factor=0, *args, **kwargs):
        self.sampler_params = sampler_params
        self.prefetch_factor = prefetch_factor
        self.lora_module = kwargs['args'].lora_module
        ####################
        self.output_dir = kwargs['args'].output_dir
        #######################################
        self.lang_type = 'model' if 'phi' in kwargs['model'].config.architectures[0].lower() else 'gpt_neox'
        super().__init__(*args, **kwargs)
        # Buffers for logging the model's separate sub-losses (llm/action/...)
        # accumulated across micro-batches between two logging steps. See
        # `compute_loss` / `log` below.
        self._custom_loss_sums = {}
        self._custom_loss_count = 0
        print("self.args.train_batch_size: ", self.args.train_batch_size)
        print("self.args.world_size: ", self.args.world_size)
        print("self.args.gradient_accumulation_steps: ", self.args.gradient_accumulation_steps)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        from functools import partial
        from transformers.trainer_utils import seed_worker
        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
            # dataloader_params["shuffle"] = True
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self.data_collator

        data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["shuffle"] = True

            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return self.accelerator.prepare(DataLoader(eval_dataset, **dataloader_params))

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.head_lr is not None:
                head_params = [name for name, _ in opt_model.named_parameters() if ("policy_head" in name)]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n not in head_params and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_head_parameters",
                        "names": [
                            n for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n not in head_params and p.requires_grad)
                        ],
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n not in head_params and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_head_parameters",
                        "names": [
                            n for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n not in head_params and p.requires_grad)
                        ]
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in head_params and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.head_lr,
                        "names": [
                            n for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in head_params and p.requires_grad)
                        ],
                        "name": "decay_no_head_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in head_params and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.head_lr,
                        "names": [
                            n for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in head_params and p.requires_grad)
                        ],
                        "name": "no_decay_no_head_parameters"
                    },
                ]
            elif self.args.non_lora_lr is not None:
                # non_lora_parameters = [name for name, _ in opt_model.named_parameters() if ("mm_projector" in name or "vision_tower" in name)]
                non_lora_parameters = []
                test = []
                for name, module in opt_model.named_parameters():

                    # if 'layers' in name and 'vision' not in name and 'gpt_neox' in name: # gptneoxl LLM的参数
                    if 'policy_head' not in name and 'layers' in name and 'vision' not in name and self.lang_type in name:  # gptneoxl LLM的参数
                        if 'llm' not in self.lora_module:
                            non_lora_parameters.append(name)
                        pass

                    elif any(key in name for key in ['vision_resampler', 'merger', 'policy_head', 'lm_head', 'proj_to_action', 'text_hidden_fcs', 'external_vit', 'input_action_proj', 'reasoning_action_proj', 'reasoning_film', 'channel_proj', 'xattn', 'expert']):  # vision adapter、action head的参数
                        non_lora_parameters.append(name)
                    elif any(key in name for key in ['expert']):
                        non_lora_parameters.append(name)

                    # elif not isinstance(module, torch.nn.Linear): #非线性层
                    #     non_lora_parameters.append(name)
                # if 'half' in self.lora_module:
                #     for n,p in opt_model.named_parameters():
                #         if ('policy_head' not in n) and ('layers' in n) and ('vision' not in n) and ('gpt_neox' in n):
                #             if int(n.split('.')[4]) % 2 == 0:
                #                 non_lora_parameters.append(n)

                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n not in non_lora_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_lora_parameters",
                        "names": [
                            n for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n not in non_lora_parameters and p.requires_grad)
                        ],
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n not in non_lora_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_lora_parameters",
                        "names": [
                            n for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n not in non_lora_parameters and p.requires_grad)
                        ]
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in non_lora_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.non_lora_lr,
                        "names": [
                            n for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in non_lora_parameters and p.requires_grad)
                        ],
                        "name": "decay_no_lora_parameters"
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in non_lora_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.non_lora_lr,
                        "names": [
                            n for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in non_lora_parameters and p.requires_grad)
                        ],
                        "name": "no_decay_no_lora_parameters"
                    },
                ]
                assert len(optimizer_grouped_parameters[1][
                               'params']) == 0, f"{optimizer_grouped_parameters[1]['names']} should be empty!!!!!"
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "name": "decay_parameters",
                        "names": [
                            n for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ]
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "name": "no_decay_parameters",
                        "names": [
                            n for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and p.requires_grad)
                        ]
                    },
                ]
            # for each in optimizer_grouped_parameters:

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)


            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            ###########################################################
            ###########################################################
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """The VLA model returns a dict of sub-losses under ``outputs.loss``
        (``loss``, ``llm_loss``, ``action_loss``, ``reasoning_loss``, ...).

        Stock ``Trainer`` assumes ``outputs.loss`` is a scalar tensor, so we
        unpack it here: hand the scalar total loss back to the (stock) training
        loop for backward/scaling, and stash the detached sub-losses so ``log``
        can report them separately. This replaces the previously vendored copies
        of ``training_step`` / ``_inner_training_loop`` / ``_maybe_log_save_evaluate``
        that were tied to an old transformers version.
        """
        outputs = model(**inputs)
        loss_dict = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        loss = loss_dict["loss"]

        # Only accumulate during training steps (skip eval/prediction) so the
        # logged averages reflect training micro-batches only.
        if model.training:
            for k, v in loss_dict.items():
                if k == "loss":
                    continue
                self._custom_loss_sums[k] = self._custom_loss_sums.get(k, 0.0) + v.detach()
            self._custom_loss_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        # Inject the averaged sub-losses accumulated since the last training log.
        # Guard on "loss" so we don't attach them to eval-only log dicts.
        if logs.get("loss") is not None and getattr(self, "_custom_loss_count", 0) > 0:
            for k, v in self._custom_loss_sums.items():
                logs[k] = round((v / self._custom_loss_count).item(), 4)
            self._custom_loss_sums = {}
            self._custom_loss_count = 0
        return super().log(logs, start_time)

    def _save_checkpoint(self, model, trial, metrics=None):
        # transformers >=5 dropped the `metrics` arg from Trainer._save_checkpoint
        # (best-metric tracking moved into _maybe_log_save_evaluate). Forwarding it
        # raises "takes 3 positional arguments but 4 were given", so don't pass it.
        super(QWen2VLATrainer, self)._save_checkpoint(model, trial)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super(QWen2VLATrainer, self)._save(output_dir, state_dict)
