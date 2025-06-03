import qwen2_vla
import policy_heads
import torch
import re
import os

from qwen2_vla import Qwen2VLForConditionalGenerationForVLA

# p = "/home/jz08/zhouzy/model_Param/Qwen2-VL-2B-Instruct"
p = "/home/jovyan/tzb/zhouzy/model_param/qwen2_vla_param_dit_diffusion_policy_results/ckpts/0503_llava_warmup/qwen2_vl_2B/0507_Qwen2_no_filmlayer_w_reasoningaction_2share/checkpoint-6000"
pattern_weights = r"model\.layers\.(\d+)\.share_moe_expert\.shared_experts\.(.*)"
save_filename = os.path.join(p, "share_mlp.bin")
model = Qwen2VLForConditionalGenerationForVLA.from_pretrained(p, torch_dtype=torch.bfloat16)
# model.layers.24.share_moe_expert.shared_experts.1.gate_proj.weight True torch.bfloat16
# model.layers.24.share_moe_expert.shared_experts.1.up_proj.weight True torch.bfloat16
# model.layers.24.share_moe_expert.shared_experts.1.down_proj.weight True torch.bfloat16
# model.layers.25.share_moe_expert.shared_experts.0.gate_proj.weight True torch.bfloat16
# model.layers.25.share_moe_expert.shared_experts.0.up_proj.weight True torch.bfloat16
# model.layers.25.share_moe_expert.shared_experts.0.down_proj.weight True torch.bfloat16
save_dict = {}
for name, param in model.named_parameters():
    match = re.match(pattern_weights, name)
    if match:
        save_dict[name] = param
torch.save(save_dict, save_filename)