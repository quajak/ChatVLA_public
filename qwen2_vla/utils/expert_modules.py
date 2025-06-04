import torch
from torch import nn
import torch.nn.functional as F

from typing import Tuple, Type, Optional



class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        # layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        # self.noise_linear = nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_ouput is the output tensor from multihead self attention block
        logits = self.topkroute_linear(mh_output)

        # # Noise logits
        # noise_logits = self.noise_linear(mh_output)
        #
        # # Adding scaled unit gaussian noise to the logits
        # noise = torch.randn_like(logits) * F.softplus(noise_logits)
        # noisy_logits = logits + noise
        #
        # top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        # zeros = torch.full_like(noisy_logits, float('-inf'))
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices

class StaticMoE(nn.Module):
    def __init__(self, config, expert_module_class: Type[nn.Module]):
        super(StaticMoE, self).__init__()
        self.experts = nn.ModuleList([expert_module_class(config) for _ in range(2)])
    def forward(self, x, vl_data_mask, eval_in_vqa=False):
        if self.training:
            vl_data_mask = vl_data_mask.type_as(x)
            mask_dim = x.shape[-2:]
            vl_data_mask = vl_data_mask.repeat(mask_dim[1], mask_dim[0], 1).transpose(0, 2)
            output = self.experts[0](x) * vl_data_mask \
                                + self.experts[1](x) * (1. - vl_data_mask)
        else:
            dim = x.shape[-1]
            if eval_in_vqa:
                output = self.experts[0](x.reshape(-1, dim)).unsqueeze(0)
            else:
                output = self.experts[1](x.reshape(-1, dim)).unsqueeze(0)
        return output



from typing import Tuple
class DeepSeekGate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, config):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()
        self.topk = config.routed_expert_num//2
        self.n_groups = 1
        self.topk_groups = 1
        self.score_func = "sigmoid"
        self.route_scale = 1
        self.weight = nn.Parameter(torch.empty(config.routed_expert_num, config.hidden_size))
        self.bias = nn.Parameter(torch.empty(config.routed_expert_num))
        nn.init.kaiming_normal_(self.weight)
        nn.init.uniform_(self.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """
        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(x), indices


class DeepSeekMoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, config, expert_module_class: Type[nn.Module]):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.dim = config.hidden_size
        self.n_routed_experts = config.routed_expert_num
        self.gate = DeepSeekGate(config)
        self.experts = nn.ModuleList([expert_module_class(config) for i in range(self.n_routed_experts)])
        self.use_shared_experts = config.shared_expert_num>0
        if self.use_shared_experts:
            self.shared_experts = expert_module_class(config, config.shared_expert_num * config.intermediate_size)
        ### TODO:PROCESS INIT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(0, self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        if self.use_shared_experts:
            z = self.shared_experts(x)
            return (y + z).view(shape)
        else:
            return y.view(shape)

