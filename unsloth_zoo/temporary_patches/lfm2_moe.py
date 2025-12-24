# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import Any, List, Optional, Tuple, Union, Dict, Set, Callable
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
from .common import (
    TEMPORARY_PATCHES,
    torch_compile,
    _torch_compile,
    get_torch_compile_options,
    UNSLOTH_ENABLE_LOGGING,
)
from .utils import (
    patch_function,
    patch_function_past_key_values,
    dedent,
    KWARGS_TYPE,
    raise_error,
    logger,
    Cache,
    process_return,
)

def patch_lfm2_moe():
    try:
        import transformers.models.lfm2_moe.modeling_lfm2_moe
    except Exception as e:
        return raise_error("transformers.models.lfm2_moe.modeling_lfm2_moe", e)

    # Pure torch loop-based implementation for MoE
    # Memory efficient but needs torch.compiler.disable due to data-dependent loop
    
    @torch.compiler.disable
    def experts_forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loop-based MoE forward pass for Lfm2Moe. 
        Loops over experts that have tokens routed to them.
        Uses @torch.compiler.disable because the loop is data-dependent (expert_hit).
        """
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # Create expert mask and find which experts have tokens
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0) # (num_experts, top_k, n_tokens)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # Only loop over experts that actually have tokens routed to them
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            
            # Find which tokens are routed to this expert
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            
            # Gather only the tokens for this expert
            current_state = hidden_states[token_idx]
            
            # Compute gate_up projection for this expert only
            # Lfm2Moe uses hardcoded Chunk(2) and SiLU as per provided source
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            
            # Compute down projection for this expert only
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            
            # Apply routing weights
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            
            # Scatter back to final output
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

    # Also disable compilation for the SparseMoeBlock
    # since fullgraph=True cannot inline a torch.compiler.disable'd function
    @torch.compiler.disable
    def sparse_moe_block_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_reshaped)
        selected_experts, routing_weights = self.route_tokens_to_experts(router_logits)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

    # Apply patches
    patch_function(transformers.models.lfm2_moe.modeling_lfm2_moe.Lfm2MoeExperts, "forward", experts_forward)
    patch_function(transformers.models.lfm2_moe.modeling_lfm2_moe.Lfm2MoeSparseMoeBlock, "forward", sparse_moe_block_forward)
pass
TEMPORARY_PATCHES.append(patch_lfm2_moe)
