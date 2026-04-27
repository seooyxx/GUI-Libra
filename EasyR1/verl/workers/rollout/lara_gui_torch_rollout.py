"""Torch/FSDP rollout for true LaRA-GUI latent GRPO.

This is the correctness path for latent reasoning RL.  It deliberately keeps
GUI-Libra's response format and reward contract unchanged, but avoids vLLM
because vLLM cannot execute the iterative hidden-state feedback used by
LaRA-GUI `<|thinking|>` tokens.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from tensordict import TensorDict
from transformers import PreTrainedTokenizer

from ...protocol import DataProto, batch_collate
from ...utils import torch_functional as VF
from ..lara_gui_latent import get_thinking_token_id, prepare_latent_inputs_embeds
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    return np.repeat(value, repeats, axis=0)


class LaRAGUITorchRollout(BaseRollout):
    def __init__(self, module, config: RolloutConfig, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.module = module
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.thinking_token_id = get_thinking_token_id(tokenizer, config.latent_thinking_token)
        self.freed_bytes = 0

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old_values = {}
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                old_values[key] = getattr(self.config, key)
                setattr(self.config, key, value)
        yield
        for key, value in old_values.items():
            setattr(self.config, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        multi_modal_inputs = {}
        batch_multi_modal_inputs = prompts.non_tensor_batch.get("multi_modal_inputs", None)
        if batch_multi_modal_inputs is not None:
            multi_modal_inputs = {
                key: value.to(input_ids.device) if torch.is_tensor(value) else value
                for key, value in batch_collate(batch_multi_modal_inputs).items()
            }
            multi_modal_inputs = {
                key: torch.cat(value, dim=0) if isinstance(value, list) and value and torch.is_tensor(value[0]) else value
                for key, value in multi_modal_inputs.items()
            }

        n = int(getattr(self.config, "n", 1))
        if n > 1:
            batch_size = batch_size * n
            input_ids = _repeat_interleave(input_ids, n)
            attention_mask = _repeat_interleave(attention_mask, n)
            position_ids = _repeat_interleave(position_ids, n)
            multi_modal_inputs = {
                key: _repeat_interleave(value, n) if torch.is_tensor(value) and value.shape[0] == batch_size // n else value
                for key, value in multi_modal_inputs.items()
            }

        with self.update_sampling_params(**prompts.meta_info):
            response_ids = self._decode(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                multi_modal_inputs=multi_modal_inputs,
                eos_token_id=eos_token_id,
            )

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        full_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        submask_dict = self._build_action_submask(response_ids, response_mask) if self.config.use_action_weight else {}
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": full_attention_mask,
                "response_mask": response_mask,
                "position_ids": full_position_ids,
                **submask_dict,
            },
            batch_size=batch_size,
        )
        non_tensor_batch = {}
        if batch_multi_modal_inputs is not None:
            non_tensor_batch["multi_modal_data"] = _repeat_interleave(
                prompts.non_tensor_batch["multi_modal_data"], n
            ) if n > 1 and "multi_modal_data" in prompts.non_tensor_batch else prompts.non_tensor_batch.get(
                "multi_modal_data"
            )
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def _decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        multi_modal_inputs: dict[str, Any],
        eos_token_id: int,
    ) -> torch.Tensor:
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask
        cur_position_ids = position_ids
        response_chunks = []
        finished = torch.zeros(input_ids.shape[0], device=input_ids.device, dtype=torch.bool)

        for _ in range(int(self.config.response_length)):
            inputs_embeds, _ = prepare_latent_inputs_embeds(
                self.module,
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                position_ids=cur_position_ids,
                thinking_token_id=self.thinking_token_id,
                **multi_modal_inputs,
            )
            if inputs_embeds is not None:
                outputs = self.module(
                    inputs_embeds=inputs_embeds,
                    attention_mask=cur_attention_mask,
                    position_ids=cur_position_ids,
                    use_cache=False,
                    return_dict=True,
                )
            else:
                outputs = self.module(
                    input_ids=cur_input_ids,
                    attention_mask=cur_attention_mask,
                    position_ids=cur_position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    return_dict=True,
                )
            logits = outputs.logits[:, -1, :]
            next_token = self._sample_next(logits)
            next_token = torch.where(finished, torch.full_like(next_token, self.pad_token_id), next_token)
            response_chunks.append(next_token.unsqueeze(-1))

            just_finished = next_token.eq(eos_token_id)
            finished = finished | just_finished
            cur_input_ids = torch.cat([cur_input_ids, next_token.unsqueeze(-1)], dim=-1)
            next_mask = (~finished).to(cur_attention_mask.dtype).unsqueeze(-1)
            cur_attention_mask = torch.cat([cur_attention_mask, next_mask], dim=-1)
            cur_position_ids = self._append_position_ids(cur_position_ids)
            if bool(finished.all().item()) and not self.config.ignore_eos:
                break

        if response_chunks:
            response_ids = torch.cat(response_chunks, dim=-1)
        else:
            response_ids = input_ids.new_empty((input_ids.shape[0], 0))
        if response_ids.shape[1] < self.config.response_length:
            pad = input_ids.new_full(
                (input_ids.shape[0], self.config.response_length - response_ids.shape[1]), self.pad_token_id
            )
            response_ids = torch.cat([response_ids, pad], dim=-1)
        return response_ids[:, : self.config.response_length]

    def _sample_next(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = float(self.config.temperature)
        if temperature <= 0:
            return torch.argmax(logits, dim=-1)
        logits = logits / max(temperature, 1e-6)
        top_k = int(self.config.top_k)
        if top_k > 0:
            values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(-1, indices, values)
        top_p = float(self.config.top_p)
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            keep = probs.cumsum(dim=-1) <= top_p
            keep[..., 0] = True
            filtered = torch.full_like(sorted_logits, float("-inf"))
            filtered = torch.where(keep, sorted_logits, filtered)
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, filtered)
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def _append_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
        next_pos = position_ids[..., -1:] + 1
        return torch.cat([position_ids, next_pos], dim=-1)

    def _build_action_submask(self, response_ids: torch.Tensor, response_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        def get_pat_ids(s: str):
            return self.tokenizer(s, add_special_tokens=False).input_ids

        def find_first(seq, pat, start=0):
            if not pat or len(seq) < len(pat):
                return None
            for i in range(start, len(seq) - len(pat) + 1):
                if seq[i : i + len(pat)] == pat:
                    return i
            return None

        ans_start = get_pat_ids("<answer>")[:-1]
        ans_end = get_pat_ids("</answer>")[:-1]
        grounding_start = get_pat_ids('"point_2d":')[1:]
        grounding_end = get_pat_ids("]")
        bsz, response_length = response_ids.size()
        response_length_per_sample = response_mask.sum(dim=1)
        action_mask = torch.zeros((bsz, response_length), dtype=response_ids.dtype, device=response_ids.device)
        grounding_mask = torch.zeros_like(action_mask)
        for i, resp in enumerate(response_ids.tolist()):
            start_idx = find_first(resp, ans_start, 0)
            end_idx = None
            if start_idx is not None:
                end_idx = find_first(resp, ans_end, start_idx + len(ans_start))
                if end_idx is not None:
                    action_mask[i, start_idx + len(ans_start) : end_idx] = 1
            g_start_idx = find_first(resp, grounding_start, start_idx + len(ans_start) if start_idx is not None else 0)
            if g_start_idx is not None:
                g_end_idx = find_first(resp, grounding_end, g_start_idx + len(grounding_start))
                if g_end_idx is None:
                    g_end_idx = end_idx if end_idx is not None else response_length_per_sample[i]
                grounding_mask[i, g_start_idx + len(grounding_start) : g_end_idx] = 1
        if response_mask.dtype == torch.bool:
            action_mask = action_mask.bool() & response_mask
            grounding_mask = grounding_mask.bool() & response_mask
        else:
            action_mask = action_mask * response_mask
            grounding_mask = grounding_mask * response_mask
        return {"response_action_mask": action_mask, "response_grounding_mask": grounding_mask}
