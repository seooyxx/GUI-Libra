"""LaRA-GUI latent-feedback helpers for GUI-Libra EasyR1.

This module intentionally keeps the GUI-Libra action interface unchanged:
rollouts still decode `<answer>{...}</answer>` as response tokens.  The only
change is that `<|thinking|>` prompt embeddings are replaced by the preceding
hidden state before answer-token log-probabilities are computed.
"""

from __future__ import annotations

from typing import Optional

import torch


def _unwrap_module(module):
    return getattr(module, "module", module)


def get_thinking_token_id(tokenizer, token: str = "<|thinking|>") -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"Tokenizer does not contain LaRA-GUI thinking token: {token}")
    return int(token_id)


def prepare_latent_inputs_embeds(
    module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    thinking_token_id: int,
    position_ids: Optional[torch.Tensor] = None,
    **model_kwargs,
) -> tuple[Optional[torch.Tensor], int]:
    """Return embeddings after LaRA-VLA-style iterative thinking feedback.

    The implementation favors robustness inside FSDP/EasyR1 over speed: for
    each thinking-token position it recomputes the current prefix, then writes
    the hidden state at `pos - 1` back into the thinking-token embedding.
    """
    if input_ids is None or thinking_token_id is None:
        return None, 0

    latent_lists = [
        torch.nonzero(input_ids[i].eq(int(thinking_token_id)), as_tuple=False).squeeze(-1).tolist()
        for i in range(input_ids.shape[0])
    ]
    latent_positions = sorted({int(pos) for items in latent_lists for pos in items})
    if not latent_positions or latent_positions[0] <= 0:
        return None, 0

    base = _unwrap_module(module)
    embedding_layer = base.get_input_embeddings()
    inputs_embeds = embedding_layer(input_ids)

    clean_kwargs = dict(model_kwargs)
    for reserved in (
        "input_ids",
        "inputs_embeds",
        "attention_mask",
        "position_ids",
        "labels",
        "output_hidden_states",
        "return_dict",
        "use_cache",
        "past_key_values",
        "cache_position",
    ):
        clean_kwargs.pop(reserved, None)

    for latent_pos in latent_positions:
        end = int(latent_pos)
        prefix_kwargs = _slice_model_kwargs(clean_kwargs, end)
        outputs = module(
            inputs_embeds=inputs_embeds[:, :end, :],
            attention_mask=attention_mask[:, :end],
            position_ids=_slice_position_ids(position_ids, end),
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            **prefix_kwargs,
        )
        hidden = outputs.hidden_states[-1]
        updated = inputs_embeds.clone()
        for batch_idx, positions in enumerate(latent_lists):
            if latent_pos in positions:
                updated[batch_idx, latent_pos, :] = hidden[batch_idx, latent_pos - 1, :]
        inputs_embeds = updated

    return inputs_embeds, len(latent_positions)


def _slice_position_ids(position_ids: Optional[torch.Tensor], end: int) -> Optional[torch.Tensor]:
    if position_ids is None:
        return None
    if position_ids.dim() == 3:
        return position_ids[..., :end]
    return position_ids[:, :end]


def _slice_model_kwargs(kwargs: dict, end: int) -> dict:
    """Slice token-aligned kwargs while leaving image tensors intact."""
    out = {}
    token_aligned = {"mm_token_type_ids", "token_type_ids"}
    for key, value in kwargs.items():
        if torch.is_tensor(value) and key in token_aligned and value.dim() >= 2:
            out[key] = value[:, :end]
        else:
            out[key] = value
    return out

