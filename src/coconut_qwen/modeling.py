from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


BOT_TOKEN = "<bot>"
EOT_TOKEN = "<eot>"
LATENT_TOKEN = "<latent>"


@dataclass
class CoconutConfig:
    latent_steps: int = 2
    bot_token_id: int | None = None
    eot_token_id: int | None = None
    latent_token_id: int | None = None


def add_coconut_tokens(tokenizer: PreTrainedTokenizerBase, model: PreTrainedModel) -> None:
    """Add Coconut marker tokens and resize the model embeddings if needed."""

    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [BOT_TOKEN, EOT_TOKEN, LATENT_TOKEN]}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if added:
        model.resize_token_embeddings(len(tokenizer))


def load_tokenizer_and_model(
    model_name: str,
    *,
    dtype: torch.dtype | str = "auto",
    device_map: str | dict | None = "auto",
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    add_coconut_tokens(tokenizer, model)
    return tokenizer, model


class CoconutForCausalLM(nn.Module):
    """Coconut latent forward wrapper for decoder-only causal LMs.

    The latent segment is computed sequentially. After the prefix ending in
    `<bot>`, each latent input embedding is the previous position's final
    hidden state. No token lookup or sampling happens inside that segment.
    """

    def __init__(self, base_model: PreTrainedModel, config: CoconutConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def get_input_embeddings(self) -> nn.Embedding:
        return self.base_model.get_input_embeddings()

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings()(ids.to(self.device))

    def build_inputs_embeds(
        self,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor | None,
        *,
        latent_steps: int | None = None,
    ) -> torch.Tensor:
        if prefix_ids.ndim != 2:
            raise ValueError("prefix_ids must be shaped [batch, seq]")
        if prefix_ids.shape[0] != 1:
            raise ValueError("This reference implementation handles batch_size=1.")

        steps = self.config.latent_steps if latent_steps is None else latent_steps
        embeds = self._embed(prefix_ids)

        for _ in range(steps):
            out = self.base_model(inputs_embeds=embeds, output_hidden_states=True, use_cache=False)
            next_embed = out.hidden_states[-1][:, -1:, :]
            embeds = torch.cat([embeds, next_embed], dim=1)

        if suffix_ids is not None and suffix_ids.numel() > 0:
            embeds = torch.cat([embeds, self._embed(suffix_ids)], dim=1)
        return embeds

    def forward(
        self,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        *,
        latent_steps: int | None = None,
    ):
        inputs_embeds = self.build_inputs_embeds(prefix_ids, suffix_ids, latent_steps=latent_steps)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        out = self.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        loss = None
        if labels is not None:
            labels = labels.to(self.device)
            shift_logits = out.logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": out.logits}

    def forward_cached(
        self,
        prefix_ids: torch.Tensor,
        suffix_ids: torch.Tensor,
        labels: torch.Tensor,
        *,
        latent_steps: int | None = None,
    ):
        """Memory/time friendlier Coconut training path.

        This still keeps gradients through the fed-back hidden states. The cache
        only avoids recomputing fixed prefix attention keys/values at each latent
        step; it does not detach or discretize the latent vectors.
        """

        if prefix_ids.shape[0] != 1:
            raise ValueError("This reference implementation handles batch_size=1.")

        steps = self.config.latent_steps if latent_steps is None else latent_steps
        prefix_embeds = self._embed(prefix_ids)
        out = self.base_model(
            inputs_embeds=prefix_embeds,
            output_hidden_states=True,
            use_cache=True,
        )
        past = out.past_key_values
        prev_hidden = out.hidden_states[-1][:, -1:, :]
        prev_logits = out.logits[:, -1:, :]

        for _ in range(steps):
            out = self.base_model(
                inputs_embeds=prev_hidden,
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
            )
            past = out.past_key_values
            prev_hidden = out.hidden_states[-1][:, -1:, :]
            prev_logits = out.logits[:, -1:, :]

        suffix_embeds = self._embed(suffix_ids)
        suffix_out = self.base_model(
            inputs_embeds=suffix_embeds,
            past_key_values=past,
            output_hidden_states=False,
            use_cache=False,
        )
        suffix_logits = torch.cat([prev_logits, suffix_out.logits[:, :-1, :]], dim=1)
        suffix_labels = labels[:, prefix_ids.shape[1] + steps :].to(self.device)
        loss = F.cross_entropy(
            suffix_logits.contiguous().view(-1, suffix_logits.size(-1)),
            suffix_labels.contiguous().view(-1),
            ignore_index=-100,
        )
        return {"loss": loss, "logits": suffix_logits}

    @torch.no_grad()
    def generate_cached(
        self,
        prefix_ids: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        steps = self.config.latent_steps
        prefix_embeds = self._embed(prefix_ids.to(self.device))
        out = self.base_model(inputs_embeds=prefix_embeds, output_hidden_states=True, use_cache=True)
        past = out.past_key_values
        prev_hidden = out.hidden_states[-1][:, -1:, :]

        for _ in range(steps):
            out = self.base_model(
                inputs_embeds=prev_hidden,
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
            )
            past = out.past_key_values
            prev_hidden = out.hidden_states[-1][:, -1:, :]

        eot = torch.tensor([[self.config.eot_token_id]], dtype=torch.long, device=self.device)
        out = self.base_model(input_ids=eot, past_key_values=past, use_cache=True)
        past = out.past_key_values
        generated: list[int] = [int(eot.item())]

        for _ in range(max_new_tokens):
            logits = out.logits[:, -1, :]
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            token = int(next_id.item())
            generated.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break
            out = self.base_model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values

        return torch.tensor([generated], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def generate(
        self,
        prefix_ids: torch.Tensor,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        suffix = torch.tensor([[self.config.eot_token_id]], dtype=torch.long, device=self.device)
        inputs_embeds = self.build_inputs_embeds(prefix_ids.to(self.device), suffix)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            out = self.base_model(inputs_embeds=inputs_embeds, use_cache=False)
            logits = out.logits[:, -1, :]
            if temperature and temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            token = int(next_id.item())
            generated.append(token)
            if eos_token_id is not None and token == eos_token_id:
                break
            inputs_embeds = torch.cat([inputs_embeds, self._embed(next_id)], dim=1)

        return torch.tensor([generated], dtype=torch.long, device=self.device)


def make_labels(
    prefix_len: int,
    latent_steps: int,
    suffix_ids: Iterable[int],
    supervised_suffix_mask: Iterable[bool],
) -> torch.Tensor:
    suffix = list(suffix_ids)
    mask = list(supervised_suffix_mask)
    if len(suffix) != len(mask):
        raise ValueError("suffix_ids and supervised_suffix_mask must have the same length")

    labels = [-100] * (prefix_len + latent_steps)
    for token_id, supervise in zip(suffix, mask):
        labels.append(int(token_id) if supervise else -100)
    return torch.tensor([labels], dtype=torch.long)
