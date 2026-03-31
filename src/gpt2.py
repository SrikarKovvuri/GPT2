"""
GPT-2 Implementation from Scratch

This module implements the GPT-2 transformer architecture using only PyTorch.
No HuggingFace dependencies are allowed in this file.
"""

import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# =========================
# Config + output dataclasses
# =========================

@dataclass
class GPT2Config:
    vocab_size: int = 50257
    max_ctx_len: int = 1024
    d_model: int = 768
    d_head: int = 64
    d_mlp_intermediate: int = 3072
    n_layer: int = 12
    n_head: int = 12
    num_labels: int = 20


@dataclass
class CausalLMOutput:
    logits: Tensor


@dataclass
class ModelOutput:
    sequences: Tensor


@dataclass
class SequenceClassifierOutput:
    logits: Tensor


# =========================
# Helper modules
# =========================

class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.d_mlp_intermediate)
        self.c_proj = nn.Linear(config.d_mlp_intermediate, config.d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = F.gelu(x, approximate="tanh")
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()

        assert config.d_model == config.n_head * config.d_head, \
            "d_model must equal n_head * d_head"

        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_model = config.d_model

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)

        mask = torch.tril(torch.ones(config.max_ctx_len, config.max_ctx_len))
        self.register_buffer(
            "bias",
            mask.view(1, 1, config.max_ctx_len, config.max_ctx_len)
        )

    def forward(
        self,
        x: Tensor,
        past_key_values: Optional[Tuple[Optional[Tensor], Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        past_len = 0
        if past_key_values is not None:
            past_k, past_v = past_key_values
            if past_k is not None and past_v is not None:
                past_len = past_k.size(-2)
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)

        total_k_len = k.size(-2)
        causal_mask = self.bias[:, :, past_len:past_len + T, :total_k_len]
        att = att.masked_fill(causal_mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        return y, (k, v)


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        x: Tensor,
        past_key_values: Optional[Tuple[Optional[Tensor], Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        attn_out, new_past = self.attn(self.ln_1(x), past_key_values=past_key_values)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, new_past


# =========================
# GPT-2 LM model
# =========================

class GPT2LMHeadModel(nn.Module):
    """
    GPT-2 Language Model with a language modeling head.
    This corresponds to HF's GPT2LMHeadModel.
    """

    def __init__(self, config: GPT2Config = GPT2Config(), bin_path: Optional[str] = None):
        super().__init__()

        self.config = config

        self.word_token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_ctx_len, config.d_model)

        self.h = nn.ModuleList([
            GPT2Block(config) for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.word_token_embedding.weight  # weight tying

        if bin_path is not None:
            self.load_weights(bin_path)

    def load_weights(self, bin_path: str):
        ckpt = torch.load(bin_path, map_location="cpu")

        with torch.no_grad():
            # embeddings
            self.word_token_embedding.weight.copy_(ckpt["wte.weight"])
            self.position_embedding.weight.copy_(ckpt["wpe.weight"])

            # blocks
            for i, block in enumerate(self.h):
                # layer norms
                block.ln_1.weight.copy_(ckpt[f"h.{i}.ln_1.weight"])
                block.ln_1.bias.copy_(ckpt[f"h.{i}.ln_1.bias"])
                block.ln_2.weight.copy_(ckpt[f"h.{i}.ln_2.weight"])
                block.ln_2.bias.copy_(ckpt[f"h.{i}.ln_2.bias"])

                # attention
                block.attn.c_attn.weight.copy_(ckpt[f"h.{i}.attn.c_attn.weight"].t())
                block.attn.c_attn.bias.copy_(ckpt[f"h.{i}.attn.c_attn.bias"])
                block.attn.c_proj.weight.copy_(ckpt[f"h.{i}.attn.c_proj.weight"].t())
                block.attn.c_proj.bias.copy_(ckpt[f"h.{i}.attn.c_proj.bias"])

                # causal mask buffer
                if f"h.{i}.attn.bias" in ckpt:
                    block.attn.bias.copy_(ckpt[f"h.{i}.attn.bias"])

                # mlp
                block.mlp.c_fc.weight.copy_(ckpt[f"h.{i}.mlp.c_fc.weight"].t())
                block.mlp.c_fc.bias.copy_(ckpt[f"h.{i}.mlp.c_fc.bias"])
                block.mlp.c_proj.weight.copy_(ckpt[f"h.{i}.mlp.c_proj.weight"].t())
                block.mlp.c_proj.bias.copy_(ckpt[f"h.{i}.mlp.c_proj.bias"])

            # final layer norm
            self.ln_f.weight.copy_(ckpt["ln_f.weight"])
            self.ln_f.bias.copy_(ckpt["ln_f.bias"])

    def forward(
    self,
    input_ids: Tensor,
    past_key_values: Optional[List[Tuple[Optional[Tensor], Optional[Tensor]]]] = None,
) -> CausalLMOutput:
        B, T = input_ids.shape
        device = input_ids.device

        past_len = 0
        if past_key_values is not None and len(past_key_values) > 0:
            first_layer_k, _ = past_key_values[0]
            if first_layer_k is not None:
                past_len = first_layer_k.size(-2)

        position_ids = torch.arange(past_len, past_len + T, device=device)

        tok_emb = self.word_token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids).unsqueeze(0)

        x = tok_emb + pos_emb

        for i, block in enumerate(self.h):
            layer_past = None
            if past_key_values is not None:
                layer_past = past_key_values[i]

            x, new_past = block(x, past_key_values=layer_past)

            if past_key_values is not None:
                past_key_values[i] = new_past

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return CausalLMOutput(logits=logits)

    def generate(
        self,
        input_ids: Tensor,
        temperature: float = 1.0,
        top_p: float = 0.95,
        max_new_tokens: int = 128
    ) -> ModelOutput:
        sequences = input_ids

        past_key_values = [(None, None) for _ in range(self.config.n_layer)]

        logits = self.forward(sequences, past_key_values=past_key_values).logits
        next_token_logits = logits[:, -1, :]

        if temperature == 0.0:
            next_token = self.greedy_sampling(next_token_logits)
        else:
            scaled_logits = next_token_logits / temperature
            next_token = self.nucleus_sampling(scaled_logits, top_p)

        sequences = torch.cat([sequences, next_token], dim=1)

        for _ in range(max_new_tokens - 1):
            logits = self.forward(next_token, past_key_values=past_key_values).logits
            next_token_logits = logits[:, -1, :]

            if temperature == 0.0:
                next_token = self.greedy_sampling(next_token_logits)
            else:
                scaled_logits = next_token_logits / temperature
                next_token = self.nucleus_sampling(scaled_logits, top_p)

            sequences = torch.cat([sequences, next_token], dim=1)

        return ModelOutput(sequences=sequences)

    def greedy_sampling(self, logits: Tensor) -> Tensor:
        return torch.argmax(logits, dim=-1, keepdim=True)

    def nucleus_sampling(self, logits: Tensor, top_p: float) -> Tensor:
        probs = F.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        nucleus_mask = cumulative_probs <= top_p
        nucleus_mask[:, 0] = True

        filtered_probs = sorted_probs.masked_fill(~nucleus_mask, 0.0)
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)

        sampled_idx = torch.multinomial(filtered_probs, num_samples=1)
        next_token = torch.gather(sorted_indices, dim=-1, index=sampled_idx)

        return next_token
# =========================
# GPT-2 classifier
# =========================

class GPT2ForSequenceClassification(nn.Module):
    def __init__(
        self,
        config: GPT2Config = GPT2Config(),
        classifier_bin_path: Optional[str] = None,
        lm_bin_path: Optional[str] = None
    ):
        super().__init__()

        assert not (classifier_bin_path and lm_bin_path), \
            "Only one of `classifier_bin_path` and `lm_bin_path` can be provided."

        # TODO: implement in Part 2
        # GPT-2 backbone
        self.config = config
        self.word_token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_ctx_len, config.d_model)
        self.h = nn.ModuleList([
            GPT2Block(config) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.d_model)

        # Classification head
        self.score = nn.Linear(config.d_model, config.num_labels)

        # Load weights if provided
        if classifier_bin_path is not None:
            self.load_classifier_weights(classifier_bin_path)
        elif lm_bin_path is not None:
            self.load_lm_weights(lm_bin_path)

    def load_lm_weights(self, lm_bin_path: str):
        ckpt = torch.load(lm_bin_path, map_location="cpu")

        with torch.no_grad():
            self.word_token_embedding.weight.copy_(ckpt["wte.weight"])
            self.position_embedding.weight.copy_(ckpt["wpe.weight"])

            for i, block in enumerate(self.h):
                block.ln_1.weight.copy_(ckpt[f"h.{i}.ln_1.weight"])
                block.ln_1.bias.copy_(ckpt[f"h.{i}.ln_1.bias"])
                block.ln_2.weight.copy_(ckpt[f"h.{i}.ln_2.weight"])
                block.ln_2.bias.copy_(ckpt[f"h.{i}.ln_2.bias"])

                block.attn.c_attn.weight.copy_(ckpt[f"h.{i}.attn.c_attn.weight"].t())
                block.attn.c_attn.bias.copy_(ckpt[f"h.{i}.attn.c_attn.bias"])
                block.attn.c_proj.weight.copy_(ckpt[f"h.{i}.attn.c_proj.weight"].t())
                block.attn.c_proj.bias.copy_(ckpt[f"h.{i}.attn.c_proj.bias"])

                if f"h.{i}.attn.bias" in ckpt:
                    block.attn.bias.copy_(ckpt[f"h.{i}.attn.bias"])

                block.mlp.c_fc.weight.copy_(ckpt[f"h.{i}.mlp.c_fc.weight"].t())
                block.mlp.c_fc.bias.copy_(ckpt[f"h.{i}.mlp.c_fc.bias"])
                block.mlp.c_proj.weight.copy_(ckpt[f"h.{i}.mlp.c_proj.weight"].t())
                block.mlp.c_proj.bias.copy_(ckpt[f"h.{i}.mlp.c_proj.bias"])

            self.ln_f.weight.copy_(ckpt["ln_f.weight"])
            self.ln_f.bias.copy_(ckpt["ln_f.bias"])

    def load_classifier_weights(self, classifier_bin_path: str):
        ckpt = torch.load(classifier_bin_path, map_location="cpu")
        self.load_state_dict(ckpt)

    def forward(self, input_ids: Tensor) -> SequenceClassifierOutput:
        B, T = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(T, device=device)

        tok_emb = self.word_token_embedding(input_ids)
        pos_emb = self.position_embedding(position_ids).unsqueeze(0)

        x = tok_emb + pos_emb

        for block in self.h:
            x, _ = block(x)

        x = self.ln_f(x)
        logits = self.score(x[:, -1, :])

        return SequenceClassifierOutput(logits=logits)