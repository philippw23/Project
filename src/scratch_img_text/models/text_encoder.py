from __future__ import annotations

import open_clip
import torch
import torch.nn as nn

from biomedclip.utils.misc import MODEL_TAG


def build_tokenizer():
    """Return the underlying HuggingFace PubMedBERT tokenizer from BiomedCLIP.

    open_clip wraps it in an HFTokenizer shim that does not expose the full
    HuggingFace kwargs (max_length, padding, truncation, return_tensors).
    Unwrapping gives a standard PreTrainedTokenizer that supports all of them.
    """
    wrapper = open_clip.get_tokenizer(MODEL_TAG)
    if hasattr(wrapper, "tokenizer"):
        return wrapper.tokenizer
    return wrapper


def get_vocab_size(tokenizer) -> int:
    """Return vocabulary size for a HuggingFace PreTrainedTokenizer."""
    if hasattr(tokenizer, "vocab_size"):
        return tokenizer.vocab_size
    return len(tokenizer)


class TinyTextTransformer(nn.Module):
    """Randomly-initialised transformer text encoder.

    Architecture:
        token_emb  [vocab_size, hidden_dim]
        pos_emb    [max_len,    hidden_dim]
        TransformerEncoder (n_layers × pre-norm TransformerEncoderLayer)
        → CLS token output (position 0, inserted by BiomedCLIP/PubMedBERT tokenizer)

    All weights are initialised randomly; no pretrained checkpoint is loaded.
    The PubMedBERT tokenizer prepends [CLS] to every sequence, so position 0
    of the output carries the sentence-level representation.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        max_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_emb   = nn.Embedding(max_len, hidden_dim)
        self.max_len   = max_len
        self.out_dim   = hidden_dim

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,    # pre-norm (more stable for from-scratch training)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers,
                                              enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_dim)

        self._init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print(
            f"TinyTextTransformer: vocab={vocab_size}, hidden={hidden_dim}, "
            f"layers={n_layers}, heads={n_heads}, max_len={max_len} | "
            f"params: {n_params:,}"
        )

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.zeros_(self.token_emb.weight[0])  # keep padding_idx = 0 vector zeroed
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,      # (B, L)
        attention_mask: torch.Tensor,  # (B, L)  1=real token, 0=pad
    ) -> torch.Tensor:                 # (B, hidden_dim)  CLS representation
        L = min(input_ids.size(1), self.max_len)
        input_ids      = input_ids[:, :L]
        attention_mask = attention_mask[:, :L]

        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x   = self.token_emb(input_ids) + self.pos_emb(pos)

        # TransformerEncoder expects key_padding_mask where True = ignore (pad)
        key_padding_mask = attention_mask == 0

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        return x[:, 0, :]  # CLS token at position 0
