"""GLoRIA's global+local contrastive loss, ported onto BiomedCLIP's ViT-B/16 + PubMedBERT.

attention_fn / local_loss / global_loss are a verbatim port of GLoRIA's original
algorithm (src/gloria/gloria/loss/gloria_loss.py, itself adapted from
https://github.com/mrlibw/ControlGAN) — same word-region attention, same
log-sum-exp word aggregation, same softmax temperatures. Two adaptations were
needed to fit BiomedCLIP's architecture instead of GLoRIA's ResNet50+BERT:

  - Image side: GLoRIA's local features come from a CNN's spatial feature map
    (its own ResNet50 layer3, 19x19). BiomedCLIP's ViT-B/16 has no such map, so
    the 196 patch tokens from `visual.trunk.forward_features` (14x14 grid) are
    reshaped into the same [B, D, H, W] convention local_loss expects.
  - Text side: GLoRIA aggregates BPE sub-word pieces into whole-word embeddings
    before the local loss. That aggregation is not reproduced here — every
    sub-word token is treated as its own "word" — since PubMedBERT's own
    (non-aggregated) hidden states are what's available from BiomedCLIP's text
    tower. [CLS]/[SEP]/[PAD] tokens are excluded from the attention explicitly
    (GLoRIA's original code left a dead TODO for [SEP] and never did this).

Both towers already share hidden dim 768, so no extra projection heads are
introduced — attention_fn/local_loss operate on the raw trunk features exactly
as GLoRIA's do on its own raw CNN/BERT features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Variable


def cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity between x1 and x2 along dim. (GLoRIA gloria_loss.py:11)"""
    w12 = torch.sum(x1 * x2, dim)
    w1  = torch.norm(x1, 2, dim)
    w2  = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def attention_fn(query: torch.Tensor, context: torch.Tensor, temp1: float):
    """Word-to-region soft attention. (GLoRIA gloria_loss.py:19)

    query:   batch x ndf x queryL   (word embeddings for one caption, repeated over the batch)
    context: batch x ndf x ih x iw  (image patch/region feature map, sourceL = ih*iw)
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    context  = context.view(batch_size, -1, sourceL)          # batch x ndf x sourceL
    contextT = torch.transpose(context, 1, 2).contiguous()    # batch x sourceL x ndf

    attn = torch.bmm(contextT, query)                         # batch x sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    attn = attn.view(batch_size, sourceL, queryL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    attnT = torch.transpose(attn, 1, 2).contiguous()          # batch x sourceL x queryL

    weightedContext = torch.bmm(context, attnT)               # batch x ndf x queryL
    return weightedContext, attn.view(batch_size, -1, ih, iw)


def global_loss(cnn_code: torch.Tensor, rnn_code: torch.Tensor, eps: float = 1e-8, temp3: float = 10.0):
    """Sentence/CLS-level symmetric contrastive loss. (GLoRIA gloria_loss.py:60)"""
    batch_size = cnn_code.shape[0]
    labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0   = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3
    scores0 = scores0.squeeze()

    scores1 = scores0.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)
    return loss0, loss1


def local_loss(
    img_features: torch.Tensor,
    words_emb: torch.Tensor,
    cap_lens: list[int],
    temp1: float = 4.0,
    temp2: float = 5.0,
    temp3: float = 10.0,
    agg: str = "sum",
):
    """Word-region contrastive loss. (GLoRIA gloria_loss.py:85)

    img_features: batch x ndf x ih x iw
    words_emb:    batch x ndf x max_words  (real words packed contiguously at
                  the front of dim 2 for each sample; padding/CLS/SEP already
                  stripped by the caller — see cap_lens)
    cap_lens:     number of real word tokens per sample
    """
    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    for i in range(words_emb.shape[0]):
        words_num = cap_lens[i]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # 1 x ndf x words_num
        word = word.repeat(batch_size, 1, 1)                          # batch x ndf x words_num
        context = img_features

        weiContext, attn = attention_fn(word, context, temp1)

        att_maps.append(attn[i].unsqueeze(0).contiguous())
        word        = word.transpose(1, 2).contiguous()
        weiContext  = weiContext.transpose(1, 2).contiguous()

        word       = word.view(batch_size * words_num, -1)
        weiContext = weiContext.view(batch_size * words_num, -1)

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)

        row_sim.mul_(temp2).exp_()
        row_sim = row_sim.sum(dim=1, keepdim=True) if agg == "sum" else row_sim.mean(dim=1, keepdim=True)
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities  = torch.cat(similarities, 1)
    similarities  = similarities * temp3
    similarities1 = similarities.transpose(0, 1)

    labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)
    loss0 = nn.CrossEntropyLoss()(similarities, labels)
    loss1 = nn.CrossEntropyLoss()(similarities1, labels)
    return loss0, loss1, att_maps


def word_mask_to_cap_lens(word_mask: torch.Tensor) -> list[int]:
    """Per-sample count of real word tokens, given a [B, T] bool mask with all
    real-word positions packed contiguously at the front (see build_word_mask)."""
    return word_mask.sum(dim=1).tolist()


def build_word_mask(input_ids: torch.Tensor, pad_token_id: int, cls_token_id: int, sep_token_id: int) -> torch.Tensor:
    """[B, T] bool mask selecting real word tokens (excludes CLS/SEP/PAD).

    Relies on PubMedBERT's tokenization layout — CLS at position 0, then real
    words contiguously, then SEP, then PAD — so after dropping the CLS column
    from the hidden-state tensor, `word_mask[:, :n]` for n = cap_len selects
    exactly the real words. See build_local_text_inputs.
    """
    return (input_ids != pad_token_id) & (input_ids != cls_token_id) & (input_ids != sep_token_id)


def build_local_image_features(patch_tokens: torch.Tensor) -> torch.Tensor:
    """[B, N, D] ViT patch tokens (CLS already dropped) -> [B, D, H, W] map."""
    b, n, d = patch_tokens.shape
    side = int(round(n ** 0.5))
    assert side * side == n, f"patch token count {n} is not a perfect square"
    return patch_tokens.transpose(1, 2).reshape(b, d, side, side)


def gloria_combined_loss(
    image_feat: torch.Tensor,
    text_feat: torch.Tensor,
    patch_tokens: torch.Tensor,
    word_tokens: torch.Tensor,
    input_ids: torch.Tensor,
    pad_token_id: int,
    cls_token_id: int,
    sep_token_id: int,
    temp1: float,
    temp2: float,
    temp3: float,
    local_loss_weight: float,
    global_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """GLoRIA's combined global+local loss (gloria_model.py:24-26,72-80):

        loss = (l0 + l1) * local_loss_weight + (g0 + g1) * global_loss_weight

    image_feat:   [B, D] pooled/projected image embedding (model.encode_image)
    text_feat:    [B, D] pooled/projected text embedding  (model.encode_text)
    patch_tokens: [B, 196, 768] ViT patch tokens, CLS dropped (visual.trunk.forward_features()[:, 1:, :])
    word_tokens:  [B, 256, 768] PubMedBERT per-token hidden states (model.text with output_tokens=True)
    input_ids:    [B, 256] token ids for the same batch (to build the word mask)
    """
    g0, g1 = global_loss(image_feat, text_feat, temp3=temp3)

    img_map    = build_local_image_features(patch_tokens)
    word_mask  = build_word_mask(input_ids, pad_token_id, cls_token_id, sep_token_id)
    cap_lens   = word_mask_to_cap_lens(word_mask)

    # Words are already contiguous right after [CLS] in PubMedBERT's tokenization
    # (real words, then [SEP], then [PAD]); dropping the CLS column at index 0
    # lines a sample's cap_lens[i] real words up with words_emb[i, :, :cap_lens[i]].
    words_emb = word_tokens[:, 1:, :].transpose(1, 2).contiguous()  # [B, 768, 255]

    valid = [n > 0 for n in cap_lens]
    if not all(valid):
        raise ValueError("gloria_combined_loss: found a sample with zero real word tokens")

    l0, l1, _ = local_loss(img_map, words_emb, cap_lens, temp1=temp1, temp2=temp2, temp3=temp3)

    loss = (l0 + l1) * local_loss_weight + (g0 + g1) * global_loss_weight
    parts = {
        "loss/local_i2t":  l0.item(),
        "loss/local_t2i":  l1.item(),
        "loss/global_i2t": g0.item(),
        "loss/global_t2i": g1.item(),
    }
    return loss, parts
