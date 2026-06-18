# LACE Mask Token Decoder — Implementation Plan

Agreed design from architecture review. All decisions are final — implement directly from this document.

---

## Summary of Design Decisions

| Decision | Choice |
|---|---|
| Base file | Update `pretrain_v2.py` in place |
| Architecture | v1 losses + new `MaskTokenDecoder` (no curriculum staging) |
| L_ITA | Unchanged — symmetric soft semantic (v1 style) |
| L_sim | `m_fg` replaces GLoRIA patch attention loop entirely |
| L_ortho | fg token vs mean of N-1 bg tokens (cosine push-apart) |
| L_dice | Dice + BCE on fg token mask prediction vs GT patch labels |
| Self-attention scope | Tokens only (NOT concat with patches) |
| Projection for `m_fg` | Repurpose existing `vit.patch_proj` (768→512, L2-norm) |
| `patch_proj` role | No longer used for patches — solely for projecting `m_fg` |
| Image input | Single image per sample — no global/crop split anywhere |
| BTXRD | Participates in both L_ortho and L_dice via decoder |
| Inference fg_idx | `mask_logits.sigmoid().sum(-1).argmax(1)` (no GT needed) |
| Loss weights | `--learn_loss_weights` flag covers all lambdas incl. new ones |
| No curriculum | All losses active from epoch 1 |
| Downstream visual | `--downstream_visual_mode cls \| fg \| cls_fg` |
| Downstream file | Update `downstream.py` v2 path in place |

---

## Files to Change

| File | Action |
|---|---|
| `src/LACE/models/mask_tokens.py` | Replace `MaskTokenModule` with `MaskTokenDecoder` + `MaskPredictionHead` + losses |
| `src/LACE/train/pretrain_v2.py` | Rewrite training loop — no stages, new losses |
| `src/LACE/loss/objectives.py` | Add `mask_token_ortho_loss`, `compute_l_dice_ce` |
| `src/LACE/models/downstream.py` | Replace `LACEv2Classifier` to use new decoder |
| `src/LACE/train/downstream.py` | Update `build_v2_model`, `extract_v2_representations`, `train_one_epoch_v2` |
| `src/LACE/train/sweep_pretrain_v2.yaml` | Add new hyperparameters |

`pretrain.py` (v1) is **not touched** — it stays as ablation baseline.

---

## 1. `src/LACE/models/mask_tokens.py` — Full Replacement

Delete the entire `MaskTokenModule` class. Replace with the following three components.

### 1a. `MaskTokenDecoder`

```python
class MaskTokenDecoder(nn.Module):
    """
    N learnable mask tokens cross-attend to ViT patch tokens, then self-attend
    among themselves. Returns refined tokens and cross-attention weights.

    Architecture (one block, no FFN):
        cross_attn: Q=mask_tokens, K=V=patch_tokens  → upd [B,N,d]  + weights [B,N,P]
        layer_norm + residual
        self_attn:  Q=K=V=refined tokens              → upd2 [B,N,d]
        layer_norm + residual

    cross_attn_weights [B,N,P] are the free spatial heatmap used for visualisation.
    They are softmax-normalised per row (sum to 1 over P patches per token).
    They are NOT used as the segmentation loss target — see MaskPredictionHead.
    """

    def __init__(
        self,
        n_tokens: int = 4,
        vit_dim: int = 768,
        n_heads: int = 8,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.vit_dim  = vit_dim

        self.mask_tokens = nn.Parameter(torch.randn(1, n_tokens, vit_dim) * 0.02)

        self.cross_attn = nn.MultiheadAttention(vit_dim, n_heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(vit_dim)

        self.self_attn  = nn.MultiheadAttention(vit_dim, n_heads, batch_first=True)
        self.self_norm  = nn.LayerNorm(vit_dim)

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: [B, P, vit_dim]  raw ViT patch embeddings (P=196)
        Returns:
            tokens:            [B, N, vit_dim]  refined mask token features
            cross_attn_weights:[B, N, P]         softmax spatial map (for viz)
        """
        B = patch_tokens.shape[0]
        tokens = self.mask_tokens.expand(B, -1, -1)               # [B, N, d]

        upd, cross_w = self.cross_attn(
            tokens, patch_tokens, patch_tokens, need_weights=True, average_attn_weights=False
        )
        # cross_w shape from MHA with average_attn_weights=False: [B, n_heads, N, P]
        # Average over heads → [B, N, P]
        cross_attn_weights = cross_w.mean(dim=1)
        tokens = self.cross_norm(tokens + upd)                     # [B, N, d]

        upd2, _ = self.self_attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.self_norm(tokens + upd2)                     # [B, N, d]

        return tokens, cross_attn_weights

    def __repr__(self) -> str:
        return (f"MaskTokenDecoder(n_tokens={self.n_tokens}, "
                f"vit_dim={self.vit_dim}, "
                f"n_heads={self.cross_attn.num_heads})")
```

### 1b. `MaskPredictionHead`

```python
class MaskPredictionHead(nn.Module):
    """Projects mask tokens and computes dot-product similarity with patch embeddings.

    Output mask_logits [B, N, P] are raw logits (no sigmoid) for use in dice+BCE.
    These are independent of the cross_attn_weights in MaskTokenDecoder.
    A linear projection of tokens before the dot product gives the head its own
    learnable mapping from token space to patch-comparison space.
    """

    def __init__(self, vit_dim: int = 768) -> None:
        super().__init__()
        self.proj = nn.Linear(vit_dim, vit_dim, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,       # [B, N, vit_dim]
        patch_tokens: torch.Tensor, # [B, P, vit_dim]
    ) -> torch.Tensor:
        """Returns mask_logits [B, N, P]."""
        t_proj  = self.proj(tokens)                                # [B, N, d]
        p_norm  = F.normalize(patch_tokens, dim=-1)                # [B, P, d]
        t_norm  = F.normalize(t_proj, dim=-1)                      # [B, N, d]
        return torch.bmm(t_norm, p_norm.transpose(1, 2))           # [B, N, P]
```

### 1c. `get_fg_idx`

```python
def get_fg_idx(
    mask_logits: torch.Tensor,       # [B, N, P]
    gt_patch_labels: torch.Tensor,   # [B, P]  float32, 1=lesion, 0=background
) -> torch.Tensor:
    """Select foreground token index per batch element (detached — no gradient).

    Training: token with highest sigmoid overlap with GT mask.
    Inference (no GT): token with most total activated patches — call with
        gt_patch_labels = mask_logits.sigmoid().sum(dim=1) (soft proxy).
    """
    with torch.no_grad():
        pred_probs = torch.sigmoid(mask_logits)                    # [B, N, P]
        overlap    = (pred_probs * gt_patch_labels.unsqueeze(1)).sum(-1)  # [B, N]
        return overlap.argmax(dim=1)                               # [B,]  detached
```

---

## 2. `src/LACE/loss/objectives.py` — New Functions

Add to the existing file (do not remove existing functions — `gloria_local_loss` and `ortho_loss` are still used by `pretrain.py` v1).

### 2a. `compute_l_dice_ce`

```python
def compute_l_dice_ce(
    mask_logits: torch.Tensor,       # [B, N, P]
    gt_patch_labels: torch.Tensor,   # [B, P]  float32
    fg_idx: torch.Tensor,            # [B,]    int64, detached
) -> torch.Tensor:
    """Dice + BCE segmentation loss.

    Dice:  on foreground token prediction vs GT mask.
    BCE:   on all tokens — fg token target = gt_patch_labels, bg tokens target = zeros.

    Samples with empty GT masks (sum==0) are excluded from Dice; BCE still applies.
    Returns scalar. Returns 0.0 if B==0.
    """
    B, N, P = mask_logits.shape
    if B == 0:
        return mask_logits.sum() * 0.0

    gt = gt_patch_labels.float()                                   # [B, P]

    # Dice on fg token
    fg_logits = mask_logits[torch.arange(B), fg_idx]              # [B, P]
    fg_pred   = torch.sigmoid(fg_logits)                           # [B, P]
    nonempty  = gt.sum(dim=-1) > 0                                 # [B,]
    if nonempty.any():
        fp = fg_pred[nonempty]
        gp = gt[nonempty]
        inter = (fp * gp).sum(-1)
        dice = 1.0 - (2.0 * inter / (fp.sum(-1) + gp.sum(-1) + 1e-6)).mean()
    else:
        dice = mask_logits.sum() * 0.0

    # BCE on all N tokens
    # Build targets: zeros for bg tokens, gt for fg token
    targets = torch.zeros_like(mask_logits)                        # [B, N, P]
    targets[torch.arange(B), fg_idx] = gt
    bce = F.binary_cross_entropy_with_logits(mask_logits, targets)

    return dice + bce
```

### 2b. `mask_token_ortho_loss`

```python
def mask_token_ortho_loss(
    tokens: torch.Tensor,   # [B, N, vit_dim]
    fg_idx: torch.Tensor,   # [B,]  int64, detached
) -> torch.Tensor:
    """Push fg token embedding away from mean of bg token embeddings.

    Operates in raw 768-dim mask token space (same space as vit.patch_proj input).
    Returns cosine_similarity + 1 (range [0,2]; 0 = maximally separated).
    Returns 0.0 if N==1 (no background tokens to contrast against).
    """
    B, N, D = tokens.shape
    if N == 1:
        return tokens.sum() * 0.0

    fg = tokens[torch.arange(B), fg_idx]                          # [B, D]

    # Mean of non-fg tokens per batch element
    mask = torch.ones(B, N, dtype=torch.bool, device=tokens.device)
    mask[torch.arange(B), fg_idx] = False
    bg = tokens[mask].reshape(B, N - 1, D).mean(dim=1)            # [B, D]

    fg_n = F.normalize(fg, dim=-1)
    bg_n = F.normalize(bg, dim=-1)
    return (F.cosine_similarity(fg_n, bg_n, dim=-1) + 1.0).mean()
```

---

## 3. `src/LACE/train/pretrain_v2.py` — Training Loop Rewrite

### 3a. Imports to change

```python
# Remove:
from LACE.loss.objectives import multi_positive_soft_semantic_loss, seg_loss
from LACE.models.mask_tokens import MaskTokenModule

# Add:
from LACE.loss.objectives import (
    symmetric_soft_semantic_loss,
    compute_l_dice_ce,
    mask_token_ortho_loss,
)
from LACE.models.mask_tokens import MaskTokenDecoder, MaskPredictionHead, get_fg_idx
```

### 3b. `_encode_text_v2` — simplify

Remove the `stage` parameter. Befund phrases are always encoded (no curriculum gating).

```python
def _encode_text_v2(
    batch: dict,
    text_enc: BiomedCLIPTextEncoder,
    device: torch.device,
    text_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode beurteilung for L_ITA and befund phrases for L_sim.

    Returns:
        z_beur_phrases [B, J, D]   — beurteilung phrase embeddings
        beur_pmask     [B, J] bool
        z_bef_phrases  [B, J, D]   — befund phrase embeddings
        bef_pmask      [B, J] bool
    """
    if text_mode == "phrase":
        # beurteilung
        z_beur_phrases = text_enc._encode_phrase_batch(
            batch["beur_phrase_ids"].to(device),
            batch["beur_phrase_attn"].to(device),
            batch["beur_phrase_mask"].to(device),
        )
        beur_pmask = batch["beur_phrase_mask"].to(device)
        # befund
        _, z_bef_phrases = text_enc.encode_befund_phrases(
            batch["bef_phrase_ids"].to(device),
            batch["bef_phrase_attn"].to(device),
            batch["bef_phrase_mask"].to(device),
        )
        bef_pmask = batch["bef_phrase_mask"].to(device)
    else:
        # full-text fallback
        z_text = text_enc.encode_beurteilung(
            batch["beurteilung_ids"].to(device), batch["beurteilung_mask"].to(device)
        )
        z_beur_phrases = z_text.unsqueeze(1)
        beur_pmask = torch.ones(z_beur_phrases.shape[:2], dtype=torch.bool, device=device)
        _, z_bef_phrases = text_enc.encode_befund(
            batch["befund_ids"].to(device), batch["befund_mask"].to(device)
        )
        bef_pmask = batch["befund_mask"].bool().to(device)
    return z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask
```

### 3c. `train_one_epoch` — new signature and body

```python
def train_one_epoch(
    vit: SharedViT,
    text_enc: BiomedCLIPTextEncoder,
    mask_decoder: MaskTokenDecoder,
    mask_head: MaskPredictionHead,
    internal_loader: DataLoader,
    btxrd_loader: DataLoader | None,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    τ: nn.Parameter,
    log_lambda_ita: nn.Parameter,
    log_lambda_sim: nn.Parameter,
    log_lambda_ortho: nn.Parameter,
    log_lambda_dice: nn.Parameter,
    trainable_params: list,
    device: torch.device,
    learn_loss_weights: bool,
    text_mode: str = "phrase",
    max_grad_norm: float = 1.0,
    τ_s_beur: float = 0.015,
    τ_s_bef: float = 0.07,
    τ_s_img_full: float = 0.07,
    λ_t2i: float = 1.0,
    same_image_boost: float = 0.0,
    reweight_by_n_phrases: bool = False,
    t2i_mode: str = "image_image",
) -> dict[str, float]:
    vit.train()
    text_enc.train()
    mask_decoder.train()
    mask_head.train()

    total_ita = total_sim = total_ortho = total_dice = total_total = 0.0
    n_batches = 0
    btxrd_cycle = itertools.cycle(btxrd_loader) if btxrd_loader is not None else None
    zero = torch.zeros(1, device=device)[0]

    for batch in pbar:
        img      = batch["full_image"].to(device)        # [B, 3, 224, 224]
        plabels  = batch["patch_labels"].to(device)      # [B, 196]  float32
        has_mask = batch["has_mask"].to(device)          # [B,]  bool
        has_bef  = batch["has_befund"].to(device)        # [B,]  bool

        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, dtype=torch.float16):

            # ── Single ViT pass ───────────────────────────────────────────────
            cls_raw, patch_feat = vit.forward_all(img)   # [B,768], [B,196,768]
            z_img = vit.img_proj(cls_raw)                # [B, D]

            # ── Mask token decoder ────────────────────────────────────────────
            tokens, cross_attn_w = mask_decoder(patch_feat)     # [B,N,768], [B,N,196]
            mask_logits = mask_head(tokens, patch_feat)          # [B, N, 196]

            # ── Text encoding ─────────────────────────────────────────────────
            z_beur_phrases, beur_pmask, z_bef_phrases, bef_pmask = _encode_text_v2(
                batch, text_enc, device, text_mode
            )

            # ── L_ITA: symmetric soft semantic (v1 style) ─────────────────────
            p2i_ita = _phrase_to_image(beur_pmask)
            l_ita, _, _ = symmetric_soft_semantic_loss(
                z_img[p2i_ita], z_beur_phrases, beur_pmask, τ,
                τ_s_phrase=τ_s_beur, τ_s_img=τ_s_img_full,
                phrase_to_image=p2i_ita, same_image_boost=same_image_boost,
                reweight_by_n_phrases=reweight_by_n_phrases,
                t2i_mode=t2i_mode,
            )

            # ── fg_idx for internal batch (needs GT mask) ─────────────────────
            # Assign fg_idx only for samples with a GT mask; others get token 0.
            fg_idx = torch.zeros(img.shape[0], dtype=torch.long, device=device)
            if has_mask.any():
                fg_idx[has_mask] = get_fg_idx(
                    mask_logits[has_mask], plabels[has_mask]
                )
            m_fg = tokens[torch.arange(img.shape[0]), fg_idx]   # [B, 768]

            # ── L_sim: m_fg projected → matched against befund phrases ─────────
            l_sim = zero
            sim_valid = has_mask & has_bef
            if sim_valid.sum() >= 2:
                m_fg_proj = vit.patch_proj(m_fg[sim_valid])      # [B_sim, D] L2-norm
                bef_sub   = z_bef_phrases[sim_valid]
                bef_pmask_sub = bef_pmask[sim_valid]
                p2i_sim = _phrase_to_image(bef_pmask_sub)
                l_sim, _, _ = symmetric_soft_semantic_loss(
                    m_fg_proj[p2i_sim], bef_sub, bef_pmask_sub, τ,
                    τ_s_phrase=τ_s_bef, τ_s_img=τ_s_img_full,
                    phrase_to_image=p2i_sim, same_image_boost=same_image_boost,
                    reweight_by_n_phrases=reweight_by_n_phrases,
                    t2i_mode=t2i_mode,
                )

            # ── L_ortho: fg token vs mean bg tokens ───────────────────────────
            l_ortho = mask_token_ortho_loss(tokens, fg_idx)

            # ── L_dice: spatial supervision on fg token ───────────────────────
            l_dice = zero
            if has_mask.any():
                l_dice = compute_l_dice_ce(
                    mask_logits[has_mask], plabels[has_mask], fg_idx[has_mask]
                )

            # ── BTXRD: decoder pass for L_ortho + L_dice ─────────────────────
            if btxrd_cycle is not None:
                btxrd_b      = next(btxrd_cycle)
                btxrd_img    = btxrd_b["image"].to(device)
                btxrd_labels = btxrd_b["patch_labels"].to(device)
                _, btxrd_patches    = vit.forward_all(btxrd_img)
                btxrd_tokens, _     = mask_decoder(btxrd_patches)
                btxrd_logits        = mask_head(btxrd_tokens, btxrd_patches)
                btxrd_fg_idx        = get_fg_idx(btxrd_logits, btxrd_labels)
                l_ortho = l_ortho + mask_token_ortho_loss(btxrd_tokens, btxrd_fg_idx)
                l_dice  = l_dice  + compute_l_dice_ce(btxrd_logits, btxrd_labels, btxrd_fg_idx)

            # ── Total loss ────────────────────────────────────────────────────
            loss = (log_lambda_ita.exp()   * l_ita
                  + log_lambda_sim.exp()   * l_sim
                  + log_lambda_ortho.exp() * l_ortho
                  + log_lambda_dice.exp()  * l_dice)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        ...
```

### 3d. Optimizer setup

```python
# Decoder and head are new modules — fully trainable (no LoRA)
decoder_params    = list(mask_decoder.parameters())
head_params       = list(mask_head.parameters())

# Log-lambda parameters (learnable or fixed)
log_lambda_ita   = nn.Parameter(torch.zeros(1))
log_lambda_sim   = nn.Parameter(torch.zeros(1))
log_lambda_ortho = nn.Parameter(torch.zeros(1))
log_lambda_dice  = nn.Parameter(torch.zeros(1))

if learn_loss_weights:
    lambda_params = [log_lambda_ita, log_lambda_sim, log_lambda_ortho, log_lambda_dice]
else:
    # Set to log of desired fixed values, detach — no grad
    log_lambda_ita.data.fill_(0.0)           # exp(0) = 1.0
    log_lambda_sim.data.fill_(0.0)           # exp(0) = 1.0
    log_lambda_ortho.data.fill_(math.log(0.1))
    log_lambda_dice.data.fill_(0.0)          # exp(0) = 1.0
    lambda_params = []
    for p in [log_lambda_ita, log_lambda_sim, log_lambda_ortho, log_lambda_dice]:
        p.requires_grad_(False)

# Weight decay split (same pattern as v1)
no_decay_keys = ("bias", "norm.weight", "norm.bias", "ln_1.weight", "ln_1.bias",
                 "ln_2.weight", "ln_2.bias")
vit_decay, vit_nd  = _split_params(vit)
txt_decay, txt_nd  = _split_params(text_enc)
dec_decay, dec_nd  = _split_params(mask_decoder)
hd_decay,  hd_nd   = _split_params(mask_head)

optimizer = torch.optim.AdamW(
    [
        {"params": vit_decay + txt_decay + dec_decay + hd_decay,
         "weight_decay": args.weight_decay},
        {"params": vit_nd + txt_nd + dec_nd + hd_nd + [τ] + lambda_params,
         "weight_decay": 0.0},
    ],
    lr=args.lr, betas=(0.9, 0.98), eps=1e-6,
)
```

### 3e. Checkpoint format

```python
checkpoint = {
    "epoch":               epoch,
    "vit_state":           vit.state_dict(),
    "mask_decoder_state":  mask_decoder.state_dict(),
    "mask_head_state":     mask_head.state_dict(),
    "text_enc_state":      text_enc.state_dict(),
    "optimizer_state":     optimizer.state_dict(),
    "lora_config": {
        "lora_layers": args.lora_layers,
        "lora_r":      args.lora_r,
        "lora_alpha":  args.lora_alpha,
        "embed_dim":   args.embed_dim,
    },
    "n_mask_tokens":       args.n_mask_tokens,
    "n_mask_heads":        args.n_mask_heads,
    "val_loss":            val_metrics["val/loss"],
}
```

### 3f. New CLI args to add

```python
# Mask token decoder
parser.add_argument("--n_mask_tokens",  type=int,   default=4)
parser.add_argument("--n_mask_heads",   type=int,   default=8)

# Loss weights (initial values when learn_loss_weights=False)
parser.add_argument("--lambda_sim",     type=float, default=1.0)
parser.add_argument("--lambda_ortho",   type=float, default=0.1)
parser.add_argument("--lambda_dice",    type=float, default=1.0)
parser.add_argument("--learn_loss_weights", action="store_true", default=False)

# L_ITA mode (carried over from v1)
parser.add_argument("--t2i_mode",       type=str,   default="image_image",
                    choices=["image_image", "text_text", "descriptor", "infonce"])
parser.add_argument("--tau_s_beur",     type=float, default=0.015)
parser.add_argument("--tau_s_bef",      type=float, default=0.07)
parser.add_argument("--tau_s_img_full", type=float, default=0.07)
parser.add_argument("--lambda_t2i",     type=float, default=1.0)
parser.add_argument("--same_image_boost", type=float, default=0.0)
parser.add_argument("--reweight_by_n_phrases", action="store_true", default=False)
```

Remove `--stage1_epochs`, `--stage2_epochs`, `--stage3_epochs`, `--lambda_seg` — no staging.

---

## 4. `src/LACE/models/downstream.py` — Replace `LACEv2Classifier`

```python
from LACE.models.mask_tokens import MaskTokenDecoder, MaskPredictionHead, get_fg_idx

class LACEv2Classifier(nn.Module):
    """
    Downstream classifier: frozen ViT + frozen MaskTokenDecoder/Head → MLP.

    visual_mode:
        "cls"    → vit.img_proj(cls_raw)              [B, 512]
        "fg"     → vit.patch_proj(m_fg)               [B, 512]
        "cls_fg" → cat([img_proj(cls), patch_proj(m_fg)])  [B, 1024]

    At inference, fg_idx is determined without GT:
        mask_logits.sigmoid().sum(-1).argmax(1)
    """

    def __init__(
        self,
        vit: SharedViT,
        mask_decoder: MaskTokenDecoder,
        mask_head: MaskPredictionHead,
        n_meta_dim: int = 32,
        n_classes: int = 3,
        use_meta: bool = True,
        linear_head: bool = False,
        visual_mode: str = "cls_fg",  # "cls" | "fg" | "cls_fg"
    ) -> None:
        super().__init__()
        self.vit          = vit
        self.mask_decoder = mask_decoder
        self.mask_head    = mask_head
        self.visual_mode  = visual_mode
        self.use_meta     = use_meta

        for p in self.vit.parameters():
            p.requires_grad_(False)
        for p in self.mask_decoder.parameters():
            p.requires_grad_(False)
        for p in self.mask_head.parameters():
            p.requires_grad_(False)

        vit_emb_dim = vit.proj_dim  # 512
        if visual_mode == "cls_fg":
            feat_dim = vit_emb_dim * 2
        else:
            feat_dim = vit_emb_dim

        if use_meta:
            self.age_emb = nn.Linear(1, n_meta_dim // 2)
            self.sex_emb = nn.Embedding(2, n_meta_dim // 2)
            feat_dim += n_meta_dim
        else:
            self.age_emb = None
            self.sex_emb = None

        if linear_head:
            self.head = nn.Linear(feat_dim, n_classes)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim),
                nn.Linear(feat_dim, 256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, n_classes),
            )

    def _get_visual(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            cls_raw, patch_feat = self.vit.forward_all(images)     # [B,768], [B,196,768]
            tokens, _           = self.mask_decoder(patch_feat)    # [B, N, 768]
            mask_logits         = self.mask_head(tokens, patch_feat)  # [B, N, 196]

            # Inference fg selection — no GT
            fg_idx = mask_logits.sigmoid().sum(-1).argmax(1)       # [B,]
            m_fg   = tokens[torch.arange(images.shape[0]), fg_idx] # [B, 768]

        if self.visual_mode == "cls":
            return self.vit.img_proj(cls_raw)                      # [B, 512]
        elif self.visual_mode == "fg":
            return self.vit.patch_proj(m_fg)                       # [B, 512]
        else:  # cls_fg
            return torch.cat([
                self.vit.img_proj(cls_raw),
                self.vit.patch_proj(m_fg),
            ], dim=-1)                                             # [B, 1024]

    def forward(
        self,
        images: torch.Tensor,   # [B, 3, 224, 224]
        age: torch.Tensor,      # [B, 1]  float
        sex: torch.Tensor,      # [B,]    long
    ) -> torch.Tensor:
        visual = self._get_visual(images)
        if self.use_meta:
            age_feat   = self.age_emb(age.float())
            sex_feat   = self.sex_emb(sex.long())
            x = torch.cat([visual, age_feat, sex_feat], dim=-1)
        else:
            x = visual
        return self.head(x)
```

---

## 5. `src/LACE/train/downstream.py` — Update v2 Path

### `build_v2_model`

```python
def build_v2_model(args, device, num_classes=NUM_CLASSES):
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    vit = _load_vit(ckpt, device)                                  # frozen
    preprocess_val   = vit.preprocess_val
    preprocess_train = build_train_transform_lace(preprocess_val)

    n_tokens = ckpt.get("n_mask_tokens", 4)
    n_heads  = ckpt.get("n_mask_heads",  8)

    mask_decoder = MaskTokenDecoder(n_tokens=n_tokens, n_heads=n_heads)
    mask_decoder.load_state_dict(ckpt["mask_decoder_state"])
    mask_decoder.to(device)

    mask_head = MaskPredictionHead()
    mask_head.load_state_dict(ckpt["mask_head_state"])
    mask_head.to(device)

    use_meta     = args.head == "mlp"
    linear_head  = args.head == "linear"
    visual_mode  = getattr(args, "downstream_visual_mode", "cls_fg")

    classifier = LACEv2Classifier(
        vit=vit,
        mask_decoder=mask_decoder,
        mask_head=mask_head,
        n_meta_dim=args.meta_embed_dim,
        n_classes=num_classes,
        use_meta=use_meta,
        linear_head=linear_head,
        visual_mode=visual_mode,
    ).to(device)

    head_wrapper = _V2HeadWrapper(classifier)
    return classifier, head_wrapper, preprocess_train, preprocess_val
```

### `extract_v2_representations`

```python
@torch.no_grad()
def extract_v2_representations(vit, mask_decoder, mask_head, loader, device, visual_mode="cls_fg"):
    vit.eval(); mask_decoder.eval(); mask_head.eval()
    all_repr, all_age, all_sex, all_lbl = [], [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        cls_raw, patch_feat = vit.forward_all(images)
        tokens, _           = mask_decoder(patch_feat)
        mask_logits         = mask_head(tokens, patch_feat)
        fg_idx = mask_logits.sigmoid().sum(-1).argmax(1)
        m_fg   = tokens[torch.arange(images.shape[0]), fg_idx]

        if visual_mode == "cls":
            repr_ = vit.img_proj(cls_raw)
        elif visual_mode == "fg":
            repr_ = vit.patch_proj(m_fg)
        else:
            repr_ = torch.cat([vit.img_proj(cls_raw), vit.patch_proj(m_fg)], dim=-1)

        all_repr.append(repr_.cpu())
        all_age.append(batch["age"])
        all_sex.append(batch["sex"])
        all_lbl.append(batch["label"])
    return torch.cat(all_repr), torch.cat(all_age), torch.cat(all_sex), torch.cat(all_lbl)
```

### New CLI arg for downstream

```python
parser.add_argument("--downstream_visual_mode", type=str, default="cls_fg",
                    choices=["cls", "fg", "cls_fg"])
```

Update `_V2HeadWrapper` to handle variable input dim (currently hardcodes `vit_dim=768`). Change to read `feat_dim` from `classifier.head[1].in_features` (the Linear after LayerNorm) or pass it explicitly.

---

## 6. `src/LACE/train/sweep_pretrain_v2.yaml` — Updated Hyperparameters

```yaml
# Remove: stage1_epochs, stage2_epochs, stage3_epochs, lambda_seg
# Add:
parameters:
  n_mask_tokens:
    values: [2, 4, 8]
  n_mask_heads:
    value: 8
  lambda_dice:
    values: [0.5, 1.0, 2.0]
  lambda_ortho:
    values: [0.05, 0.1, 0.2]
  lambda_sim:
    values: [0.5, 1.0, 2.0]
  learn_loss_weights:
    value: false
  t2i_mode:
    values: ["image_image", "text_text", "infonce"]
  tau_s_beur:
    distribution: log_uniform_values
    min: 0.01
    max: 0.07
  tau_s_bef:
    distribution: log_uniform_values
    min: 0.01
    max: 0.07
```

---

## Key Shape Reference

| Tensor | Shape | Source |
|---|---|---|
| `img` | `[B, 3, 224, 224]` | dataloader `full_image` |
| `cls_raw` | `[B, 768]` | `vit.forward_all` |
| `patch_feat` | `[B, 196, 768]` | `vit.forward_all` |
| `z_img` | `[B, 512]` | `vit.img_proj(cls_raw)` |
| `tokens` | `[B, N, 768]` | `mask_decoder(patch_feat)` |
| `cross_attn_w` | `[B, N, 196]` | `mask_decoder(patch_feat)` |
| `mask_logits` | `[B, N, 196]` | `mask_head(tokens, patch_feat)` |
| `fg_idx` | `[B,]` int64 | `get_fg_idx(...)`, detached |
| `m_fg` | `[B, 768]` | `tokens[range(B), fg_idx]` |
| `m_fg_proj` | `[B_sim, 512]` | `vit.patch_proj(m_fg[sim_valid])` |
| `z_beur_phrases` | `[B, J, 512]` | text encoder |
| `z_bef_phrases` | `[B, J, 512]` | text encoder |
| `beur_pmask` | `[B, J]` bool | dataloader |
| `bef_pmask` | `[B, J]` bool | dataloader |

Default: N=4, P=196, D=512, vit_dim=768, J=16.

---

## What Is Removed vs Kept

| Component | Status |
|---|---|
| `MaskTokenModule` | **Removed** — replaced by `MaskTokenDecoder` + `MaskPredictionHead` |
| `gloria_local_loss` | **Kept in objectives.py** — still used by `pretrain.py` v1 |
| `ortho_loss` | **Kept in objectives.py** — still used by `pretrain.py` v1 |
| `seg_loss` | **Kept in objectives.py** — no longer used but harmless |
| `_compute_crop_features` | **Removed from pretrain_v2.py** — no longer needed |
| `vit.patch_proj` | **Repurposed** — now projects `m_fg` (768→512) instead of raw patches |
| Global/crop image split | **Removed** — single `full_image` input everywhere |
| 3-stage curriculum | **Removed** — all losses active from epoch 1 |
| `multi_positive_soft_semantic_loss` | **No longer used in v2** — `symmetric_soft_semantic_loss` used instead |

---

## Sanity Checks After Implementation

Run these before any full training:

```python
# 1. Shape test
B, N, P, D = 2, 4, 196, 768
patch_feat = torch.randn(B, P, D)
decoder = MaskTokenDecoder(n_tokens=N)
head    = MaskPredictionHead()
tokens, cross_w = decoder(patch_feat)
assert tokens.shape  == (B, N, D)
assert cross_w.shape == (B, N, P)
mask_logits = head(tokens, patch_feat)
assert mask_logits.shape == (B, N, P)

# 2. fg_idx detached
gt = torch.zeros(B, P); gt[0, :10] = 1.0
fg_idx = get_fg_idx(mask_logits, gt)
assert fg_idx.requires_grad == False

# 3. Losses finite
l_dice = compute_l_dice_ce(mask_logits, gt, fg_idx)
l_ortho = mask_token_ortho_loss(tokens, fg_idx)
assert l_dice.isfinite() and l_ortho.isfinite()

# 4. Gradient to mask_tokens
loss = l_dice + l_ortho
loss.backward()
assert decoder.mask_tokens.grad is not None
assert not decoder.mask_tokens.grad.eq(0).all()

# 5. Parameter count
n_new = sum(p.numel() for p in decoder.parameters()) + \
        sum(p.numel() for p in head.parameters())
print(f"New params: {n_new:,}")   # expect ~3-4M
```
