# LACE evidence-prototype loss (`evid`) — design plan

An LGDEA-style ([LLM-Guided Diagnostic Evidence Alignment](https://arxiv.org/abs/2602.07540),
2026) diagnostic-prototype space, added to LACE v2 as a new `evid` loss term that is an
**alternative to `sim`** for local befund→image grounding.

## Summary of decisions

| Branch | Decision |
|--------|----------|
| Mechanism | LGDEA-faithful: K learnable prototypes, soft assignment, reconstruction + evidence KL |
| Scope | **Prototype space only** (L_rec + L_evid_p). No graph label-propagation, no L_evi-align, no L_evid_u |
| Role | `evid` is an **alternative to `sim`**. `ita` stays and still carries retrieval / early-stopping; `ortho`+`dice` stay |
| Text source | **Befund findings phrases** (the LGDEA "diagnostic evidence" analog; already used by L_sim) |
| Image side | **Reuse `MaskTokenDecoder` tokens** as lesion queries (dice/ortho already ground them) + one new φ head |
| Evidence dim | **Reuse embed_dim=512** — prototypes live in the existing contrastive space |
| Evid composition | **L_rec + L_evid_p only** (paper-paired; no cross-image negatives). Accepted collapse risk → monitor |
| L_rec gradient | **Joint** (updates µ_k *and* cls_proj), paper-faithful |
| Curriculum | **L_rec in stage 1** (warm up µ_k), **L_evid_p in stage 2** (once tokens are dice-grounded) |
| K (prototypes) | **32**, sweepable {16, 32, 64} |
| Assignment temp | **Shared fixed τ_proto = 0.1** (τ_t = τ_p), sweepable {0.05, 0.1, 0.2} |
| Prototype init | **N(0, 0.02), unnormalized**, with Eq-5 `Σ‖µ_k‖²` shrinkage (λ_µ ≈ 0.01) |

## Mechanism (end to end)

Prototypes: `C = {µ_k}`, `K=32`, `µ_k ∈ R^512`, init `N(0, 0.02)`, unnormalized.

**Text side** (befund phrases `z_n`, from existing `encode_befund_phrases` → `cls_proj`, L2-normalised):
- `p(k|z_n) = softmax(z_n · µ_k / τ_proto)`                     (Eq 4)
- `Q̄_R(k) = mean over befund phrases n of the image` of `p(k|z_n)`  (Eq 8)

**Image side** (reuse `MaskTokenDecoder` tokens `v_ℓ`, `[B, N, 768]`):
- `φ = ProjectionHead(768 → 512)` (new head, L2-normalises output)
- `Q_I(ℓ,k) = softmax(φ(v_ℓ) · µ_k / τ_proto)`                  (Eq 6)
- `Q̄_I = mean over N mask tokens` of `Q_I(ℓ,·)`                 (Eq 7)

**Losses:**
- `L_rec = ‖stop? z_n − Σ_k p(k|z_n) µ_k‖² + λ_µ Σ_k ‖µ_k‖²`    (Eq 5, joint → no stop-grad on z_n)
- `L_evid_p = KL(Q̄_R ‖ Q̄_I)` per pair, report is teacher (detach `Q̄_R`)  (Eq 9)

`τ_proto = 0.1` fixed, shared across both modalities so the two distributions have equal
sharpness and their KL is well-posed.

## Curriculum

- **Stage 1** (`dice, ortho, ita, L_rec`): prototypes warm up on the fixed findings
  distribution while the mask decoder learns to localise.
- **Stage 2** (`+ L_evid_p`): image prototype-distributions align to report ones, now that
  both µ_k and the mask tokens are meaningful.

`rec` (L_rec) and `evid` (L_evid_p) are separate, independently-stageable keys in
`pretrain_v2.py`. The two-stage split above is the **default**; any loss can be reassigned to
specific stages (or disabled) via `--loss_stages`, e.g.
`--loss_stages dice:1,2 ortho:1,2 rec:1 ita:2 evid:2 sim:none`. When `--loss_stages` is
omitted, the map is derived from `--losses` using the default curriculum (the `evid` selector
expands to `rec` + `evid`).

## Watch-items (accepted, not blockers)

1. **cls_proj coupling.** Joint L_rec + active L_ITA both shape the shared `cls_proj`.
   Reconstruction pressure competes with contrastive alignment. Monitor L_ITA and retrieval
   when `evid` is on.
2. **Collapse under no-discrimination L_evid_p.** With only per-pair KL and no batch
   negatives, `Q̄_I` can collapse to the batch-average distribution. Dice-supervised mask
   tokens give implicit grounding that mitigates this. **Log prototype-usage entropy**
   `H(mean_n p(k|z_n))` and per-image Q̄ entropy to catch collapse. Escape hatch: add the
   evidence-space InfoNCE term (declined for now) behind a flag.

## Implementation surface

- **New module** `src/LACE/models/prototypes.py`: `PrototypeBank(nn.Module)` owning `µ_k`,
  the φ image head, `assign()` (softmax over prototypes), and aggregation helpers.
- **New objective** in `src/LACE/loss/objectives.py`: `evidence_prototype_loss(...)`
  returning `(l_rec, l_evid_p, usage_entropy)`.
- **`pretrain_v2.py`**: instantiate `PrototypeBank`; add args `--n_prototypes 32`,
  `--tau_proto 0.1`, `--lambda_mu 0.01`, `--lambda_rec`, `--lambda_evid`; `"evid"` added to
  `--losses` choices; curriculum gating (rec stage 1, evid_p stage 2); `log_lambda_rec` and
  `log_lambda_evid` weights; log `l_rec`, `l_evid_p`, `proto_entropy`; add prototype + φ
  state to the checkpoint.
- **BTXRD unchanged** (dice+ortho only; no befund → no evidence term).
- **Downstream / kNN / retrieval eval unchanged** (CLS-based).
- **New sbatch** `sbatch/lace/run_lace_pretrain_v2_evid.sh` with `LOSSES="ita evid ortho dice"`.
