# patch_seg_head — Entfernte Idee (Ablation)

## Was es war

`patch_seg_head = nn.Linear(768, 1, bias=False)` — ein linearer Probe direkt auf den ViT Patch-Features ohne Zwischenschicht.

**Forward:**
```python
patch_logits = patch_seg_head(patch_feat).squeeze(-1)  # [B, 196]
```

**Dice Loss (primary path):**
```python
l_dice = compute_l_dice_ce(
    patch_logits[has_mask].unsqueeze(1), plabels[has_mask],
    pixel_mask=pxmask[has_mask],
)
```

## Idee dahinter

Kurzer Gradientenpfad direkt zum ViT: ohne `MaskTokenDecoder` + `MaskPredictionHead` dazwischen bekommen die ViT Patch-Features einen direkten, einfachen Supervisionssignal für Läsionssegmentierung. Die Mask Token Pipeline war als "auxiliary" zusätzlich da.

**Visualisierung:** `sigmoid(patch_seg_head(patch_feat))` → 14×14 Heatmap — einfachste, interpretierbarste räumliche Ausgabe.

**Downstream `extract_v2_representations`:**
```python
patch_probs = torch.sigmoid(patch_seg_head(patch_feat).squeeze(-1))  # [B, 196]
w    = patch_probs / (patch_probs.sum(-1, keepdim=True) + 1e-8)
m_fg = (w.unsqueeze(-1) * patch_feat).sum(1)                          # [B, 768]
repr_ = vit.patch_proj(m_fg)                                           # [B, 512]
```

## Warum entfernt

- Architektur inkonsistent: `patch_seg_head` und `MaskPredictionHead` machen semantisch das Gleiche (Läsions-Heatmap), aber auf verschiedenen Wegen
- `vit.patch_proj(m_fg)` für L_sim war falsch — `patch_proj` ist für CLS-Features trainiert, nicht für gewichtete Patch-Aggregate
- Durch Vereinheitlichung auf `MaskPredictionHead`-Heatmap ist Training (L_dice, L_sim) und Inferenz (downstream, Visualisierung) konsistent

## Mögliche Wiederverwendung

Falls der Dice Loss über `MaskPredictionHead` nicht konvergiert (Gradienten zu schwach durch den langen Pfad), könnte `patch_seg_head` als **Warmstart-Hilfe in Stage 1** wieder eingeführt werden: erst mit direktem linearem Probe trainieren, dann in Stage 2 nur noch Mask Tokens.
