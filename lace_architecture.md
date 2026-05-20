# LACE Architecture Diagrams

## Comparison at a Glance

| | LACE v1 | LACE v2 |
|---|---|---|
| **Stage 1** | L_ITA | L_ITA |
| **Stage 2** | L_ITA + L_sim | L_ITA + L_seg |
| **Stage 3** | L_ITA + L_sim + L_ortho | L_ITA + L_seg + L_sim |
| **Image inputs** | Full image + explicit lesion crop | Full image only |
| **Local alignment** | GLoRIA attn pool on crop patches ↔ befund words | Heatmap-guided pool on full patches ↔ befund tokens |
| **Structural loss** | L_ortho — cosine-sim regulariser (no labels) | L_seg — Dice+BCE vs GT patch labels |
| **BTXRD active from** | Stage 3 | Stage 2 |
| **Extra module** | — | MaskTokenModule |

---

## LACE v1 — `src/LACE/train/pretrain.py`

Three-stage curriculum: **L_ITA → L_ITA + L_sim → L_ITA + L_sim + L_ortho**

```mermaid
flowchart TD
    classDef inp  fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    classDef mod  fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef emb  fill:#ede9fe,stroke:#7c3aed,color:#3730a3
    classDef loss fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef tot  fill:#fee2e2,stroke:#dc2626,color:#7f1d1d

    I1([Full Image\n224×224])
    I2([Lesion Crop\n224×224  —  stage 2+])
    I3([Beurteilung\nassessment text])
    I4([Befund\nfindings text  —  stage 2+])
    I5([BTXRD images\n+ patch labels  —  stage 3+])
    class I1,I2,I3,I4,I5 inp

    subgraph SV["SharedViT  (BiomedCLIP ViT-B/16 + LoRA)"]
        VT[ViT-B/16 trunk\nLoRA injected into Q and V of last k transformer blocks\nonly LoRA delta weights are trained]
        IP[img_proj\nLinear 768→256  +  L2-norm]
        PP[patch_proj\nLinear 768→256  +  L2-norm]
    end
    class VT,IP,PP mod

    subgraph TE["BiomedCLIPTextEncoder  (frozen PubMedBERT)"]
        BT[PubMedBERT transformer\nall weights frozen]
        CP[cls_proj\nLinear 768→256  +  L2-norm]
        WP[word_proj\nLinear 768→256  +  L2-norm]
    end
    class BT,CP,WP mod

    ZI["z_img\n[B, 256]"]
    ZT["z_text\n[B, 256]"]
    ZBC["z_bef_cls\n[B, 256]"]
    PPATCHES["proj_patches\n[B, 196, 256]  lesion crop"]
    PWORDS["proj_words\n[B, L, 256]"]
    RAWP["raw patch tokens\n[B, 196, 768]  full image  +  BTXRD"]
    class ZI,ZT,ZBC,PPATCHES,PWORDS,RAWP emb

    subgraph S1["Stage 1  (always active)"]
        L_ITA["L_ITA\nMedCLIP-style soft InfoNCE\nsoft targets = pairwise text-text similarity\nboth image→text and text→image directions"]
    end
    class L_ITA loss

    subgraph S2["Stage 2+  (requires segmentation mask + befund)"]
        GLORIA["GLoRIA attention pooling\nA  =  softmax( proj_patches · proj_words^T / √D,  dim=patches )\ncontext  =  A^T · proj_patches      →      v_lesion  =  norm( mean(context) )"]
        L_SIM["L_sim\nSymmetric InfoNCE\nv_lesion  ↔  z_bef_cls"]
    end
    class GLORIA,L_SIM loss

    subgraph S3["Stage 3+  (requires GT patch labels; BTXRD active)"]
        L_ORTHO["L_ortho\nOrthogonality regulariser  (no explicit label supervision)\nv_lesion  =  mean patch token where label = 1\nv_bg  =  mean patch token where label = 0\nminimise  cosine_sim( v_lesion,  v_bg )  in raw 768-d ViT space"]
    end
    class L_ORTHO loss

    LTOT["L  =  exp(log_λ_ITA) · L_ITA  +  λ_sim · L_sim  +  λ_reg · L_ortho\nlog_λ_ITA is a learnable scalar parameter"]
    class LTOT tot

    I1 -->|forward_all → CLS token| VT --> IP --> ZI
    I1 --> VT -->|196 raw patch tokens| RAWP
    I2 -->|forward_patches| VT --> PP --> PPATCHES
    I5 --> VT

    I3 --> BT --> CP --> ZT
    I4 --> BT
    BT --> CP --> ZBC
    BT --> WP --> PWORDS

    ZI  --> L_ITA
    ZT  --> L_ITA

    PPATCHES --> GLORIA
    PWORDS   --> GLORIA
    GLORIA   --> L_SIM
    ZBC      --> L_SIM

    RAWP --> L_ORTHO

    L_ITA   --> LTOT
    L_SIM   --> LTOT
    L_ORTHO --> LTOT
```

---

## LACE v2 — `src/LACE/train/pretrain_v2.py`

Three-stage curriculum: **L_ITA → L_ITA + L_seg → L_ITA + L_seg + L_sim**

Key change: replaces the explicit lesion crop + L_ortho with a learnable **MaskTokenModule** that learns spatial heatmaps under explicit segmentation supervision (L_seg), then uses those heatmaps for local phrase alignment (L_sim).

```mermaid
flowchart TD
    classDef inp  fill:#dbeafe,stroke:#2563eb,color:#1e3a5f
    classDef mod  fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef emb  fill:#ede9fe,stroke:#7c3aed,color:#3730a3
    classDef loss fill:#fef3c7,stroke:#d97706,color:#78350f
    classDef tot  fill:#fee2e2,stroke:#dc2626,color:#7f1d1d
    classDef new  fill:#fce7f3,stroke:#db2777,color:#831843

    I1([Full Image\n224×224])
    I3([Beurteilung\nassessment text])
    I4([Befund\nfindings text  —  stage 3+])
    I5([BTXRD images\n+ GT patch labels  —  stage 2+])
    I6([Internal GT patch labels\nfrom segmentation masks  —  stage 2+])
    class I1,I3,I4,I5,I6 inp

    subgraph SV["SharedViT  (same as v1)"]
        VT[ViT-B/16 trunk\nLoRA on Q and V of last k blocks]
        IP[img_proj\nLinear 768→256  +  L2-norm]
        PP2[patch_proj\nLinear 768→256  +  L2-norm\nStage 3 only]
    end
    class VT,IP,PP2 mod

    subgraph MTM["MaskTokenModule  ←  NEW in v2"]
        MTM_DESC["N learnable mask tokens  [N, 768]\n\n1.  Cross-attention:   Q = mask tokens,  K/V = patch tokens\n    → updated mask features  [B, N, 768]\n\n2.  Self-attention on concat( patches, updated_mask )  [B, m+N, 768]\n    → final mask features  mask_feats  [B, N, 768]\n\n3.  Heatmap:  M_proj = L2_norm( mask_proj( mask_feats ) )\n    H_logits = M_proj · P_norm^T / τ     [B, N, m]\n    H_soft   = softmax( H_logits, dim=-1 )   [B, N, m]"]
    end
    class MTM_DESC new

    subgraph TE["BiomedCLIPTextEncoder  (same as v1)"]
        BT[PubMedBERT transformer frozen]
        CP2[cls_proj\nLinear 768→256  +  L2-norm]
        WP2[word_proj\nLinear 768→256  +  L2-norm]
    end
    class BT,CP2,WP2 mod

    ZI["z_img\n[B, 256]"]
    ZT["z_text\n[B, 256]"]
    HS["H_soft   [B, N, m]\nH_logits  [B, N, m]"]
    PP_emb2["proj_patches\n[B, 196, 256]"]
    PW_emb2["proj_words\n[B, J, 256]"]
    class ZI,ZT,HS,PP_emb2,PW_emb2 emb

    subgraph S1v2["Stage 1  (always active)"]
        L_ITAv2["L_ITA\nSoft-label InfoNCE  (same as v1)"]
    end
    class L_ITAv2 loss

    subgraph S2v2["Stage 2+  (requires GT patch labels; BTXRD active from here)"]
        L_SEG["L_seg\nDice  +  BCE\npred = sigmoid( H_logits.mean(dim=1) )   [B, m]\nvs GT patch labels  [B, m]\napplied to internal samples with masks  +  all BTXRD samples"]
    end
    class L_SEG loss

    subgraph S3v2["Stage 3+  (requires befund text)"]
        SLV2["sim_loss_v2  pooling\nH_mean  =  H_soft.mean( dim=1 )                       [B, m]\nv_local =  norm( H_mean @ proj_patches )              [B, D]\nα       =  softmax( v_local · proj_words^T / τ_s )    [B, J]\nt̂      =  norm( α @ proj_words )                     [B, D]\nsoft targets w = softmax( t̂ · t̂^T / τ_s )"]
        L_SIMv2["L_sim\nSoft InfoNCE\nv_local  ↔  t̂"]
    end
    class SLV2,L_SIMv2 new

    LTOTv2["L  =  exp(log_λ_ITA) · L_ITA  +  λ_seg · L_seg  +  λ_sim · L_sim\nlog_λ_ITA is a learnable scalar parameter"]
    class LTOTv2 tot

    I1 -->|forward_all → CLS| VT --> IP --> ZI
    I1 --> VT -->|196 patch tokens| MTM_DESC
    I5 --> VT -->|BTXRD patch tokens| MTM_DESC
    MTM_DESC --> HS

    I1 --> VT -->|196 patch tokens  stage 3| PP2 --> PP_emb2

    I3 --> BT --> CP2 --> ZT
    I4 --> BT --> WP2 --> PW_emb2

    ZI     --> L_ITAv2
    ZT     --> L_ITAv2

    HS     --> L_SEG
    I6     --> L_SEG
    I5     -->|GT labels| L_SEG

    HS       --> SLV2
    PP_emb2  --> SLV2
    PW_emb2  --> SLV2
    SLV2     --> L_SIMv2

    L_ITAv2 --> LTOTv2
    L_SEG   --> LTOTv2
    L_SIMv2 --> LTOTv2
```

---

## Key Architectural Differences

### Local alignment strategy

| | LACE v1 | LACE v2 |
|---|---|---|
| **How lesion region is identified** | Explicit crop from segmentation mask (separate ViT forward pass) | MaskTokenModule learns spatial heatmaps H_soft supervised by GT labels |
| **Patch pool for L_sim** | `patch_proj(crop_patches)` — projection of lesion-crop patch tokens | `H_mean @ patch_proj(full_patches)` — heatmap-weighted sum of full-image patch tokens |
| **Text representation for L_sim** | GLoRIA: attn over befund word tokens → context vectors → v_lesion | Phrase attention: v_local attends over befund tokens → t̂ |
| **Contrastive pair** | v_lesion ↔ z_bef_cls | v_local ↔ t̂ |

### MaskTokenModule internals (v2 only)

```
patch_tokens [B, 196, 768]
        │
        ▼  Cross-Attention  (Q = N mask tokens,  K/V = patch tokens)
updated_mask [B, N, 768]
        │
        ▼  concat([patch_tokens, updated_mask])  →  Self-Attention
seq      [B, 196+N, 768]   split → mask_feats [B, N, 768]
        │
        ▼  mask_proj + L2-norm  →  cosine sim with L2-normed patch tokens
H_logits [B, N, 196]     ← raw logits, used for L_seg  (Dice + BCE)
H_soft   [B, N, 196]     ← softmax over patches, used for L_sim pooling
```
