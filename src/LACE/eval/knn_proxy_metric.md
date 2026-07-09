# k-NN Proxy Metric — Implementation (v2)

Cheap, downstream-aligned proxy metric computed during LACE v2 pretraining to
observe checkpoint quality alongside `mean_r1`. **Log-only** — it does not drive
checkpoint selection or early stopping (that stays keyed on `retrieval/mean_r1`).

## Idea

Fit a cosine k-NN malignancy classifier on frozen **train** CLS embeddings and
score it on the **val** set. If pretraining makes the visual embedding space more
malignancy-separable, this number should rise — giving a per-epoch proxy for how
useful the encoder will be downstream, without training a probe.

## Resolved design

- **Scope**: v2 only (`train/pretrain_v2.py`). v1 untouched.
- **Labels**: GT malignancy label (`benign`/`intermediate`/`malignant`) threaded
  through `InternalDatasetV2.__getitem__` as an int via `LABEL_TO_INT`
  (`data/datasets.py`). Correspondence embedding↔label is guaranteed by
  construction (read from the batch), not by index alignment — safe under the
  phrase-mode sample dropping.
- **Reference bank**: a dedicated **no-augmentation, `shuffle=False`** train-eval
  loader (`preprocess_val` over `pretrain_samples`), so the bank is a clean
  function of the encoder weights only.
- **Embedding**: **image-only** CLS via `vit.forward_cls` (no age/sex metadata).
  Measures pure visual-encoder separability — the thing pretraining changes.
- **Classifier**: `KNeighborsClassifier(metric="cosine", algorithm="brute",
  weights="uniform")`. Cosine matches the contrastive space; uniform voting, no
  neighbor weighting (imbalance is handled on the eval side).
- **k**: both **k=5 and k=20**. Treat the k5–k20 gap as an imbalance canary, not
  two competing metrics.
- **Scoring**: **balanced accuracy + macro F1** on val.
- **Cadence**: every epoch, alongside retrieval.

## Logged keys (`knn/` namespace)

Merged into the existing `wandb.log(...)` and printed under the retrieval line:

```
knn/bal_acc_k5   knn/f1_k5   knn/bal_acc_k20   knn/f1_k20   knn/n_train   knn/n_val
```

## Files touched

- `data/datasets.py` — `LABEL_TO_INT`; `label` added to `InternalDatasetV2` batch.
- `eval/knn_probe.py` — `evaluate_knn_probe(...)` (new).
- `train/pretrain_v2.py` — no-aug `knn_train_loader`; call + print + W&B merge in
  the epoch eval loop.

## Deferred

The correlation study (does `knn/bal_acc` predict downstream accuracy better than
`mean_r1`?) is **not** implemented. See `knn_proxy_correlation_study.md`.
