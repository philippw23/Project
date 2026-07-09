# k-NN Proxy — Correlation Study (deferred TODO)

Validate whether the logged `knn/bal_acc` proxy predicts downstream malignancy
accuracy better than `retrieval/mean_r1`, then decide which to use for checkpoint
selection. Not yet implemented.

## Plan

- Scatter: `mean_r1` (x) vs `knn/bal_acc` (y), colored/sized by final downstream
  probe accuracy, across saved checkpoints.
- Spearman: `mean_r1` vs downstream acc, and `knn/bal_acc` vs downstream acc.
- Pick whichever proxy correlates better; only then consider keying a
  `best_knn_checkpoint.pt` / early stopping on it.

## The trap to remember

You **cannot** correlate on the 3 near-best checkpoints v2 currently saves
(`best`, `best_retrieval`, `final`) — they cluster and the Spearman is noise. You
need a **quality gradient**: add periodic checkpoint dumps (e.g. `--proxy_ckpt_every N`,
epoch 5 bad → epoch 50 good), run the real downstream probe on each, and correlate.

Also fix which downstream number is ground truth: `downstream.py` reports both
`val/balanced_acc` and `test/balanced_acc`. Selecting a checkpoint on a val-based
proxy and validating against val is optimistic — prefer `test/balanced_acc` as the
ground-truth target.
