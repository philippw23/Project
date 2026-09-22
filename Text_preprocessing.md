# Text Preprocessing Pipeline

Documents everything that happens to the German radiology reports before they are used in
training (contrastive pretraining text, LACE phrase alignment, downstream text auxiliary loss).

Overall flow:

```
data/text/reports.json
  → src/preprocess_reports.py            → data/text/sanitized_reports.json
  → src/translate_reports.py             → data/text/translated_reports.json
  → src/llm_extractor.py / _seperated.py → full_reports.json / full_reports_separated.json
  → src/create_dataset.py                → data/internal_dataset/dataset_full.json
  → src/create_split.py                  → data/internal_dataset/split.json (split_binary.json)
  → Dataset classes (__getitem__)        → tokenized tensors fed to the model
```

Note: `src/build_dataset_json.py` referenced in CLAUDE.md does not currently exist in the repo;
the functional equivalent is `src/create_dataset.py`.

---

## Stage 0 — `src/preprocess_reports.py`

Filters raw scraped `data/text/reports.json` down to usable reports before translation.

- **Filter** (lines 12–15): keeps a report `r` only if:
  - `r.get("befund", "").strip()` is non-empty, **and**
  - the placeholder string `"Die Bilder wurden bereitgestellt."` (meaning "no real findings were
    dictated") is **not** contained in `befund`.
- Output: `data/text/sanitized_reports.json`. Prints input/removed/output counts.

This is the first "leave out instances without text" gate: reports with an empty or
placeholder-only `befund` never make it past this stage.

---

## Stage 1 — `src/translate_reports.py` (German → English)

Translates `befund`/`beurteilung` fields using a local Qwen instruct LLM (default
`Qwen/Qwen2.5-7B-Instruct`).

- I/O: `data/text/sanitized_reports.json` → `data/text/translated_reports.json` (same entries +
  `befund_en`, `beurteilung_en`).
- Key functions:
  - `load_model(model_name, quantize)` — float16 GPU, optional 4-bit NF4 quantization, float32 CPU
    fallback.
  - `translate_report(befund, beurteilung, tokenizer, model, max_new_tokens=1024)` — builds a chat
    prompt, greedy-decodes (`do_sample=False`), extracts the first `{...}` JSON blob
    (`raw.find("{")`/`raw.rfind("}")`), parses with `json.loads`; on failure returns
    `{"befund_en": "", "beurteilung_en": "", "_error": raw}`.
  - `main()` — resumable: loads any existing output, skips patids already translated
    (keyed by `str(patid)`), saves after **every** entry (crash-safe), `--max` for quick test runs.
- **Empty-input handling**: if `befund`/`beurteilung` is empty, the prompt substitutes the literal
  string `"(empty)"` — every report is still sent to the model, even placeholder-only ones. No
  sample is dropped at this stage.
- System prompt: "You are a medical translator specializing in radiology... Preserve all medical
  terminology... Do not summarize, interpret, or add information... If the input is empty or not
  present, return an empty string. Reply with valid JSON only."
- Params: `--model`, `--quantize` (4-bit NF4), `--max`, `--input`/`--output`.

---

## Stage 2 — LLM phrase extraction

Two CLI entry points, both thin wrappers over `src/qwen_llm_extractor/`:
- `src/llm_extractor.py` → `qwen_llm_extractor.extract.joint`
- `src/llm_extractor_seperated.py` → `qwen_llm_extractor.extract.separated`

### 2a. Joint pipeline — `src/qwen_llm_extractor/extract/joint.py`

One LLM call per report (befund + beurteilung concatenated) →
`{"befund_phrases": [...], "beurteilung_phrases": [...]}`.

- `_entry_key(e)`: dedupe/resume key = `accnr` if present else `patid`.
- `query_llm_batch(...)`: left-pads a batch of chat-templated prompts, greedy `model.generate`,
  parses with `_parse_response` (see 2e). Batched via `--batch_size` (default 4).
- Optional **two-stage mode** (`--two_stage`): stage 1 = atomic phrase extraction, stage 2 =
  per-phrase category+relevance classification (`high`/`medium`/`low`), assembled and sorted by
  relevance; unmatched phrases default to `("befund", "low")` — nothing from stage 1 is dropped.
- **Filtering — which reports get processed** (`main()`, lines 268–272):
  ```python
  to_process = [
      e for e in source_reports
      if _entry_key(e) not in completed
      and (e.get(befund_key, "").strip() or e.get(beur_key, "").strip())
  ]
  ```
  A report is skipped **only if both** befund and beurteilung (or `befund_en`/`beurteilung_en` in
  `--english` mode) are empty/whitespace. Already-completed entries (resume) are also skipped.
- Output merges phrases back onto the source entry, written incrementally after every batch.
- Params: `--english`, `--model` (default `Qwen/Qwen2.5-14B-Instruct`), `--quantize` /
  `--quantize_8bit`, `--max`, `--max_new_tokens` (512), `--two_stage`,
  `--stage2_max_new_tokens` (1024), `--batch_size` (4).

### 2b. Separated pipeline — `src/qwen_llm_extractor/extract/separated.py`

Queries befund and beurteilung independently with section-specific prompts.

- One-shot path (`main()`): calls `query_llm` separately for befund and beurteilung.
  **Fallback** (lines 401–405): if beurteilung yields no phrases, re-derives a diagnosis summary
  from befund text via `BEURTEILUNG_SUMMARY_PROMPT_TEMPLATE`.
- Loads via `load_reports_separated()`, which drops reports where **both** sections are empty (see
  2c).
- Two-stage path (`--two_stage`): stage 1 = atomic per-section extraction, stage 2 = importance
  ranking (`high`/`medium`/`low`/`none`). Phrases explicitly tagged `relevance == "none"` are
  **dropped**; everything else is kept (unmatched → defaults to `"low"`).
- **Filtering — which reports get processed** (lines 289–293):
  ```python
  to_process = [
      e for e in source_reports
      if _entry_key(e) not in completed
      and ((e.get("befund_en") or "").strip() or (e.get("beurteilung_en") or "").strip())
  ]
  ```
  (Uses the English fields; both must be empty to skip.)
- `_batch_extract()`: for a whitespace-only *section* (not whole report), the LLM call is
  **skipped entirely** and `({"phrases": []}, "")` is assigned directly — avoids the model
  replying with non-JSON prose for empty input.
- Writes a debug ranking file to `results/llm_extraction/<output-stem>_ranking.json` including
  dropped `"none"`-relevance phrases.
- Params mirror the joint pipeline: `--reports`, `--model`, `--quantize`/`--quantize_8bit`,
  `--max`, `--max_new_tokens` (512), `--batch_size` (1 default), `--two_stage`, `--out_dir`
  (`results`), `--output`.

### 2c. Loaders — `src/qwen_llm_extractor/data/reports.py`

- `load_reports_joint(source, english=False)`: concatenates befund+beurteilung (double-newline
  separated); **skips reports where both fields are empty** (`if parts:` guard) — silently
  dropped. Supports a legacy plain-string-list JSON format.
- `load_reports_separated(source)`: returns three parallel lists `(befunds, beurteilungs,
  patids)`; **skips reports where both `befund_en` and `beurteilung_en` are empty**. Asserts the
  three lists stay the same length.

### 2d. Model loading — `src/qwen_llm_extractor/models/loader.py`

`load_model(model_id, quantize, quantize_8bit=False)`. `DEFAULT_MODEL =
"Qwen/Qwen2.5-14B-Instruct"`. Precision selection order: CUDA+8bit (bitsandbytes, ~14GB) →
CUDA+4bit (NF4, ~8GB) → CUDA plain float16 (~28GB) → CPU float32.

### 2e. Robust JSON parsing — `src/qwen_llm_extractor/utils/json_repair.py`

`_parse_response(raw)`: strips markdown fences, tries `_extract_json_objects()`
(`json.JSONDecoder.raw_decode` scanning for `{`, tolerating leading/trailing prose and multiple
concatenated JSON objects, merging list-valued keys), falls back to `_repair_json()` (fixes
invalid backslash escapes, missing commas between adjacent strings/after closing brackets, `}`
used to close an array instead of `]`). On total failure returns `{"error": ..., "raw": raw}`.

### 2f. Analysis — `src/qwen_llm_extractor/eval/analysis.py`

`flatten_to_dataframe(results, patids)`: builds a tidy DataFrame (`patid, report_idx, category,
phrase`); a category with no phrases gets a placeholder row `phrase="not reported"`; phrases are
lowercased. `print_summary(df)`: per-category phrase counts/top-10 frequent phrases, plus
failed-parse ("error") report counts.

### 2g. Prompts — `src/qwen_llm_extractor/prompts/`

**`joint.py`**:
- German→German (`SYSTEM_PROMPT`/`USER_PROMPT_TEMPLATE`): separates `befund_phrases`
  (descriptive) vs `beurteilung_phrases` (diagnostic, incl. suspected diagnoses, DDs, negative
  findings, recommendations); 2–8 words per phrase; **excludes numeric size/measurement phrases**
  (e.g. "ca. 4 cm", "35 mm"); requires ≥1 phrase per category with a fallback instruction; 5
  worked examples.
- English (`SYSTEM_PROMPT_ENGLISH`/`USER_PROMPT_TEMPLATE_ENGLISH`, used with `--english` on
  `translated_reports.json`): classifies by content type not section; excludes
  measurements/sizes ("6 x 8 mm osteolytic lesion" → "osteolytic lesion"); special fracture rule
  (rule 9) — excludes all fracture phrases by default except "pathological fracture" /
  "insufficiency fracture" describing a present finding; negated fracture phrases excluded
  entirely; requires ≥1 phrase per category; 4 worked examples.
- Two-stage variants: stage 1 (`SYSTEM_PROMPT_EXTRACT_ENGLISH`) does recall-first atomic
  segmentation into 2–6 word phrases, deduplicated; stage 2 (`SYSTEM_PROMPT_CLASSIFY_ENGLISH`)
  tags each phrase `{phrase, category, relevance}` without dropping/reordering/rewording.
- Shared taxonomy (`DESCRIPTOR_CATEGORIES_ENGLISH`/`CATEGORY_DEFINITIONS_ENGLISH`): margin/border,
  periosteal reaction, matrix/density, lesion geometry, host bone response, soft tissue status,
  surface lesions, skeletal deformity, multifocal disease, diagnosis.

**`separated.py`** (German input, English output):
- `BEFUND_PROMPT_TEMPLATE`: descriptive phrases (morphology, location, size, matrix,
  aggressiveness signs), min 2 phrases, 2–8 words each.
- `BEURTEILUNG_PROMPT_TEMPLATE`: diagnoses/DDs/malignancy assessment/negations/clinical
  impressions; standardizes "V.a." → "suspected", "DD" → "differential", "kein Anhalt für" →
  "no evidence of"; empty list allowed if nothing present.
- `BEURTEILUNG_SUMMARY_PROMPT_TEMPLATE`: fallback used when beurteilung yields nothing — derives
  1–3 diagnostic phrases from befund text alone.
- Two-stage extras: `SYSTEM_EXTRACTION_PROMPT_TEMPLATE` (verbatim, contiguous phrase segmentation,
  no classification), `BEFUND_RANK_*`/`BEURTEILUNG_RANK_*` (assign relevance without
  adding/dropping/rewording phrases).

**`joint_backup.py`**: unused backup/older prompt file, not imported anywhere active.

---

## Stage 3 — `src/create_dataset.py`

Assembles `data/internal_dataset/dataset_full.json` by joining `metadata.xlsx`
(sheet `internal_data_matched`) with `full_reports.json`.

- Report lookup: builds `accnr_lookup`/`patid_lookup` dicts, preferring `report_accnr` match,
  falling back to normalized `patid` (`_normalise_id`).
- **Filtering** (lines 109–111): a metadata row is **skipped** (only hard drop in this stage) if
  the corresponding image PNG (`images_dir/{stem}.png`) does not exist on disk.
- If no report match is found, `befund`/`beurteilung`/`befund_en`/`beurteilung_en`/`report` are
  set to `None` and phrase lists to `[]` — the sample is **still included** as an image-only
  sample, not dropped.
- `report` field (lines 129–131) = `" ".join(filter(None, [befund_en, beurteilung_en])) or None`
  — the canonical pretraining text field; `None` if both are empty.
- `age`: parsed as float, else `"unknown"`; `sex`: `"m"`/`"f"`/`"unknown"`.
- Output entries carry: `image, mask, report, label, age, sex, patid, befund, beurteilung,
  befund_en, beurteilung_en, befund_phrases, beurteilung_phrases`.
- Prints coverage stats: total entries, with-report count, with-label count, unknown-age/sex
  counts.

---

## Stage 4 — `src/create_split.py` → `biomedclip.data.splits.build_stratified_splits`

`src/create_split.py` is a thin CLI (`--dataset` default `dataset_full.json`, `--out_dir`, split
fractions, `--seed`, `--binary`) that writes `split.json` / `split_binary.json`.

**Core filtering happens in `src/biomedclip/data/splits.py::build_stratified_splits`**, not in
`create_split.py` itself (lines 459–477):

```python
for e in all_entries:
    if not e.get("report"):
        skipped["no_report"] += 1; continue
    if not e.get("label"):
        skipped["no_label"] += 1; continue
    if binary and e["label"] == "intermediate":
        skipped["intermediate"] += 1; continue
    if not _mask_has_pixels(Path(e.get("mask") or "")):
        skipped["no_mask"] += 1; continue
    samples.append(...)
```

This is **the single biggest "excluded because no text" gate in the whole pipeline**: a sample is
dropped from every train/val/test split if it has:
- **no `report` text** — `splits.py:460-462`
- **no `label`** — `splits.py:463-465`
- (binary mode only) **`label == "intermediate"`** — `splits.py:466-468`
- **no mask, or an all-zero mask** — `splits.py:469-471`, via `_mask_has_pixels()` (loads the PNG
  and checks `np.any(mask > 0)`)

Every downstream train/val/test split therefore only contains samples with both report text and a
non-empty segmentation mask.

- `age="unknown"` → later imputed with the training-set mean age; `sex="unknown"` → encoded as
  `0.5`, not dropped.
- Splitting is **patient-level** (`_group_by_patient`) with majority-label stratification
  (`_majority_label`) to prevent leakage, via `sklearn.train_test_split`
  (`_stratified_split_two`, falls back to non-stratified split on `ValueError` when a class has
  too few samples).
- Legacy/dead code in the same file (`_old_load_all_samples`, `old_build_stratified_splits`,
  `old_build_pretrain_datasets`) builds pretraining pairs directly from Excel+reports without
  `dataset_full.json`; there, `report_text` is required non-empty for a sample to enter
  `pretrain_cands`, while `downstream_cands` are gated by valid age+sex. Appears superseded by the
  `dataset_full.json`-based path but remains in the codebase.
- `build_pretrain_datasets()` does a 90/10 random split of `pretrain_samples` into
  train/monitor-val `BoneTumorPairDataset`s — no additional text filtering (already validated
  upstream).

---

## Stage 5 — Dataset `__getitem__` (text handling at training time)

### `src/biomedclip/data/datasets.py`

- `BoneTumorPairDataset`: `(image, mask, text)` → `preprocess(image)` +
  `tokenizer([text], context_length=256).squeeze(0)`. No further filtering — assumes upstream
  `report_text` already validated non-empty.
- `DownstreamDataset`: filters `samples` at construction to those with `stem in age_sex_lookup`
  **and** `label in label_to_idx` — drops samples missing age/sex metadata or with unmapped
  labels. No text used.
- `DownstreamDatasetWithText`: same age/sex/label filter as above. At `__getitem__`:
  ```python
  text = " ".join(filter(None, [s.get("befund_en"), s.get("beurteilung_en")])) or s.get("report") or ""
  ```
  Falls back from split English fields → pre-joined `report` field → empty string if nothing is
  available. No sample-level drop for missing text at `__getitem__` time. Tokenized at
  `context_length=256`.
- `EmbeddingDataset`: wraps precomputed embeddings; no text handling.

### `src/LACE/data/datasets.py`

- `InternalDatasetV2.__init__`: filtering depends on `text_mode` (default `"phrase"`):
  - **`"phrase"` mode**: keeps a sample only if **both** `befund_phrases` and `beurteilung_phrases`
    are non-empty (`s.get("befund_phrases") and s.get("beurteilung_phrases")`). Drop count is
    printed: `"InternalDatasetV2: dropped {dropped} samples missing befund/beurteilung phrases."`
  - **`"mixed"` mode**: keeps a sample only if `befund_phrases` is non-empty — beurteilung may be
    empty since full beurteilung text is used instead of phrases.
  - **`"full"` mode**: **no filtering** — all samples kept.
- `_tok(text, max_length)`: if `text` is falsy, substitutes the placeholder token string
  `"[PAD]"` before tokenizing — handling for empty *individual* text fields (as opposed to
  whole-sample exclusion). Uses `padding="max_length"`, `truncation=True`.
- `_encode_phrase_list(phrases, max_phrases)`: tokenizes up to `max_phrases` phrases (default
  `max_bef_phrases=16`, `max_beur_phrases=16`) at `phrase_tok_len=32` tokens each; unused slots
  zero-padded; returns a boolean `mask` marking real vs padding phrases. Phrases beyond the cap
  are dropped by simple list-order truncation (no random sampling); order was already
  relevance-sorted upstream when the two-stage LLM pipeline was used.
- `__getitem__` text-mode branches:
  - `"full"`: tokenizes full `befund_en`/`beurteilung_en` at `max_text_len=128`;
    `has_befund = bool(befund)`.
  - `"mixed"`: full beurteilung at `max_beur_text_len=256` + befund phrase list;
    `concat_full = " ".join(filter(None, [befund, beurteilung]))`.
  - `"phrase"` (else): both phrase lists tokenized separately;
    `all_text = ", ".join(beur_phrases + bef_phrases)` also tokenized at `max_text_len`.
  - `has_befund = torch.tensor(bool(bef_phrases))` — used downstream to mask out image-only
    samples from text losses even when the sample wasn't hard-dropped in `"full"` mode.
- `InternalTripleDataset`: same filtering pattern for `"phrase"`/`"mixed"` text modes, plus a
  `"concat"` mode (joins phrase lists with `", "`, tokenizes as one string;
  `has_beurteilung`/`has_befund` flags from `bool(...)`).
- `BTXRDOrthoDataset`: image+mask only, no text (external BTXRD dataset, unrelated to German
  reports).

---

## Cross-cutting notes

- **German umlauts**: no explicit normalization/sanitization step found anywhere. All JSON I/O
  uses `ensure_ascii=False` (`translate_reports.py`, `extract/joint.py`, `extract/separated.py`)
  so umlauts/German characters are preserved as literal UTF-8 rather than escaped.
- **Deduplication**: phrase-level deduplication is instructed at the prompt level ("Deduplicate:
  emit each distinct concept once") but not code-enforced. `_norm_phrase()`
  (whitespace-collapse + lowercase) is used only to match stage-1 phrases to stage-2 relevance
  tags, not to remove duplicates from output lists.
- **Phrase counts / truncation**: hard caps `max_bef_phrases=16`, `max_beur_phrases=16` (LACE
  datasets), `phrase_tok_len=32` tokens/phrase; truncation beyond the cap is a simple index slice
  (first N kept).
- **Tokenizer max lengths**: `max_text_len=128` (LACE full/phrase text), `max_beur_text_len=256`
  (LACE mixed-mode beurteilung), `context_length=256` (BiomedCLIP `open_clip` tokenizer),
  `phrase_tok_len=32` (per-phrase).
- **Measurement/size stripping** is done entirely via LLM prompt instruction (not regex
  post-processing).
- **Resume/incremental-save pattern**, consistent across `translate_reports.py`,
  `extract/joint.py`, `extract/separated.py`: writes full output after every batch/entry; on
  restart, loads the existing output file and skips already-completed keys (`patid` for
  translation, `accnr`-or-`patid` for extraction).

## Summary of every "no text → excluded" gate

| Stage | Condition | Effect |
|---|---|---|
| `preprocess_reports.py:12-15` | empty `befund`, or `befund` contains placeholder "Die Bilder wurden bereitgestellt." | report dropped before translation |
| `extract/joint.py:268-272`, `extract/separated.py:289-293` | both befund and beurteilung (or their `_en` fields) empty/whitespace | report skipped for LLM phrase extraction |
| `qwen_llm_extractor/data/reports.py` (`load_reports_joint`/`load_reports_separated`) | both sections empty | report silently skipped when loading for extraction |
| `create_dataset.py:109-111` | corresponding image PNG missing on disk | metadata row dropped (not text-related, but the only hard drop at this stage) |
| `biomedclip/data/splits.py:459-477` | no `report` text, no `label`, (binary) `label == "intermediate"`, or empty/all-zero mask | sample excluded from train/val/test split — the primary text-driven exclusion gate |
| `LACE/data/datasets.py` `InternalDatasetV2`/`InternalTripleDataset` (`text_mode="phrase"`) | empty `befund_phrases` or `beurteilung_phrases` | sample dropped at dataset construction (count printed) |
| `LACE/data/datasets.py` (`text_mode="mixed"`) | empty `befund_phrases` | sample dropped (beurteilung allowed empty) |
| `LACE/data/datasets.py` (`text_mode="full"`) | — | no filtering; empty individual fields padded with `"[PAD]"` at tokenization time |
