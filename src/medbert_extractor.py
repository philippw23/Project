"""
Bone Tumor Feature Extractor using German Medical BERT (medbert-512)
=====================================================================
Pipeline:
  1. Load German-MedBERT (encoder-only) for contextual embeddings
  2. Use KeyBERT (backed by the BERT model) to extract imaging features
     guided by bone-tumor–specific seed terms
  3. Cluster extracted terms across all reports with HDBSCAN / K-Means
  4. Visualise clusters with UMAP

Note on model choice:
  medbert-512 (smanjil/German-MedBERT) is an *encoder* model — it cannot
  answer generative prompts. Instead it powers semantic similarity search
  inside KeyBERT, which is a more reliable extraction approach than prompting
  for structured lists from an instruction-tuned LLM.

Usage:
    python medbert_extractor.py                        # uses bundled sample reports
    python medbert_extractor.py --reports my_dir/      # load *.txt from a directory
    python medbert_extractor.py --reports reports.json # load JSON list of strings
"""

import argparse
import json
import os
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ── Model identifier ─────────────────────────────────────────────────────────
MEDBERT_MODEL = "smanjil/German-MedBERT"

# ── Seed phrases that anchor KeyBERT toward bone-tumor imaging features ───────
# These are the kinds of terms a radiologist would use — KeyBERT finds text
# passages semantically close to these seeds.
BONE_TUMOR_SEEDS = [
    "Kortikalisdestruktion",
    "Periostreaktion",
    "Weichteilkomponente",
    "osteolytische Läsion",
    "sklerotische Läsion",
    "Matrixmineralisierung",
    "Knochenmarkinfiltration",
    "Randsklerose",
    "Kalzifikation",
    "Kontrastmittelanreicherung",
    "Tumorausdehnung",
    "Signalintensität",
    "Ödemmuster",
    "Codman-Dreieck",
    "peritumorales Ödem",
]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Report loading
# ─────────────────────────────────────────────────────────────────────────────

def load_reports(source: Optional[str] = None) -> list[str]:
    """Load reports from a directory of .txt files, a JSON file, or built-in samples."""
    if source is None:
        from sample_reports import SAMPLE_REPORTS
        print(f"Using {len(SAMPLE_REPORTS)} bundled synthetic reports.")
        return SAMPLE_REPORTS

    p = Path(source)
    if p.is_dir():
        txts = sorted(p.glob("*.txt"))
        reports = [f.read_text(encoding="utf-8") for f in txts]
        print(f"Loaded {len(reports)} reports from {p}")
        return reports

    if p.suffix == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON must be a list.")
        # Plain list of strings (old format)
        if data and isinstance(data[0], str):
            return data
        # List of report dicts — combine befund + beurteilung
        reports = []
        for entry in data:
            parts = []
            if entry.get("befund", "").strip():
                parts.append(entry["befund"].strip())
            if entry.get("beurteilung", "").strip():
                parts.append(entry["beurteilung"].strip())
            if parts:
                reports.append("\n\n".join(parts))
        print(f"Loaded {len(reports)} reports from {p} (befund + beurteilung).")
        return reports

    raise ValueError(f"Unsupported source: {source}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  KeyBERT-based feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def build_keybert_model():
    """
    Build a KeyBERT instance backed by German-MedBERT.

    KeyBERT encodes the document and candidate n-grams with the same BERT model,
    then ranks candidates by cosine similarity to the document embedding.
    Passing `candidates` (our seed phrases) makes it a *guided* extraction.
    """
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    print(f"\nLoading model: {MEDBERT_MODEL}")
    print("(First run will download ~440 MB — subsequent runs use the cache.)\n")

    # SentenceTransformer wraps any HuggingFace model and adds mean-pooling
    st_model = SentenceTransformer(MEDBERT_MODEL)
    kw_model = KeyBERT(model=st_model)
    return kw_model, st_model


def extract_features_from_report(
    report: str,
    kw_model,
    top_n: int = 15,
    ngram_range: tuple = (1, 4),
    use_mmr: bool = True,
    diversity: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Extract keyphrases from one report, sentence-by-sentence.

    Running KeyBERT per sentence (rather than passing explicit candidates)
    prevents n-grams from spanning sentence boundaries while keeping
    CountVectorizer's standard n-gram generation intact.

    Parameters
    ----------
    top_n       : max keyphrases returned across all sentences
    ngram_range : unigrams up to trigrams
    use_mmr     : Maximal Marginal Relevance reduces redundancy
    diversity   : MMR diversity weight (0 = similar, 1 = diverse)
    """
    sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", report.strip())
        if len(s.strip()) > 15
    ]
    if not sentences:
        sentences = [report]

    # How many phrases to request per sentence (at least 3, scales with sentence count)
    per_sent = max(3, top_n // len(sentences))

    best: dict[str, float] = {}
    for sent in sentences:
        kws = kw_model.extract_keywords(
            sent,
            keyphrase_ngram_range=ngram_range,
            stop_words=None,
            top_n=per_sent,
            use_mmr=use_mmr,
            diversity=diversity,
            seed_keywords=BONE_TUMOR_SEEDS,
        )
        for phrase, score in kws:
            if phrase not in best or score > best[phrase]:
                best[phrase] = score

    return sorted(best.items(), key=lambda x: -x[1])[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Post-processing & frequency analysis
# ─────────────────────────────────────────────────────────────────────────────

GERMAN_STOP_WORDS = {
    "der", "die", "das", "des", "dem", "den", "ein", "eine", "eines",
    "einer", "einem", "einen", "und", "oder", "aber", "mit", "von",
    "zu", "in", "an", "auf", "bei", "nach", "ist", "sind", "war",
    "nicht", "kein", "keine", "im", "am", "als", "des", "sich", "zum",
    "zur", "aus", "für", "wird", "werden", "wurde", "hat", "haben",
    "noch", "auch", "kein", "keine", "sowohl",
}


def clean_phrase(phrase: str) -> str:
    phrase = phrase.strip().lower()
    phrase = re.sub(r"[^\w\s\-äöüß]", "", phrase)
    return phrase


def filter_phrase(phrase: str, min_chars: int = 4) -> bool:
    if len(phrase) < min_chars:
        return False
    tokens = phrase.split()
    # keep if at least one token is not a stop word
    return any(t not in GERMAN_STOP_WORDS for t in tokens)


def build_term_dataframe(
    all_keywords: list[list[tuple[str, float]]],
    reports: list[str],
) -> pd.DataFrame:
    """Flatten per-report keywords into a tidy DataFrame."""
    rows = []
    for report_idx, kws in enumerate(all_keywords):
        for phrase, score in kws:
            cleaned = clean_phrase(phrase)
            if filter_phrase(cleaned):
                rows.append({
                    "report_idx": report_idx,
                    "phrase": cleaned,
                    "score": score,
                })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Embedding & clustering
# ─────────────────────────────────────────────────────────────────────────────

def embed_phrases(phrases: list[str], st_model) -> np.ndarray:
    """Encode unique phrases with German-MedBERT via SentenceTransformer."""
    print(f"\nEmbedding {len(phrases)} unique phrases with {MEDBERT_MODEL}…")
    embeddings = st_model.encode(
        phrases,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings  # shape (N, hidden_dim)


def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 3):
    """HDBSCAN: discovers cluster count automatically; robust to noise."""
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)
        return labels
    except ImportError:
        print("hdbscan not installed — falling back to K-Means (k=12).")
        return cluster_kmeans(embeddings, k=12)


def cluster_kmeans(embeddings: np.ndarray, k: int = 12):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    return km.fit_predict(embeddings)


def reduce_umap(embeddings: np.ndarray, n_components: int = 2):
    try:
        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(embeddings)
    except ImportError:
        print("umap-learn not installed — using PCA for 2-D projection.")
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=42).fit_transform(embeddings)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_clusters(
    xy: np.ndarray,
    labels: np.ndarray,
    phrases: list[str],
    output_path: str = "cluster_plot.png",
    annotate_top_n: int = 5,
):
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(n_clusters + 1)

    fig, ax = plt.subplots(figsize=(14, 10))
    for label in sorted(set(labels)):
        mask = labels == label
        color = "lightgrey" if label == -1 else cmap(label)
        lname = "Noise" if label == -1 else f"Cluster {label}"
        ax.scatter(xy[mask, 0], xy[mask, 1], c=[color], s=40, alpha=0.7, label=lname)

    # annotate a few points per cluster for readability
    from collections import defaultdict
    cluster_indices = defaultdict(list)
    for i, lbl in enumerate(labels):
        cluster_indices[lbl].append(i)

    for lbl, idxs in cluster_indices.items():
        if lbl == -1:
            continue
        for i in idxs[:annotate_top_n]:
            ax.annotate(
                phrases[i],
                xy=(xy[i, 0], xy[i, 1]),
                fontsize=6,
                alpha=0.85,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_title(
        f"Bone Tumor Imaging Features — {n_clusters} clusters\n"
        f"(German-MedBERT embeddings + UMAP)",
        fontsize=13,
    )
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"\nCluster plot saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_cluster_summary(
    phrases: list[str],
    labels: np.ndarray,
    freq_map: dict[str, int],
    top_n: int = 10,
):
    from collections import defaultdict

    cluster_phrases = defaultdict(list)
    for phrase, lbl in zip(phrases, labels):
        cluster_phrases[lbl].append(phrase)

    print("\n" + "=" * 70)
    print("CLUSTER SUMMARY — Top imaging terms per cluster")
    print("=" * 70)

    for lbl in sorted(cluster_phrases):
        if lbl == -1:
            cluster_name = "Noise (unclustered)"
        else:
            cluster_name = f"Cluster {lbl:>2}"

        members = cluster_phrases[lbl]
        # sort by frequency across reports
        members_sorted = sorted(members, key=lambda p: freq_map.get(p, 0), reverse=True)

        print(f"\n{cluster_name}  ({len(members)} terms)")
        print("-" * 50)
        for p in members_sorted[:top_n]:
            freq = freq_map.get(p, 0)
            print(f"  [{freq:>3}×]  {p}")


def save_results(
    df: pd.DataFrame,
    phrases: list[str],
    labels: np.ndarray,
    freq_map: dict[str, int],
    out_dir: str = ".",
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # full term table
    df.to_csv(out / "extracted_terms.csv", index=False, encoding="utf-8-sig")

    # cluster assignments
    cluster_df = pd.DataFrame({
        "phrase": phrases,
        "cluster": labels,
        "freq_across_reports": [freq_map.get(p, 0) for p in phrases],
    }).sort_values(["cluster", "freq_across_reports"], ascending=[True, False])
    cluster_df.to_csv(out / "cluster_assignments.csv", index=False, encoding="utf-8-sig")

    print(f"\nResults saved to '{out}/'")
    print(f"  extracted_terms.csv       ({len(df)} rows)")
    print(f"  cluster_assignments.csv   ({len(cluster_df)} unique phrases)")


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load reports ──────────────────────────────────────────────────────────
    reports = load_reports(args.reports)
    if not reports:
        raise ValueError("No reports loaded.")
    if args.max_reports is not None:
        reports = reports[: args.max_reports]
        print(f"Limiting to {len(reports)} reports (--max_reports).")

    # ── Build model ───────────────────────────────────────────────────────────
    kw_model, st_model = build_keybert_model()

    # ── Extract per-report keyphrases ─────────────────────────────────────────
    print(f"\nExtracting features from {len(reports)} reports…")
    all_keywords: list[list[tuple[str, float]]] = []
    for report in tqdm(reports, desc="Reports"):
        kws = extract_features_from_report(
            report,
            kw_model,
            top_n=args.top_n,
            use_mmr=True,
            diversity=0.5,
        )
        all_keywords.append(kws)

    # ── Flatten to DataFrame ──────────────────────────────────────────────────
    df = build_term_dataframe(all_keywords, reports)
    print(f"\nTotal extracted (phrase, report) pairs: {len(df)}")
    print(f"Unique phrases before clustering:       {df['phrase'].nunique()}")

    # ── Frequency map: how many distinct reports mention each phrase ──────────
    freq_map: dict[str, int] = (
        df.groupby("phrase")["report_idx"]
        .nunique()
        .to_dict()
    )

    # ── Unique phrases to embed ───────────────────────────────────────────────
    unique_phrases = df["phrase"].unique().tolist()

    # ── Embed ─────────────────────────────────────────────────────────────────
    embeddings = embed_phrases(unique_phrases, st_model)

    # ── Cluster ───────────────────────────────────────────────────────────────
    print("\nClustering…")
    if args.cluster_method == "kmeans":
        labels = cluster_kmeans(embeddings, k=args.n_clusters)
    else:
        labels = cluster_hdbscan(embeddings, min_cluster_size=args.min_cluster_size)

    n_found = len(set(labels)) - (1 if -1 in labels else 0)
    noise_n = int((labels == -1).sum())
    print(f"Clusters found: {n_found}  (noise points: {noise_n})")

    # ── 2-D projection ────────────────────────────────────────────────────────
    print("Reducing to 2-D for visualisation…")
    xy = reduce_umap(embeddings)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_cluster_summary(unique_phrases, labels, freq_map)

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_path = str(Path(args.out_dir) / "cluster_plot.png")
    plot_clusters(xy, labels, unique_phrases, output_path=plot_path)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    save_results(df, unique_phrases, labels, freq_map, out_dir=args.out_dir)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and cluster bone-tumor imaging features from German radiology reports "
                    "using German-MedBERT (smanjil/German-MedBERT)."
    )
    parser.add_argument(
        "--reports",
        default=None,
        help="Path to a directory of *.txt files, or a JSON file (list of strings). "
             "Omit to use the bundled synthetic sample reports.",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=15,
        help="Max keyphrases to extract per report (default: 15).",
    )
    parser.add_argument(
        "--cluster_method",
        choices=["hdbscan", "kmeans"],
        default="hdbscan",
        help="Clustering algorithm (default: hdbscan).",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=12,
        help="Number of clusters for K-Means (ignored for HDBSCAN, default: 12).",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=3,
        help="Minimum cluster size for HDBSCAN (default: 3).",
    )
    parser.add_argument(
        "--max_reports",
        type=int,
        default=None,
        help="Limit the number of reports to process (e.g. 5 for a quick test).",
    )
    parser.add_argument(
        "--out_dir",
        default="results",
        help="Output directory for CSVs and plot (default: results/).",
    )

    args = parser.parse_args()
    main(args)
