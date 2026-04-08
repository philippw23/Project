"""Analyse and visualise word count distributions for radiology report fields."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Path to the radiology reports JSON file
DATA_PATH = Path(__file__).parent.parent / "data" / "text" / "reports.json"

# Load all reports from disk
with open(DATA_PATH, encoding="utf-8") as f:
    reports = json.load(f)

# Scan the raw file line-by-line to record line numbers of empty/whitespace-only field entries.
# We decode the captured JSON string value so escape sequences like \n or \t are handled correctly.
import re

_field_pattern = re.compile(r'"(befund|beurteilung)"\s*:\s*(".*?")')
empty_befund_lines = []
empty_beurteilung_lines = []
with open(DATA_PATH, encoding="utf-8") as f:
    for lineno, line in enumerate(f, start=1):
        m = _field_pattern.search(line)
        if m:
            try:
                value = json.loads(m.group(2))
            except json.JSONDecodeError:
                continue
            if not value.strip():
                if m.group(1) == "befund":
                    empty_befund_lines.append(lineno)
                else:
                    empty_beurteilung_lines.append(lineno)

print(f"Total reports:       {len(reports)}")
print(f"\nEmpty findings ({len(empty_befund_lines)})"
      f"   — JSON line numbers: {empty_befund_lines}")
print(f"Empty assessments ({len(empty_beurteilung_lines)})"
      f" — JSON line numbers: {empty_beurteilung_lines}")

# Count words in each "befund" (findings) and "beurteilung" (assessment) field,
# skipping records where the field is missing, empty, whitespace-only,
# or only contains the placeholder "Die Bilder wurden bereitgestellt."
PLACEHOLDER = "Die Bilder wurden bereitgestellt."
befund_counts = [
    len(r["befund"].split()) for r in reports
    if r.get("befund", "").strip() and PLACEHOLDER not in r["befund"]
]
beurteilung_counts = [
    len(r["beurteilung"].split()) for r in reports if r.get("beurteilung", "").strip()
]

# Create a side-by-side figure with one plot per field
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, counts, label in [
    (axes[0], befund_counts, "Findings"),
    (axes[1], beurteilung_counts, "Assessment"),
]:
    # Plot the word count histogram
    ax.hist(counts, bins=40, color="steelblue", edgecolor="white", alpha=0.85)
    ax.set_title(f"Word Count Distribution: {label}", fontsize=13)
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Number of Reports")

    # Overlay mean and median as vertical dashed lines
    ax.axvline(np.mean(counts), color="crimson", linestyle="--", linewidth=1.5,
               label=f"Mean: {np.mean(counts):.0f}")
    ax.axvline(np.median(counts), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median: {np.median(counts):.0f}")
    ax.legend()

    # Print descriptive statistics to the console
    print(f"\n--- {label} ---")
    print(f"  N:      {len(counts)}")
    print(f"  Min:    {min(counts)}")
    print(f"  Max:    {max(counts)}")
    print(f"  Mean:   {np.mean(counts):.1f}")
    print(f"  Median: {np.median(counts):.1f}")
    print(f"  Std:    {np.std(counts):.1f}")


plt.tight_layout()

# Save the figure next to the data directory
output_path = Path(__file__).parent.parent / "data" / "word_count_distribution.png"
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved: {output_path}")
plt.show()
