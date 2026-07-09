


import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


split1 = "data/internal_dataset/split.json"
split2 = "data/internal_dataset/split_new.json"

with open(split1, 'r') as f1, open(split2, 'r') as f2:
    data1 = json.load(f1)
    data2 = json.load(f2)


images1 = {}
images2 = {}
for key in ["train", "val", "test"]:
    images1[key] = [entry["image"] for entry in data1[key]]
    images2[key] = [entry["image"] for entry in data2[key]]

for key in ["train", "val", "test"]:
    set1 = set(images1[key])
    set2 = set(images2[key])
    if set1 == set2:
        print(f"{key}: identical ({len(set1)} images)")
    else:
        only1 = set1 - set2
        only2 = set2 - set1
        print(f"{key}: DIFFERENT ({len(set1)} vs {len(set2)} images)")
        for img in sorted(only1):
            print(f"  only in split1: {img}")
        for img in sorted(only2):
            print(f"  only in split2: {img}")