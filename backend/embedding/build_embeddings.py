"""
Builds dish embeddings with an off-the-shelf model (no training/fine-tuning --
see DATA_SOURCES.md / earlier discussion for why that's the right call here).

Prefers data/enriched_dishes.csv (has spice_level/flavor_profile from
enrich.py) and falls back to data/sample_dishes.csv if enrichment hasn't been
run yet (that step needs an ANTHROPIC_API_KEY this environment doesn't have).
Either way, the output is clearly labeled with which input it used.

Usage:
    python build_embeddings.py
"""

import csv
import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ENRICHED_PATH = os.path.join(DATA_DIR, "enriched_dishes.csv")
SAMPLE_PATH = os.path.join(DATA_DIR, "sample_dishes.csv")
VECTORS_OUT = os.path.join(DATA_DIR, "dish_embeddings.npy")
META_OUT = os.path.join(DATA_DIR, "dish_embeddings_meta.json")


def pick_input():
    if os.path.exists(ENRICHED_PATH):
        return ENRICHED_PATH, True
    if os.path.exists(SAMPLE_PATH):
        return SAMPLE_PATH, False
    sys.exit(f"No dish data found at {ENRICHED_PATH} or {SAMPLE_PATH}")


def dish_text(row, enriched):
    ingredients = row["ingredients"].replace("+", ", ")
    parts = [
        row["en_name"],
        f"a {row['course']} from {row['province']}, {row['region']} Thailand",
        f"made with {ingredients}",
    ]
    if enriched:
        if row.get("flavor_profile"):
            parts.append(f"tastes {row['flavor_profile'].replace('+', ', ')}")
        if row.get("spice_level"):
            parts.append(f"spice level {row['spice_level']} out of 5")
    return ". ".join(parts)


def main():
    input_path, enriched = pick_input()
    print(f"Using input: {input_path} ({'enriched' if enriched else 'RAW, not enriched -- run enrichment/enrich.py first for richer embeddings'})")

    with open(input_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    texts = [dish_text(r, enriched) for r in rows]

    print(f"Loading embedding model: {MODEL_NAME} (one-time download, then fully local/offline)...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Embedding {len(texts)} dishes...")
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    np.save(VECTORS_OUT, vectors.astype(np.float32))
    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_file": os.path.basename(input_path),
                "enriched": enriched,
                "model": MODEL_NAME,
                "dishes": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved {len(rows)} vectors to {VECTORS_OUT}")
    print(f"Saved metadata to {META_OUT}")


if __name__ == "__main__":
    main()
