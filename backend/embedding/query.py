"""
Terminal prototype: ask free-text questions about Thai food, get the closest
matching dishes by embedding similarity (see build_embeddings.py -- run that
first). No LLM call involved here -- this is pure local vector search, so it
costs nothing and needs no API key.

Usage:
    python query.py
"""

import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # older Python without reconfigure(); Thai script may not print correctly on this console

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORS_PATH = os.path.join(DATA_DIR, "dish_embeddings.npy")
META_PATH = os.path.join(DATA_DIR, "dish_embeddings_meta.json")
TOP_K = 5


def load():
    if not os.path.exists(VECTORS_PATH) or not os.path.exists(META_PATH):
        sys.exit("No embeddings found. Run build_embeddings.py first.")
    vectors = np.load(VECTORS_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return vectors, meta


def top_matches(query, model, vectors, dishes, k=TOP_K):
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    scores = vectors @ query_vec  # vectors are pre-normalized, so this is cosine similarity
    ranked = np.argsort(-scores)[:k]
    return [(dishes[i], float(scores[i])) for i in ranked]


def main():
    vectors, meta = load()
    dishes = meta["dishes"]
    print(f"Loaded {len(dishes)} dishes (source: {meta['source_file']}, enriched: {meta['enriched']})")
    if not meta["enriched"]:
        print("Note: this is running on RAW (non-enriched) data -- no verified allergen/spice filtering here, "
              "just similarity over name/ingredients/region. Run enrichment/enrich.py with an API key for richer results.")

    print(f"Loading embedding model: {meta['model']}...")
    model = SentenceTransformer(meta["model"])

    print("\n=== Thai Food Query Prototype ===")
    print("Ask about Thai food in plain language (e.g. \"something sour and spicy\", "
          "\"vegetarian dessert\", \"dishes like Pad Thai\"). Type 'quit' to exit.\n")

    while True:
        query = input("> ").strip()
        if not query or query.lower() in ("quit", "exit"):
            break

        for dish, score in top_matches(query, model, vectors, dishes):
            print(f"  [{score:.3f}] {dish['en_name']} ({dish['th_name']}) -- "
                  f"{dish['course']}, {dish['province']} ({dish['region']})")
            print(f"           ingredients: {dish['ingredients'].replace('+', ', ')}")
        print()


if __name__ == "__main__":
    main()
