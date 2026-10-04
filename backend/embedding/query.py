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
    """Load the precomputed vectors AND their metadata together, since
    they're two halves of one dataset -- vectors[i] only means something in
    relation to meta["dishes"][i]. Returning them separately (rather than
    zipping them into one structure) matches how build_embeddings.py saved
    them, and keeps the large numpy array out of JSON (which can't hold it
    efficiently anyway)."""
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


def rank_candidates_by_preference(text, model, all_vectors, all_dishes, candidate_dishes):
    """Retrieval step for RAG itinerary generation.

    `candidate_dishes` is a SUBSET of `all_dishes` (e.g. already passed the
    deterministic allergen/diet/region/spice filter -- that hard filter must
    run before this, never after, since embedding similarity cannot be
    trusted to enforce safety constraints -- see DATA_SOURCES.md / earlier
    testing). This function only re-ranks within that already-safe subset by
    relevance to free-text `text`, using each candidate's precomputed vector
    looked up by en_name (candidates aren't necessarily index-aligned with
    all_dishes/all_vectors since filtering already dropped rows).

    Returns a list of (dish, score) sorted best-first, same length as
    candidate_dishes (nothing is dropped here, only reordered).
    """
    # assumes en_name is unique across all_dishes (true for this dataset --
    # sample_dishes.csv was deduped by Thai name when rows were merged in).
    # If a name collided, this dict comprehension would silently keep only
    # the LAST matching index -- fine here, but worth knowing if the dataset
    # ever grows from a source that doesn't guarantee unique names.
    name_to_index = {d["en_name"]: i for i, d in enumerate(all_dishes)}
    query_vec = model.encode([text], normalize_embeddings=True)[0]

    scored = []
    for dish in candidate_dishes:
        idx = name_to_index.get(dish["en_name"])
        if idx is None:
            continue  # shouldn't happen if candidate_dishes really is a subset, but don't crash if it does
        score = float(all_vectors[idx] @ query_vec)
        scored.append((dish, score))

    scored.sort(key=lambda pair: -pair[1])
    return scored


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
