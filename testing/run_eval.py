"""
Runs every case in test_cases.py down two paths and saves the raw results
for analyze_results.py to score:

  (A) "our_model"  -- the real pipeline, imported directly from backend/
      (not re-implemented here): filter_dishes -> [rank_candidates_by_preference,
      if preference_text is set] -> build_daily_plan -> generate_narrative /
      generate_narrative_rag, using anthropic/claude-sonnet-5.
  (B) "baseline"   -- one naive prompt straight to anthropic/claude-haiku-4.5
      (the "low cost foundation model"), stating the same constraints in
      plain language, with no filter step and no retrieval step at all.

Saves everything -- including the full narrative text for both arms, token
usage, and wall-clock latency -- to results.json. No scoring happens here;
that's analyze_results.py's job, kept separate so the raw model outputs stay
inspectable on their own.

Usage (needs backend/.env's OPENROUTER_API_KEY):
    python run_eval.py
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

TESTING_DIR = Path(__file__).parent
BACKEND_DIR = TESTING_DIR.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from itinerary.planner import (  # noqa: E402
    load_dishes, filter_dishes, build_daily_plan, generate_narrative, generate_narrative_rag,
)
from embedding.query import load as load_embeddings, rank_candidates_by_preference  # noqa: E402

from test_cases import TEST_CASES  # noqa: E402

load_dotenv(BACKEND_DIR / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
BASELINE_MODEL = "anthropic/claude-haiku-4.5"  # the "low cost foundation model" side of this eval

BASELINE_PROMPT_TEMPLATE = """You are a Thai food travel guide. Write a short, friendly day-by-day
Thai food itinerary for a traveler with these requirements:

- Regions to visit: {regions}
- Trip length: {days} day(s)
- Allergens to strictly avoid: {exclude_allergens}
- Maximum spice level: {max_spice} out of 5
- Dietary requirement: {diet}
{preference_line}

For each day, recommend 2-3 real Thai dishes that satisfy ALL of the above
requirements. For each dish, write 2-4 sentences: what it is, why it fits the
province/region, and one thing to notice about it. Do not claim any specific
restaurant, vendor, or price -- just what dishes to seek out."""


def usage_from_completion(completion):
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def run_our_model(case, client, dishes, embed_model, vectors, embed_meta):
    filtered = filter_dishes(
        dishes,
        regions=case["regions"],
        exclude_allergens=case["exclude_allergens"],
        max_spice=case["max_spice"],
        required_diet=case["diet"],
    )
    if not filtered:
        return {
            "narrative": None,
            "error": "No dishes in the dataset match these constraints.",
            "matched_dish_count": 0,
            "usage": None,
            "latency_seconds": 0.0,
        }

    preference_text = case["preference_text"].strip()
    start = time.time()

    if preference_text:
        ranked = rank_candidates_by_preference(preference_text, embed_model, vectors, embed_meta["dishes"], filtered)
        retrieval_scores = {d["en_name"]: s for d, s in ranked}
        ranked_dishes = [d for d, _ in ranked]
        plan = build_daily_plan(ranked_dishes, case["days"], shuffle=False)
        narrative, usage = generate_narrative_rag(
            client, plan, {k: v for k, v in case.items() if k != "id"}, preference_text, retrieval_scores,
        )
    else:
        plan = build_daily_plan(filtered, case["days"])
        narrative, usage = generate_narrative(client, plan, {k: v for k, v in case.items() if k != "id"})

    latency = time.time() - start
    return {
        "narrative": narrative,
        "error": None,
        "matched_dish_count": len(filtered),
        "usage": usage,
        "latency_seconds": round(latency, 2),
    }


def run_baseline(case, client):
    preference_line = (
        f'- Traveler also mentioned they are craving: "{case["preference_text"]}"'
        if case["preference_text"].strip() else ""
    )
    prompt = BASELINE_PROMPT_TEMPLATE.format(
        regions=", ".join(case["regions"]) if case["regions"] else "any region in Thailand",
        days=case["days"],
        exclude_allergens=", ".join(case["exclude_allergens"]) if case["exclude_allergens"] else "none",
        max_spice=case["max_spice"],
        diet=case["diet"],
        preference_line=preference_line,
    )

    start = time.time()
    completion = client.chat.completions.create(
        model=BASELINE_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.time() - start

    return {
        "narrative": completion.choices[0].message.content,
        "error": None,
        "prompt_used": prompt,
        "usage": usage_from_completion(completion),
        "latency_seconds": round(latency, 2),
    }


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY in backend/.env before running the eval.")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    print("Loading dish data and embedding model...")
    dishes = load_dishes(str(BACKEND_DIR / "data" / "enriched_dishes.csv"))
    vectors, embed_meta = load_embeddings()
    embed_model = SentenceTransformer(embed_meta["model"])

    results = []
    for case in TEST_CASES:
        print(f"\n=== {case['id']} ===")

        print("  running our_model...")
        our_result = run_our_model(case, client, dishes, embed_model, vectors, embed_meta)

        print("  running baseline...")
        baseline_result = run_baseline(case, client)

        results.append({
            "case": case,
            "our_model": our_result,
            "baseline": baseline_result,
        })

    out_path = TESTING_DIR / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(results)} test case results to {out_path}")
    print("Run analyze_results.py next to score them and generate REPORT.md.")


if __name__ == "__main__":
    main()
