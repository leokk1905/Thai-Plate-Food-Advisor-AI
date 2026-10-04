"""
Phase 0 demo entry point: a CLI Thai food travel buddy.

Run (with backend/.env containing OPENROUTER_API_KEY=...):
    python cli.py

If backend/data/enriched_dishes.csv doesn't exist yet, run the enrichment
step first:
    python enrichment/enrich.py data/sample_dishes.csv data/enriched_dishes.csv
"""

import os
import sys

from openai import OpenAI
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass  # older Python without reconfigure(); Thai script may not print correctly on this console

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

sys.path.insert(0, os.path.dirname(__file__))
from itinerary.planner import load_dishes, filter_dishes, build_daily_plan, generate_narrative

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
REGIONS = ["Central", "North", "Northeast", "South"]
ALLERGENS = [
    "milk", "egg", "fish", "crustacean_shellfish", "molluscs", "tree_nut",
    "peanut", "wheat", "soybean", "sesame", "mustard", "celery",
]


def ask(prompt, options=None):
    if options:
        print(f"{prompt} ({'/'.join(options)})")
    return input("> ").strip()


def ask_int(prompt, default):
    while True:
        raw = ask(prompt) or str(default)
        try:
            return int(raw)
        except ValueError:
            print(f"'{raw}' isn't a number -- try again.")


def main():
    enriched_path = os.path.join(DATA_DIR, "enriched_dishes.csv")
    if not os.path.exists(enriched_path):
        sys.exit(
            f"Missing {enriched_path}.\n"
            "Run enrichment first:\n"
            "  python enrichment/enrich.py data/sample_dishes.csv data/enriched_dishes.csv"
        )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY (in backend/.env) before running.")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    dishes = load_dishes(enriched_path)

    print("=== Thai Food Travel Buddy (Phase 0 demo) ===\n")

    print(f"Which region(s) are you visiting? Options: {', '.join(REGIONS)}")
    region_input = ask("Comma-separated, or blank for all")
    regions = [r.strip() for r in region_input.split(",") if r.strip()] or None

    days = ask_int("How many days is the trip?", 3)

    print(f"\nAny allergens to avoid? Options: {', '.join(ALLERGENS)}")
    allergy_input = ask("Comma-separated, or blank for none")
    exclude_allergens = [a.strip() for a in allergy_input.split(",") if a.strip()]

    max_spice = ask_int("Max spice level (1=none, 5=very spicy)", 5)

    diet_input = ask("Dietary requirement (vegan/vegetarian/pescatarian/none)") or "none"

    preferences = {
        "regions": regions or "any",
        "days": days,
        "exclude_allergens": exclude_allergens,
        "max_spice": max_spice,
        "dietary_requirement": diet_input,
    }

    filtered = filter_dishes(
        dishes,
        regions=regions,
        exclude_allergens=exclude_allergens,
        max_spice=max_spice,
        required_diet=diet_input,
    )

    if not filtered:
        print("\nNo dishes in the sample dataset match those constraints. "
              "(This is a small demo dataset — a real database would have far more coverage.)")
        return

    plan = build_daily_plan(filtered, days)

    print("\nGenerating your itinerary...\n")
    narrative = generate_narrative(client, plan, preferences)
    print(narrative)

    print("\n---")
    print("Note: allergen/spice tags in this demo are LLM-inferred drafts, not verified. "
          "Do not use this output for real allergy-safety decisions yet.")


if __name__ == "__main__":
    main()
