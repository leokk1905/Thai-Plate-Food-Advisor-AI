"""
Phase 0 enrichment: adds allergens, dietary_tags, spice_level, and
flavor_profile to a dish CSV.

Architecture (see DATA_SOURCES.md for why):
- allergens / dietary_tags come from a DETERMINISTIC keyword lookup against
  backend/data/reference/ingredient_allergen_map.json. This is the safety
  net -- no LLM guessing for allergen-relevant fields.
- spice_level / flavor_profile come from an LLM call, because there is no
  reference dataset for "how spicy does this dish taste" -- this is
  genuinely subjective and fine for a model to estimate.
- Any ingredient token not found in the map is recorded in
  "unmatched_ingredients" instead of being silently guessed, so a human can
  extend the map for real coverage instead of trusting an LLM allergen guess.

Usage (with backend/.env containing OPENROUTER_API_KEY=...):
    python enrich.py ../data/sample_dishes.csv ../data/enriched_dishes.csv

Calls the model through OpenRouter (https://openrouter.ai), which exposes an
OpenAI-compatible API in front of many providers including Anthropic -- hence
using the `openai` SDK here with OpenRouter's base_url, not the `anthropic` SDK.
"""

import csv
import json
import os
import sys

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = os.environ.get("ENRICH_MODEL", "anthropic/claude-haiku-4.5")
MAP_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reference", "ingredient_allergen_map.json")

SYSTEM_PROMPT = """You estimate two subjective qualities of a Thai dish from its name,
course, and ingredient list: spice_level and flavor_profile. Do NOT comment on
allergens or dietary status -- that is handled separately from a reference table.

- spice_level: integer 1-5. 1 = no chili, 5 = very spicy by reputation.
- flavor_profile: list of dominant tastes from ["sour", "sweet", "salty", "spicy", "umami", "bitter"].

Respond using the record_flavor tool."""

TOOL = {
    "type": "function",
    "function": {
        "name": "record_flavor",
        "description": "Record subjective flavor tags for one dish.",
        "parameters": {
            "type": "object",
            "properties": {
                "spice_level": {"type": "integer", "minimum": 1, "maximum": 5},
                "flavor_profile": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["spice_level", "flavor_profile"],
        },
    },
}


def normalize_token(token):
    return token.strip().lower()


def load_allergen_map():
    with open(MAP_PATH, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_meta", None)  # _meta is documentation for humans reading the JSON file, not a real ingredient entry
    return data


def lookup_allergens_and_diet(ingredients_str, allergen_map):
    """The deterministic half of enrichment -- no model call in this
    function at all. Every ingredient token either matches a known entry in
    ingredient_allergen_map.json (and contributes its allergens/diet
    exclusions) or it doesn't, in which case it's reported as unmatched
    rather than guessed at. See ingredient_allergen_map.json's own "_meta"
    block for where that map's categories came from."""
    tokens = [normalize_token(t) for t in ingredients_str.split("+") if t.strip()]

    allergens = set()
    excluded_diets = set()
    unmatched = []
    low_confidence_notes = []

    for token in tokens:
        entry = allergen_map.get(token)
        if entry is None:
            unmatched.append(token)
            continue
        allergens.update(entry.get("allergens", []))
        excluded_diets.update(entry.get("excludes_diets", []))
        if entry.get("confidence") == "low":
            low_confidence_notes.append(f"{token}: {entry.get('note', 'low confidence')}")

    # A dish qualifies for a diet unless some ingredient explicitly excludes
    # it -- e.g. a dish made of nothing but vegetables accumulates zero
    # exclusions, so it ends up tagged vegan AND vegetarian AND pescatarian,
    # which is correct: vegan is a subset of what vegetarian/pescatarian
    # allow, so anything vegan automatically qualifies as both of those too.
    all_diets = {"vegan", "vegetarian", "pescatarian"}
    possible_diets = sorted(all_diets - excluded_diets) or ["none"]

    return {
        "allergens": sorted(allergens),
        "dietary_tags": possible_diets,
        "unmatched_ingredients": unmatched,
        "low_confidence_notes": low_confidence_notes,
    }


def infer_flavor(client, en_name, ingredients, course):
    """The LLM half of enrichment -- only ever asked for spice_level and
    flavor_profile, which genuinely have no reference dataset to look up
    (unlike allergens). tool_choice forces the model to respond through the
    record_flavor function instead of free text, so the result is always
    parseable structured JSON rather than something like "pretty spicy, I'd
    say a 4" that main() would then have to parse itself."""
    completion = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Dish: {en_name}\nCourse: {course}\nIngredients: {ingredients}"},
        ],
        tools=[TOOL],
        tool_choice={"type": "function", "function": {"name": "record_flavor"}},
    )
    tool_calls = completion.choices[0].message.tool_calls
    if not tool_calls:
        raise RuntimeError(f"No tool call returned for {en_name}")
    # arguments comes back as a JSON *string* (that's how tool-calling args
    # are transmitted), not already-parsed -- json.loads turns it back into
    # the {"spice_level": ..., "flavor_profile": [...]} dict main() expects.
    return json.loads(tool_calls[0].function.arguments)


def main(in_path, out_path):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("Set OPENROUTER_API_KEY (in backend/.env) before running enrichment.")

    client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    allergen_map = load_allergen_map()

    with open(in_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # keep every original column (en_name, ingredients, etc.) and append the
    # new enrichment columns -- this is additive, nothing from the input is
    # dropped or overwritten.
    fieldnames = list(rows[0].keys()) + [
        "allergens", "dietary_tags", "spice_level", "flavor_profile",
        "unmatched_ingredients", "low_confidence_notes",
        "allergen_source", "flavor_source",
    ]

    all_unmatched = set()  # tracked across ALL dishes so the end-of-run summary only lists each unmatched token once

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            print(f"Enriching: {row['en_name']}...")

            # deterministic lookup first (no API call, no cost)...
            lookup = lookup_allergens_and_diet(row["ingredients"], allergen_map)
            all_unmatched.update(lookup["unmatched_ingredients"])

            # ...then the one real API call per dish, for the two fields
            # that genuinely need a model's judgment.
            flavor = infer_flavor(client, row["en_name"], row["ingredients"], row["course"])

            # lists get joined back into "+"-separated strings to match the
            # CSV convention already used for `ingredients` in the input data
            row["allergens"] = "+".join(lookup["allergens"])
            row["dietary_tags"] = "+".join(lookup["dietary_tags"])
            row["unmatched_ingredients"] = "+".join(lookup["unmatched_ingredients"])
            row["low_confidence_notes"] = "; ".join(lookup["low_confidence_notes"])
            row["spice_level"] = flavor["spice_level"]
            row["flavor_profile"] = "+".join(flavor["flavor_profile"])
            # provenance columns -- so anyone looking at the output later can
            # tell which fields are a verified lookup vs. a model's guess
            # without having to re-read this script to find out
            row["allergen_source"] = "keyword_lookup:ingredient_allergen_map.json"
            row["flavor_source"] = "llm_inferred_unverified"
            writer.writerow(row)

    print(f"\nWrote {len(rows)} enriched rows to {out_path}")
    if all_unmatched:
        print(f"\n{len(all_unmatched)} ingredient token(s) had no entry in ingredient_allergen_map.json "
              "and were NOT allergen-tagged (treated as unknown, not 'safe'):")
        for token in sorted(all_unmatched):
            print(f"  - {token}")
        print("Add these to backend/data/reference/ingredient_allergen_map.json before trusting allergen filtering on this data.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: python enrich.py <input_csv> <output_csv>")
    main(sys.argv[1], sys.argv[2])
