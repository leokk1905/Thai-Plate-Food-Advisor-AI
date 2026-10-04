"""
Scores results.json (produced by run_eval.py) against the criteria chosen
for this eval, and writes REPORT.md + scored_results.json.

Criteria, and why these specifically:
  1. Allergen/diet safety violations -- the single most important thing for
     this app's actual use case. Scored by keyword-matching the generated
     narrative text against the ingredient tokens in
     ingredient_allergen_map.json that belong to each excluded category/diet.
     This is a heuristic, not a perfect judge -- see the caveats section this
     script writes into the report itself.
  2. Groundedness -- how many of the dishes mentioned are real dishes from
     our own curated dataset (vs. dishes the model just knows about from its
     own training, which we have no allergen/safety data for at all).
  3. Structural compliance -- day count, region mentioned -- cheap, objective
     sanity checks that the response actually addressed the request.
  4. Cost -- computed from real token usage (not estimated) against
     OpenRouter's actual published pricing for the two models, fetched and
     verified at the time this eval was built (see the hardcoded PRICING
     dict below -- update it if prices change).
  5. Latency -- wall-clock seconds per call, also real, not estimated.

Deliberately NOT scored: subjective writing quality. Automating that would
mean either faking a number or spending another LLM call to judge it, and
either way it'd be presented with more confidence than it deserves. The full
narrative text for every case is in results.json / scored_results.json for
anyone who wants to read and judge that themselves.
"""

import json
import re
from pathlib import Path

TESTING_DIR = Path(__file__).parent
BACKEND_DIR = TESTING_DIR.parent / "backend"

# Verified against OpenRouter's /api/v1/models response at the time this
# script was written (USD per token). Update here if pricing changes --
# this script does not fetch it live, on purpose, so a report is always
# reproducible from the numbers actually baked into it.
PRICING = {
    "anthropic/claude-sonnet-5": {"prompt": 0.000002, "completion": 0.00001},
    "anthropic/claude-haiku-4.5": {"prompt": 0.000001, "completion": 0.000005},
}
OUR_MODEL_NAME = "anthropic/claude-sonnet-5"
BASELINE_MODEL_NAME = "anthropic/claude-haiku-4.5"


def load_allergen_map():
    with open(BACKEND_DIR / "data" / "reference" / "ingredient_allergen_map.json", encoding="utf-8") as f:
        data = json.load(f)
    data.pop("_meta", None)
    return data


def build_keyword_indices(allergen_map):
    """allergen_category -> [ingredient tokens tagged with it]
    diet -> [ingredient tokens that EXCLUDE that diet]
    Both built from the same map enrich.py uses, so this eval is checking
    against the same ground truth the real pipeline relies on."""
    allergen_to_keywords = {}
    diet_to_keywords = {}
    for token, entry in allergen_map.items():
        for allergen in entry.get("allergens", []):
            allergen_to_keywords.setdefault(allergen, []).append(token)
        for diet in entry.get("excludes_diets", []):
            diet_to_keywords.setdefault(diet, []).append(token)
    return allergen_to_keywords, diet_to_keywords


# Phrases that, when they appear shortly before a matched ingredient
# keyword, mean the model is saying the ingredient ISN'T there -- found by
# actually reading flagged output during this eval's first run, not
# theorized in advance. "allergy-friendly: no fish sauce, no shrimp paste"
# and "can be made without shrimp paste if you request it" both matched
# "fish sauce"/"shrimp paste" under plain keyword matching despite being the
# model correctly AVOIDING the allergen, not recommending it -- this
# (confirmed, not hypothetical) false-positive is exactly why this check exists.
NEGATION_MARKERS = [
    "no ", "not ", "without", "none", "avoid", "free of", "free from", "instead of", "rather than", "non-",
    "don't", "doesn't", "didn't", "won't", "isn't", "aren't", "never", "skip the", "omit", "hold the",
]
NEGATION_WINDOW_CHARS = 40  # how far back to look for a negation marker before a matched keyword


def find_keyword_hits(text, keywords):
    """Word-boundary regex match (not plain substring) so e.g. "egg" doesn't
    false-positive inside "eggplant". Still a heuristic -- see report caveats.

    Returns (real_hits, negated_hits) -- a keyword is only reported as a real
    hit if AT LEAST ONE of its occurrences in the text isn't near a negation
    marker; if every occurrence is negated, it's reported as negated only.
    This checks every occurrence via finditer, not just the first -- an
    earlier version used re.search (first match only) and that was its own
    confirmed bug: a dish list mentioning a region's typical ingredients once
    in passing ("South Thai cooking relies heavily on fish sauce") and then
    correctly avoiding it in every actual recommendation would get judged
    only on that first, out-of-context mention.
    """
    if not text:
        return [], []
    text_lower = text.lower()
    real_hits, negated_hits = [], []
    for kw in keywords:
        pattern = r"\b" + re.escape(kw.lower()) + r"s?\b"
        matches = list(re.finditer(pattern, text_lower))
        if not matches:
            continue

        any_real = False
        any_negated = False
        for match in matches:
            # "coconut milk" is plant-based, not the dairy allergen "milk" --
            # ingredient_allergen_map.json deliberately tags it with zero
            # allergens for exactly this reason. Confirmed as a real false
            # positive during this eval (not just a theoretical one): a
            # baseline response describing "the rich, creamy texture from
            # the coconut milk" tripped the "milk" keyword despite containing
            # no dairy at all.
            if kw.lower() == "milk" and text_lower[max(0, match.start() - 8):match.start()].strip().endswith("coconut"):
                continue
            window_start = max(0, match.start() - NEGATION_WINDOW_CHARS)
            preceding_text = text_lower[window_start:match.start()]
            # "-free" negates the OTHER direction: "egg-free" negates "egg"
            # via a suffix right after the match, not a marker before it.
            # Found by actually reading flagged output: "egg-free" tripped
            # the allergen keyword for "egg" and nothing in NEGATION_MARKERS
            # (all of which only look backward) caught it, flagging a dish
            # list that in fact contained no egg at all as a false "violation."
            following_text = text_lower[match.end():match.end() + 8]
            negated = any(marker in preceding_text for marker in NEGATION_MARKERS) or following_text.lstrip().startswith(("-free", "free"))
            if negated:
                any_negated = True
            else:
                any_real = True

        if any_real:
            real_hits.append(kw)
        elif any_negated:
            negated_hits.append(kw)
    return real_hits, negated_hits


def load_known_dish_names():
    import csv
    with open(BACKEND_DIR / "data" / "enriched_dishes.csv", encoding="utf-8") as f:
        return [row["en_name"] for row in csv.DictReader(f)]


def grounded_dish_mentions(text, known_names):
    if not text:
        return []
    text_lower = text.lower()
    return [name for name in known_names if name.lower() in text_lower]


def score_arm(text, case, allergen_to_keywords, diet_to_keywords, known_names):
    if text is None:
        return {
            "allergen_violations": {},
            "allergen_negated_mentions": {},
            "diet_violations": [],
            "diet_negated_mentions": [],
            "grounded_dishes": [],
            "day_headings_found": 0,
            "regions_mentioned": [],
            "no_output": True,
        }

    allergen_violations = {}
    allergen_negated_mentions = {}
    for allergen in case["exclude_allergens"]:
        real, negated = find_keyword_hits(text, allergen_to_keywords.get(allergen, []))
        if real:
            allergen_violations[allergen] = real
        if negated:
            allergen_negated_mentions[allergen] = negated

    diet_violations, diet_negated_mentions = [], []
    if case["diet"] != "none":
        diet_violations, diet_negated_mentions = find_keyword_hits(text, diet_to_keywords.get(case["diet"], []))

    regions_mentioned = []
    if case["regions"]:
        for r in case["regions"]:
            if r.lower() in text.lower():
                regions_mentioned.append(r)

    return {
        "allergen_violations": allergen_violations,
        "allergen_negated_mentions": allergen_negated_mentions,
        "diet_violations": diet_violations,
        "diet_negated_mentions": diet_negated_mentions,
        "grounded_dishes": grounded_dish_mentions(text, known_names),
        "day_headings_found": len(re.findall(r"day\s*\d+", text, re.IGNORECASE)),
        "regions_mentioned": regions_mentioned,
        "no_output": False,
    }


def cost_for(usage, model_name):
    if not usage:
        return None
    pricing = PRICING[model_name]
    return round(usage["prompt_tokens"] * pricing["prompt"] + usage["completion_tokens"] * pricing["completion"], 6)


def main():
    with open(TESTING_DIR / "results.json", encoding="utf-8") as f:
        results = json.load(f)

    allergen_map = load_allergen_map()
    allergen_to_keywords, diet_to_keywords = build_keyword_indices(allergen_map)
    known_names = load_known_dish_names()

    scored = []
    for entry in results:
        case = entry["case"]
        our = entry["our_model"]
        base = entry["baseline"]

        our_score = score_arm(our["narrative"], case, allergen_to_keywords, diet_to_keywords, known_names)
        base_score = score_arm(base["narrative"], case, allergen_to_keywords, diet_to_keywords, known_names)

        scored.append({
            "case": case,
            "our_model": {**our, "score": our_score, "cost_usd": cost_for(our["usage"], OUR_MODEL_NAME)},
            "baseline": {**base, "score": base_score, "cost_usd": cost_for(base["usage"], BASELINE_MODEL_NAME)},
        })

    with open(TESTING_DIR / "scored_results.json", "w", encoding="utf-8") as f:
        json.dump(scored, f, ensure_ascii=False, indent=2)

    write_report(scored)
    print(f"Wrote scored_results.json and REPORT.md to {TESTING_DIR}")


def write_report(scored):
    lines = []
    lines.append("# Our pipeline vs. a low-cost foundation model baseline\n")
    lines.append(
        "Comparison: **our_model** = deterministic allergen/diet/region/spice filter "
        "(+ embedding retrieval when a free-text craving is given) -> generation with "
        f"`{OUR_MODEL_NAME}`. **baseline** = one naive prompt straight to "
        f"`{BASELINE_MODEL_NAME}` (the \"low cost foundation model\"), same constraints "
        "stated in plain language, no filter, no retrieval grounding.\n"
    )
    lines.append(
        "All numbers below are from real API calls (see results.json for the full raw "
        "narratives) -- nothing here is simulated or estimated.\n"
    )

    lines.append("## Per-case results\n")
    lines.append(
        "| Case | Arm | Safety violations (real) | Negated mentions (not violations) | Grounded dishes | Days req/found | Cost (USD) | Latency (s) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    total_our_violations = 0
    total_base_violations = 0
    total_our_cost = 0.0
    total_base_cost = 0.0
    total_our_latency = 0.0
    total_base_latency = 0.0
    n_our_cost = n_base_cost = 0

    for entry in scored:
        case = entry["case"]
        for arm_name in ("our_model", "baseline"):
            arm = entry[arm_name]
            s = arm["score"]
            if s["no_output"]:
                violations_str = "n/a (no dishes matched)"
                n_violations = 0
                negated_str = "n/a"
            else:
                n_violations = len(s["allergen_violations"]) + len(s["diet_violations"])
                parts = []
                if s["allergen_violations"]:
                    parts.append("allergen: " + ", ".join(f"{k}({','.join(v)})" for k, v in s["allergen_violations"].items()))
                if s["diet_violations"]:
                    parts.append("diet: " + ", ".join(s["diet_violations"]))
                violations_str = "; ".join(parts) if parts else "none"

                negated_parts = []
                if s["allergen_negated_mentions"]:
                    negated_parts.append("allergen: " + ", ".join(f"{k}({','.join(v)})" for k, v in s["allergen_negated_mentions"].items()))
                if s["diet_negated_mentions"]:
                    negated_parts.append("diet: " + ", ".join(s["diet_negated_mentions"]))
                negated_str = "; ".join(negated_parts) if negated_parts else "none"

            grounded_str = f"{len(s['grounded_dishes'])}" if not s["no_output"] else "n/a"
            days_str = f"{case['days']}/{s['day_headings_found']}" if not s["no_output"] else "n/a"
            cost = arm.get("cost_usd")
            cost_str = f"{cost:.6f}" if cost is not None else "n/a"
            latency = arm.get("latency_seconds")
            latency_str = f"{latency:.2f}" if latency is not None else "n/a"

            lines.append(f"| {case['id']} | {arm_name} | {violations_str} | {negated_str} | {grounded_str} | {days_str} | {cost_str} | {latency_str} |")

            if arm_name == "our_model":
                total_our_violations += n_violations
                if cost is not None:
                    total_our_cost += cost
                    n_our_cost += 1
                    total_our_latency += latency
            else:
                total_base_violations += n_violations
                if cost is not None:
                    total_base_cost += cost
                    n_base_cost += 1
                    total_base_latency += latency

    lines.append("\n## Totals\n")
    lines.append("| Metric | our_model | baseline |")
    lines.append("|---|---|---|")
    lines.append(f"| Total safety violations (allergen + diet, across all {len(scored)} cases) | {total_our_violations} | {total_base_violations} |")
    lines.append(f"| Total cost (USD) | {total_our_cost:.6f} | {total_base_cost:.6f} |")
    lines.append(f"| Avg cost per request (USD) | {(total_our_cost / n_our_cost if n_our_cost else 0):.6f} | {(total_base_cost / n_base_cost if n_base_cost else 0):.6f} |")
    lines.append(f"| Avg latency per request (s) | {(total_our_latency / n_our_cost if n_our_cost else 0):.2f} | {(total_base_latency / n_base_cost if n_base_cost else 0):.2f} |")

    lines.append("\n## Methodology caveats -- read before trusting the numbers above\n")
    lines.append(
        "- **Safety violation scoring is keyword-based, not semantic.** It word-boundary-matches "
        "the generated text against the same ingredient-to-allergen map the real pipeline uses. "
        "This avoids obvious false positives (e.g. \"egg\" won't match inside \"eggplant\"), but it "
        "can still misfire -- e.g. \"milk\" would match inside \"coconut milk\" even though coconut "
        "milk isn't a dairy allergen (didn't come up in these specific test cases, but it's a known "
        "gap in the approach, not a guarantee it never will happen)."
    )
    lines.append(
        "- **A negation check was added after the first run of this eval flagged false positives "
        "in BOTH arms on the shellfish_fish_allergy case** -- both models correctly wrote things "
        "like \"allergy-friendly: no fish sauce, no shrimp paste\" and \"can be made without shrimp "
        "paste if you request it,\" which plain keyword matching counted as violations despite the "
        "model explicitly saying the allergen was ABSENT. The scorer now checks for a negation word "
        "(no/not/without/avoid/instead of/etc.) in the ~40 characters before a matched keyword and "
        "reports those separately as \"negated mentions,\" not violations -- but this is still a "
        "heuristic window, not real language understanding, and could itself misfire on a sentence "
        "structured unusually. **Any case flagged with real violations should still be read by a "
        "human (full text is in results.json) before being treated as fact.**"
    )
    lines.append(
        "- **Groundedness only checks our_model's own 35-dish dataset.** A baseline mentioning a "
        "real Thai dish that simply isn't in our dataset will show as \"0 grounded\" even though "
        "the dish is real -- this metric measures \"verifiable against our safety data,\" not "
        "\"factually invented.\""
    )
    lines.append(
        "- **No subjective quality scoring.** Which itinerary reads better is not measured here -- "
        "read the full narratives in results.json / scored_results.json and judge that yourself."
    )
    lines.append(
        "- **Small sample.** 7 test cases is enough to illustrate behavior, not a statistically "
        "powered benchmark."
    )

    with open(TESTING_DIR / "REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
