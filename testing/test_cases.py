"""
Test scenarios for the "our pipeline vs. a low-cost foundation model" eval.

Each case is the same traveler request sent down two different paths (see
run_eval.py):
  (A) "our model"   -- the real pipeline: deterministic allergen/diet/region/
      spice filter -> (embedding retrieval, if preference_text is set) ->
      generation with anthropic/claude-sonnet-5.
  (B) "baseline"    -- a single naive prompt straight to a cheap foundation
      model (anthropic/claude-haiku-4.5), stating the same constraints in
      plain language, with NO filter and NO retrieval grounding -- it has
      to rely entirely on its own general knowledge and instruction-following.

Chosen to cover, deliberately: an easy case, the hardest realistic allergy
case for Thai food (fish + shellfish -- fish sauce is in almost everything),
a diet+region combo narrow enough that very few dishes may qualify, a spice
ceiling that cuts against Isan food's reputation, a multi-constraint
intersection, and two free-text "craving" cases (which only exercise
retrieval on the "our model" side -- the baseline has no retrieval step to
test at all, which is itself part of what this eval is checking).
"""

TEST_CASES = [
    {
        "id": "baseline_easy",
        "description": "Easy case, no hard constraints beyond region -- sanity check both arms can do the basic job.",
        "regions": ["Central"],
        "days": 2,
        "exclude_allergens": [],
        "max_spice": 5,
        "diet": "none",
        "preference_text": "",
    },
    {
        "id": "shellfish_fish_allergy",
        "description": "The hardest realistic allergy case for Thai food -- fish sauce and shrimp paste are in most savory dishes.",
        "regions": None,
        "days": 3,
        "exclude_allergens": ["crustacean_shellfish", "fish"],
        "max_spice": 5,
        "diet": "none",
        "preference_text": "",
    },
    {
        "id": "vegan_south",
        "description": "Narrow diet+region combo -- South Thai cooking leans heavily on fish sauce/shrimp paste, so vegan options may be scarce or absent.",
        "regions": ["South"],
        "days": 2,
        "exclude_allergens": [],
        "max_spice": 5,
        "diet": "vegan",
        "preference_text": "",
    },
    {
        "id": "mild_northeast",
        "description": "Spice ceiling that cuts against Isan (Northeast) food's reputation for heat -- tests whether the constraint actually holds.",
        "regions": ["Northeast"],
        "days": 2,
        "exclude_allergens": [],
        "max_spice": 1,
        "diet": "none",
        "preference_text": "",
    },
    {
        "id": "peanut_egg_vegetarian",
        "description": "Multi-constraint intersection -- vegetarian AND no peanut AND no egg, across a 4-day trip.",
        "regions": None,
        "days": 4,
        "exclude_allergens": ["peanut", "egg"],
        "max_spice": 3,
        "diet": "vegetarian",
        "preference_text": "",
    },
    {
        "id": "craving_sour_funky",
        "description": "Free-text preference case -- exercises retrieval on the 'our model' side; baseline has no retrieval step at all.",
        "regions": None,
        "days": 2,
        "exclude_allergens": ["crustacean_shellfish"],
        "max_spice": 5,
        "diet": "none",
        "preference_text": "something sour and funky, lots of fermented flavor",
    },
    {
        "id": "craving_comfort_mild",
        "description": "Second free-text preference case, different region/spice mix, to avoid drawing conclusions from a single craving example.",
        "regions": ["Central", "North"],
        "days": 3,
        "exclude_allergens": [],
        "max_spice": 2,
        "diet": "none",
        "preference_text": "comforting, mild, like a hug in a bowl",
    },
]
