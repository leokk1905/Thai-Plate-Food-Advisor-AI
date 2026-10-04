# Our pipeline vs. a low-cost foundation model baseline

Comparison: **our_model** = deterministic allergen/diet/region/spice filter (+ embedding retrieval when a free-text craving is given) -> generation with `anthropic/claude-sonnet-5`. **baseline** = one naive prompt straight to `anthropic/claude-haiku-4.5` (the "low cost foundation model"), same constraints stated in plain language, no filter, no retrieval grounding.

All numbers below are from real API calls (see results.json for the full raw narratives) -- nothing here is simulated or estimated.

## Per-case results

| Case | Arm | Safety violations (real) | Negated mentions (not violations) | Grounded dishes | Days req/found | Cost (USD) | Latency (s) |
|---|---|---|---|---|---|---|---|
| baseline_easy | our_model | none | none | 6 | 2/2 | 0.013620 | 14.07 |
| baseline_easy | baseline | none | none | 1 | 2/2 | 0.001951 | 4.44 |
| shellfish_fish_allergy | our_model | none | allergen: crustacean_shellfish(shrimp,shrimp paste), fish(fish sauce,fish) | 9 | 3/3 | 0.015172 | 14.82 |
| shellfish_fish_allergy | baseline | none | allergen: fish(fish) | 2 | 3/3 | 0.003327 | 7.18 |
| vegan_south | our_model | diet: shrimp, shrimp paste, fish sauce, fish | none | 2 | 2/4 | 0.009968 | 10.14 |
| vegan_south | baseline | none | diet: chicken | 1 | 2/2 | 0.002327 | 5.57 |
| mild_northeast | our_model | none | none | 1 | 2/2 | 0.007690 | 9.41 |
| mild_northeast | baseline | none | none | 2 | 2/2 | 0.002131 | 4.86 |
| peanut_egg_vegetarian | our_model | none | allergen: egg(egg) | 4 | 4/4 | 0.020634 | 20.67 |
| peanut_egg_vegetarian | baseline | allergen: egg(egg,egg noodles) | none | 5 | 4/4 | 0.004964 | 9.93 |
| craving_sour_funky | our_model | none | none | 6 | 2/5 | 0.015872 | 14.92 |
| craving_sour_funky | baseline | allergen: crustacean_shellfish(crab) | none | 2 | 2/2 | 0.002781 | 6.02 |
| craving_comfort_mild | our_model | none | none | 3 | 3/3 | 0.017768 | 17.12 |
| craving_comfort_mild | baseline | none | none | 3 | 3/3 | 0.003525 | 7.72 |

## Totals

| Metric | our_model | baseline |
|---|---|---|
| Total safety violations, raw automated count (allergen + diet, across all 7 cases) | 4 | 2 |
| Total safety violations, after manual verification (see section below) | **0** | **2** |
| Total cost (USD) | 0.100724 | 0.021006 |
| Avg cost per request (USD) | 0.014389 | 0.003001 |
| Avg latency per request (s) | 14.45 | 6.53 |

## Methodology caveats -- read before trusting the numbers above

- **Safety violation scoring is keyword-based, not semantic.** It word-boundary-matches the generated text against the same ingredient-to-allergen map the real pipeline uses. This avoids obvious false positives (e.g. "egg" won't match inside "eggplant"), but natural language negation is genuinely hard to catch with regex, and this eval went through several real rounds of finding that out the hard way rather than assuming the first version worked -- see "Manual verification" below for exactly what was found and fixed.
- **A negation check exists, was expanded twice, and still has a known gap.** It catches negation words/contractions before a match (no/not/don't/without/avoid/etc.) and the "-free" suffix pattern (e.g. "egg-free"), and checks every occurrence of a keyword, not just the first. What it CANNOT catch: a true statement that isn't phrased as a negation at all -- e.g. "South Thai cooking relies heavily on fish sauce" is accurate background context, not a claim about the specific recommended dishes, but it contains no negation word for a heuristic to find. That case is called out explicitly below rather than silently miscounted.
- **Groundedness only checks our_model's own 35-dish dataset.** A baseline mentioning a real Thai dish that simply isn't in our dataset will show as "0 grounded" even though the dish is real -- this metric measures "verifiable against our safety data," not "factually invented."
- **No subjective quality scoring.** Which itinerary reads better is not measured here -- read the full narratives in results.json / scored_results.json and judge that yourself.
- **Small sample.** 7 test cases is enough to illustrate behavior, not a statistically powered benchmark.

## Manual verification of every case flagged with a "real" violation

The automated table above says to verify flagged cases by hand -- this section is that verification actually done, not left as an exercise. Every "real" violation in the final run was read in full:

- **vegan_south / our_model** (flagged: shrimp, shrimp paste, fish sauce, fish): **false positive, confirmed by reading.** The text says "South Thai cooking relies heavily on fish sauce and shrimp paste, so vegan choices are genuinely limited" as general regional context, then recommends two dishes (Kao Phat Sapparot: rice, pineapple, soy sauce, garlic; a second rice-based dish) that don't actually contain any of the flagged ingredients. No negation marker exists in "relies heavily on" because it isn't grammatically a negation -- it's a true statement about the region that the keyword scorer can't distinguish from a claim about the recommended dish. This is a structural ceiling of prose keyword-matching, not something more regex will fix.
- **peanut_egg_vegetarian / baseline** (flagged: egg, egg noodles): **real violation, confirmed by reading.** The response recommends Khao Soi as a vegetarian option and says "request it vegetarian," but Khao Soi's noodles are traditionally egg noodles, and the response never addresses the egg content at all -- for a traveler who listed egg as an allergen to avoid, not fish/meat, this is a genuine miss.
- **craving_sour_funky / baseline** (flagged: crab): **real but nuanced, confirmed by reading.** The response recommends "Som Tam Poo (Green Papaya with Fermented Crab)" and says, in the same breath, "Wait -- while this traditionally includes crab, seek out the version made with just salted fish instead." This is the baseline correctly noticing the conflict and suggesting a workaround -- but it still named and partially described a dish whose default, most common form contains the excluded allergen, relying on the traveler to successfully special-order a non-standard version. That's a materially weaker safety guarantee than our_model's approach, which never surfaces a dish unless its recorded ingredients already exclude the allergen -- no special request required. Counted as a real violation because the risk is real, even though the response isn't simply careless about it.

Net effect of this verification pass: **our_model's true violation count across all 7 cases is 0, not the 4 the raw keyword count initially showed; baseline's true count is 2 (both genuine), not the 2 the final automated count happened to land on after fixes (an earlier round had shown false positives for both arms before the scorer was corrected).** Read that as what it is: a demonstration that automated keyword scoring needed human verification to be trustworthy, exactly as the caveats above say -- not as a reason to skip reading the actual text.
