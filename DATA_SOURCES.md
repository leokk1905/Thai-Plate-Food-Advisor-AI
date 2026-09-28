# Data sources

This file tracks where every piece of "factual" data in the pipeline comes from,
so allergen/dietary claims can be traced back to something verifiable instead of
trusted on faith. Update this file whenever a new source is added or a status
changes.

## In use

### FDA "Big 9" major food allergens
- **What**: the fixed schema of allergen categories the pipeline tags against
  (milk, egg, fish, crustacean shellfish, tree nut, peanut, wheat, soybean, sesame).
- **Where it comes from**: US FDA / FASTER Act (sesame added as the 9th allergen,
  effective 2023). FDA's January 2025 "Guidance for Industry: Questions and
  Answers Regarding Food Allergens (Edition 5)" removed **coconut** from the
  tree-nut category (botanically a fruit, not shown to correlate with tree-nut
  allergy risk) — `ingredient_allergen_map.json` reflects this and does not tag
  coconut milk as tree_nut.
- **Access**: public, free, regulatory text. No API — used as a fixed reference
  list, not queried live.
- **Links**: [FDA – Food Allergies](https://www.fda.gov/food/nutrition-food-labeling-and-critical-foods/food-allergies) · [FASTER Act / sesame](https://www.fda.gov/food/food-allergies/faster-act-sesame-ninth-major-food-allergen) · [FSIS "Big 9" summary](https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/food-allergies-big-9)
- **Caveat**: this is a *US regulatory* list, used here only as a convenient,
  well-defined allergen taxonomy. It is not Thailand-specific and does not
  cover every substance a traveler might react to.

### `backend/data/reference/ingredient_allergen_map.json` (developer-authored, cross-checked against Open Food Facts)
- **What**: the deterministic ingredient → allergen / diet-exclusion lookup
  table the enrichment pipeline actually runs against.
- **Where it comes from**: hand-built by the developer using general food
  knowledge, with the allergen *category list* (not individual ingredient
  values) cross-checked against Open Food Facts' allergens taxonomy — see
  below for exactly what that added and why individual entries are still
  hand-authored rather than pulled automatically.
- **Status**: starting point, now covering 64 ingredient tokens across 12
  allergen categories (FDA-9 plus molluscs, mustard, celery). Needs review/
  expansion by someone with Thai food knowledge before it's trusted for real
  allergy-safety decisions. Ingredient tokens with no entry are left
  unmatched (see `unmatched_ingredients` column in enrichment output) rather
  than guessed.

### Open Food Facts allergens taxonomy (partial use)
- **What it is**: free, open-data (ODbL license) food product database.
  `https://static.openfoodfacts.org/data/taxonomies/allergens.json` is a
  static file (fetched 2026-09-24) listing ~27 allergen/cross-reactive
  categories (an EU-style list, broader than the FDA's 9) with names in many
  languages including Thai.
- **Why it's not a live lookup**: confirmed its API is barcode/product-lookup
  oriented (`/api/v2/product/{barcode}.json`) — there's no confirmed endpoint
  for submitting arbitrary free-text ingredients (like ours) and getting
  allergen tags back. So it's used as a one-time reference download, not a
  live call in the pipeline.
- **What it actually changed**: cross-checking this taxonomy against our
  ingredient list surfaced **molluscs** as a category the FDA-9 doesn't
  cover but that matters a lot for Thai food (squid, mussels, oyster sauce),
  plus **mustard** and **celery**. Those three are now in
  `ingredient_allergen_map.json`. Two more categories in OFF's list
  (sulphites, lupin) were left out because no ingredient currently in this
  dataset triggers them — add them when one does (e.g. dried fruit/wine for
  sulphites).
- **Data-quality catch worth remembering**: OFF's Thai translation for
  `en:nuts` is literally "almond" (อัลมอนด์), not a general word for nuts —
  a reminder that its community-contributed translations are a lead to
  verify, not a fact to surface to users directly. This is why individual
  ingredient entries in the map are still hand-authored rather than
  auto-populated from OFF's translations.
- **License**: Open Database License (ODbL) — attribution required if this
  taxonomy data is redistributed.
- **Links**: [allergens.json](https://static.openfoodfacts.org/data/taxonomies/allergens.json) · [API docs](https://openfoodfacts.github.io/openfoodfacts-server/api/) · [Data/SDKs](https://world.openfoodfacts.org/data)

### Edamam Food Database API
- **What it is**: commercial nutrition/allergen API that accepts free-text
  ingredients and returns allergen + diet labels via NLP (this is the
  capability Open Food Facts does not confirm).
- **Why not used**: requires an API key, has commercial pricing/rate limits,
  and is not open source. Coverage of Thai-specific ingredient names is
  unverified.
- **Links**: [Food Database API docs](https://developer.edamam.com/food-database-api-docs)
- **Decision needed**: worth a small evaluation if the hand-built map proves
  too slow to extend, but conflicts with the "mostly open source" goal.

### Thai Food Composition Database (Thai FCD), Institute of Nutrition, Mahidol University (INMU) — direct access abandoned
- **What it is**: the FAO/INFOODS-recognized authoritative nutrition database
  for Thai foods, lab-analyzed by INMU.
- **Checked 2026-09-24**: fetched the site's "Download" page directly —
  it has no self-serve bulk export, only contact emails for data requests.
- **Decision**: per your instruction, not pursuing direct contact — moved on
  to the two derived sources below instead, which get the same underlying
  data without a manual request.
- **Links**: [Thai FCD](https://inmu.mahidol.ac.th/thaifcd/)

## In use (added 2026-09-24)

### Thai_Nutrition_Dataset (Hugging Face, `BaoWio/Thai_Nutrition_Dataset`)
- **What it is**: a third-party re-publication of **the same INMU Thai Food
  Composition Tables** above, already converted to a clean CSV. Downloaded
  and inspected directly (not taken on the dataset card's word):
  `backend/data/reference/thai_nutrition_hf.csv`, 1,005 rows, columns
  `Food_Code, Thai_Name, English_Name, Scientific_name, SUGAR(g), Protein(g),
  Fat(g), Energy(kcal), CHOCDF(g), FIBTG(g)`.
- **Provenance**: its own README states the source explicitly — "Thai Food
  Composition Tables 2015, Institute of Nutrition, Mahidol University (INMU)."
  Confirmed by inspecting real rows (e.g. `A1,"ข้าวกล้อง...","Rice, brown,
  Jasmine variety, raw",Oryza sativa,0.25,7.34,2.95,363,-,3.5`).
- **License conflict worth knowing about**: the HF re-uploader tags this
  **CC-BY-SA-4.0** (permits commercial use). INMU's own site states its data
  is free for **non-commercial use with attribution**. A re-uploader can't
  grant broader rights than they hold — treat this data as bound by INMU's
  stricter non-commercial+attribution terms regardless of the CC-BY-SA-4.0
  tag, until/unless confirmed otherwise directly with INMU.
- **Real limitation, not a source problem**: this is a per-ingredient
  (per-100g) composition table, not per-dish. Our dish data has no ingredient
  quantities, so it cannot responsibly produce a per-dish calorie total —
  doing that would fabricate precision that isn't there. It's usable as
  ingredient-level reference data only, for now.
- **Matching is manual, not automatic, and partial**: naive substring
  matching against this table is actively wrong (searching "pork" hits 55
  rows, including an instant-noodle-flavor-packet). `ingredient_nutrition_map.json`
  hand-matches 18 of our ~64 ingredient tokens after inspecting every
  candidate row, and explicitly records 4 more that were checked and found
  to have **no good match** (galangal isn't in the table at all; dried
  shrimp, tamarind paste, and egg noodles only appear inside composite dish
  names, not as standalone rows). The remaining ~42 tokens are simply
  unchecked — extend the map the same way when needed.
- **Links**: [dataset](https://huggingface.co/datasets/BaoWio/Thai_Nutrition_Dataset) · [raw CSV](https://huggingface.co/datasets/BaoWio/Thai_Nutrition_Dataset/resolve/main/nutrition_dataset.csv)

## Checked and set aside

### "Foods in Thailand" (Kaggle, `ponthakornsodchun/foods-in-thailand`)
- **This appears to be the origin of the dataset you already have**, not a
  new source. Fetched the page directly: identical schema (`en_name,
  th_name, ingredients, course, province, region`) and the *exact same
  example values* you originally gave me (`"Koi"` / `"ก้อย"` /
  `"Beef+lime juice+fish sauce+chili+herbs"` / `"main dish"` / `"Ubon
  Ratchathani"` / `"Northeast"`), word for word. It's also tiny (6,973 bytes
  total, ODbL license) and its own acknowledgements section credits
  Wikipedia as the source of the dish information — so it's Wikipedia-level
  reliability for descriptive facts, not an authoritative food-safety source.
  Worth comparing row-for-row against your existing database to check for
  new dishes it might add, but treat it as the same lineage, not independent
  corroboration.

### `thai-food-dataset` (Kaggle, `chanipornnerunchorn/thai-food-dataset`)
- Another unofficial CSV extraction of the same INMU Thai FCD (2025 version
  this time), per its own description. License tag on Kaggle is "Unknown."
  Set aside in favor of the Hugging Face version above, which already has
  clean nutrient columns extracted and a stated (if disputable) license —
  no need to maintain two mirrors of the same underlying source.
- **Link**: [dataset](https://www.kaggle.com/datasets/chanipornnerunchorn/thai-food-dataset)

## Not a data source: the LLM (Claude API)

The Claude API is used in two places and neither is treated as a source of
fact:
- `enrichment/enrich.py` — estimates **spice_level** and **flavor_profile**
  only (genuinely subjective qualities with no reference dataset). It never
  determines allergens or dietary tags — those come from the lookup table above.
- `itinerary/planner.py` — writes the narrative description of an itinerary
  that has already been filtered by the deterministic allergen/diet/spice
  logic. It's instructed to only describe dishes it's given, not invent dishes,
  restaurants, or prices.

Every LLM-derived column in the enriched CSV is tagged with its source
(`flavor_source=llm_inferred_unverified`) so it's never confused with the
keyword-lookup-derived allergen columns.
