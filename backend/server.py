"""
Local test server for the Phase 0 prototype -- a thin HTTP wrapper around the
same functions the CLI scripts already use (itinerary/planner.py,
embedding/query.py). No new logic lives here; this just exposes what already
existed to the browser page in static/index.html.

The OpenRouter key stays server-side only -- the browser never sees it, it
only ever talks to this server's own /api/* routes.

Run:
    python server.py
Then open http://127.0.0.1:8000
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
load_dotenv(BASE_DIR / ".env")

sys.path.insert(0, str(BASE_DIR))
from itinerary.planner import (  # noqa: E402
    load_dishes, filter_dishes, build_daily_plan, generate_narrative, generate_narrative_rag,
)
from embedding.query import load as load_embeddings, top_matches, rank_candidates_by_preference  # noqa: E402

ENRICHED_PATH = BASE_DIR / "data" / "enriched_dishes.csv"

app = FastAPI()

# Everything below runs once, at import time (module load), not per-request --
# the dish CSV, the embedding vectors, and the embedding model itself are all
# expensive-ish to load, so every request reuses these same in-memory objects
# rather than re-reading files or re-loading the model each time.
dishes = load_dishes(str(ENRICHED_PATH))
vectors, embed_meta = load_embeddings()
embed_model = SentenceTransformer(embed_meta["model"])

api_key = os.environ.get("OPENROUTER_API_KEY")
llm_client = None
if api_key:
    from openai import OpenAI
    llm_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
# if no key is configured, llm_client stays None and /api/itinerary reports
# that clearly instead of crashing the whole server at startup -- /api/search
# and /api/meta don't need the LLM at all, so they still work either way.


class SearchRequest(BaseModel):
    query: str


class ItineraryRequest(BaseModel):
    # list defaults like `[]` here are safe with Pydantic (unlike a plain
    # Python function signature, where a mutable default is shared across
    # every call) -- Pydantic gives each request its own fresh copy.
    regions: Optional[List[str]] = None
    days: int = 3
    exclude_allergens: List[str] = []
    max_spice: int = 5
    diet: str = "none"
    preference_text: str = ""  # non-empty turns this into the RAG path -- see the branch in itinerary() below


@app.post("/api/search")
def search(req: SearchRequest):
    if not req.query.strip():
        return {"results": []}
    matches = top_matches(req.query, embed_model, vectors, embed_meta["dishes"], k=5)
    return {"results": [{"dish": d, "score": round(s, 3)} for d, s in matches]}


@app.post("/api/itinerary")
def itinerary(req: ItineraryRequest):
    if llm_client is None:
        return {"error": "Server has no OPENROUTER_API_KEY configured -- set it in backend/.env and restart."}

    # Hard filter ALWAYS runs first, for both the plain and RAG paths below --
    # this is the one step that must never be skipped or reordered, since
    # it's what actually guarantees allergen/diet/region/spice safety.
    filtered = filter_dishes(
        dishes,
        regions=req.regions,
        exclude_allergens=req.exclude_allergens,
        max_spice=req.max_spice,
        required_diet=req.diet,
    )
    if not filtered:
        return {"error": "No dishes in this demo dataset match those constraints."}

    # Note: req.days isn't validated for being >= 1 anywhere below -- the
    # frontend's <input min="1"> enforces that for the page, but a direct
    # API call with days=0 would just produce an empty plan (empty
    # narrative), not a crash. Not worth defending against for a local test
    # server with one client.
    preference_text = req.preference_text.strip()

    if preference_text:
        # RAG path: re-rank the already-safety-filtered dishes by embedding
        # similarity to the free-text craving, THEN generate grounded in
        # that retrieval order. The hard filter above already ran first --
        # retrieval only reorders within the safe set, never expands it.
        ranked = rank_candidates_by_preference(
            preference_text, embed_model, vectors, embed_meta["dishes"], filtered,
        )
        retrieval_scores = {d["en_name"]: s for d, s in ranked}
        ranked_dishes = [d for d, _ in ranked]

        plan = build_daily_plan(ranked_dishes, req.days, shuffle=False)
        narrative, _usage = generate_narrative_rag(llm_client, plan, req.dict(), preference_text, retrieval_scores)
        return {
            "narrative": narrative,
            "matched_dish_count": len(filtered),
            "retrieved": [{"en_name": d["en_name"], "score": round(s, 3)} for d, s in ranked[:10]],
        }

    plan = build_daily_plan(filtered, req.days)
    narrative, _usage = generate_narrative(llm_client, plan, req.dict())
    return {"narrative": narrative, "matched_dish_count": len(filtered)}


@app.get("/api/meta")
def get_meta():
    """So the page can show real numbers instead of hardcoded ones."""
    from collections import Counter
    regions = sorted(Counter(d["region"] for d in dishes))
    allergens = sorted({a for d in dishes for a in d.get("allergens", "").split("+") if a})
    return {
        "dish_count": len(dishes),
        "regions": regions,
        "allergens": allergens,
        "llm_configured": llm_client is not None,
    }


# Must be registered LAST: this mount catches every path not already
# matched above (including "/" itself, serving index.html, via html=True).
# If this were registered before the /api/* routes, it would swallow those
# requests too and they'd never reach the handlers above.
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
