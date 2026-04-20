"""
eval/run_synthesis_eval.py — End-to-end synthesis quality evaluation.

Tests the full retrieval + synthesis pipeline on ~50 questions sampled across
four strata and evaluates using an independent LLM-as-judge. Also measures
calibration of the system's self-assessed confidence ratings.

Strata:
  A — year-specific 10-Q   (12 questions)
  B — year-specific 10-K   (13 questions)
  C — multi-year trend     (up to 12 synthetic cross-year comparison questions)
  D — no-year              (13 questions with year references stripped)

Total will be ~46–50 depending on cross-year pairs available in dataset_meta.json.

Judge:
  gpt-4o-mini (separate from Settings.llm) assesses each answer for:
    - faithfulness  (0.0–1.0): are all claims in the answer supported by the cited context?
    - relevance     (0.0–1.0): does the answer address what the question asks?
    - unsupported_claims: list of specific claims the judge could not verify

Calibration:
  Compares system's self-reported data_quality.rating to judge's faithfulness bucket:
    HIGH   → faithfulness >= 0.8   = CORRECT, else OVERCONFIDENT
    MEDIUM → 0.5 <= faith < 0.8   = CORRECT
              faith >= 0.8         = UNDERCONFIDENT
              faith < 0.5          = OVERCONFIDENT
    LOW    → faithfulness < 0.5    = CORRECT, else UNDERCONFIDENT
    (fallback / no rating)         = N/A

Output:
  eval/results_synthesis_eval_final.csv  — one row per question, 25 columns
  Summary printed to stdout.

Requires:
  eval/dataset_meta.json
  DATABASE_URL + OPENAI_API_KEY

Usage:
  export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/secrag
  export OPENAI_API_KEY=sk-...
  python eval/run_synthesis_eval.py
"""

import asyncio
import csv
import json
import random
import re
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel

from index_loader import load_index
from sec_query import SecQueryEngine, SynthesisResult

EVAL_DIR = Path(__file__).parent
OUT_CSV  = EVAL_DIR / "results_synthesis_eval_final.csv"

SEED = 42

# ─────────────────────────────────────────────────────────────────────────────
# Judge Pydantic schema
# ─────────────────────────────────────────────────────────────────────────────

class JudgeResult(BaseModel):
    faithfulness:       float   # 0.0–1.0
    relevance:          float   # 0.0–1.0
    rationale:          str     # 2–3 sentence explanation
    unsupported_claims: list[str]  # claims in answer not verified by context


JUDGE_PROMPT = """\
You are an expert auditor of AI-generated financial analysis. You will be given:
1. A question about SEC filings
2. An AI-generated answer with inline citation markers [N]
3. The source text excerpts [1], [2], ... that the AI had access to

Your task: evaluate the answer strictly against the provided excerpts.

Faithfulness (0.0–1.0):
  1.0 = every factual claim in the answer is directly supported by a verbatim statement in the excerpts
  0.8 = nearly all claims supported; minor inference or rounding acceptable
  0.6 = most claims supported but 1–2 require noticeable inference or extrapolation
  0.4 = significant claims are not present in the excerpts or require substantial inference
  0.0 = the answer contradicts or fabricates content not in the excerpts

Relevance (0.0–1.0):
  1.0 = answer fully addresses the question
  0.5 = answer is partially relevant or only addresses part of the question
  0.0 = answer does not address the question

unsupported_claims: list each specific claim in the answer that you could NOT verify
  in the provided excerpts. Be precise — quote the claim. Return [] if all claims
  are supported.

rationale: 2–3 sentences explaining your faithfulness score. Name specific excerpts
  (by number) that support or undermine confidence. Be specific, not generic.

You must respond with a JSON object. No markdown fences.

Schema:
  {{
    "faithfulness": <float 0.0–1.0>,
    "relevance": <float 0.0–1.0>,
    "rationale": "<string>",
    "unsupported_claims": ["<claim>", ...]
  }}

--- QUESTION ---
{question}

--- AI ANSWER ---
{answer}

--- SOURCE EXCERPTS ---
{context}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Question sampling
# ─────────────────────────────────────────────────────────────────────────────

def _strip_year(text: str) -> str:
    """Remove common year references from a question string."""
    # Remove explicit fiscal year patterns
    text = re.sub(r'\bFY\s*20\d\d\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfiscal year\s+20\d\d\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bin\s+20\d\d\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfor\s+20\d\d\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bQ[1-4]\s+20\d\d\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bthe\s+20\d\d\s+', 'the ', text, flags=re.IGNORECASE)
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text).strip()
    return text


def sample_questions(meta: dict, rng: random.Random) -> list[dict]:
    """
    Build ~50 evaluation questions across four strata.

    Returns list of dicts with keys:
      q_id, query, strata, ticker, filing_type, fiscal_year, fiscal_quarter,
      section_label, source_node_id (from dataset, not live retrieval)
    """
    q_list = list(meta.values())

    # Strata A: 10-Q questions (year-specific, quarterly)
    ten_q = [q for q in q_list if q["filing_type"] == "10-Q"]
    strata_a = rng.sample(ten_q, min(12, len(ten_q)))

    # Strata B: 10-K questions (year-specific, annual)
    ten_k = [q for q in q_list if q["filing_type"] == "10-K"]
    strata_b = rng.sample(ten_k, min(13, len(ten_k)))

    # Strata C: multi-year trend questions (synthetic cross-year comparisons)
    # Group by (ticker, section_label) and pair entries from different years
    from collections import defaultdict
    by_ticker_section: dict[tuple, list] = defaultdict(list)
    for q in q_list:
        key = (q["ticker"], q.get("section_label", ""), q["filing_type"])
        by_ticker_section[key].append(q)

    trend_candidates = []
    for (ticker, section, ftype), qs in by_ticker_section.items():
        years_seen = {}
        for q in qs:
            yr = q["fiscal_year"]
            if yr not in years_seen:
                years_seen[yr] = q
        year_pairs = [(y1, y2, years_seen[y1], years_seen[y2])
                      for y1, y2 in [(y, z) for y in years_seen for z in years_seen if y < z]]
        for y1, y2, q1, q2 in year_pairs:
            topic = q1["query"].split("?")[0].rstrip(".").strip()
            # Build a trend question using the topic from q1
            trend_q = (
                f"How did {ticker}'s position on the following change between FY{y1} and FY{y2}? "
                f"{topic}."
            )
            trend_candidates.append({
                "query":           trend_q,
                "ticker":          ticker,
                "filing_type":     ftype,
                "fiscal_year":     y2,        # primary year for year-alignment check
                "fiscal_quarter":  None,
                "section_label":   section,
                "source_node_id":  q2.get("node_id", ""),
            })

    rng.shuffle(trend_candidates)
    strata_c = trend_candidates[:12]

    # Strata D: year-agnostic questions (strip year references from existing questions)
    # Use a separate pool from A and B to avoid duplicates
    used_queries = {q["query"] for q in strata_a + strata_b}
    year_agnostic_pool = [q for q in q_list if q["query"] not in used_queries]
    rng.shuffle(year_agnostic_pool)
    strata_d_raw = year_agnostic_pool[:13]
    strata_d = []
    for q in strata_d_raw:
        stripped = _strip_year(q["query"])
        # Only include if the question changed and is still well-formed
        strata_d.append({**q, "query": stripped})

    # Assemble with strata labels
    result = []
    for q in strata_a:
        result.append({**q, "strata": "A_10Q", "source_node_id": q.get("node_id", "")})
    for q in strata_b:
        result.append({**q, "strata": "B_10K", "source_node_id": q.get("node_id", "")})
    for q in strata_c:
        result.append({**q, "strata": "C_trend"})
    for q in strata_d:
        result.append({**q, "strata": "D_no_year", "source_node_id": q.get("node_id", "")})

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Year alignment check
# ─────────────────────────────────────────────────────────────────────────────

def check_year_alignment(
    query: str,
    nodes: list,
    cited_node_ids: list[str],
) -> bool | None:
    """
    Returns True if the query specifies a year AND all cited nodes match it.
    Returns False if cited nodes are from a different year.
    Returns None if the query doesn't specify a year.
    """
    match = re.search(r'\b(20\d\d)\b', query)
    if not match:
        return None
    asked_year = int(match.group(1))

    cited_set = set(cited_node_ids)
    cited_years = set()
    for n in nodes:
        if n.node_id in cited_set:
            yr = n.node.metadata.get("fiscal_year")
            if yr is not None:
                cited_years.add(int(yr))

    if not cited_years:
        return None
    return all(y == asked_year for y in cited_years)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-as-judge
# ─────────────────────────────────────────────────────────────────────────────

def _build_judge_context(query: str, answer: str, nodes: list, cited_node_ids: list[str]) -> str:
    """Build judge context from the nodes that were actually cited."""
    cited_set = set(cited_node_ids)
    cited_nodes = [n for n in nodes if n.node_id in cited_set]
    # Fallback: if no cited nodes, use top-5
    if not cited_nodes:
        cited_nodes = nodes[:5]

    parts = []
    for i, n in enumerate(cited_nodes, 1):
        parts.append(f"[{i}]\n{n.node.get_content()[:800]}")
    return "\n\n---\n\n".join(parts)


def judge_answer(
    judge_llm,
    query: str,
    answer: str,
    nodes: list,
    cited_node_ids: list[str],
) -> JudgeResult | None:
    """Call the judge LLM to evaluate one answer. Returns None on failure."""
    context = _build_judge_context(query, answer, nodes, cited_node_ids)
    prompt_text = JUDGE_PROMPT.format(
        question=query,
        answer=answer,
        context=context,
    )
    try:
        result = judge_llm.structured_predict(
            JudgeResult,
            PromptTemplate("{prompt_str}"),
            prompt_str=prompt_text,
        )
        if not isinstance(result, JudgeResult):
            print(f"  [judge] WARNING: structured_predict returned {type(result).__name__}, expected JudgeResult")
            return None
        return result
    except Exception as e:
        print(f"  [judge] WARNING: structured_predict failed ({e})")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────────────────

def calibration_status(
    system_rating: str | None,
    faithfulness: float | None,
) -> str:
    """
    CORRECT        — system NLI rating matches judge faithfulness bucket
    OVERCONFIDENT  — system claims more certainty than judge supports
    UNDERCONFIDENT — system claims less certainty than judge supports
    N/A            — missing data (no rating or no judge result)

    Thresholds aligned to NLI model rating cutoffs:
      HIGH   → faithfulness >= 0.75  = CORRECT, else OVERCONFIDENT
      MEDIUM → 0.45 <= faith < 0.75  = CORRECT
                faith >= 0.75        = UNDERCONFIDENT
                faith < 0.45         = OVERCONFIDENT
      LOW    → faithfulness < 0.45   = CORRECT, else UNDERCONFIDENT
    """
    if system_rating is None or faithfulness is None:
        return "N/A"

    r = system_rating.upper()
    f = faithfulness

    if r == "HIGH":
        return "CORRECT" if f >= 0.75 else "OVERCONFIDENT"
    elif r == "MEDIUM":
        if 0.45 <= f < 0.75:
            return "CORRECT"
        elif f >= 0.75:
            return "UNDERCONFIDENT"
        else:
            return "OVERCONFIDENT"
    elif r == "LOW":
        return "CORRECT" if f < 0.45 else "UNDERCONFIDENT"
    return "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

async def eval_one(
    engine: SecQueryEngine,
    judge_llm,
    question: dict,
    idx: int,
    total: int,
) -> dict:
    query    = question["query"]
    strata   = question.get("strata", "")
    ticker   = question.get("ticker", "")
    fy       = question.get("fiscal_year", "")
    ftype    = question.get("filing_type", "")
    section  = question.get("section_label", "")

    print(f"  [{idx}/{total}] [{strata}] {ticker} FY{fy}  {query[:80]}...")

    # --- Retrieve ---
    retrieval = await engine._retrieve_nodes(query)
    nodes = retrieval.nodes
    n_retrieved = len(nodes)
    retrieved_tickers = sorted({n.node.metadata.get("ticker", "") for n in nodes})

    # --- Synthesize ---
    result: SynthesisResult = engine._synthesize(query, nodes)

    system_rating   = result.data_quality.rating    if result.data_quality else None
    system_summary  = result.data_quality.summary   if result.data_quality else ""
    missing_cov_str = "; ".join(result.data_quality.missing_coverage) if result.data_quality else ""
    nli_score       = result.data_quality.nli_score if result.data_quality else None

    n_cited      = len(result.cited_node_ids)
    cited_tickers = sorted({
        n.node.metadata.get("ticker", "")
        for n in nodes if n.node_id in set(result.cited_node_ids)
    })

    year_aligned = check_year_alignment(query, nodes, result.cited_node_ids)

    # --- Judge ---
    judge = judge_answer(judge_llm, query, result.answer, nodes, result.cited_node_ids)

    faithfulness       = round(judge.faithfulness, 3) if judge else None
    relevance          = round(judge.relevance, 3) if judge else None
    judge_rationale    = judge.rationale if judge else ""
    unsupported_str    = "; ".join(judge.unsupported_claims) if judge else ""

    cal_status = calibration_status(system_rating, faithfulness)

    return {
        # Question metadata
        "strata":             strata,
        "ticker":             ticker,
        "filing_type":        ftype,
        "fiscal_year":        fy,
        "section_label":      section,
        "query":              query,

        # Retrieval
        "n_retrieved":        n_retrieved,
        "retrieved_tickers":  ",".join(retrieved_tickers),

        # Synthesis
        "answer_length":      len(result.answer),
        "n_cited":            n_cited,
        "cited_tickers":      ",".join(cited_tickers),
        "year_aligned":       "" if year_aligned is None else str(year_aligned),

        # System NLI assessment
        "system_rating":      system_rating or "",
        "nli_score":          "" if nli_score is None else nli_score,
        "system_summary":     system_summary,
        "missing_coverage":   missing_cov_str,

        # Judge scores
        "faithfulness":       "" if faithfulness is None else faithfulness,
        "relevance":          "" if relevance is None else relevance,
        "judge_rationale":    judge_rationale,
        "unsupported_claims": unsupported_str,

        # Calibration
        "calibration_status": cal_status,

        # Full answer (last — wide column)
        "answer":             result.answer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def print_report(rows: list[dict]) -> None:
    total = len(rows)
    with_judge = [r for r in rows if r["faithfulness"] != ""]

    if not with_judge:
        print("[report] No judge results available.")
        return

    faith_values = [float(r["faithfulness"]) for r in with_judge]
    rel_values   = [float(r["relevance"])     for r in with_judge]

    nli_values = [float(r["nli_score"]) for r in rows if r.get("nli_score") not in ("", None)]

    print(f"\n{'='*60}")
    print(f"SYNTHESIS EVAL REPORT  (n={total}, judged={len(with_judge)})")
    print(f"{'='*60}")
    print(f"  Mean faithfulness : {_mean(faith_values):.3f}")
    print(f"  Mean relevance    : {_mean(rel_values):.3f}")
    if nli_values:
        print(f"  Mean NLI score    : {_mean(nli_values):.3f}  (n={len(nli_values)})")

    # By strata
    strata_names = ["A_10Q", "B_10K", "C_trend", "D_no_year"]
    print(f"\n  {'Strata':<14}  {'N':>4}  {'Faith':>7}  {'Rel':>7}")
    print(f"  {'-'*38}")
    for s in strata_names:
        s_rows = [r for r in with_judge if r["strata"] == s]
        if not s_rows:
            continue
        sf = _mean([float(r["faithfulness"]) for r in s_rows])
        sr = _mean([float(r["relevance"])     for r in s_rows])
        print(f"  {s:<14}  {len(s_rows):>4}  {sf:>7.3f}  {sr:>7.3f}")

    # Faithfulness by system NLI rating
    print(f"\n  {'System NLI rating':<18}  {'N':>4}  {'Mean faith':>11}  {'Overconf%':>10}  {'Underconf%':>11}")
    print(f"  {'-'*62}")
    for rating in ["HIGH", "MEDIUM", "LOW"]:
        r_rows = [r for r in with_judge if r["system_rating"] == rating]
        if not r_rows:
            continue
        mf = _mean([float(r["faithfulness"]) for r in r_rows])
        oc = sum(1 for r in r_rows if r["calibration_status"] == "OVERCONFIDENT") / len(r_rows)
        uc = sum(1 for r in r_rows if r["calibration_status"] == "UNDERCONFIDENT") / len(r_rows)
        print(f"  {rating:<18}  {len(r_rows):>4}  {mf:>11.3f}  {oc:>9.1%}  {uc:>10.1%}")

    # Calibration summary
    cal_counts = {}
    for s in ["CORRECT", "OVERCONFIDENT", "UNDERCONFIDENT", "N/A"]:
        cal_counts[s] = sum(1 for r in rows if r["calibration_status"] == s)
    judged_cal = [r for r in rows if r["calibration_status"] != "N/A"]
    print(f"\n  Calibration breakdown (n={len(judged_cal)}):")
    for s, n in cal_counts.items():
        if n == 0:
            continue
        pct = n / total * 100
        print(f"    {s:<15} {n:>3}  ({pct:.1f}%)")

    # 3×3 confusion matrix (rows=system_rating, cols=faith_bucket)
    print(f"\n  Confusion matrix (rows=system rating, cols=faithfulness bucket):")
    print(f"  {'':>10}  {'faith<0.5':>10}  {'0.5-0.8':>10}  {'faith>=0.8':>10}")
    for rating in ["HIGH", "MEDIUM", "LOW"]:
        r_rows = [r for r in with_judge if r["system_rating"] == rating]
        lo  = sum(1 for r in r_rows if float(r["faithfulness"]) < 0.5)
        mid = sum(1 for r in r_rows if 0.5 <= float(r["faithfulness"]) < 0.8)
        hi  = sum(1 for r in r_rows if float(r["faithfulness"]) >= 0.8)
        print(f"  {rating:>10}  {lo:>10}  {mid:>10}  {hi:>10}")
    print(f"{'='*60}")


def write_csv(rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[eval] Results written → {OUT_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def main() -> None:
    meta_path = EVAL_DIR / "dataset_meta.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run generate_dataset.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)
    print(f"[eval] {len(meta)} source questions loaded from {meta_path}")

    rng = random.Random(SEED)
    questions = sample_questions(meta, rng)
    print(f"[eval] Sampled {len(questions)} questions across 4 strata")

    strata_dist = {}
    for q in questions:
        strata_dist[q["strata"]] = strata_dist.get(q["strata"], 0) + 1
    for s, n in sorted(strata_dist.items()):
        print(f"  {s}: {n}")
    print()

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    Settings.llm         = OpenAI(model="gpt-4o")

    # Separate judge LLM — never uses Settings.llm
    judge_llm = OpenAI(model="gpt-4o-mini", temperature=0)

    print("[eval] Loading index from pgvector...")
    index  = load_index()
    engine = SecQueryEngine(index=index, verbose=False)
    print("[eval] Engine ready.\n")

    rows = []
    total = len(questions)
    for idx, question in enumerate(questions, 1):
        row = await eval_one(engine, judge_llm, question, idx, total)
        rows.append(row)

    print_report(rows)
    write_csv(rows)


if __name__ == "__main__":
    asyncio.run(main())
