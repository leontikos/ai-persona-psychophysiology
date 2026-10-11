#!/usr/bin/env python
"""
Cross-model coder-reliability check for the automated dialogue coding.

WHAT THIS DOES AND DOES NOT ESTABLISH
-------------------------------------
The behavioural results in the manuscript rest on a single LLM coder (gpt-4.1-2025-04-14)
with no validated ground truth. This script re-codes a sample of the same conversations
with two independent, architecturally different models and reports inter-coder agreement:

    coder A : gpt-4.1-2025-04-14   (the archived original codes, read from disk)
    coder B : claude-sonnet-5      (Anthropic Messages API, structured outputs)
    coder C : gpt-5.6-terra        (OpenAI Responses API, structured outputs)

High agreement bounds *coder noise* — it shows the taxonomy is applied consistently and
that the distributions are not an artefact of one model's idiosyncrasies. It does NOT
establish validity: three models sharing a systematic bias would agree with each other and
still be wrong. Only human double-coding establishes that the codes mean what the taxonomy
says they mean. Report this as a reliability check, never as validation, and pair it with a
human-coded subset if reviewers ask for validity.

The coding taxonomy is imported unchanged from the production pipeline so that all three
coders answer exactly the same question with exactly the same label set.

USAGE
-----
    # 1. See what it would cost before spending anything (default)
    python recode_reliability.py --n-conversations 12

    # 2. Actually run it
    python recode_reliability.py --n-conversations 12 --execute

    # 3. Only one coder
    python recode_reliability.py --n-conversations 12 --execute --coders claude


Environment: ANTHROPIC_API_KEY, OPENAI_API_KEY
Requires:    anthropic (verified on 0.112.0; messages.parse and count_tokens both present),
             openai (verified on 2.43.0; responses.parse present), pandas, scikit-learn
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
# Same schemas the production pipeline used -- do not redefine them here, or the
# comparison stops being apples-to-apples.
from analysis_pipeline_with_gpt import (  # noqa: E402
    ChunkAnalysisResult,
    EmotionalTone,
    PrimaryStrategy,
    TaskEngagement,
)

# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
BASE = Path(os.environ.get("STUDY_DATA_ROOT", "data"))
CONV_LOGS = BASE / "conversation_logs"
ARCHIVED = BASE / "CONVERSATION_ANALYSES" / "interactions_dataframe.csv"
EXTRACTED = BASE / "CONVERSATION_ANALYSES" / "extracted_participants.csv"
OUT = Path(__file__).resolve().parents[1] / "statistics" / "recode"

CLAUDE_MODEL = "claude-sonnet-5"
OPENAI_MODEL = "gpt-5.6-terra"          # NOT the "gpt-5.6" alias, which routes to Sol
CHUNK_SIZE = 16                          # identical to the production pipeline
MAX_OUTPUT_TOKENS = 16384

# Pricing per 1M tokens, verified 2026-08-23. Sonnet 5's $2/$10 was made permanent on
# 2026-08-11 -- the increase to $3/$15 that had been scheduled for 1 September was
# cancelled. GPT-5.6 Terra moved to $2/$12 on 2026-07-30, down from $2.50/$15.
PRICING = {
    CLAUDE_MODEL: {"in": 2.00, "out": 10.00},
    OPENAI_MODEL: {"in": 2.00, "out": 12.00},
}

CODING_INSTRUCTIONS = """You are coding a transcript of a conversation between a study \
participant and an AI chatbot that supervised them on a marketing task.

Code EVERY participant message on three dimensions:
  - primary_strategy: the single dominant behavioural strategy (choose exactly one label)
  - emotional_tone:   all emotional indicators present (may be more than one)
  - task_engagement:  HIGH, MEDIUM or LOW engagement with the substantive task

Also code every chatbot message on its leadership tactics, tone, response type and task
guidance style, using only the enumerated labels.

Apply the label definitions literally. Do not infer intent beyond what the text supports.
Where a message is ambiguous, choose the label with the most direct textual evidence and
say what that evidence was in the `evidence` field. Code the participant's behaviour, not
the chatbot's effect on you as a reader."""


# --------------------------------------------------------------------------- helpers
def build_id_resolver() -> dict[str, Path]:
    """Map archived conversation_id values onto files in conversation_logs/.

    The archived ids are heterogeneous: 39 of 57 are literal `conversation_*.json`
    filenames, the other 18 are `participant_id` + date/timestamp forms
    (e.g. "umbrella_feather-2025-06-09", "yellow_meadow_2025-06-24T09:50:13.246035").
    Resolving them by string manipulation alone silently drops the second group, so we
    go through extracted_participants.csv, which carries participant_id -> filepath.
    """
    resolver: dict[str, Path] = {}
    if EXTRACTED.exists():
        ext = pd.read_csv(EXTRACTED)
        by_participant: dict[str, Path] = {}
        for _, r in ext.iterrows():
            fp = Path(str(r["filepath"]))
            if fp.exists():
                # a participant with two files (restarted session) keeps the longer one
                prev = by_participant.get(str(r["participant_id"]))
                if prev is None or int(r.get("num_messages", 0) or 0) >= 0 and fp.stat().st_size > prev.stat().st_size:
                    by_participant[str(r["participant_id"])] = fp
        for pid, fp in by_participant.items():
            resolver[pid] = fp
    for fp in CONV_LOGS.glob("conversation_*.json"):
        resolver[fp.name] = fp
        resolver[fp.stem] = fp
    return resolver


def resolve_conversation(cid: str, resolver: dict[str, Path]) -> Path | None:
    cid = str(cid)
    if cid in resolver:
        return resolver[cid]
    # participant_id prefixed forms: strip a trailing -DATE or _DATE(THH:MM:SS...)
    for sep in ("-", "_"):
        head = cid.split(sep + "20")[0]
        if head in resolver:
            return resolver[head]
    return None


def load_conversation(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["messages"]


def chunk_messages(messages: list[dict], size: int = CHUNK_SIZE):
    for i in range(0, len(messages), size):
        yield i, messages[i : i + size]


def render_chunk(chunk: list[dict], start_index: int) -> str:
    lines = []
    for offset, m in enumerate(chunk):
        who = "PARTICIPANT" if m["role"] == "user" else "CHATBOT"
        lines.append(f"[message {start_index + offset}] {who}: {m['content']}")
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- coders
def code_chunk_claude(client, chunk_text: str, chunk_id: int, start: int, end: int):
    """Claude Sonnet 5 via the Messages API with structured outputs.

    NOTE: temperature/top_p/top_k are REMOVED on Sonnet 5 and return HTTP 400 -- the
    production pipeline's temperature=0.0 cannot be carried over. Determinism is instead
    approached by keeping the prompt fixed. Adaptive thinking is the only on-mode; the
    older {"type": "enabled", "budget_tokens": N} form is also rejected with a 400.
    """
    response = client.messages.parse(
        model=CLAUDE_MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=CODING_INSTRUCTIONS,
        thinking={"type": "adaptive"},
        messages=[
            {
                "role": "user",
                "content": (
                    f"This is chunk {chunk_id} covering global message indices "
                    f"{start}-{end}.\n\n{chunk_text}"
                ),
            }
        ],
        output_format=ChunkAnalysisResult,
    )
    return response.parsed_output, response.usage


def code_chunk_openai(client, chunk_text: str, chunk_id: int, start: int, end: int):
    """gpt-5.6-terra via the Responses API with structured outputs.

    OpenAI directs GPT-5.6 reasoning work to the Responses API rather than Chat
    Completions; reasoning depth is set with reasoning={"effort": ...} (none | low |
    medium | high | xhigh | max). temperature is not set -- reasoning models do not take
    a sampling temperature the way gpt-4.1 did. Requires the openai Python SDK v2.x for
    responses.parse / text_format.
    """
    response = client.responses.parse(
        model=OPENAI_MODEL,
        reasoning={"effort": "medium"},
        max_output_tokens=MAX_OUTPUT_TOKENS,
        input=[
            {"role": "system", "content": CODING_INSTRUCTIONS},
            {
                "role": "user",
                "content": (
                    f"This is chunk {chunk_id} covering global message indices "
                    f"{start}-{end}.\n\n{chunk_text}"
                ),
            },
        ],
        text_format=ChunkAnalysisResult,
    )
    return response.output_parsed, response.usage


# --------------------------------------------------------------------------- metrics
def krippendorff_alpha_nominal(units: dict[str, list[str]]) -> float | None:
    """Krippendorff's alpha for nominal data, computed from the coincidence matrix.

    `units` maps unit id -> labels assigned by the available coders. Units coded by fewer
    than two coders are skipped, so this tolerates the missing data that arises when one
    coder fails on a chunk. Implemented directly to avoid a dependency.

    alpha = 1 - Do/De, where Do is observed disagreement in the coincidence matrix and De
    is the disagreement expected from the marginals. alpha = 1 is perfect agreement, 0 is
    chance, negative is systematic disagreement.
    """
    o: dict[tuple[str, str], float] = defaultdict(float)
    for labels in units.values():
        m = len(labels)
        if m < 2:
            continue
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i == j:          # a coder is never paired with itself; compare by
                    continue        # position, not by value -- two coders agreeing must count
                o[(a, b)] += 1.0 / (m - 1)
    if not o:
        return None
    marginals: dict[str, float] = defaultdict(float)
    for (a, _b), v in o.items():
        marginals[a] += v
    n = sum(marginals.values())
    if n < 2:
        return None
    observed_disagreement = sum(v for (a, b), v in o.items() if a != b)
    expected_disagreement = sum(
        marginals[a] * marginals[b] for a in marginals for b in marginals if a != b
    ) / (n - 1)
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return 1.0 - observed_disagreement / expected_disagreement


def agreement_report(frames: dict[str, pd.DataFrame], dimension: str) -> dict:
    """Pairwise Cohen's kappa + overall Krippendorff's alpha for one coded dimension."""
    from itertools import combinations

    from sklearn.metrics import cohen_kappa_score

    keyed = {
        name: df.set_index(["conversation_id", "message_index"])[dimension]
        for name, df in frames.items()
        if dimension in df.columns
    }
    out = {"dimension": dimension, "pairwise": {}, "n_units": {}}
    for a, b in combinations(sorted(keyed), 2):
        common = keyed[a].index.intersection(keyed[b].index)
        if len(common) < 2:
            continue
        ka, kb = keyed[a].loc[common], keyed[b].loc[common]
        out["pairwise"][f"{a} vs {b}"] = {
            "kappa": float(cohen_kappa_score(ka.astype(str), kb.astype(str))),
            "raw_agreement": float((ka.astype(str) == kb.astype(str)).mean()),
            "n": int(len(common)),
        }
        out["n_units"][f"{a} vs {b}"] = int(len(common))

    units: dict[str, list[str]] = defaultdict(list)
    for name, series in keyed.items():
        for idx, val in series.items():
            units[f"{idx[0]}::{idx[1]}"].append(str(val))
    out["krippendorff_alpha"] = krippendorff_alpha_nominal(units)
    return out


# --------------------------------------------------------------------------- driver
def flatten(parsed: ChunkAnalysisResult, conversation_id: str) -> list[dict]:
    rows = []
    for a in parsed.participant_analyses:
        rows.append(
            {
                "conversation_id": conversation_id,
                "message_index": a.message_number,
                "primary_strategy": a.primary_strategy.value,
                "strategy_family": a.primary_strategy.value.split("_")[0],
                "task_engagement": a.task_engagement.value,
                "emotional_tone": "|".join(sorted(e.value for e in a.emotional_tone)),
                # alphabetically first tone; the coders do not rank tones, so this is not a
                # dominant tone. Kept for the file format of version 1.1.0; not used in the analysis.
                "dominant_emotion": (
                    sorted(e.value for e in a.emotional_tone)[0] if a.emotional_tone else "NONE"
                ),
            }
        )
    return rows


def estimate_cost(conversations: list[Path]) -> None:
    total_chars = 0
    total_chunks = 0
    for p in conversations:
        msgs = load_conversation(p)
        total_chars += sum(len(m["content"]) for m in msgs)
        total_chunks += (len(msgs) + CHUNK_SIZE - 1) // CHUNK_SIZE
    # ~4 chars/token for English; add the fixed instruction overhead per chunk
    in_tok = total_chars / 4 + total_chunks * len(CODING_INSTRUCTIONS) / 4
    out_tok = total_chunks * 2500          # observed structured-output size, order of magnitude
    print(f"\n  conversations      : {len(conversations)}")
    print(f"  chunks             : {total_chunks}")
    print(f"  est. input tokens  : {in_tok:,.0f}")
    print(f"  est. output tokens : {out_tok:,.0f}  (assumes ~2,500/chunk)")
    grand = 0.0
    for model, pr in PRICING.items():
        cost = in_tok / 1e6 * pr["in"] + out_tok / 1e6 * pr["out"]
        grand += cost
        print(f"  {model:20s}: ~${cost:,.2f}")
    print(f"  {'BOTH CODERS':20s}: ~${grand:,.2f}")
    print("\n  Rough estimate: 4 chars/token and a flat 2,500 output tokens per chunk, with no")
    print("  reasoning tokens, which bill as output. The batch APIs (recode_batch.py) cost half.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conversations", type=int, default=12,
                    help="how many conversations to re-code (stratified by condition)")
    ap.add_argument("--coders", default="claude,openai",
                    help="comma-separated subset of: claude, openai")
    ap.add_argument("--execute", action="store_true",
                    help="actually call the APIs; without this only the cost estimate runs")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    random.seed(args.seed)
    archived = pd.read_csv(ARCHIVED)
    conv_ids = sorted(archived["conversation_id"].dropna().unique())

    by_condition: dict[str, list[str]] = defaultdict(list)
    for cid in conv_ids:
        cond = archived.loc[archived.conversation_id == cid, "condition"].iloc[0]
        by_condition[str(cond)].append(cid)
    picked: list[str] = []
    per_cond = max(1, args.n_conversations // max(1, len(by_condition)))
    for cond, ids in sorted(by_condition.items()):
        picked += random.sample(ids, min(per_cond, len(ids)))
    print(f"sampled {len(picked)} conversations across {len(by_condition)} conditions")

    resolver = build_id_resolver()
    resolved = [(cid, resolve_conversation(cid, resolver)) for cid in picked]
    paths = [p for _, p in resolved if p is not None]
    unresolved = [cid for cid, p in resolved if p is None]
    if unresolved:
        print(f"  warning: could not resolve {len(unresolved)} id(s) to a log file: "
              f"{unresolved[:4]}")
    # keep the archived id alongside the file so the coders and the archive align
    id_for_path = {p: cid for cid, p in resolved if p is not None}

    if not args.execute:
        print("\nDRY RUN -- no API calls made. Re-run with --execute to code for real.")
        estimate_cost(paths)
        return

    estimate_cost(paths)
    coders = [c.strip() for c in args.coders.split(",") if c.strip()]
    clients: dict[str, object] = {}
    if "claude" in coders:
        import anthropic
        clients["claude-sonnet-5"] = anthropic.Anthropic()
    if "openai" in coders:
        from openai import OpenAI
        clients["gpt-5.6-terra"] = OpenAI()

    results: dict[str, list[dict]] = {name: [] for name in clients}
    usage_log: list[dict] = []

    for path in paths:
        msgs = load_conversation(path)
        for chunk_id, (start, chunk) in enumerate(chunk_messages(msgs), start=1):
            text = render_chunk(chunk, start)
            end = start + len(chunk) - 1
            for name, client in clients.items():
                fn = code_chunk_claude if name == CLAUDE_MODEL else code_chunk_openai
                try:
                    parsed, usage = fn(client, text, chunk_id, start, end)
                    results[name].extend(flatten(parsed, id_for_path[path]))
                    usage_log.append({"model": name, "conversation": id_for_path[path],
                                      "chunk": chunk_id, "usage": str(usage)})
                except Exception as exc:  # keep going; a failed chunk becomes missing data
                    print(f"  [{name}] {id_for_path[path]} chunk {chunk_id} FAILED: "
                          f"{type(exc).__name__}: {exc}")
        print(f"  coded {id_for_path[path]} ({len(msgs)} messages)")

    OUT.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for name, rows in results.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"recode_{name.replace('.', '_')}.csv", index=False)
        frames[name] = df

    # archived gpt-4.1 codes, aligned onto the same keys
    arch = archived[archived.conversation_id.isin(list(id_for_path.values()))].copy()
    if "turn_number" in arch.columns and "participant_strategy" in arch.columns:
        arch = arch.rename(columns={"turn_number": "message_index",
                                    "participant_strategy": "primary_strategy",
                                    "participant_engagement": "task_engagement"})
        arch["strategy_family"] = arch["primary_strategy"].astype(str).str.split("_").str[0]
        frames["gpt-4.1 (archived)"] = arch

    report = {
        "n_conversations": len(paths),
        "coders": list(frames),
        "dimensions": [
            agreement_report(frames, dim)
            for dim in ("primary_strategy", "strategy_family", "task_engagement",
                        "dominant_emotion")
        ],
    }
    (OUT / "recode_reliability_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(usage_log).to_csv(OUT / "recode_usage_log.csv", index=False)

    print("\n" + "=" * 74)
    print("INTER-CODER AGREEMENT")
    print("=" * 74)
    for d in report["dimensions"]:
        alpha = d["krippendorff_alpha"]
        print(f"\n{d['dimension']}   Krippendorff alpha = "
              f"{alpha:.3f}" if alpha is not None else f"\n{d['dimension']}   alpha = n/a")
        for pair, v in d["pairwise"].items():
            print(f"    {pair:34s} kappa={v['kappa']:+.3f}  "
                  f"raw={v['raw_agreement']:.1%}  n={v['n']}")
    print(f"\nwrote {OUT}/recode_reliability_report.json")
    print("\nInterpretation guide: kappa >= .80 excellent, .60-.79 substantial, "
          ".40-.59 moderate, < .40 poor.\nRemember this bounds coder NOISE, not validity "
          "-- see the module docstring.")


if __name__ == "__main__":
    main()
