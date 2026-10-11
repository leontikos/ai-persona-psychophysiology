#!/usr/bin/env python
"""
Run the cross-model coder-reliability re-code over the whole corpus, through both
providers' batch APIs.

Same question as recode_reliability.py, same taxonomy, same chunking -- but asynchronous.
Both providers price batch work at 50% of the synchronous rate and return within 24 h
(usually far less), and nothing about re-coding an archived corpus needs to be synchronous.

    archived : gpt-4.1-2025-04-14   the original codes, read from disk, not re-run
    re-coders: claude-opus-5, claude-sonnet-5   Message Batches API  (POST /v1/messages/batches)
               gpt-5.6-sol, gpt-5.6-terra       Batch API + Responses (POST /v1/batches, /v1/responses)

WHAT THIS ESTABLISHES: agreement among independent coders bounds *coder noise*. It does not
establish validity -- models sharing a systematic bias would agree with each other and still
be wrong. Only human double-coding does that. Report accordingly.

STAGES (each is separately re-runnable; state lives in recode_batch_state.json)

    python recode_batch.py submit     build the requests, send the batches, save their ids
    python recode_batch.py status     poll the providers
    python recode_batch.py collect    download results, parse, write per-coder CSVs
    python recode_batch.py resync --model gpt-5.6-terra
                                      re-run any chunks the batch did not return, this time
                                      synchronously, and merge them into that coder's CSV
    python recode_batch.py report     Cohen's kappa + Krippendorff's alpha across coders

SCHEMA HANDLING. Neither batch API runs the SDK's client-side .parse() helper, so the
response schema has to go on the wire explicitly. Rather than hand-writing it twice, each
provider's own converter is reused -- anthropic.transform_schema and openai's
type_to_text_format_param -- applied to the *unmodified* production Pydantic model. The
batch requests therefore carry exactly the schema the synchronous helpers would have sent,
and all three coders answer the same question with the same label set.

Environment: ANTHROPIC_API_KEY and OPENAI_API_KEY, read from the repository .env.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_pipeline_with_gpt import ChunkAnalysisResult  # noqa: E402
from recode_reliability import (  # noqa: E402
    ARCHIVED,
    BASE,
    CHUNK_SIZE,
    CLAUDE_MODEL,
    CODING_INSTRUCTIONS,
    OPENAI_MODEL,
    OUT,
    agreement_report,
    build_id_resolver,
    chunk_messages,
    flatten,
    load_conversation,
    render_chunk,
    resolve_conversation,
)

REPO = Path(__file__).resolve().parents[2]
STATE = OUT / "recode_batch_state.json"
RAW = OUT / "recode_batch_raw"

# Synchronous list prices per 1M tokens, verified 2026-08-23. Batch is exactly half of
# these on both platforms.
#   claude-sonnet-5  $2 / $10   locked permanently 2026-08-11
#   claude-opus-5    $5 / $25   Opus 4.8 pricing, carried over
#   gpt-5.6-terra    $2 / $12   since 2026-07-30
#   gpt-5.6-sol      $4 / $20   cut from $5/$25 on 2026-08-22 (short context; >272K input
#                               tokens bills at $10/$45, which no chunk here approaches)
SYNC_PRICING = {
    "claude-sonnet-5": {"in": 2.00, "out": 10.00},
    "claude-opus-5":   {"in": 5.00, "out": 25.00},
    "gpt-5.6-terra":   {"in": 2.00, "out": 12.00},
    "gpt-5.6-sol":     {"in": 4.00, "out": 20.00},
}
BATCH_PRICING = {m: {k: v / 2 for k, v in pr.items()} for m, pr in SYNC_PRICING.items()}

# The four re-coders. Two providers x two capability tiers, so the design separates
# "does the coding depend on the model?" from "does it depend on the provider?".
CODERS = {
    "claude-sonnet-5": "anthropic",
    "claude-opus-5":   "anthropic",
    "gpt-5.6-terra":   "openai",
    "gpt-5.6-sol":     "openai",
}


# The archived coder is OpenAI too, which matters for the within- vs cross-provider split.
CODER_PROVIDER = dict(CODERS, **{"gpt-4.1 (archived)": "openai"})


def provider(model: str) -> str:
    if model in CODERS:
        return CODERS[model]
    return "anthropic" if model.startswith("claude") else "openai"
# Adaptive thinking and effort=medium both emit reasoning tokens, which bill as output and
# count against max_tokens. Headroom costs nothing -- only generated tokens are billed --
# and truncating a chunk mid-JSON would cost a whole request.
MAX_TOKENS = 32000


def load_env() -> None:
    """Read KEY="value" pairs from the repository .env without clobbering a real env var."""
    for env_path in (REPO / ".env",):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def build_requests(limit: int | None = None) -> list[dict]:
    """One request per chunk, in a provider-neutral form. custom_id is deliberately opaque
    (`c003_k002`): Anthropic constrains it to ^[a-zA-Z0-9_-]{1,64}$, and the archived
    conversation ids contain characters that would not survive that."""
    resolver = build_id_resolver()
    archived = pd.read_csv(ARCHIVED)
    conv_ids = sorted(archived["conversation_id"].dropna().unique())

    seen: set[Path] = set()
    convs: list[tuple[str, Path]] = []
    for cid in conv_ids:
        p = resolve_conversation(str(cid), resolver)
        if p is not None and p not in seen:
            seen.add(p)
            convs.append((str(cid), p))
    convs.sort(key=lambda t: t[1].name)
    if limit:
        convs = convs[:limit]

    reqs: list[dict] = []
    for ci, (cid, path) in enumerate(convs):
        msgs = load_conversation(path)
        for k, (start, chunk) in enumerate(chunk_messages(msgs, CHUNK_SIZE), start=1):
            end = start + len(chunk) - 1
            reqs.append({
                "custom_id": f"c{ci:03d}_k{k:03d}",
                "conversation_id": cid,
                "file": path.name,
                "chunk_id": k,
                "start": start,
                "end": end,
                "n_messages": len(chunk),
                "user": (f"This is chunk {k} covering global message indices "
                         f"{start}-{end}.\n\n{render_chunk(chunk, start)}"),
            })
    return reqs


# --------------------------------------------------------------------------- submit
def submit_claude(reqs: list[dict], model: str) -> str:
    import anthropic
    from anthropic import transform_schema
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    from pydantic import TypeAdapter

    schema = transform_schema(TypeAdapter(ChunkAnalysisResult).json_schema())
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=[
        Request(
            custom_id=r["custom_id"],
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=MAX_TOKENS,
                system=CODING_INSTRUCTIONS,
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": schema}},
                messages=[{"role": "user", "content": r["user"]}],
            ),
        )
        for r in reqs
    ])
    return batch.id


def submit_openai(reqs: list[dict], model: str) -> str:
    from openai import OpenAI
    from openai.lib._parsing._responses import type_to_text_format_param

    text_format = type_to_text_format_param(ChunkAnalysisResult)
    client = OpenAI()
    lines = [
        json.dumps({
            "custom_id": r["custom_id"],
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": model,
                "reasoning": {"effort": "medium"},
                "max_output_tokens": MAX_TOKENS,
                "input": [
                    {"role": "system", "content": CODING_INSTRUCTIONS},
                    {"role": "user", "content": r["user"]},
                ],
                "text": {"format": text_format},
            },
        })
        for r in reqs
    ]
    RAW.mkdir(parents=True, exist_ok=True)
    jsonl = RAW / f"requests_{model.replace('.', '_')}.jsonl"
    jsonl.write_text("\n".join(lines) + "\n", encoding="utf-8")
    up = client.files.create(file=jsonl.open("rb"), purpose="batch")
    batch = client.batches.create(input_file_id=up.id, endpoint="/v1/responses",
                                  completion_window="24h")
    return batch.id


def cmd_submit(args) -> None:
    reqs = build_requests(args.limit)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in SYNC_PRICING]
    if unknown:
        raise SystemExit(f"unpriced model(s): {unknown}. Add them to SYNC_PRICING first -- "
                         f"running a coder whose price you have not checked is how a "
                         f"reliability check turns into a surprise invoice.")
    print(f"built {len(reqs)} requests over "
          f"{len({r['file'] for r in reqs})} conversations, "
          f"{sum(r['n_messages'] for r in reqs)} messages")
    if args.dry_run:
        print("dry run -- nothing submitted")
        return

    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    state["submitted_at"] = datetime.now(timezone.utc).isoformat()
    state["requests"] = [{k: v for k, v in r.items() if k != "user"} for r in reqs]
    state.setdefault("batches", {})
    for model in models:
        if model in state["batches"] and not args.resubmit:
            print(f"  {model:16s} already submitted ({state['batches'][model]['id']}) -- "
                  f"skipping; pass --resubmit to run it again")
            continue
        fn = submit_claude if provider(model) == "anthropic" else submit_openai
        bid = fn(reqs, model)
        state["batches"][model] = {"id": bid, "submitted_at":
                                   datetime.now(timezone.utc).isoformat()}
        print(f"  {model:16s} batch {bid}")
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))
    print(f"\nstate -> {STATE}\npoll with:  python recode_batch.py status")


# --------------------------------------------------------------------------- status
def status_claude(bid: str) -> dict:
    import anthropic
    b = anthropic.Anthropic().messages.batches.retrieve(bid)
    rc = b.request_counts
    return {"status": b.processing_status,
            "counts": {"processing": rc.processing, "succeeded": rc.succeeded,
                       "errored": rc.errored, "canceled": rc.canceled,
                       "expired": rc.expired},
            "done": b.processing_status == "ended"}


def status_openai(bid: str) -> dict:
    from openai import OpenAI
    b = OpenAI().batches.retrieve(bid)
    rc = b.request_counts
    return {"status": b.status,
            "counts": {"total": rc.total, "completed": rc.completed, "failed": rc.failed},
            "done": b.status in ("completed", "failed", "expired", "cancelled"),
            "output_file_id": b.output_file_id, "error_file_id": b.error_file_id}


def cmd_status(args) -> None:
    state = json.loads(STATE.read_text())
    alldone = True
    for model, meta in state["batches"].items():
        st = (status_claude if provider(model) == "anthropic" else status_openai)(meta["id"])
        alldone &= st["done"]
        print(f"{model:16s} {meta['id']}  {st['status']:12s} {st['counts']}")
    print("\nall batches ended" if alldone else "\nstill processing")
    sys.exit(0 if alldone else 3)



def fetch_claude_results(bid: str, path: Path, attempts: int = 5) -> int:
    """Download a Message Batch's results to disk before parsing them.

    The SDK streams results as JSONL over a single long-lived connection, so one transient
    read error loses the whole pass. Results stay retrievable for 29 days, so the fix is to
    persist first and parse from the file: a dropped connection then costs a retry, not the
    run. If the file already holds every expected line, the download is skipped entirely.
    """
    import anthropic
    expected = anthropic.Anthropic().messages.batches.retrieve(bid).request_counts
    want = expected.succeeded + expected.errored + expected.canceled + expected.expired
    if path.exists():
        have = sum(1 for line in path.open(encoding="utf-8") if line.strip())
        if have >= want:
            print(f"                 reusing {path.name} ({have} results already on disk)")
            return have
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            client = anthropic.Anthropic(max_retries=5)
            n = 0
            with path.open("w", encoding="utf-8") as fh:
                for res in client.messages.batches.results(bid):
                    fh.write(res.model_dump_json() + "\n")
                    n += 1
            if n < want:
                raise RuntimeError(f"stream ended early: {n}/{want} results")
            return n
        except Exception as exc:
            last = exc
            print(f"                 results download attempt {attempt}/{attempts} failed "
                  f"({type(exc).__name__}); retrying")
            time.sleep(3 * attempt)
    raise RuntimeError(f"could not download batch results after {attempts} attempts") from last


# --------------------------------------------------------------------------- collect
def extract_claude(msg) -> tuple[str | None, dict]:
    """With output_config.format the JSON arrives as a text block; thinking blocks precede
    it, so take the last text block rather than content[0]."""
    texts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    u = msg.usage
    usage = {"in": u.input_tokens, "out": u.output_tokens,
             "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0}
    return (texts[-1] if texts else None), usage


def extract_openai(body: dict) -> tuple[str | None, dict]:
    text = None
    for item in body.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    text = c.get("text")
    u = body.get("usage", {}) or {}
    usage = {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0),
             "reasoning": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}
    return text, usage


def cmd_collect(args) -> None:
    state = json.loads(STATE.read_text())
    index = {r["custom_id"]: r for r in state["requests"]}
    RAW.mkdir(parents=True, exist_ok=True)
    summary = {}

    for model, meta in state["batches"].items():
        st = (status_claude if provider(model) == "anthropic" else status_openai)(meta["id"])
        if args.only and model not in args.only:
            print(f"{model:16s} skipped (--only)")
            continue
        if not st["done"]:
            # Overwriting a good CSV with an empty one because the batch is still running
            # would silently destroy results. Skip and leave whatever is on disk alone.
            print(f"{model:16s} still {st['status']} -- skipped, nothing overwritten")
            continue
        rows, usage_rows, failures = [], [], []
        if provider(model) == "anthropic":
            from anthropic.types.messages import MessageBatchIndividualResponse
            jsonl = RAW / f"results_{model.replace('.', '_')}.jsonl"
            n = fetch_claude_results(meta["id"], jsonl)
            print(f"                 {n} results on disk -> {jsonl.name}")
            for line in jsonl.open(encoding="utf-8"):
                if not line.strip():
                    continue
                res = MessageBatchIndividualResponse.model_validate_json(line)
                cid = res.custom_id
                if res.result.type != "succeeded":
                    failures.append({"custom_id": cid, "type": res.result.type,
                                     "detail": str(getattr(res.result, "error", ""))[:300]})
                    continue
                text, usage = extract_claude(res.result.message)
                rows.append((cid, text))
                usage_rows.append({"custom_id": cid, **usage})
        else:
            from openai import OpenAI
            client = OpenAI()
            b = client.batches.retrieve(meta["id"])
            for fid, sink in ((b.output_file_id, "out"), (b.error_file_id, "err")):
                if not fid:
                    continue
                blob = client.files.content(fid).text
                (RAW / f"{model.replace('.', '_')}_{sink}.jsonl").write_text(blob, encoding="utf-8")
                if sink == "err":
                    continue
                for line in blob.splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    cid = rec["custom_id"]
                    resp = rec.get("response") or {}
                    if rec.get("error") or resp.get("status_code") != 200:
                        failures.append({"custom_id": cid, "type": "errored",
                                         "detail": str(rec.get("error"))[:300]})
                        continue
                    text, usage = extract_openai(resp.get("body", {}))
                    rows.append((cid, text))
                    usage_rows.append({"custom_id": cid, **usage})

        coded, parse_fail = [], []
        for cid, text in rows:
            if not text:
                parse_fail.append({"custom_id": cid, "reason": "no text block"})
                continue
            try:
                parsed = ChunkAnalysisResult.model_validate_json(text)
            except Exception as exc:
                parse_fail.append({"custom_id": cid, "reason": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            coded.extend(flatten(parsed, index[cid]["conversation_id"]))

        tag = model.replace(".", "_")
        got = {cid for cid, _ in rows} - {f["custom_id"] for f in parse_fail}
        meta["returned_ids"] = sorted(got)
        meta["missing_ids"] = sorted({r["custom_id"] for r in state["requests"]} - got)
        meta["collected_status"] = st["status"]
        pd.DataFrame(coded).to_csv(OUT / f"recode_{tag}.csv", index=False)
        pd.DataFrame(usage_rows).to_csv(OUT / f"recode_usage_{tag}.csv", index=False)
        if failures or parse_fail:
            (RAW / f"failures_{tag}.json").write_text(
                json.dumps({"api": failures, "parse": parse_fail}, indent=2))

        u = pd.DataFrame(usage_rows)
        pr = BATCH_PRICING[model]
        cost = (u["in"].sum() / 1e6 * pr["in"] + u["out"].sum() / 1e6 * pr["out"]) if len(u) else 0
        summary[model] = {"requests": len(state["requests"]), "returned": len(rows),
                          "api_failures": len(failures), "parse_failures": len(parse_fail),
                          "messages_coded": len(coded),
                          "input_tokens": int(u["in"].sum()) if len(u) else 0,
                          "output_tokens": int(u["out"].sum()) if len(u) else 0,
                          "reasoning_tokens": int(u["reasoning"].sum()) if "reasoning" in u else None,
                          "batch_cost_usd": round(cost, 2)}
        print(f"{model:16s} coded {len(coded):,} messages from {len(got)}/"
              f"{len(state['requests'])} chunks; ${cost:,.2f}")
        if meta["missing_ids"]:
            print(f"                 {len(meta['missing_ids'])} chunks missing: "
                  f"{', '.join(meta['missing_ids'][:8])}")
            print(f"                 fill them with: python recode_batch.py resync "
                  f"--model {model}")
        if failures:
            print(f"                 {len(failures)} API failures -> {RAW}/failures_{tag}.json")
        if parse_fail:
            print(f"                 {len(parse_fail)} unparseable -> {RAW}/failures_{tag}.json")

    state["collected_at"] = datetime.now(timezone.utc).isoformat()
    # merge, never replace: a --only collect must not erase the record of coders it skipped
    state["summary"] = {**state.get("summary", {}), **summary}
    STATE.write_text(json.dumps(state, indent=2))
    total = sum(v.get("batch_cost_usd", 0) or 0 for v in state["summary"].values())
    topup = sum(v.get("sync_topup_cost_usd", 0) or 0 for v in state["summary"].values())
    print(f"\ncumulative spend across all coders: ${total + topup:,.2f} "
          f"(batch ${total:,.2f} + synchronous top-ups ${topup:,.2f})")


# --------------------------------------------------------------------------- report
ANALYSES = BASE / "CONVERSATION_ANALYSES" / "successful_analyses"
LOGS = BASE / "conversation_logs"


def _norm(t) -> str:
    return " ".join(str(t).split()).strip().lower()


def user_positions(fname: str) -> list[int]:
    """0-based positions of the participant's messages in the source transcript."""
    msgs = json.loads((LOGS / fname).read_text(encoding="utf-8"))["messages"]
    return [i for i, m in enumerate(msgs) if m.get("role") == "user"]


def _source_text(fname: str) -> dict[int, str]:
    msgs = json.loads((LOGS / fname).read_text(encoding="utf-8"))["messages"]
    return {i: _norm(m.get("content", "")) for i, m in enumerate(msgs)}


def anchor_archived(fname: str) -> list[dict]:
    """Anchor the archived gpt-4.1 codes to positions in the source transcript.

    gpt-4.1's `message_number` cannot be used as a join key, because it is not one scheme:
    across the 58 analyses, 39 conversations number participant messages by their GLOBAL
    position in the transcript (2, 4, 6, ...) and 19 number them ORDINALLY (1, 2, 3, ...).
    Joining a re-code's global index onto that column silently pairs different messages in
    a third of the corpus -- and the resulting kappa measures alignment error, not coder
    disagreement.

    The archived records carry the message TEXT, so anchor on that instead: match each coded
    record to the participant message it quotes. Repeated texts ("ok", "yes") are consumed
    in order so duplicates cannot collapse onto one position. Records whose text does not
    match (9 of 1,332 corpus-wide, where the coder lightly paraphrased) fall back to ordinal
    position, which is correct whenever the coder emitted one record per participant message
    in order -- true for every conversation here.
    """
    apath = ANALYSES / fname.replace(".json", "_analyzed.json")
    if not apath.exists():
        return []
    pa = json.loads(apath.read_text(encoding="utf-8")).get("participant_message_analyses", [])
    upos = user_positions(fname)
    texts = _source_text(fname)
    by_text: dict[str, list[int]] = defaultdict(list)
    for i in upos:
        by_text[texts[i]].append(i)

    used: set[int] = set()
    rows = []
    for k, a in enumerate(pa):
        cand = [c for c in by_text.get(_norm(a.get("participant_message", "")), [])
                if c not in used]
        pos = cand[0] if cand else (upos[k] if k < len(upos) else None)
        if pos is None:
            continue
        used.add(pos)
        tones = sorted(str(t) for t in (a.get("emotional_tone") or []))
        strategy = str(a.get("primary_strategy"))
        rows.append({"file": fname, "source_index": pos,
                     "primary_strategy": strategy,
                     "strategy_family": strategy.split("_")[0],
                     "task_engagement": str(a.get("task_engagement")),
                     "emotional_tone": "|".join(tones),
                     "frustrated": "FRUSTRATED" if "FRUSTRATED" in tones else "no",
                     "anchor": "text" if cand else "ordinal"})
    return rows


def anchor_recode(df: pd.DataFrame) -> pd.DataFrame:
    """Anchor a re-coded frame to the same source positions.

    The batch prompt labels every message with its global 0-based position
    ("[message 37] PARTICIPANT: ..."), so a well-behaved coder's message_index already IS
    the source position -- this verifies that rather than assuming it. Per conversation, if
    the indices do not land on participant messages, the conversation falls back to ordinal
    anchoring and is counted in the report so the fallback is never silent.
    """
    out, fallbacks = [], []
    for fname, g in df.groupby("file"):
        upos = set(user_positions(fname))
        g = g.sort_values("message_index")
        hit = g["message_index"].isin(upos).mean()
        if hit >= 0.9:
            g = g.assign(source_index=g["message_index"], anchor="index")
        else:
            ordered = sorted(upos)
            n = min(len(g), len(ordered))
            g = g.iloc[:n].assign(source_index=ordered[:n], anchor="ordinal")
            fallbacks.append((fname, round(float(hit), 3)))
        out.append(g)
    res = pd.concat(out, ignore_index=True) if out else df.assign(source_index=[])
    res.attrs["ordinal_fallbacks"] = fallbacks
    return res




def condition_replication(frames: dict, file_condition: dict) -> list[dict]:
    """Does the Dark Triad vs Servant Leader contrast survive a change of coder?

    Inter-coder kappa bounds how consistently the taxonomy is applied. It does not answer
    the question the manuscript actually rests on, which is whether the CONDITION
    DIFFERENCE is a property of the data or of one coder. A coder can disagree about where
    the HELP/NEGOTIATION boundary sits, shifting every marginal, and still recover the same
    between-condition contrast -- because a systematic labelling shift applies to both arms
    and largely cancels in the difference. This computes, per coder, the participant-level
    proportion of each strategy family and engagement level by condition, and Cohen's d for
    the contrast, so the two coders' effect sizes can be read side by side.
    """
    out = []
    for name, df in frames.items():
        d = df.copy()
        d["condition"] = d["file"].map(file_condition)
        d = d.dropna(subset=["condition"])
        for dim, levels in (("strategy_family", None), ("task_engagement", None)):
            if dim not in d.columns:
                continue
            for level in sorted(d[dim].dropna().astype(str).unique()) if levels is None else levels:
                # participant-level proportions: one value per conversation, then compare
                per_conv = (d.assign(hit=(d[dim].astype(str) == level).astype(float))
                              .groupby(["file", "condition"])["hit"].mean().reset_index())
                dt = per_conv.loc[per_conv.condition.astype(str).str.upper().str.startswith(("DARK", "DT")), "hit"]
                sl = per_conv.loc[~per_conv.condition.astype(str).str.upper().str.startswith(("DARK", "DT")), "hit"]
                if len(dt) < 3 or len(sl) < 3:
                    continue
                import numpy as np
                sp = np.sqrt(((len(dt) - 1) * dt.var(ddof=1) + (len(sl) - 1) * sl.var(ddof=1))
                             / (len(dt) + len(sl) - 2))
                d_val = float((dt.mean() - sl.mean()) / sp) if sp > 0 else 0.0
                out.append({"coder": name, "dimension": dim, "level": level,
                            "M_DT": round(float(dt.mean()) * 100, 1),
                            "M_SL": round(float(sl.mean()) * 100, 1),
                            "cohens_d_DT_minus_SL": round(d_val, 3),
                            "n_DT": int(len(dt)), "n_SL": int(len(sl))})
    return out


def derived_metrics(frames: dict, file_condition: dict) -> pd.DataFrame:
    """Re-derive the manuscript's headline behavioural metrics under each coder.

    Strategy diversity (Shannon entropy, bits, over the 17 strategy subtypes), turning points
    (switches in strategy family between consecutive participant messages) and the share of
    messages coded as resistance or tagged FRUSTRATED are not raw codes -- they are computed
    FROM the codes. If the codes move, they move.
    H3a and H3b in the manuscript rest on entropy and turning points respectively, so a
    reliability check that stops at kappa never touches them. This recomputes each metric
    per conversation, per coder, exactly as the pipeline defines it, and reports the
    Dark Triad - Servant Leader contrast so the three coders can be compared directly.
    """
    import numpy as np
    rows = []
    for name, df in frames.items():
        d = df.copy()
        d["condition"] = d["file"].map(file_condition)
        d = d.dropna(subset=["condition"]).sort_values(["file", "source_index"])
        per = []
        for f, g in d.groupby("file"):
            fam = g["strategy_family"].astype(str).tolist()
            counts = g["primary_strategy"].astype(str).value_counts(normalize=True)
            entropy = float(-(counts * np.log2(counts)).sum())
            switches = sum(1 for a, b in zip(fam, fam[1:]) if a != b)
            emo = g["emotional_tone"].fillna("").astype(str)
            per.append({"file": f, "condition": g["condition"].iloc[0],
                        "entropy": entropy, "turning_points": switches,
                        "resistance_pct": 100 * float(np.mean([x == "RESISTANCE" for x in fam])),
                        "frustrated_pct": 100 * float(emo.str.contains("FRUSTRATED").mean())})
        per = pd.DataFrame(per)
        is_dt = per.condition.astype(str).str.upper().str.startswith(("DARK", "DT"))
        for metric in ("entropy", "turning_points", "resistance_pct", "frustrated_pct"):
            dt, sl = per.loc[is_dt, metric].dropna(), per.loc[~is_dt, metric].dropna()
            if len(dt) < 3 or len(sl) < 3:
                continue
            sp = np.sqrt(((len(dt) - 1) * dt.var(ddof=1) + (len(sl) - 1) * sl.var(ddof=1))
                         / (len(dt) + len(sl) - 2))
            rows.append({"coder": name, "metric": metric,
                         "M_DT": round(float(dt.mean()), 2), "M_SL": round(float(sl.mean()), 2),
                         "cohens_d": round(float((dt.mean() - sl.mean()) / sp) if sp else 0.0, 3)})
    return pd.DataFrame(rows)


def provider_split(dimensions: list[dict]) -> pd.DataFrame:
    """Is the disagreement between MODELS, or between PROVIDERS?

    Pooled kappa cannot tell those apart, because every pair in the pool mixes both. Split
    the pairwise kappas by whether the two coders come from the same lab: if agreement is
    systematically higher within a provider than across, the taxonomy is being read through
    a house style, and adding a second model from the same lab buys less independence than
    it appears to. Note gpt-4.1 counts as OpenAI, so the within-provider OpenAI cell also
    carries the generation gap between gpt-4.1 and GPT-5.6.
    """
    rows = []
    for d in dimensions:
        for pair, v in d["pairwise"].items():
            a, b = pair.split(" vs ")
            pa, pb = CODER_PROVIDER.get(a), CODER_PROVIDER.get(b)
            if pa is None or pb is None:
                continue
            rows.append({"dimension": d["dimension"], "pair": pair,
                         "kind": f"within {pa}" if pa == pb else "cross-provider",
                         "kappa": v["kappa"]})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.pivot_table(index="dimension", columns="kind", values="kappa",
                          aggfunc="mean").round(3)


TONES = ("ANXIOUS", "CONFIDENT", "DEFENSIVE", "FRUSTRATED", "NEUTRAL", "RESIGNED")


def pair_class(a: str, b: str) -> str | None:
    """Relationship between two coders: same provider and model generation (Opus 5 / Sonnet 5,
    GPT-5.6 Sol / Terra), same provider across a generation (GPT-4.1 / GPT-5.6), or different
    providers."""
    pa, pb = CODER_PROVIDER.get(a), CODER_PROVIDER.get(b)
    if pa is None or pb is None:
        return None
    if pa != pb:
        return "cross-provider"
    if ("gpt-4.1" in a) != ("gpt-4.1" in b):
        return "same provider, across generation"
    return "same provider, same generation"


def agreement_by_class(dimensions: list[dict]) -> pd.DataFrame:
    """Mean pairwise kappa per dimension within each class of coder pair (Supplementary Table 12)."""
    rows = [{"dimension": d["dimension"], "class": pair_class(*pair.split(" vs ")), "kappa": v["kappa"]}
            for d in dimensions for pair, v in d["pairwise"].items()]
    df = pd.DataFrame([r for r in rows if r["class"]])
    return df.pivot_table(index="dimension", columns="class", values="kappa", aggfunc="mean")


def cmd_report(args) -> None:
    state = json.loads(STATE.read_text())
    # key every coder by source FILE, not by the archived conversation_id: those ids are
    # heterogeneous (39 filenames, 18 participant/date forms) and are not a stable key.
    file_for_conv = {r["conversation_id"]: r["file"] for r in state["requests"]}
    files = set(file_for_conv.values())

    frames: dict[str, pd.DataFrame] = {}
    for model in state["batches"]:
        f = OUT / f"recode_{model.replace('.', '_')}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if not len(df):
            continue
        df["file"] = df["conversation_id"].map(file_for_conv)
        df = anchor_recode(df.dropna(subset=["file"]))
        fb = df.attrs.get("ordinal_fallbacks") or []
        if fb:
            print(f"  note: {model} fell back to ordinal anchoring in "
                  f"{len(fb)} conversation(s): {fb[:4]}")
        # agreement_report() keys on (conversation_id, message_index); feed it the anchored
        # source position so all coders key the same physical message.
        df["conversation_id"], df["message_index"] = df["file"], df["source_index"]
        df["frustrated"] = df["emotional_tone"].fillna("").astype(str).str.contains("FRUSTRATED").map(
            {True: "FRUSTRATED", False: "no"})
        frames[model] = df

    arch = pd.DataFrame([r for f in sorted(files) for r in anchor_archived(f)])
    if len(arch):
        n_ord = int((arch["anchor"] == "ordinal").sum())
        print(f"  note: gpt-4.1 anchored by quoted text; {n_ord} record(s) fell back "
              f"to ordinal position")
        arch["conversation_id"], arch["message_index"] = arch["file"], arch["source_index"]
        frames["gpt-4.1 (archived)"] = arch

    # Emotional tone is multi-label: a message can carry several tones, which the coders do not
    # rank. Agreement is therefore computed on whether a message expresses frustration (the tone
    # the hypotheses concern) and, for every tone, on whether the message carries it. The
    # "first tone" column reproduces the summary used by code version 1.1.0 (the alphabetically
    # first tone of each message); it is kept for comparison only and has no substantive meaning.
    for df in frames.values():
        tone_sets = df["emotional_tone"].fillna("").astype(str).str.split("|")
        for t in TONES:
            df[f"tone_{t.lower()}"] = tone_sets.apply(lambda s, t=t: t if t in s else "no")
        df["first_tone_alphabetical"] = tone_sets.apply(
            lambda s: sorted(x for x in s if x)[0] if any(s) else "NONE")
    dims = ("primary_strategy", "strategy_family", "task_engagement", "frustrated")
    report = {"n_conversations": len(files),
              "coders": {k: int(len(v)) for k, v in frames.items()},
              "batch_summary": state.get("summary", {}),
              "dimensions": [agreement_report(frames, d) for d in dims],
              "tone_dimensions": [agreement_report(frames, f"tone_{t.lower()}") for t in TONES],
              "legacy_first_tone": agreement_report(frames, "first_tone_alphabetical")}
    by_class = agreement_by_class(report["dimensions"] + report["tone_dimensions"]
                                  + [report["legacy_first_tone"]])
    report["agreement_by_class"] = by_class.round(3).reset_index().to_dict("records")
    (OUT / "recode_reliability_report.json").write_text(json.dumps(report, indent=2))

    print("=" * 78)
    print("CODERS")
    print("=" * 78)
    for k, v in frames.items():
        print(f"  {k:22s} {len(v):>6,} participant messages, "
              f"{v['file'].nunique()} conversations")

    print("\n" + "=" * 78)
    print("INTER-CODER AGREEMENT")
    print("=" * 78)
    for d in report["dimensions"]:
        a = d["krippendorff_alpha"]
        print(f"\n{d['dimension']}   Krippendorff alpha = "
              + (f"{a:.3f}" if a is not None else "n/a"))
        for pair, v in d["pairwise"].items():
            print(f"    {pair:44s} kappa={v['kappa']:+.3f}  raw={v['raw_agreement']:.1%}  "
                  f"n={v['n']}")

    print("\nper tone (does the message carry it?), Krippendorff alpha: " + ", ".join(
        f"{d['dimension'][5:]} {d['krippendorff_alpha']:.3f}" for d in report["tone_dimensions"]))
    print(f"alphabetically first tone (v1.1.0 summary; comparison only), Krippendorff alpha = "
          f"{report['legacy_first_tone']['krippendorff_alpha']:.3f}")

    print("\n" + "=" * 78)
    print("MEAN PAIRWISE KAPPA BY CLASS OF CODER PAIR")
    print("=" * 78)
    print(by_class.round(3).to_string())

    print("\n" + "=" * 78)
    print("MARGINAL DISTRIBUTIONS  (% of coded participant messages)")
    print("=" * 78)
    for dim in ("strategy_family", "task_engagement"):
        print(f"\n{dim}")
        tab = pd.DataFrame({name: (df[dim].value_counts(normalize=True) * 100).round(1)
                            for name, df in frames.items() if dim in df.columns})
        print(tab.fillna(0.0).to_string())

    # Condition from the session metadata written by the chatbot ("1" = Servant Leader,
    # "2" = Dark Triad), not from any coder's inference.
    file_condition = {}
    for f in files:
        meta = json.loads((LOGS / f).read_text(encoding="utf-8")).get("metadata", {})
        file_condition[f] = {"1": "servant_leader", "2": "dark_triad"}.get(str(meta.get("chatbot_version")))
    rep = condition_replication(frames, file_condition)
    report["condition_replication"] = rep
    (OUT / "recode_reliability_report.json").write_text(json.dumps(report, indent=2))
    if rep:
        print("\n" + "=" * 78)
        print("DOES THE CONDITION CONTRAST SURVIVE A CHANGE OF CODER?")
        print("=" * 78)
        rd = pd.DataFrame(rep)
        for dim in rd.dimension.unique():
            print(f"\n{dim}   (participant-level %, and Cohen's d for Dark Triad - Servant Leader)")
            piv = rd[rd.dimension == dim].pivot_table(
                index="level", columns="coder",
                values=["M_DT", "M_SL", "cohens_d_DT_minus_SL"])
            print(piv.round(2).to_string())

    ps = provider_split(report["dimensions"])
    if len(ps):
        report["provider_split"] = ps.reset_index().to_dict("records")
        print("\n" + "=" * 78)
        print("IS IT THE MODEL OR THE PROVIDER?  (mean pairwise kappa)")
        print("=" * 78)
        print(ps.to_string())

    dm = derived_metrics(frames, file_condition)
    if len(dm):
        report["derived_metrics"] = dm.to_dict("records")
        (OUT / "recode_reliability_report.json").write_text(json.dumps(report, indent=2))
        print("\n" + "=" * 78)
        print("MANUSCRIPT'S DERIVED METRICS, RECOMPUTED UNDER EACH CODER")
        print("=" * 78)
        for metric in dm.metric.unique():
            sub = dm[dm.metric == metric].set_index("coder")[["M_DT", "M_SL", "cohens_d"]]
            print(f"\n{metric}")
            print(sub.to_string())

    print(f"\nwrote {OUT}/recode_reliability_report.json")
    print("\nkappa >= .80 excellent, .60-.79 substantial, .40-.59 moderate, < .40 poor.")
    print("This bounds coder NOISE, not validity -- see the module docstring.")


# --------------------------------------------------------------------------- resync
def sync_claude(user: str, model: str):
    """Synchronous equivalent of the batch request. Streams because a non-streaming call
    with max_tokens=32000 plus adaptive thinking trips the SDK's 10-minute ceiling; batch
    requests are exempt from that limit precisely because they cannot stream."""
    import anthropic
    from anthropic import transform_schema
    from pydantic import TypeAdapter
    schema = transform_schema(TypeAdapter(ChunkAnalysisResult).json_schema())
    with anthropic.Anthropic().messages.stream(
            model=model, max_tokens=MAX_TOKENS, system=CODING_INSTRUCTIONS,
            thinking={"type": "adaptive"},
            output_config={"format": {"type": "json_schema", "schema": schema}},
            messages=[{"role": "user", "content": user}]) as st:
        return extract_claude(st.get_final_message())


def sync_openai(user: str, model: str):
    from openai import OpenAI
    from openai.lib._parsing._responses import type_to_text_format_param
    r = OpenAI().responses.create(
        model=model, reasoning={"effort": "medium"}, max_output_tokens=MAX_TOKENS,
        input=[{"role": "system", "content": CODING_INSTRUCTIONS},
               {"role": "user", "content": user}],
        text={"format": type_to_text_format_param(ChunkAnalysisResult)})
    return extract_openai(json.loads(r.model_dump_json()))


def cmd_resync(args) -> None:
    """Fill the chunks a batch did not return, by calling the model synchronously.

    A batch that is cancelled or expires leaves holes. Rather than resubmit a whole batch
    for a handful of chunks -- another wait, for requests that are individually cheap --
    these are run through the synchronous endpoint with identical parameters. The only
    difference is price: synchronous is 2x batch, which on a few chunks is pennies.
    Results are APPENDED to the coder's existing CSV, never overwriting the batch rows.
    """
    state = json.loads(STATE.read_text())
    model = args.model
    meta = state["batches"][model]
    missing = list(meta.get("missing_ids") or [])
    if not missing:
        print(f"{model}: nothing missing -- run collect first if that looks wrong")
        return

    by_id = {r["custom_id"]: r for r in build_requests()}
    tag = model.replace(".", "_")
    csv = OUT / f"recode_{tag}.csv"
    existing = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
    index = {r["custom_id"]: r for r in state["requests"]}
    fn = sync_claude if provider(model) == "anthropic" else sync_openai

    print(f"{model}: re-running {len(missing)} chunk(s) synchronously")
    coded, usage_rows, still_failed = [], [], []
    for cid in missing:
        req, meta_r = by_id.get(cid), index[cid]
        if req is None:
            still_failed.append({"custom_id": cid, "reason": "request could not be rebuilt"})
            continue
        for attempt in range(1, args.retries + 1):
            try:
                text, usage = fn(req["user"], model)
                parsed = ChunkAnalysisResult.model_validate_json(text)
                rows = flatten(parsed, meta_r["conversation_id"])
                coded.extend(rows)
                usage_rows.append({"custom_id": cid, **usage})
                print(f"  {cid}  {meta_r['conversation_id'][:34]:34s} "
                      f"chunk {meta_r['chunk_id']}  ->  {len(rows)} participant messages")
                break
            except Exception as exc:
                if attempt == args.retries:
                    still_failed.append({"custom_id": cid,
                                         "reason": f"{type(exc).__name__}: {exc}"[:300]})
                    print(f"  {cid}  FAILED after {attempt} attempts: {type(exc).__name__}")
                else:
                    time.sleep(5 * attempt)

    if coded:
        merged = pd.concat([existing, pd.DataFrame(coded)], ignore_index=True)
        merged = merged.drop_duplicates(subset=["conversation_id", "message_index"],
                                        keep="first")
        merged.to_csv(csv, index=False)
        print(f"\n{csv.name}: {len(existing):,} -> {len(merged):,} rows")

    u = pd.DataFrame(usage_rows)
    pr = SYNC_PRICING[model]
    cost = (u["in"].sum() / 1e6 * pr["in"] + u["out"].sum() / 1e6 * pr["out"]) if len(u) else 0.0
    if len(u):
        upath = OUT / f"recode_usage_{tag}_resync.csv"
        u.to_csv(upath, index=False)
        print(f"synchronous top-up: {int(u['in'].sum()):,} in / {int(u['out'].sum()):,} out "
              f"tokens, ${cost:,.2f} at sync rates")

    done = {r["custom_id"] for r in usage_rows}
    meta["returned_ids"] = sorted(set(meta.get("returned_ids", [])) | done)
    meta["missing_ids"] = sorted(set(missing) - done)
    meta.setdefault("resync", {})[datetime.now(timezone.utc).isoformat()] = {
        "attempted": len(missing), "recovered": len(done),
        "still_failed": still_failed, "sync_cost_usd": round(cost, 2)}
    state["summary"].setdefault(model, {})["resync_recovered"] = len(done)
    state["summary"][model]["sync_topup_cost_usd"] = round(cost, 2)
    state["summary"][model]["messages_coded"] = int(
        len(pd.read_csv(csv)) if csv.exists() else 0)
    STATE.write_text(json.dumps(state, indent=2))
    if still_failed:
        print(f"\nstill missing after resync: {len(still_failed)}")
        for f in still_failed:
            print(f"  {f['custom_id']}: {f['reason'][:120]}")
    else:
        print(f"\nall {len(missing)} recovered -- {model} is now complete")


def main() -> None:
    load_env()
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.set_defaults(fn=cmd_submit)
    s.add_argument("--models", default="claude-sonnet-5,gpt-5.6-terra",
                   help="comma-separated model ids from CODERS")
    s.add_argument("--resubmit", action="store_true",
                   help="re-submit a model that already has a batch in the state file")
    s.add_argument("--limit", type=int, default=None,
                   help="only the first N conversations (smoke test)")
    s.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("resync"); r.set_defaults(fn=cmd_resync)
    r.add_argument("--model", required=True, help="model id, e.g. gpt-5.6-terra")
    r.add_argument("--retries", type=int, default=3)
    cl = sub.add_parser("collect"); cl.set_defaults(fn=cmd_collect)
    cl.add_argument("--only", default=None, type=lambda v: [x.strip() for x in v.split(",")],
                    help="collect only these model ids; protects finished coders' CSVs")
    for name, fn in (("status", cmd_status), ("report", cmd_report)):
        p = sub.add_parser(name); p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
