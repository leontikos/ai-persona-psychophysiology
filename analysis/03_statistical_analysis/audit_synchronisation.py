#!/usr/bin/env python
"""
Synchronisation audit.

The event-related pipeline aligns BIOPAC recordings to conversation logs by taking the
BIOPAC segment-start event marker, adding a fixed +2 h, and treating the result as the
zero point against which conversation-log timestamps are offset:

    time_diff    = (event_time - biopac_start).total_seconds()
    event_sample = int(time_diff * sampling_rate)

This script quantifies how accurate that alignment is, rather than asserting it. It checks:

  1. Is the +2 h shift the correct UTC->local offset?  AcqKnowledge writes the LOCAL wall
     clock into the marker's raw name string while bioread exposes date_created_utc, so the
     two can be compared directly and the true offset recovered per participant.
  2. Is the offset constant across the study period, or does any session fall in CET
     (UTC+1) rather than CEST (UTC+2)?  A fixed +2 h would be an hour wrong for those.
  3. Does the marker sit at sample_index 0 (i.e. is it really the recording origin)?
  4. Timestamp resolution on both sides.
  5. Does the conversation fall inside the recording window, and with how much margin?
  6. How many assistant events were dropped because their epoch ran past the end of the
     recording -- the practical consequence of any misalignment. Assistant events are read from
     the conversation logs (one merged log per participant).

Outputs: synchronisation_audit.csv + printed summary.
"""
import glob
import json
import os
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
BASE = os.environ.get("STUDY_DATA_ROOT", "data")
CLEAN = f"{BASE}/cleaned_signals/cleaned_signals"
LOGS = f"{BASE}/conversation_logs"
OUT = os.environ.get("SYNC_AUDIT_OUT", os.path.join(BASE, "statistics_output"))

# matches e.g. "EventMarker Segment 1 śr. kwi 23 2025 12:47:44.711, MP160 0027BD:"
LOCAL_RE = re.compile(r"Segment\s+\d+\s+\S+\s+\w+\s+(\d{1,2})\s+(\d{4})\s+(\d{2}):(\d{2}):(\d{2})\.(\d+)")
PRE, POST = 2.0, 18.0          # EDA epoch window used in the analysis


def parse_iso(s):
    if s is None:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00").split("+")[0])


def assistant_messages():
    """Assistant messages per participant, from the (merged) conversation logs."""
    out = {}
    for f in sorted(glob.glob(f"{LOGS}/*.json")):
        j = json.load(open(f))
        uid = j.get("user_id") or j.get("metadata", {}).get("user_id")
        out[uid] = [m for m in j["messages"] if m.get("role") == "assistant"]
    return out


def main():
    messages = assistant_messages()
    rows = []
    for info_path in sorted(glob.glob(f"{CLEAN}/*/*_experimental_info.json")):
        info = json.load(open(info_path))
        pid = info["participant_id"]
        rec = {"pid": pid, "group": info.get("experimental_group"),
               "study_date": info.get("study_date")}

        main_ts = info.get("biopac_recording_timestamps", {}).get("main", {})
        markers = main_ts.get("event_markers", [])
        if not markers:
            rec["problem"] = "no event markers"
            rows.append(rec)
            continue
        m0 = markers[0]

        # --- 1/2. recover the true UTC->local offset from the marker string -------
        raw = m0.get("raw_name", "")
        utc_match = re.search(r"date_created_utc:\s*([0-9T:\.\-]+)", raw)
        loc_match = LOCAL_RE.search(raw)
        if utc_match and loc_match:
            utc_dt = parse_iso(utc_match.group(1))
            day, year, hh, mm, ss, frac = loc_match.groups()
            local_dt = utc_dt.replace(year=int(year), day=int(day),
                                      hour=int(hh), minute=int(mm), second=int(ss),
                                      microsecond=int(frac.ljust(6, "0")[:6]))
            rec["true_offset_hours"] = round((local_dt - utc_dt).total_seconds() / 3600.0, 4)
            rec["offset_residual_ms"] = round(
                ((local_dt - utc_dt).total_seconds() - 7200.0) * 1000.0, 3)
        rec["marker_sample_index"] = m0.get("sample_index")

        # --- 3/4. resolution + origin -------------------------------------------
        start = parse_iso(main_ts.get("recording_start_time"))
        end = parse_iso(main_ts.get("recording_end_time"))
        rec["biopac_start"] = start
        rec["biopac_ms_resolution"] = (start.microsecond % 1000 == 0) if start else None
        si = main_ts.get("sampling_info", {})
        rec["sampling_interval"] = si.get("sampling_interval")
        rec["duration_s"] = main_ts.get("duration_seconds")

        # --- 5/6. conversation coverage -----------------------------------------
        if pid in messages and start and end:
            resp = messages[pid]
            times = [parse_iso(m["timestamp_start"]) for m in resp if m.get("timestamp_start")]
            times = [t for t in times if t]
            if times:
                rec["n_assistant_events"] = len(times)
                rec["conv_start_offset_s"] = (min(times) - start).total_seconds()
                rec["conv_end_offset_s"] = (max(times) - start).total_seconds()
                rec["margin_before_first_event_s"] = rec["conv_start_offset_s"] - PRE
                rec["margin_after_last_event_s"] = rec["duration_s"] - (rec["conv_end_offset_s"] + POST)
                # events whose epoch would fall outside the recording
                usable = [t for t in times
                          if (t - start).total_seconds() - PRE >= 0
                          and (t - start).total_seconds() + POST < rec["duration_s"]]
                rec["n_events_usable"] = len(usable)
                rec["n_events_dropped"] = len(times) - len(usable)
                # microsecond resolution on the conversation side?
                rec["conv_us_resolution"] = any(t.microsecond % 1000 != 0 for t in times)
        rows.append(rec)

    df = pd.DataFrame(rows)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(f"{OUT}/synchronisation_audit.csv", index=False)
    print(f"wrote {OUT}/synchronisation_audit.csv  (n = {len(df)})\n")

    print("=" * 74)
    print("1-2. UTC -> local offset recovered from the AcqKnowledge marker string")
    print("=" * 74)
    if "true_offset_hours" in df:
        vc = df["true_offset_hours"].value_counts(dropna=False)
        for val, n in vc.items():
            print(f"   offset = {val} h : {n} participants")
        bad = df[df["true_offset_hours"].notna() & (df["true_offset_hours"] != 2.0)]
        print(f"   participants where the applied +2 h is WRONG: {len(bad)}")
        if len(bad):
            print(bad[["pid", "study_date", "true_offset_hours"]].to_string(index=False))
        res = df["offset_residual_ms"].dropna()
        if len(res):
            print(f"   residual after removing exactly 2 h: max |{res.abs().max():.3f}| ms")
    print(f"\n   study dates: {df['study_date'].min()} .. {df['study_date'].max()}")
    print("   (Poland observes CEST = UTC+2 from the last Sunday in March to the last "
          "Sunday in October)")

    print("\n" + "=" * 74)
    print("3-4. Recording origin and timestamp resolution")
    print("=" * 74)
    print(f"   marker at sample_index 0: {(df['marker_sample_index'] == 0).sum()} / {len(df)}")
    print(f"   BIOPAC start times on an exact-millisecond grid: "
          f"{df['biopac_ms_resolution'].sum()} / {df['biopac_ms_resolution'].notna().sum()}")
    if "conv_us_resolution" in df:
        print(f"   conversation timestamps with sub-millisecond digits: "
              f"{df['conv_us_resolution'].sum()} / {df['conv_us_resolution'].notna().sum()}")
    si = df["sampling_interval"].dropna()
    if len(si):
        print(f"   sampling interval: {si.min():.10f} .. {si.max():.10f} s "
              f"(nominal 0.0005 s = 2000 Hz)")
        print(f"   implied drift over a {df['duration_s'].max()/60:.0f}-min recording: "
              f"{(si.max()-0.0005)/0.0005 * df['duration_s'].max()*1000:.1f} ms")

    print("\n" + "=" * 74)
    print("5-6. Conversation coverage within the recording")
    print("=" * 74)
    for col, label in [("conv_start_offset_s", "first assistant event, s after BIOPAC start"),
                       ("margin_before_first_event_s", "margin before first epoch (s)"),
                       ("margin_after_last_event_s", "margin after last epoch (s)")]:
        if col in df:
            v = df[col].dropna()
            print(f"   {label:48s} min={v.min():9.1f}  median={v.median():9.1f}  max={v.max():9.1f}")
    if "n_events_dropped" in df:
        tot, drop = df["n_assistant_events"].sum(), df["n_events_dropped"].sum()
        print(f"\n   assistant events total: {int(tot)}; epochs droppable at window edges: "
              f"{int(drop)} ({100*drop/tot:.1f}%)")
        worst = df.nlargest(5, "n_events_dropped")[
            ["pid", "n_assistant_events", "n_events_dropped", "margin_after_last_event_s"]]
        print(worst.to_string(index=False))
        neg = df[df["margin_before_first_event_s"] < 0]
        print(f"\n   participants whose first event precedes the recording start: {len(neg)}")
        if len(neg):
            print(neg[["pid", "study_date", "margin_before_first_event_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
