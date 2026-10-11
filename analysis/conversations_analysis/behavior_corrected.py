#!/usr/bin/env python
"""
Corrected behavioural (dialogue) metrics: conversation logs + archived GPT-4.1 codes ->
per-message and per-conversation tables, condition comparisons, and the dialogue figures.

This supersedes, for every dialogue-derived number and figure in the paper, the path
gpt_assessment_analysis.py -> conversations_dataframe.csv / interactions_dataframe.csv ->
compute_behavioral.py, the earlier make_main_fig2.py and regenerate_manuscript_figures.py, and corrects six
problems in that path:

  1. Message counts, mean message lengths and the condition label of the conversation-level
     table were copied from the coding model's own conversation summary, i.e. they were LLM
     estimates, not counts. The per-message `message_length` fields are LLM estimates too.
     Here counts and lengths are counted from the logs (characters of `content`), and the
     condition is the log's metadata.chatbot_version ("1" = Servant Leader, "2" = Dark Triad).
  2. `num_turning_points` was the number of "key turning points" the model listed in its
     summary, not a count of strategy switches. Here a strategy switch is an adjacent pair of
     the participant's messages, in transcript order, whose strategy FAMILY differs
     (families = label prefix: COMPLIANCE, RESISTANCE, NEGOTIATION, HELP_SEEKING, COPING,
     JAILBREAKING); switch rate = switches / (n - 1).
  3. Emotional-tone "proportions" were each tone's share of all emotion labels in the
     conversation. A message can carry several tones, so that is not a share of messages.
     Here a tone's proportion is the share of the participant's messages tagged with it.
  4. Chatbot message length included the fixed opening greeting "How may I help you?"
     (twice in the one restarted session, raspberry_daisy) and the coded
     "[SESSION CONTINUATION]" system marker. Here chatbot lengths and all chatbot code shares
     use the persona replies only (greeting(s) and system marker excluded). The chatbot
     message COUNT is all logged assistant messages (greeting included, marker excluded);
     the count without the greeting is reported alongside (tests are unchanged by it).
  5. Interaction dynamics (Supplementary Note 5 / Fig. 4) used the model's
     `interaction_turns`, which exist for 13 of 58 conversations (194 turns). Here every
     participant message is paired with the chatbot message that PRECEDES it (the most
     recent chatbot message before it in the transcript), in all 58 conversations; messages
     whose preceding chatbot message is the greeting are excluded (59 messages).
  6. Phases (Supplementary Note 6 / Figs. 5-6) were cut from message_number / max(message_number)
     with bins (0, .33], (.33, .67], (.67, 1]; the numbering scheme differs between analysis
     files (39 count every message, 19 count participant messages only), and LLM lengths were
     averaged. Here phases are ordinal thirds: the k-th of a conversation's n participant
     messages (k = 0..n-1) is in phase floor(3k / n) (0 = early, 1 = middle, 2 = late);
     chatbot persona replies are split the same way over their own ordinal positions.

The archived per-message codes (participant: primary_strategy, emotional_tone,
task_engagement; chatbot: leadership_tactics, primary_tone, response_to_participant) are used
unchanged. Each code record is anchored to its transcript message by its quoted text
(exact > prefix > fuzzy > ordinal fallback); for all 58 files this is one-to-one and
reproduces the archived list order.

Statistics: Mann-Whitney U (U of the Dark Triad sample, as the manuscript prints it), tie-
corrected Z, p, rank-biserial r with BCa 95% CI (5,000 resamples, seed 20260704, one RNG
stream through the variables in the published order), Hodges-Lehmann SL - DT with the
distribution-free 95% CI, Cohen's d (SL - DT) with the noncentral-t 95% CI, and the Welch p.
Positive Z, r, HL and d mean Servant Leader > Dark Triad. Chi-square tests of code x
condition report Cramer's V with a cluster (conversation) bootstrap 95% CI (5,000, seed 7).
The helper functions are copied verbatim from analysis/statistics/recompute_stats.py and
compute_behavioral.py (importing recompute_stats runs the survey reconstruction).
Conversations are processed in the row order of the archived conversations_dataframe.csv
(read for its conversation_id column only), so the bootstrap draws of every unchanged
variable are identical to the published run.

Outputs (BEHAV_CORRECTED_DIR):
  participant_messages_corrected.csv   one row per participant message (1,332)
  chatbot_messages_corrected.csv       one row per coded chatbot message (1,382)
  conversation_metrics_corrected.csv   one row per conversation (58)
  conversation_level_tests_corrected.csv
  chisquare_corrected.csv, message_level_percentages_corrected.csv
  note5_*.csv, note6_*.csv, correlation_matrix_corrected.csv, behavior_corrected_log.txt
Figures (FIGS): figure_main02_behavior.png, figure03.png, figure04.png, figure05.png, figure06.png,
  figure07.png, figure08.png, figure09.png, figure10.png, figure11.png (dpi 300; same file names
  as the manuscript), and figure02.png (all six strategy families; not shown in the paper). Brackets on the box plots mark the conversation-level Mann-Whitney p of the
  tables, and on the jailbreaking bar plot Fisher's exact p (uncorrected; * < .05, ** < .01,
  *** < .001, n.s. otherwise).

Run:  python behavior_corrected.py   (tables, tests, figures)
"""
import difflib
import glob
import json
import os
import re
import sys
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import chi2_contingency, mannwhitneyu, nct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import seaborn as sns                                              # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA = os.environ.get("STUDY_DATA_ROOT", "data")
LOGS = f"{DATA}/conversation_logs"
ANALYSES = f"{DATA}/CONVERSATION_ANALYSES/successful_analyses"
ROW_ORDER = f"{DATA}/CONVERSATION_ANALYSES/conversations_dataframe.csv"   # conversation_id order only
OUT = os.environ.get("BEHAV_CORRECTED_DIR", f"{DATA}/CONVERSATION_ANALYSES/behavior_corrected")
FIGS = os.environ.get("FIGS", os.path.join(DATA, "figures"))

SEED_BOOT, SEED_V, NBOOT = 20260704, 7, 5000
GREETING = "How may I help you?"
COND = {"1": "servant_leader", "2": "dark_triad"}
FAMILIES = ["COMPLIANCE", "RESISTANCE", "NEGOTIATION", "HELP_SEEKING", "COPING", "JAILBREAKING"]
EMOS = ["FRUSTRATED", "ANXIOUS", "CONFIDENT", "NEUTRAL", "RESIGNED", "DEFENSIVE"]
ENG = ["HIGH", "MEDIUM", "LOW"]
DT_T = ["INTIMIDATION", "CRITICISM_HARSH", "MANIPULATION", "DISMISSIVENESS", "MICROMANAGEMENT",
        "BLAME_SHIFTING", "GRANDIOSITY"]
SL_T = ["ENCOURAGEMENT", "EMPOWERMENT", "CONSTRUCTIVE_FEEDBACK", "ACTIVE_LISTENING", "COLLABORATION",
        "SUPPORT_OFFERING", "RECOGNITION"]
NT_T = ["TASK_DIRECTION", "INFORMATION_PROVISION", "CLARIFICATION"]
TONES = ["HOSTILE", "CONDESCENDING", "COLD", "NEUTRAL", "WARM", "SUPPORTIVE", "SARCASTIC"]
NEG_TONES, POS_TONES = ["HOSTILE", "CONDESCENDING", "COLD", "SARCASTIC"], ["WARM", "SUPPORTIVE"]
RESPONSES = ["VALIDATES", "REJECTS", "PARTIALLY_ACCEPTS", "IGNORES", "ESCALATES", "DE_ESCALATES", "REDIRECTS"]
PHASES = ["Early", "Middle", "Late"]

LOG_LINES = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG_LINES.append(s)


# =============================================================================================
# 1. Building the tables
# =============================================================================================
def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00").split("+")[0])


def norm(s):
    return re.sub(r"\s+", " ", (s or "")).strip()


def family(label):
    for f in FAMILIES:
        if label.startswith(f):
            return f
    return "OTHER"


def entropy_bits(labels):
    """Shannon entropy (log2) of the label distribution; same formula as the archived table.
    Rounded to 12 decimals: the summation order follows the order in which labels first appear,
    so mathematically equal entropies can otherwise differ in the last bit and split rank ties."""
    if len(labels) == 0:
        return 0.0
    p = np.array(list(Counter(labels).values())) / len(labels)
    return round(float(-np.sum(p * np.log2(p + 1e-10))), 12) + 0.0


def ordinal_phase(k, n):
    """k-th of n messages (k = 0..n-1) -> 0 early, 1 middle, 2 late (ordinal thirds)."""
    return int(np.floor(3 * k / n))


def anchor(records, key, transcript):
    """Match archived code records to transcript messages by quoted text.
    Preference exact > prefix > prefix before an elision marker > fuzzy (>= 0.6) > ordinal."""
    n = len(transcript)
    used, out, last = set(), [], -1
    tnorm = [norm(m["content"]) for m in transcript]
    for rec in records:
        q = norm(rec.get(key, ""))
        q_stripped = re.sub(r"(\.\.\.|…)\s*(\[.*?\])?\s*$", "", q).strip()
        cand = [i for i in range(n) if i not in used and tnorm[i] == q]
        method = "exact"
        if not cand and len(q_stripped) >= 8:
            cand = [i for i in range(n) if i not in used and tnorm[i].startswith(q_stripped)]
            method = "prefix"
        if not cand and len(q_stripped) >= 8:
            head = re.split(r"\s*(\.\.\.|…)", q)[0].strip()
            if len(head) >= 8:
                cand = [i for i in range(n) if i not in used and tnorm[i].startswith(head)]
                method = "prefix_head"
        if not cand:
            best, bi = 0.0, None
            for i in range(n):
                if i in used:
                    continue
                r = difflib.SequenceMatcher(None, q[:400], tnorm[i][:400]).ratio()
                if r > best:
                    best, bi = r, i
            if bi is not None and best >= 0.6:
                cand, method = [bi], "fuzzy"
        if cand:
            after = [i for i in cand if i > last]
            i = after[0] if after else cand[0]
        else:
            free = [i for i in range(n) if i not in used]
            i = free[0] if free else None
            method = "ordinal_fallback"
        if i is not None:
            used.add(i)
            last = i
        out.append((i, method))
    return out


def build_tables():
    pm_rows, cb_rows, cv_rows, anchor_rows = [], [], [], []
    for f in sorted(glob.glob(f"{LOGS}/*.json")):
        fname = os.path.basename(f)
        log = json.load(open(f))
        md, msgs = log["metadata"], log["messages"]
        pid, cond = md["user_id"], COND[str(md["chatbot_version"])]
        a = json.load(open(f"{ANALYSES}/{fname.replace('.json', '_analyzed.json')}"))
        assert a["original_filename"] == fname
        cid = a["conversation_id"]
        psum = a["participant_summary"]

        users = [dict(idx=i, **m) for i, m in enumerate(msgs) if m["role"] == "user"]
        bots = [dict(idx=i, **m) for i, m in enumerate(msgs) if m["role"] in ("assistant", "system")]
        for b in bots:
            b["is_greeting"] = int(b["role"] == "assistant" and norm(b["content"]) == GREETING)
            b["is_persona"] = int(b["role"] == "assistant" and not b["is_greeting"])
        persona = [b for b in bots if b["is_persona"]]
        n_user, n_persona = len(users), len(persona)
        persona_ord = {b["idx"]: k for k, b in enumerate(persona)}

        # ---- anchor archived codes to the transcript ----
        pa, ca = a["participant_message_analyses"], a["chatbot_message_analyses"]
        panc, canc = anchor(pa, "participant_message", users), anchor(ca, "chatbot_message", bots)
        ppos, cpos = [p for p, _ in panc], [p for p, _ in canc]
        anchor_rows.append(dict(pid=pid, file=fname, n_user=n_user, n_coded_user=len(pa),
                                n_bot_incl_marker=len(bots), n_coded_bot=len(ca),
                                user_one_to_one=len(set(ppos)) == len(ppos) == n_user and None not in ppos,
                                user_in_list_order=ppos == sorted(ppos),
                                bot_one_to_one=len(set(cpos)) == len(cpos) == len(bots) and None not in cpos,
                                bot_in_list_order=cpos == sorted(cpos),
                                user_methods=dict(Counter(m for _, m in panc)),
                                bot_methods=dict(Counter(m for _, m in canc))))
        code_of_bot = {}
        pnum_max = max(r["message_number"] for r in pa)
        cnum_max = max(r["message_number"] for r in ca)
        for k, (r, (pos, meth)) in enumerate(zip(ca, canc)):
            b = bots[pos]
            tactics = r.get("leadership_tactics", []) or []
            row = dict(pid=pid, file=fname, conversation_id=cid, condition=cond, list_index=k,
                       message_number=r["message_number"],
                       archived_position=r["message_number"] / cnum_max,
                       transcript_index=b["idx"], role=b["role"], is_greeting=b["is_greeting"],
                       is_persona=b["is_persona"], anchor_method=meth,
                       persona_ordinal=persona_ord.get(b["idx"], np.nan), n_persona=n_persona,
                       primary_tone=r.get("primary_tone", "NEUTRAL"),
                       response_to_participant=r.get("response_to_participant", "NEUTRAL"),
                       tactics="|".join(tactics), n_tactics=len(tactics),
                       n_dt_tactics=sum(t in DT_T for t in tactics),
                       n_sl_tactics=sum(t in SL_T for t in tactics),
                       n_nt_tactics=sum(t in NT_T for t in tactics),
                       chars=len(b["content"]), llm_message_length=r.get("message_length"),
                       timestamp_start=b["timestamp_start"])
            row["phase"] = (PHASES[ordinal_phase(row["persona_ordinal"], n_persona)]
                            if b["is_persona"] else None)
            cb_rows.append(row)
            code_of_bot[b["idx"]] = row

        # ---- participant messages (transcript order), paired with the preceding chatbot message ----
        seq = sorted(zip(ppos, pa, [m for _, m in panc]), key=lambda t: t[0])
        for k, (pos, r, meth) in enumerate(seq):
            u = users[pos]
            prev = [b for b in bots if b["idx"] < u["idx"]]
            pb = code_of_bot[prev[-1]["idx"]] if prev else None
            row = dict(pid=pid, file=fname, conversation_id=cid, condition=cond, ordinal=k, n_in_conv=n_user,
                       phase=PHASES[ordinal_phase(k, n_user)], rel_position=(k + 1) / n_user,
                       transcript_index=u["idx"], anchor_method=meth, message_number=r["message_number"],
                       archived_position=r["message_number"] / pnum_max,
                       primary_strategy=r["primary_strategy"], family=family(r["primary_strategy"]),
                       task_engagement=r["task_engagement"], emotions="|".join(r["emotional_tone"]),
                       n_emotions=len(r["emotional_tone"]),
                       **{f"emo_{e.lower()}": int(e in r["emotional_tone"]) for e in EMOS},
                       chars=len(u["content"]), llm_message_length=r.get("message_length"),
                       timestamp_start=u["timestamp_start"],
                       prev_bot_transcript_index=(pb["transcript_index"] if pb else np.nan),
                       prev_bot_adjacent=int(bool(prev) and prev[-1]["idx"] == u["idx"] - 1),
                       prev_is_greeting_or_marker=(int(not pb["is_persona"]) if pb else 1),
                       prev_tone=(pb["primary_tone"] if pb else None),
                       prev_response=(pb["response_to_participant"] if pb else None),
                       prev_bot_chars=(pb["chars"] if pb else np.nan),
                       prev_bot_phase=(pb["phase"] if pb else None))
            pm_rows.append(row)

        # ---- conversation level ----
        subs = [r["primary_strategy"] for _, r, _ in seq]
        fams = [family(s) for s in subs]
        n = len(subs)
        sw_fam = sum(fams[i] != fams[i - 1] for i in range(1, n))
        sw_sub = sum(subs[i] != subs[i - 1] for i in range(1, n))
        ulen = np.array([len(m["content"]) for m in users])
        all_asst = [b for b in bots if b["role"] == "assistant"]
        plen = np.array([len(b["content"]) for b in persona])
        alen = np.array([len(b["content"]) for b in all_asst])
        nonsys = [m for m in msgs if m["role"] != "system"]
        dur = (max(ts(m["timestamp_end"]) for m in nonsys) - min(ts(m["timestamp_start"]) for m in nonsys)).total_seconds() / 60
        all_emos = [e for _, r, _ in seq for e in r["emotional_tone"]]
        eng = [r["task_engagement"] for _, r, _ in seq]
        row = dict(conversation_id=cid, pid=pid, file=fname, condition=cond,
                   total_participant_messages=n_user,
                   total_chatbot_messages=len(all_asst),
                   total_chatbot_messages_excl_greeting=n_persona,
                   n_greetings=sum(b["is_greeting"] for b in bots), n_system_markers=sum(b["role"] == "system" for b in bots),
                   avg_participant_msg_length=ulen.mean(),
                   avg_chatbot_msg_length=plen.mean(),
                   avg_chatbot_msg_length_incl_greeting=alen.mean(),
                   total_participant_chars=int(ulen.sum()), total_chatbot_chars=int(plen.sum()),
                   total_chatbot_chars_incl_greeting=int(alen.sum()),
                   duration_minutes=dur,
                   strategy_diversity=entropy_bits(subs),
                   strategy_switches=sw_fam,
                   switch_rate=(sw_fam / (n - 1) if n > 1 else np.nan),
                   subtype_switches=sw_sub, subtype_switch_rate=(sw_sub / (n - 1) if n > 1 else np.nan),
                   family_entropy=entropy_bits(fams),
                   n_distinct_subtypes=len(set(subs)), n_distinct_families=len(set(fams)))
        for fam in FAMILIES:
            row[f"prop_{fam.lower()}"] = sum(x == fam for x in fams) / n
        for e in EMOS:
            row[f"prop_emotion_{e.lower()}"] = sum(e in r["emotional_tone"] for _, r, _ in seq) / n
        for e in EMOS:                                              # archived definition, for reference
            row[f"labelshare_emotion_{e.lower()}"] = sum(x == e for x in all_emos) / max(len(all_emos), 1)
        for lv in ENG:
            row[f"prop_engagement_{lv.lower()}"] = eng.count(lv) / n
        # chatbot code shares, persona replies only
        pc = [code_of_bot[b["idx"]] for b in persona]
        tac = [t for c in pc for t in c["tactics"].split("|") if t]
        row["prop_dark_triad_tactics"] = sum(t in DT_T for t in tac) / max(len(tac), 1)
        row["prop_servant_tactics"] = sum(t in SL_T for t in tac) / max(len(tac), 1)
        for t in TONES:
            row[f"prop_tone_{t.lower()}"] = sum(c["primary_tone"] == t for c in pc) / max(len(pc), 1)
        for rsp in RESPONSES:
            row[f"prop_response_{rsp.lower()}"] = sum(c["response_to_participant"] == rsp for c in pc) / max(len(pc), 1)
        # what the archived table carried (LLM summary), for reference only
        row.update(llm_total_participant_messages=psum.get("total_participant_messages"),
                   llm_total_chatbot_messages=psum.get("total_chatbot_messages"),
                   llm_avg_participant_msg_length=psum.get("average_participant_message_length",
                                                           psum.get("average_message_length")),
                   llm_avg_chatbot_msg_length=psum.get("average_chatbot_message_length"),
                   llm_key_turning_points=len(psum.get("key_turning_points", [])),
                   llm_condition=psum.get("condition_inference", "").lower().replace(" ", "_"))
        cv_rows.append(row)

    pm, cb, cv, an = pd.DataFrame(pm_rows), pd.DataFrame(cb_rows), pd.DataFrame(cv_rows), pd.DataFrame(anchor_rows)
    # conversation row order of the archived table (bootstrap draws are by row position)
    # The bootstrap CIs depend on row order; the archived step-7 table fixes it. Without that table
    # (or if it lists different conversations) point estimates are unchanged but BCa CIs can differ
    # in the second decimal from the reported ones.
    if os.path.exists(ROW_ORDER):
        order = pd.read_csv(ROW_ORDER, usecols=["conversation_id"]).conversation_id.tolist()
        if sorted(order) == sorted(cv.conversation_id):
            cv = cv.set_index("conversation_id").loc[order].reset_index()
        else:
            P(f"WARNING: {ROW_ORDER} lists different conversations; using file order (BCa CIs may differ)")
    else:
        P(f"WARNING: {ROW_ORDER} not found; using file order (BCa CIs may differ from the reported ones)")
    return pm, cb, cv, an


# =============================================================================================
# 2. Statistics (helpers copied verbatim from recompute_stats.py / compute_behavioral.py)
# =============================================================================================
RNG = np.random.default_rng(SEED_BOOT)      # recompute_stats.py module RNG
RNG_V = np.random.default_rng(SEED_V)       # compute_behavioral.py module RNG


def d_ci_nct(d, n1, n2, conf=0.95):
    """95% CI for Cohen's d (independent groups) via noncentral t."""
    df = n1 + n2 - 2
    a = 1.0/n1 + 1.0/n2
    t = d / np.sqrt(a)
    alpha = 1 - conf

    def solve(target):
        f = lambda ncp: nct.cdf(t, df, ncp) - target   # noqa: E731
        # adaptively widen bracket until sign change, then brentq
        for span in (20, 40, 80, 160, 320):
            lo, hi = t - span, t + span
            try:
                flo, fhi = f(lo), f(hi)
                if np.isnan(flo) or np.isnan(fhi):
                    continue
                if flo * fhi < 0:
                    return brentq(f, lo, hi, maxiter=1000)
            except Exception:
                continue
        return np.nan
    ncp_lo = solve(1 - alpha/2)   # lower CI limit ncp
    ncp_hi = solve(alpha/2)       # upper CI limit ncp
    return ncp_lo*np.sqrt(a), ncp_hi*np.sqrt(a)


def d_ci_nct_grid(d, n1, n2, conf=0.95):
    """Same noncentral-t inversion with the root bracketed on a fine ncp grid. Used only where
    d_ci_nct returns NaN (nct.cdf evaluates to nan at its +-20 bracket ends)."""
    df = n1 + n2 - 2
    a = 1.0/n1 + 1.0/n2
    t = d/np.sqrt(a)
    alpha = 1 - conf
    grid = np.linspace(t-15, t+15, 3001)
    out = []
    for target in (1-alpha/2, alpha/2):
        vals = nct.cdf(t, df, grid) - target
        ok = ~np.isnan(vals)
        g, v = grid[ok], vals[ok]
        idx = np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0]
        if len(idx) == 0:
            out.append(np.nan)
            continue
        i = idx[0]
        out.append(brentq(lambda ncp: nct.cdf(t, df, ncp) - target, g[i], g[i+1]) * np.sqrt(a))
    return out[0], out[1]


def mw_z(x, y):
    """Standardized Z for Mann-Whitney with tie correction (normal approx)."""
    x = np.asarray(x); y = np.asarray(y)
    n1, n2 = len(x), len(y)
    U, p = mannwhitneyu(x, y, alternative="two-sided")
    all_v = np.concatenate([x, y])
    n = n1 + n2
    mu = n1*n2/2.0
    # tie correction
    _, counts = np.unique(all_v, return_counts=True)
    tie = np.sum(counts**3 - counts)
    sigma = np.sqrt((n1*n2/12.0)*((n+1) - tie/(n*(n-1))))
    z = (U - mu)/sigma if sigma > 0 else np.nan
    return U, p, z


def rank_biserial(x, y):
    """Rank-biserial r = 2U/(n1*n2) - 1, sign: positive => x>y (group1 larger)."""
    x = np.asarray(x); y = np.asarray(y)
    n1, n2 = len(x), len(y)
    U1, _ = mannwhitneyu(x, y, alternative="two-sided")  # U for x
    return (2.0*U1/(n1*n2)) - 1.0


def hodges_lehmann(x, y, conf=0.95):
    """HL median difference (x - y) with distribution-free CI."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    diffs = np.sort((x[:, None] - y[None, :]).ravel())
    K = len(diffs)
    hl = np.median(diffs)
    n1, n2 = len(x), len(y)
    z = stats.norm.ppf(1-(1-conf)/2)
    sd = np.sqrt(n1*n2*(n1+n2+1)/12.0)
    c = z*sd
    k = int(np.round(n1*n2/2.0 - c))
    k = max(1, k)
    lo = diffs[k-1]
    hi = diffs[K-k]
    return hl, lo, hi


def bca_ci(data_tuple, stat_fn, nboot=NBOOT, conf=0.95):
    """BCa bootstrap CI. data_tuple=(x,y). stat_fn(x,y)->scalar. Resample within groups."""
    x, y = [np.asarray(a) for a in data_tuple]
    theta_hat = stat_fn(x, y)
    boots = np.empty(nboot)
    n1, n2 = len(x), len(y)
    for b in range(nboot):
        xb = x[RNG.integers(0, n1, n1)]
        yb = y[RNG.integers(0, n2, n2)]
        try:
            boots[b] = stat_fn(xb, yb)
        except Exception:
            boots[b] = np.nan
    boots = boots[~np.isnan(boots)]
    if len(boots) < 100:
        return (np.nan, np.nan)
    # bias correction
    z0 = stats.norm.ppf((np.sum(boots < theta_hat)+0.5)/(len(boots)+1))
    # acceleration via jackknife
    jack = []
    for i in range(n1):
        xj = np.delete(x, i)
        jack.append(stat_fn(xj, y))
    for j in range(n2):
        yj = np.delete(y, j)
        jack.append(stat_fn(x, yj))
    jack = np.array(jack); jbar = jack.mean()
    denom = 6*(np.sum((jbar-jack)**2))**1.5
    a = np.sum((jbar-jack)**3)/denom if denom != 0 else 0.0
    alpha = 1-conf

    def adj(q):
        zq = stats.norm.ppf(q)
        return stats.norm.cdf(z0 + (z0+zq)/(1-a*(z0+zq)))
    lo_q, hi_q = adj(alpha/2), adj(1-alpha/2)
    return np.nanpercentile(boots, 100*lo_q), np.nanpercentile(boots, 100*hi_q)


def cramers_v(ct):
    ct = np.asarray(ct, float)
    # drop zero-margin rows/cols so chi2_contingency doesn't error
    ct = ct[ct.sum(1) > 0][:, ct.sum(0) > 0]
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.sum()
    k = min(ct.shape)-1
    return np.sqrt(chi2/(n*k))


def ffh_exact(table):
    """Fisher-Freeman-Halton exact p for a 2 x k table: sum of the probabilities of all tables with
    the observed margins that are no more probable than the observed one."""
    from math import comb
    table = np.asarray(table, int)
    rows, cols = table.sum(1), table.sum(0)
    n = int(table.sum())

    def prob(first_row):
        num = 1
        for c, x in zip(cols, first_row):
            num *= comb(int(c), int(x))
        return num / comb(n, int(rows[0]))

    p_obs = prob(table[0])
    total = 0.0

    def rec(i, remaining, acc):
        nonlocal total
        if i == len(cols) - 1:
            if 0 <= remaining <= cols[i]:
                pr = prob(acc + [remaining])
                if pr <= p_obs * (1 + 1e-9):
                    total += pr
            return
        for x in range(0, min(int(cols[i]), remaining) + 1):
            rec(i + 1, remaining - x, acc + [x])

    rec(0, int(rows[0]), [])
    return total


def cohens_d(a, b):
    a, b = pd.Series(a), pd.Series(b)
    n1, n2 = len(a), len(b)
    p = np.sqrt(((n1-1)*a.std(ddof=1)**2+(n2-1)*b.std(ddof=1)**2)/(n1+n2-2))
    return (a.mean()-b.mean())/p if p > 0 else 0.0


def cluster_boot_V(long, cat_col, nboot=NBOOT):
    """long: one row per coded unit with conversation_id, condition, cat_col. Cluster
    (conversation) bootstrap: resample conversations, sum their category count vectors into
    their condition row, recompute Cramer's V (compute_behavioral.py)."""
    cats = sorted(long[cat_col].dropna().unique())
    conds = sorted(long["condition"].unique())
    cat_idx = {c: i for i, c in enumerate(cats)}
    cond_idx = {c: i for i, c in enumerate(conds)}
    ct0 = np.zeros((len(conds), len(cats)))
    conv_vecs = {}
    for cid, sub in long.groupby("conversation_id"):
        row = cond_idx[sub["condition"].iloc[0]]
        vec = np.zeros(len(cats))
        for c in sub[cat_col].dropna():
            vec[cat_idx[c]] += 1
        conv_vecs[cid] = (row, vec)
        ct0[row] += vec
    v0 = cramers_v(ct0)
    convs = np.array(list(conv_vecs.keys()))
    boots = []
    for _ in range(nboot):
        samp = RNG_V.choice(convs, len(convs), replace=True)
        ct = np.zeros_like(ct0)
        for cid in samp:
            r, vec = conv_vecs[cid]
            ct[r] += vec
        if (ct.sum(1) > 0).all() and (ct.sum(0) > 0).sum() >= 2:
            b = cramers_v(ct)
            if np.isfinite(b):
                boots.append(b)
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    chi2, p, dof, _ = chi2_contingency(ct0[:, ct0.sum(0) > 0], correction=False)
    return dict(V=v0, V_lo=lo, V_hi=hi, N=int(ct0.sum()), chi2=chi2, df=dof, p=p, table=ct0, cats=cats, conds=conds)


def mw_table(cv, variables):
    """Conversation-level SL vs DT comparisons. g1 = SL, g2 = DT; Z, r, HL, d positive = SL > DT."""
    rows = []
    for v in variables:
        g1 = cv.loc[cv.condition == "servant_leader", v].dropna().values.astype(float)
        g2 = cv.loc[cv.condition == "dark_triad", v].dropna().values.astype(float)
        U, up, z = mw_z(g1, g2)
        rrb = rank_biserial(g1, g2)
        rlo, rhi = bca_ci((g1, g2), rank_biserial)
        hl, hllo, hlhi = hodges_lehmann(g1, g2)
        d = cohens_d(g1, g2)
        dlo, dhi = d_ci_nct(d, len(g1), len(g2))
        glo, ghi = d_ci_nct_grid(d, len(g1), len(g2))
        tt = stats.ttest_ind(g2, g1, equal_var=False)
        rows.append(dict(Variable=v, n_DT=len(g2), n_SL=len(g1),
                         M_DT=g2.mean(), SD_DT=g2.std(ddof=1), Mdn_DT=np.median(g2),
                         M_SL=g1.mean(), SD_SL=g1.std(ddof=1), Mdn_SL=np.median(g1),
                         U_DT=len(g1)*len(g2)-U, U_SL=U, Z=z, p=up, rrb=rrb, rrb_lo=rlo, rrb_hi=rhi,
                         HL=hl, HL_lo=hllo, HL_hi=hlhi, d=d, d_lo=dlo, d_hi=dhi,
                         d_lo_grid=glo, d_hi_grid=ghi,
                         d_lo_report=(dlo if np.isfinite(dlo) else glo), d_hi_report=(dhi if np.isfinite(dhi) else ghi),
                         welch_t=tt.statistic, welch_p=tt.pvalue))
    return pd.DataFrame(rows)


# variables of Supplementary Tables 2-5 and main Table 2, in the published order
TEST_VARS = ["strategy_diversity", "strategy_switches", "total_participant_messages", "total_chatbot_messages",
             "avg_participant_msg_length", "avg_chatbot_msg_length", "duration_minutes",
             "prop_compliance", "prop_resistance", "prop_jailbreaking", "prop_negotiation", "prop_help_seeking",
             "prop_coping", "prop_emotion_frustrated", "prop_emotion_anxious", "prop_emotion_confident",
             "prop_emotion_neutral", "prop_emotion_resigned", "prop_emotion_defensive",
             "prop_engagement_high", "prop_engagement_medium", "prop_engagement_low"]
EXTRA_VARS = ["switch_rate", "total_chatbot_messages_excl_greeting", "avg_chatbot_msg_length_incl_greeting",
              "total_participant_chars", "total_chatbot_chars", "subtype_switches", "subtype_switch_rate",
              "family_entropy"] + [f"labelshare_emotion_{e.lower()}" for e in EMOS] + ["llm_key_turning_points"]


def run_tests(cv, pm):
    global RNG, RNG_V
    RNG = np.random.default_rng(SEED_BOOT)
    res = mw_table(cv, TEST_VARS + EXTRA_VARS)
    res.to_csv(f"{OUT}/conversation_level_tests_corrected.csv", index=False)
    P("\n=== Conversation-level comparisons (U = U of DT; Z, r, HL, d positive = SL > DT) ===")
    P(res[["Variable", "M_DT", "SD_DT", "M_SL", "SD_SL", "U_DT", "Z", "p", "rrb", "rrb_lo", "rrb_hi",
           "HL", "HL_lo", "HL_hi", "d", "d_lo_report", "d_hi_report", "welch_p"]].round(3).to_string())

    RNG_V = np.random.default_rng(SEED_V)
    long_emo = pm.melt(id_vars=["conversation_id", "condition"], value_vars=[f"emo_{e.lower()}" for e in EMOS],
                       var_name="emotion", value_name="on")
    long_emo = long_emo[long_emo.on == 1].assign(emotion=lambda d: d.emotion.str.replace("emo_", ""))
    tests = [("strategy family x condition", pm, "family"),
             ("engagement x condition", pm, "task_engagement"),
             ("jailbreaking subtype x condition", pm[pm.family == "JAILBREAKING"], "primary_strategy"),
             ("emotion (multi-label) x condition", long_emo, "emotion")]
    vrows = []
    P("\n=== Chi-square, code x condition (Cramer's V, cluster bootstrap 95% CI) ===")
    for name, d, col in tests:
        r = cluster_boot_V(d, col)
        P(f"  {name:35s} chi2({r['df']}) = {r['chi2']:.2f}, p = {r['p']:.3g}, V = {r['V']:.3f} "
          f"[{r['V_lo']:.3f}, {r['V_hi']:.3f}], N = {r['N']}")
        vrows.append(dict(test=name, chi2=r["chi2"], df=r["df"], p=r["p"], V=r["V"], V_lo=r["V_lo"],
                          V_hi=r["V_hi"], N=r["N"]))
    pd.DataFrame(vrows).to_csv(f"{OUT}/chisquare_corrected.csv", index=False)

    # exact tests for the sparse jailbreaking comparisons
    jb = pm[pm.family == "JAILBREAKING"]
    tab = pd.crosstab(jb.condition, jb.primary_strategy).values
    P(f"\n=== Jailbreaking: subtype x condition, Fisher-Freeman-Halton exact p = {ffh_exact(tab):.4f} "
      f"(counts {pd.crosstab(jb.condition, jb.primary_strategy).to_dict()})")
    any_jb = cv.assign(any_jb=cv.prop_jailbreaking > 0).groupby("condition").any_jb.agg(["sum", "count"])
    a, na = int(any_jb.loc["dark_triad", "sum"]), int(any_jb.loc["dark_triad", "count"])
    b, nb = int(any_jb.loc["servant_leader", "sum"]), int(any_jb.loc["servant_leader", "count"])
    P(f"    conversations with any jailbreaking: DT {a}/{na}, SL {b}/{nb}, Fisher exact p = "
      f"{stats.fisher_exact([[a, na - a], [b, nb - b]])[1]:.4f}")

    # length adjustment: condition effect on diversity and switching with log(n participant messages)
    import statsmodels.formula.api as smf
    q = cv.assign(DT=(cv.condition == "dark_triad").astype(int), logn=np.log(cv.total_participant_messages))
    P("\n=== Adjusted for log(number of participant messages), OLS ===")
    for v in ["strategy_diversity", "strategy_switches", "switch_rate"]:
        m = smf.ols(f"{v} ~ DT + logn", data=q).fit()
        P(f"    {v:20s} B_DT = {m.params.DT:.3f}, p = {m.pvalues.DT:.4f}, partial d = {m.params.DT / np.sqrt(m.mse_resid):.2f}")

    # message-level percentages (the DT % / SL % columns of the tables) and conversation means
    ml = {}
    for cond in ["dark_triad", "servant_leader"]:
        s = pm[pm.condition == cond]
        c = cv[cv.condition == cond]
        d = {"n_messages": len(s), "n_conversations": len(c)}
        for f in FAMILIES:
            d[f"strategy_{f.lower()}_pct_msgs"] = 100 * (s.family == f).mean()
            d[f"strategy_{f.lower()}_conv_mean"] = 100 * c[f"prop_{f.lower()}"].mean()
        for e in EMOS:
            d[f"emotion_{e.lower()}_pct_msgs"] = 100 * s[f"emo_{e.lower()}"].mean()
            d[f"emotion_{e.lower()}_conv_mean"] = 100 * c[f"prop_emotion_{e.lower()}"].mean()
        for lv in ENG:
            d[f"engagement_{lv.lower()}_pct_msgs"] = 100 * (s.task_engagement == lv).mean()
            d[f"engagement_{lv.lower()}_conv_mean"] = 100 * c[f"prop_engagement_{lv.lower()}"].mean()
        ml[cond] = d
    ml = pd.DataFrame(ml)
    ml.to_csv(f"{OUT}/message_level_percentages_corrected.csv")
    P("\n=== Message-level percentages and conversation means ===")
    P(ml.round(2).to_string())
    return res


# =============================================================================================
# 3. Supplementary Notes 5 and 6
# =============================================================================================
def note5(pm):
    q = pm[pm.prev_is_greeting_or_marker == 0]
    P(f"\n=== Note 5: participant message x the chatbot message that precedes it "
      f"(greeting excluded): n = {len(q)} messages from {q.pid.nunique()} conversations; "
      f"{int((q.prev_bot_adjacent == 0).sum())} follow another participant message and are paired with the "
      f"most recent chatbot message ===")
    out = {}
    for name, rcol, ccol, cols in [("tone_x_family", "prev_tone", "family", FAMILIES),
                                   ("response_x_engagement", "prev_response", "task_engagement", ENG)]:
        ct = pd.crosstab(q[rcol], q[ccol]).reindex(columns=cols, fill_value=0)
        ct["n"] = ct.sum(1)
        pct = 100 * ct[cols].div(ct["n"], axis=0)
        pct["n"] = ct["n"]
        ct.to_csv(f"{OUT}/note5_{name}_counts.csv")
        pct.to_csv(f"{OUT}/note5_{name}_rowpct.csv")
        P(f"\n  {name} (row %, n = messages):\n" + pct.round(1).to_string())
        out[name] = pct
        for cond in ["dark_triad", "servant_leader"]:
            s = q[q.condition == cond]
            c2 = pd.crosstab(s[rcol], s[ccol]).reindex(columns=cols, fill_value=0)
            c2["n"] = c2.sum(1)
            p2 = 100 * c2[cols].div(c2["n"], axis=0)
            p2["n"] = c2["n"]
            p2.to_csv(f"{OUT}/note5_{name}_rowpct_{cond}.csv")
    return q, out


def phase_tables(df, phase_col, len_col, label):
    rows = []
    for cond in ["dark_triad", "servant_leader"]:
        s = df[df.condition == cond]
        for ph in PHASES:
            t = s[s[phase_col] == ph]
            r = dict(rule=label, condition=cond, phase=ph, n=len(t), mean_length=t[len_col].mean())
            for f in FAMILIES:
                r[f"strategy_{f.lower()}"] = 100 * (t.family == f).mean()
            for e in EMOS:
                r[f"emotion_{e.lower()}"] = 100 * t[f"emo_{e.lower()}"].mean()
            for lv in ENG:
                r[f"engagement_{lv.lower()}"] = 100 * (t.task_engagement == lv).mean()
            rows.append(r)
    return pd.DataFrame(rows)


def note6(pm, cb):
    P("\n=== Note 6: phases. Corrected rule: k-th of n participant messages (k = 0..n-1) -> phase floor(3k/n) ===")
    cor = phase_tables(pm, "phase", "chars", "corrected: ordinal thirds, true characters")
    a = pm.copy()
    a["arch_phase"] = pd.cut(a.archived_position, bins=[0, 0.33, 0.67, 1.0], labels=PHASES).astype(str)
    arc = phase_tables(a, "arch_phase", "llm_message_length", "archived: message_number/max, LLM length")
    tl = phase_tables(a, "arch_phase", "chars", "archived phase bins, true characters")
    allp = pd.concat([cor, arc, tl])
    allp.to_csv(f"{OUT}/note6_participant_phases.csv", index=False)
    cols = [c for c in cor.columns if c not in ("rule", "condition", "phase")]
    comp = cor.set_index(["condition", "phase"])[cols].stack().rename("corrected").to_frame()
    comp["archived"] = arc.set_index(["condition", "phase"])[cols].stack()
    comp["diff"] = comp.corrected - comp.archived
    comp.to_csv(f"{OUT}/note6_participant_phases_corrected_vs_archived.csv")
    P(comp.round(1).unstack("phase").reindex(columns=PHASES, level=1).to_string())
    # chi-square per phase (strategy family, engagement) x condition
    crow = []
    for ph in PHASES:
        d = pm[pm.phase == ph]
        for col in ["family", "task_engagement"]:
            ct = pd.crosstab(d.condition, d[col])
            chi2, p, dof, _ = chi2_contingency(ct)
            V = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
            crow.append(dict(phase=ph, variable=col, chi2=chi2, df=dof, p=p, V=V, N=int(ct.values.sum())))
    crow = pd.DataFrame(crow)
    crow.to_csv(f"{OUT}/note6_phase_chisquare.csv", index=False)
    P("\n  phase x condition chi-square (corrected phases):\n" + crow.round(4).to_string())
    # conversation-level late/early ratio of mean participant message length
    cl = pm.groupby(["pid", "condition", "phase"]).chars.mean().unstack()
    cl["late_over_early"] = cl["Late"] / cl["Early"]
    P("  conversation-level median late/early ratio of mean participant message length: " +
      str(cl.groupby("condition").late_over_early.median().round(3).to_dict()))

    # chatbot persona replies
    pr = cb[cb.is_persona == 1].copy()
    pr["negative_tone"] = pr.primary_tone.isin(NEG_TONES)
    pr["positive_tone"] = pr.primary_tone.isin(POS_TONES)
    ca = cb.copy()
    ca["arch_phase"] = pd.cut(ca.archived_position, bins=[0, 0.33, 0.67, 1.0], labels=PHASES).astype(str)
    ca["negative_tone"] = ca.primary_tone.isin(NEG_TONES)
    ca["positive_tone"] = ca.primary_tone.isin(POS_TONES)
    agg = dict(n=("chars", "size"), dt_tactics=("n_dt_tactics", "mean"), sl_tactics=("n_sl_tactics", "mean"),
               nt_tactics=("n_nt_tactics", "mean"), negative_tone_pct=("negative_tone", "mean"),
               positive_tone_pct=("positive_tone", "mean"))
    c1 = pr.groupby(["condition", "phase"]).agg(mean_chars=("chars", "mean"), **agg)
    c2 = ca.groupby(["condition", "arch_phase"]).agg(mean_chars=("llm_message_length", "mean"), **agg)
    for c in (c1, c2):
        c[["negative_tone_pct", "positive_tone_pct"]] *= 100
    c1 = c1.reindex(PHASES, level=1)
    c2 = c2.reindex(PHASES, level=1)
    c1.assign(rule="corrected: persona replies, ordinal thirds, true characters").to_csv(f"{OUT}/note6_chatbot_phases.csv")
    c2.assign(rule="archived: all coded chatbot records, message_number/max, LLM length").to_csv(
        f"{OUT}/note6_chatbot_phases_archived.csv")
    P("\n  chatbot by phase, corrected:\n" + c1.round(2).to_string())
    P("  chatbot by phase, archived rule:\n" + c2.round(2).to_string())
    # escalation / de-escalation counts
    esc = pr[pr.response_to_participant.isin(["ESCALATES", "DE_ESCALATES"])]
    et = pd.crosstab([esc.condition, esc.phase], esc.response_to_participant).reindex(
        pd.MultiIndex.from_product([["dark_triad", "servant_leader"], PHASES]), fill_value=0)
    P("  ESCALATES / DE_ESCALATES persona replies by phase:\n" + et.to_string())
    return cor, c1


# =============================================================================================
# 4. Figures
# =============================================================================================
SLc, DTc = "#4C9BE0", "#E4695B"                                  # box-plot figures (layout of the earlier make_main_fig2.py)
COND_COLORS = {"dark_triad": "#E74C3C", "servant_leader": "#3498DB"}  # gpt_assessment_analysis.py figures
COND_LABEL = {"dark_triad": "Dark Triad", "servant_leader": "Servant Leader"}


def savefig(fig, name):
    fig.savefig(f"{FIGS}/{name}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    P("wrote", f"{FIGS}/{name}")


def sig_label(p):
    """Marker for an uncorrected two-sided p, as in the manuscript tables."""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def sig_bracket(ax, x1, x2, y, h, p, fontsize=10):
    """Bracket from x1 to x2 with legs starting at height y, labelled with the marker for p."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", lw=0.8)
    lab = sig_label(p)
    ax.text((x1 + x2) / 2, y + h, lab, ha="center", va="bottom",
            fontsize=fontsize if lab != "n.s." else fontsize - 1)


def box_pair(ax, x0, sl, dt, rng, s=14, s_mean=42):
    """Servant Leader box at x0 and Dark Triad box at x0 + 1, with the conversations as points and
    the mean as a black diamond."""
    for j, (data, color) in enumerate([(sl, SLc), (dt, DTc)]):
        ax.boxplot(data, positions=[x0 + j], widths=0.55, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", lw=1.4),
                   boxprops=dict(facecolor=color, alpha=.35, edgecolor=color),
                   whiskerprops=dict(color=color), capprops=dict(color=color))
        xx = np.full(len(data), x0 + j) + rng.uniform(-.11, .11, len(data))
        ax.scatter(xx, data, s=s, color=color, alpha=.75, edgecolor="white", linewidth=.4, zorder=3)
        ax.scatter([x0 + j], [np.mean(data)], marker="D", s=s_mean, color="black", zorder=4)


def mw_brackets(ax, pairs, lo, hi):
    """Brackets over box pairs marking the conversation-level Mann-Whitney test (the p of the tables).
    pairs: (x of the Servant Leader box, SL values, DT values). lo, hi: range of the plotted data;
    the y axis is extended above hi to make room, and ticks are kept within the data range."""
    span = hi - lo
    for x0, sl, dt in pairs:
        y = max(sl.max(), dt.max()) + 0.04 * span
        sig_bracket(ax, x0, x0 + 1, y, 0.025 * span, mw_z(sl, dt)[1])
    ax.set_ylim(lo - 0.03 * span, hi + 0.15 * span)
    ax.set_yticks([t for t in ax.get_yticks() if lo - 1e-9 <= t <= hi + 0.02 * span])


def fig_main02(cv):
    """main Fig. 2 (layout of the earlier make_main_fig2.py)."""
    rng = np.random.default_rng(3)

    def values(col):
        return (cv.loc[cv.condition == "servant_leader", col].dropna().values,
                cv.loc[cv.condition == "dark_triad", col].dropna().values)

    def panel(ax, specs):
        pairs, ticks = [], []
        for i, (col, lab) in enumerate(specs):
            sl, dt = values(col)
            box_pair(ax, i * 2.5, sl, dt, rng, s=13, s_mean=40)
            pairs.append((i * 2.5, sl, dt))
            ticks.append((i * 2.5 + 0.5, lab))
        ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks], fontsize=9)
        mw_brackets(ax, pairs, 0.0, float(max(max(sl.max(), dt.max()) for _, sl, dt in pairs)))

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6))
    ax = axes[0]
    panel(ax, [("prop_compliance", "Compliance"), ("prop_resistance", "Resistance"),
               ("prop_help_seeking", "Help-\nseeking")])
    ax.set_ylabel("Proportion of messages")
    ax.set_title("a  Communication strategy", fontsize=11, fontweight="bold", loc="left")

    ax = axes[1]
    panel(ax, [("prop_emotion_frustrated", "Frustrated"), ("prop_emotion_anxious", "Anxious"),
               ("prop_emotion_defensive", "Defensive")])
    ax.set_ylabel("Proportion of messages")
    ax.set_title("b  Emotional tone", fontsize=11, fontweight="bold", loc="left")

    ax = axes[2]
    panel(ax, [("strategy_diversity", "")])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Servant\nLeader", "Dark\nTriad"], fontsize=9)
    ax.set_ylabel("Shannon entropy (bits)")
    ax.set_title("c  Strategy diversity", fontsize=11, fontweight="bold", loc="left")

    ax = axes[3]
    panel(ax, [("strategy_switches", "")])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Servant\nLeader", "Dark\nTriad"], fontsize=9)
    ax.set_ylabel("Strategy switches (count)")
    ax.set_title("d  Strategy switches", fontsize=11, fontweight="bold", loc="left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    nsl, ndt = (cv.condition == "servant_leader").sum(), (cv.condition == "dark_triad").sum()
    handles = [Patch(facecolor=SLc, alpha=.5, label=f"Servant Leader (n = {nsl})"),
               Patch(facecolor=DTc, alpha=.5, label=f"Dark Triad (n = {ndt})")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    savefig(fig, "figure_main02_behavior.png")


def grid_boxes(cv, specs, title, name, ncol=3):
    """One panel per variable, Servant Leader vs Dark Triad (layout of
    regenerate_manuscript_figures.grid_fig)."""
    rng = np.random.default_rng(3)
    nrow = int(np.ceil(len(specs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 4.0 * nrow), sharey=True)
    axes = np.array(axes).reshape(-1)
    hi = float(cv[[c for c, _ in specs]].max().max())
    for ax, (col, lab) in zip(axes, specs):
        sl = cv.loc[cv.condition == "servant_leader", col].dropna().values
        dt = cv.loc[cv.condition == "dark_triad", col].dropna().values
        box_pair(ax, 0, sl, dt, rng)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Servant\nLeader", "Dark\nTriad"], fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(lab, fontsize=11, fontweight="bold")
        mw_brackets(ax, [(0, sl, dt)], 0.0, hi)
    for r in range(nrow):
        axes[r * ncol].set_ylabel("Proportion of messages")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.005)
    fig.tight_layout()
    savefig(fig, name)


def fig02(cv):
    """Overview of all six strategy families (figure02.png; not shown in the paper)."""
    grid_boxes(cv, [("prop_compliance", "Compliance"), ("prop_negotiation", "Negotiation"),
                    ("prop_help_seeking", "Help-seeking"), ("prop_resistance", "Resistance"),
                    ("prop_coping", "Coping"), ("prop_jailbreaking", "Jailbreaking")],
               "Communication strategy: conversation-level proportions by condition", "figure02.png")


def fig05(cv):
    """Supp Fig. 3: emotional tone."""
    grid_boxes(cv, [("prop_emotion_neutral", "Neutral"), ("prop_emotion_confident", "Confident"),
                    ("prop_emotion_frustrated", "Frustrated"), ("prop_emotion_anxious", "Anxious"),
                    ("prop_emotion_defensive", "Defensive"), ("prop_emotion_resigned", "Resigned")],
               "Participant emotional tone: conversation-level proportions by condition", "figure05.png")


def fig06(cv):
    """Main Fig. 3: task engagement."""
    grid_boxes(cv, [("prop_engagement_high", "High engagement"), ("prop_engagement_medium", "Medium engagement"),
                    ("prop_engagement_low", "Low engagement")],
               "Task engagement: conversation-level proportions by condition", "figure06.png")


def fig04(cv):
    """Supp Fig. 2 (plot_conversation_characteristics; common bin edges for the two conditions)."""
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        chars = [("total_participant_messages", "Participant Messages per Conv", "Participant messages per conversation"),
                 ("total_chatbot_messages", "Chatbot Messages per Conv", "Chatbot messages per conversation (incl. greeting)"),
                 ("avg_participant_msg_length", "Avg Participant Msg Length", "Mean participant message length (characters)"),
                 ("avg_chatbot_msg_length", "Avg Chatbot Msg Length", "Mean chatbot message length (characters, excl. greeting)"),
                 ("strategy_diversity", "Strategy Diversity (Entropy)", "Strategy diversity (Shannon entropy, bits)"),
                 ("strategy_switches", "Strategy Switches", "Strategy switches per conversation (count)")]
        for ax, (col, title, xlab) in zip(axes, chars):
            vals = cv[col].dropna().values
            if col in ("total_participant_messages", "total_chatbot_messages", "strategy_switches"):
                lo, hi = vals.min(), vals.max()
                step = max(1, int(np.ceil((hi - lo + 1) / 15)))
                edges = np.arange(lo - 0.5, hi + step + 0.5, step)
            else:
                edges = np.histogram_bin_edges(vals, bins=15)
            for cond in ["servant_leader", "dark_triad"]:
                data = cv.loc[cv.condition == cond, col].dropna()
                ax.hist(data, alpha=0.6, label=COND_LABEL[cond], bins=edges, color=COND_COLORS[cond])
            ax.set_xlabel(xlab, fontsize=11)
            ax.set_ylabel("Number of Conversations", fontsize=11)
            ax.set_title(title, fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        savefig(fig, "figure04.png")


def fig07(pm, cb):
    """Supp Fig. 4 (plot_interaction_dynamics) with the corrected pairing."""
    q = pm[pm.prev_is_greeting_or_marker == 0].copy()
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        # 1. participant strategy family by preceding chatbot tone
        ct = pd.crosstab(q.prev_tone, q.family).reindex(columns=[f for f in sorted(FAMILIES)], fill_value=0)
        n = ct.sum(1)
        pct = ct.div(n, axis=0) * 100
        pct.plot(kind="bar", ax=axes[0, 0], colormap="tab10")
        axes[0, 0].set_xticklabels([f"{t} (n = {n[t]})" for t in pct.index], rotation=40, ha="right")
        axes[0, 0].set_xlabel("Tone of the preceding chatbot message", fontsize=12)
        axes[0, 0].set_ylabel("Percentage of Participant Strategies (%)", fontsize=12)
        axes[0, 0].set_title("Participant Strategy by Preceding Chatbot Tone", fontsize=14, fontweight="bold")
        axes[0, 0].legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
        axes[0, 0].grid(axis="y", alpha=0.3)
        # 2. engagement by preceding chatbot response type
        ct = pd.crosstab(q.prev_response, q.task_engagement).reindex(columns=ENG, fill_value=0)
        n = ct.sum(1)
        pct = ct.div(n, axis=0) * 100
        eng_colors = {"HIGH": "#2ECC71", "MEDIUM": "#F39C12", "LOW": "#E74C3C"}
        pct[ENG].plot(kind="bar", ax=axes[0, 1], color=[eng_colors[e] for e in ENG])
        axes[0, 1].set_xticklabels([f"{t} (n = {n[t]})" for t in pct.index], rotation=40, ha="right")
        axes[0, 1].set_xlabel("Response type of the preceding chatbot message", fontsize=12)
        axes[0, 1].set_ylabel("Percentage of Engagement Levels (%)", fontsize=12)
        axes[0, 1].set_title("Participant Engagement by Preceding Chatbot Response", fontsize=14, fontweight="bold")
        axes[0, 1].legend(title="Engagement")
        axes[0, 1].grid(axis="y", alpha=0.3)
        # 3. length ratio over conversation position (deciles of each conversation's participant messages)
        q["ratio"] = q.chars / q.prev_bot_chars
        q["decile"] = np.floor(10 * q.ordinal / q.n_in_conv).astype(int)
        conv_dec = q.groupby(["condition", "pid", "decile"]).ratio.mean().reset_index()
        x = np.arange(10)
        for cond in ["servant_leader", "dark_triad"]:
            g = conv_dec[conv_dec.condition == cond].groupby("decile").ratio
            m, se, k = g.mean().reindex(x), g.sem().reindex(x), g.size().reindex(x)
            axes[1, 0].plot(x, m.values, marker="o", markersize=4, label=f"{COND_LABEL[cond]}",
                            color=COND_COLORS[cond], alpha=0.85)
            axes[1, 0].fill_between(x, (m - se).values, (m + se).values, color=COND_COLORS[cond], alpha=0.12)
        axes[1, 0].axhline(y=1, color="gray", linestyle="--", alpha=0.5, label="Equal length")
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels([f"{10*i}–{10*(i+1)}%" for i in x], rotation=30)
        axes[1, 0].set_xlabel("Position in conversation (decile of the participant's messages)", fontsize=12)
        axes[1, 0].set_ylabel("Participant / preceding chatbot message length (characters)", fontsize=12)
        axes[1, 0].set_title("Message Length Ratio Over Conversation", fontsize=14, fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        # 4. escalation / de-escalation persona replies by phase
        pr = cb[cb.is_persona == 1]
        ax = axes[1, 1]
        for cond in ["dark_triad", "servant_leader"]:
            s = pr[pr.condition == cond]
            for rsp, mk, ls, al in [("ESCALATES", "^", "-", 1.0), ("DE_ESCALATES", "v", "--", 0.6)]:
                cnt = [int(((s.phase == ph) & (s.response_to_participant == rsp)).sum()) for ph in PHASES]
                ax.plot(PHASES, cnt, marker=mk, markersize=8, linestyle=ls, linewidth=2, alpha=al,
                        label=f"{COND_LABEL[cond]} - {rsp.replace('_', '-').title()}", color=COND_COLORS[cond])
                for xi, c in enumerate(cnt):
                    if c > 0:
                        ax.annotate(str(c), (xi, c), textcoords="offset points",
                                    xytext=(0, 8) if cond == "dark_triad" else (11, 3), ha="center",
                                    fontsize=9, color=COND_COLORS[cond])
        nrep = pr.groupby(["condition", "phase"]).size()
        ax.set_ylim(-2, ax.get_ylim()[1] * 1.12)
        ax.text(0.35, 0.62, "Chatbot persona replies per phase (E / M / L):\n" +
                "\n".join(f"{COND_LABEL[c]}: " + " / ".join(str(nrep.get((c, ph), 0)) for ph in PHASES)
                          for c in ["dark_triad", "servant_leader"]),
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set_xlabel("Conversation Phase", fontsize=12)
        ax.set_ylabel("Frequency (chatbot replies)", fontsize=12)
        ax.set_title("Escalation Patterns Over Conversation", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        savefig(fig, "figure07.png")


def fig08(pm):
    """Supp Fig. 5 (plot_temporal_patterns) with ordinal-third phases and true lengths."""
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for i, cond in enumerate(["dark_triad", "servant_leader"]):
            s = pm[pm.condition == cond]
            st = s.groupby(["phase", "family"]).size().unstack(fill_value=0).reindex(PHASES)
            st = st.reindex(columns=sorted(FAMILIES), fill_value=0)
            pct = st.div(st.sum(axis=1), axis=0) * 100
            pct.plot(kind="bar", stacked=True, ax=axes[0, i], colormap="tab10", legend=(i == 1))
            axes[0, i].set_title(f"Strategy Evolution - {COND_LABEL[cond]}", fontsize=14, fontweight="bold")
            axes[0, i].set_xlabel("Conversation Phase", fontsize=12)
            axes[0, i].set_ylabel("Percentage (%)", fontsize=12)
            axes[0, i].set_xticklabels([f"{ph}\n(n = {st.sum(1)[ph]})" for ph in PHASES], rotation=0)
            if i == 1:
                axes[0, i].legend(title="Strategy", bbox_to_anchor=(1.05, 1), loc="upper left")
        for cond in ["servant_leader", "dark_triad"]:
            s = pm[pm.condition == cond]
            pl = s.groupby("phase").chars.mean().reindex(PHASES)
            axes[1, 0].plot(pl.index, pl.values, marker="o", linewidth=2, markersize=8,
                            label=COND_LABEL[cond], color=COND_COLORS[cond])
            for xi, v in enumerate(pl.values):
                axes[1, 0].annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(8, 6), fontsize=9,
                                    color=COND_COLORS[cond])
        axes[1, 0].set_xlabel("Conversation Phase", fontsize=12)
        axes[1, 0].set_ylabel("Average Message Length (characters)", fontsize=12)
        axes[1, 0].set_title("Participant Message Length Evolution", fontsize=14, fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        for cond in ["servant_leader", "dark_triad"]:
            s = pm[pm.condition == cond]
            et = s.groupby(["phase", "task_engagement"]).size().unstack(fill_value=0).reindex(PHASES)
            ep = et.div(et.sum(axis=1), axis=0) * 100
            axes[1, 1].plot(ep.index, ep["HIGH"], marker="o", linewidth=2, markersize=8,
                            label=f"{COND_LABEL[cond]} - HIGH", color=COND_COLORS[cond])
            axes[1, 1].plot(ep.index, ep["LOW"], marker="s", linewidth=2, markersize=8, linestyle="--",
                            label=f"{COND_LABEL[cond]} - LOW", color=COND_COLORS[cond], alpha=0.6)
        axes[1, 1].set_xlabel("Conversation Phase", fontsize=12)
        axes[1, 1].set_ylabel("Percentage (%)", fontsize=12)
        axes[1, 1].set_title("Engagement Evolution (HIGH vs LOW)", fontsize=14, fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        plt.tight_layout()
        savefig(fig, "figure08.png")


def fig09(cb):
    """Supp Fig. 6 (plot_temporal_chatbot_patterns): persona replies, ordinal thirds, true lengths."""
    pr = cb[cb.is_persona == 1].copy()
    pr["negative_tone"] = pr.primary_tone.isin(NEG_TONES)
    pr["positive_tone"] = pr.primary_tone.isin(POS_TONES)
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        for i, cond in enumerate(["dark_triad", "servant_leader"]):
            s = pr[pr.condition == cond]
            tt = s.groupby("phase")[["n_dt_tactics", "n_sl_tactics", "n_nt_tactics"]].mean().reindex(PHASES)
            tt.columns = ["Dark Triad", "Servant", "Neutral"]
            tt.plot(kind="bar", ax=axes[0, i], color=["#E74C3C", "#3498DB", "#95A5A6"])
            nph = s.groupby("phase").size().reindex(PHASES)
            axes[0, i].set_xticklabels([f"{ph}\n(n = {nph[ph]})" for ph in PHASES], rotation=0)
            axes[0, i].set_xlabel("Conversation Phase", fontsize=12)
            axes[0, i].set_ylabel("Avg Tactics per Message", fontsize=12)
            axes[0, i].set_title(f"Tactic Evolution - {COND_LABEL[cond]}", fontsize=14, fontweight="bold")
            axes[0, i].legend(title="Tactic Type")
            axes[0, i].grid(axis="y", alpha=0.3)
        for cond in ["servant_leader", "dark_triad"]:
            s = pr[pr.condition == cond]
            tb = s.groupby("phase")[["negative_tone", "positive_tone"]].mean().reindex(PHASES) * 100
            axes[1, 0].plot(tb.index, tb.negative_tone, marker="s", markersize=8, linestyle="-", linewidth=2,
                            label=f"{COND_LABEL[cond]} - Negative", color=COND_COLORS[cond])
            axes[1, 0].plot(tb.index, tb.positive_tone, marker="o", markersize=8, linestyle="--", linewidth=2,
                            label=f"{COND_LABEL[cond]} - Positive", color=COND_COLORS[cond], alpha=0.6)
        axes[1, 0].set_xlabel("Conversation Phase", fontsize=12)
        axes[1, 0].set_ylabel("Percentage (%)", fontsize=12)
        axes[1, 0].set_title("Tone Valence Evolution", fontsize=14, fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        for cond in ["servant_leader", "dark_triad"]:
            s = pr[pr.condition == cond]
            lb = s.groupby("phase").chars.mean().reindex(PHASES)
            axes[1, 1].plot(lb.index, lb.values, marker="o", markersize=8, linewidth=2,
                            label=COND_LABEL[cond], color=COND_COLORS[cond])
            for xi, v in enumerate(lb.values):
                axes[1, 1].annotate(f"{v:.0f}", (xi, v), textcoords="offset points", xytext=(8, 6), fontsize=9,
                                    color=COND_COLORS[cond])
        axes[1, 1].set_xlabel("Conversation Phase", fontsize=12)
        axes[1, 1].set_ylabel("Average Message Length (characters)", fontsize=12)
        axes[1, 1].set_title("Chatbot Message Length Evolution", fontsize=14, fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        plt.tight_layout()
        savefig(fig, "figure09.png")


CORR_COLS = [("total_participant_messages", "Total Participant Messages"),
             ("total_chatbot_messages", "Total Chatbot Messages"),
             ("avg_participant_msg_length", "Avg Participant Msg Length"),
             ("avg_chatbot_msg_length", "Avg Chatbot Msg Length"),
             ("strategy_diversity", "Strategy Diversity"),
             ("strategy_switches", "Strategy Switches"),
             ("prop_compliance", "Compliance"), ("prop_resistance", "Resistance"),
             ("prop_jailbreaking", "Jailbreaking"), ("prop_negotiation", "Negotiation"),
             ("prop_emotion_frustrated", "Emotion Frustrated"), ("prop_emotion_anxious", "Emotion Anxious"),
             ("prop_emotion_confident", "Emotion Confident"),
             ("prop_engagement_high", "Engagement High"), ("prop_engagement_low", "Engagement Low"),
             ("prop_dark_triad_tactics", "Dark Triad Tactics"), ("prop_servant_tactics", "Servant Tactics"),
             ("prop_tone_hostile", "Tone Hostile"), ("prop_tone_supportive", "Tone Supportive"),
             ("prop_response_validates", "Response Validates"), ("prop_response_rejects", "Response Rejects"),
             ("prop_response_escalates", "Response Escalates")]


def fig10(cv):
    """Supp Fig. 7 (plot_correlation_matrix), Pearson r over the 58 conversations."""
    cols = [c for c, _ in CORR_COLS]
    corr = cv[cols].corr()
    corr.to_csv(f"{OUT}/correlation_matrix_corrected.csv")
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, ax = plt.subplots(figsize=(18, 16))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5,
                    cbar_kws={"shrink": 0.8}, ax=ax, vmin=-1, vmax=1, annot_kws={"size": 8})
        labels = [lab for _, lab in CORR_COLS]
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, rotation=0, fontsize=9)
        ax.set_title("Correlation Matrix (Participant + Chatbot Features)", fontsize=16, fontweight="bold", pad=20)
        plt.tight_layout()
        savefig(fig, "figure10.png")
    return corr


def fig03(pm, cv):
    """Supplementary Fig. 1: jailbreaking (gpt_assessment_analysis.py layout; phases = ordinal thirds)."""
    jb = pm[pm.family == "JAILBREAKING"]
    order = ["dark_triad", "servant_leader"]
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        counts = jb.groupby("condition").size().reindex(order, fill_value=0)
        axes[0, 0].bar([COND_LABEL[c] for c in order], counts.values, color=[COND_COLORS[c] for c in order])
        for i, v in enumerate(counts.values):
            axes[0, 0].text(i, v, f"{v}", ha="center", va="bottom", fontsize=11)
        axes[0, 0].set_xlabel("Condition", fontsize=12)
        axes[0, 0].set_ylabel("Number of Jailbreaking Messages", fontsize=12)
        axes[0, 0].set_title("Jailbreaking Messages by Condition", fontsize=14, fontweight="bold")
        n_conv = cv.groupby("condition").size().reindex(order)
        with_jb = jb.groupby("condition")["conversation_id"].nunique().reindex(order, fill_value=0)
        pct = 100 * with_jb / n_conv
        axes[0, 1].bar([COND_LABEL[c] for c in order], pct.values, color=[COND_COLORS[c] for c in order])
        for i, c in enumerate(order):
            axes[0, 1].text(i, pct[c], f"{with_jb[c]} of {n_conv[c]}", ha="center", va="bottom", fontsize=11)
        # Fisher's exact test, conversations with any jailbreaking (DT row first)
        p_any = stats.fisher_exact([[with_jb[c], n_conv[c] - with_jb[c]] for c in order])[1]
        sig_bracket(axes[0, 1], 0, 1, 1.12 * pct.max(), 0.03 * pct.max(), p_any, fontsize=12)
        axes[0, 1].set_ylim(0, 1.3 * pct.max())
        axes[0, 1].set_xlabel("Condition", fontsize=12)
        axes[0, 1].set_ylabel("Percentage of Conversations (%)", fontsize=12)
        axes[0, 1].set_title("Conversations with Jailbreaking", fontsize=14, fontweight="bold")
        sub = (jb.groupby(["primary_strategy", "condition"]).size().unstack(fill_value=0)
               .reindex(columns=order, fill_value=0))
        sub = sub.loc[sub.sum(axis=1).sort_values().index]
        y = np.arange(len(sub))
        for j, c in enumerate(order):
            axes[1, 0].barh(y + (j - 0.5) * 0.38, sub[c].values, height=0.38, color=COND_COLORS[c],
                            label=COND_LABEL[c])
        axes[1, 0].set_yticks(y)
        axes[1, 0].set_yticklabels([t.replace("JAILBREAKING_", "").replace("_", " ").title() for t in sub.index])
        axes[1, 0].set_xlabel("Number of Messages", fontsize=12)
        axes[1, 0].set_title("Types of Jailbreaking", fontsize=14, fontweight="bold")
        axes[1, 0].legend()
        ph = (jb.groupby(["condition", "phase"]).size().unstack(fill_value=0)
              .reindex(index=order, columns=PHASES, fill_value=0))
        x = np.arange(len(order))
        for k, (lab, col) in enumerate(zip(PHASES, ["#3498DB", "#F39C12", "#E74C3C"])):
            axes[1, 1].bar(x + (k - 1) * 0.25, ph[lab].values, width=0.25, color=col, label=lab)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([COND_LABEL[c] for c in order])
        axes[1, 1].set_xlabel("Condition", fontsize=12)
        axes[1, 1].set_ylabel("Number of Messages", fontsize=12)
        axes[1, 1].set_title("Timing of Jailbreaking", fontsize=14, fontweight="bold")
        axes[1, 1].legend(title="Phase")
        fig.tight_layout()
        savefig(fig, "figure03.png")
    P(f"Supp Fig 1: jailbreaking messages {counts.to_dict()}; conversations {with_jb.to_dict()}; "
      f"by phase {ph.to_dict(orient='index')}")


def fig11(cv):
    """Supplementary Fig. 8: dyadic scatter plots (gpt_assessment_analysis.py layout). Chatbot shares are
    over persona replies; participant frustration is the share of messages tagged FRUSTRATED."""
    panels = [("prop_servant_tactics", "prop_compliance", "Chatbot Servant Leader Tactics (share of replies)",
               "Participant Compliance Proportion", "Compliance vs Supportive Chatbot Behavior"),
              ("prop_dark_triad_tactics", "prop_emotion_frustrated", "Chatbot Dark Triad Tactics (share of replies)",
               "Participant Frustration (share of messages)", "Participant Frustration vs Hostile Chatbot Behavior"),
              ("prop_response_rejects", "prop_resistance", "Chatbot Rejection (share of replies)",
               "Participant Resistance Proportion", "Participant Resistance vs Chatbot Rejection"),
              ("prop_response_validates", "prop_engagement_high", "Chatbot Validation (share of replies)",
               "Participant High Engagement Proportion", "Participant Engagement vs Chatbot Validation")]
    with plt.style.context("seaborn-v0_8-darkgrid"):
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        for ax, (xc, yc, xl, yl, title) in zip(axes.ravel(), panels):
            for cond in ["servant_leader", "dark_triad"]:
                d = cv[cv.condition == cond]
                ax.scatter(d[xc], d[yc], alpha=0.6, s=80, color=COND_COLORS[cond], label=COND_LABEL[cond])
            dt = cv[cv.condition == "dark_triad"]
            r = np.corrcoef(dt[xc], dt[yc])[0, 1]
            P(f"Supp Fig 8 {title}: r within Dark Triad = {r:.2f}; pooled = {np.corrcoef(cv[xc], cv[yc])[0, 1]:.2f}")
            ax.set_xlabel(xl, fontsize=12)
            ax.set_ylabel(yl, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.text(0.98, 0.02, f"r within Dark Triad = {r:.2f}", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10)
            ax.legend()
        fig.tight_layout()
        savefig(fig, "figure11.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIGS, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 300)
    pm, cb, cv, an = build_tables()
    pm.to_csv(f"{OUT}/participant_messages_corrected.csv", index=False)
    cb.to_csv(f"{OUT}/chatbot_messages_corrected.csv", index=False)
    cv.to_csv(f"{OUT}/conversation_metrics_corrected.csv", index=False)
    an.to_csv(f"{OUT}/anchoring_check.csv", index=False)
    P(f"conversations {len(cv)} (DT {int((cv.condition == 'dark_triad').sum())}, "
      f"SL {int((cv.condition == 'servant_leader').sum())}); participant messages {len(pm)}; "
      f"coded chatbot records {len(cb)} (persona replies {int(cb.is_persona.sum())}, greetings "
      f"{int(cb.is_greeting.sum())}, system markers {int((cb.role == 'system').sum())})")
    P(f"anchoring: participant one-to-one in {int(an.user_one_to_one.sum())}/58 and in list order in "
      f"{int(an.user_in_list_order.sum())}/58; chatbot one-to-one in {int(an.bot_one_to_one.sum())}/58 and in "
      f"list order in {int(an.bot_in_list_order.sum())}/58; methods participant "
      f"{pm.anchor_method.value_counts().to_dict()}, chatbot {cb.anchor_method.value_counts().to_dict()}")
    P(f"condition: log metadata vs coding model's inference agree in "
      f"{int((cv.condition == cv.llm_condition).sum())}/58")

    res = run_tests(cv, pm)
    note5(pm)
    note6(pm, cb)
    fig_main02(cv)
    fig02(cv)
    fig04(cv)
    fig05(cv)
    fig06(cv)
    fig07(pm, cb)
    fig08(pm)
    fig09(cb)
    fig10(cv)
    fig03(pm, cv)
    fig11(cv)
    with open(f"{OUT}/behavior_corrected_log.txt", "w") as fh:
        fh.write("\n".join(LOG_LINES) + "\n")


if __name__ == "__main__":
    main()
