#!/usr/bin/env python
"""
Participant-level event-related analysis, step 3 of 3: figures.

Main-text Figure 4 (EDA) and Supplementary Figures 9 and 10 (corrugator, zygomaticus), drawn
from the outputs of participant_level_stats.py. Every band is a participant-level 95%
confidence interval: group curves are the mean of participant mean waveforms
+/- t(0.975, n-1) x SE across participants, and the difference wave carries a Welch interval.
Clusters that survive the cluster-based permutation test are shaded (none survived in the
reported data). PNGs are written to $STUDY_DATA_ROOT/figures/.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
RES = os.path.join(DATA_ROOT, "event_related_participant_level", "results")
FIGS = os.path.join(DATA_ROOT, "figures")
DT_C, SL_C, DIFF_C = "#C0392B", "#2E86C1", "#6C3483"


def group_band(W):
    m = W.mean(0)
    h = stats.t.ppf(0.975, len(W) - 1) * W.std(0, ddof=1) / np.sqrt(len(W))
    return m, m - h, m + h


def diff_band(A, B):
    """Dark Triad (A) minus Servant Leader (B), Welch 95% CI at each timepoint."""
    va, vb = A.var(0, ddof=1) / len(A), B.var(0, ddof=1) / len(B)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(A) - 1) + vb ** 2 / (len(B) - 1))
    d = A.mean(0) - B.mean(0)
    h = stats.t.ppf(0.975, df) * se
    return d, d - h, d + h


def pointwise_d(A, B):
    sp = np.sqrt(((len(A) - 1) * A.var(0, ddof=1) + (len(B) - 1) * B.var(0, ddof=1)) / (len(A) + len(B) - 2))
    return (A.mean(0) - B.mean(0)) / sp


def sig_clusters(cl, channel):
    c = cl[(cl.channel == channel) & (cl.p < 0.05)]
    return list(zip(c.start, c.end))


def eda_figure(wv, cl):
    t, W, dt = wv["eda_t"], wv["eda_W"], wv["is_dt"]
    A, B = W[dt], W[~dt]
    spans = sig_clusters(cl, "EDA")
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    def mark(a):
        a.axvline(0, color="0.35", ls="--", lw=1.2)
        for s, e in spans:
            a.axvspan(s, e, color="0.85", alpha=.55, zorder=0)

    a = ax[0]
    for X, c, lab in ((A, DT_C, f"Dark Triad (n = {len(A)})"), (B, SL_C, f"Servant Leader (n = {len(B)})")):
        m, lo, hi = group_band(X)
        a.plot(t, m, color=c, lw=2, label=lab)
        a.fill_between(t, lo, hi, color=c, alpha=.20, lw=0)
    mark(a)
    a.set_ylabel("Skin conductance change (μS)")
    a.legend(frameon=False, fontsize=9, loc="upper right")
    a.set_title("a", loc="left", fontweight="bold", fontsize=13)

    a = ax[1]
    d, lo, hi = diff_band(A, B)
    a.axhline(0, color="0.4", lw=1)
    a.plot(t, d, color=DIFF_C, lw=2)
    a.fill_between(t, lo, hi, color=DIFF_C, alpha=.22, lw=0, label="95% CI")
    mark(a)
    a.set_ylabel("Δ skin conductance (μS)\nDark Triad − Servant Leader")
    a.legend(frameon=False, fontsize=9, loc="upper right")
    a.set_title("b", loc="left", fontweight="bold", fontsize=13)

    a = ax[2]
    a.axhline(0, color="0.4", lw=1)
    for y in (.2, .5, .8):
        a.axhline(y, color="0.7", ls=":", lw=1)
    a.plot(t, pointwise_d(A, B), color=DIFF_C, lw=2)
    mark(a)
    a.set_ylabel("Cohen's d\nDark Triad − Servant Leader")
    a.set_title("c", loc="left", fontweight="bold", fontsize=13)

    for a_ in ax:
        a_.set_xlabel("Time from message onset (s)")
        a_.set_xlim(t[0], t[-1])
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig4_eda_event_related.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    dd = pointwise_d(A, B)
    print(f"Fig 4: max point-wise d = {dd.max():.2f} at {t[dd.argmax()]:.2f} s; clusters drawn: {spans}")


def emg_figure(wv, cl, tests, lab, muscle, out):
    t, W, dt = wv[f"{lab}_t"], wv[f"{lab}_W"], wv["is_dt"]
    A, B = W[dt], W[~dt]
    spans = sig_clusters(cl, lab)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    a = ax[0]
    for X, c, nm in ((A, DT_C, f"Dark Triad (n = {len(A)})"), (B, SL_C, f"Servant Leader (n = {len(B)})")):
        m, lo, hi = group_band(X)
        a.plot(t, m, color=c, lw=1.6, label=nm)
        a.fill_between(t, lo, hi, color=c, alpha=.20, lw=0)
    a.set_ylabel("EMG amplitude change (μV RMS)")
    a.legend(frameon=False, fontsize=9, loc="upper right")
    a.set_title("a", loc="left", fontweight="bold", fontsize=13)

    b = ax[1]
    d, lo, hi = diff_band(A, B)
    b.axhline(0, color="0.4", lw=1)
    b.plot(t, d, color=DIFF_C, lw=1.6)
    b.fill_between(t, lo, hi, color=DIFF_C, alpha=.22, lw=0, label="95% CI")
    b.set_ylabel("Δ EMG amplitude (μV)\nDark Triad − Servant Leader")
    b.legend(frameon=False, fontsize=9, loc="upper right")
    b.set_title("b", loc="left", fontweight="bold", fontsize=13)
    for a_ in (ax[0], ax[1]):
        a_.axvline(0, color="0.35", ls="--", lw=1.2)
        for s, e in spans:
            a_.axvspan(s, e, color="0.85", alpha=.55, zorder=0)
        a_.set_xlabel("Time from message onset (s)")
        a_.set_xlim(t[0], t[-1])

    c = ax[2]
    names = [("peak change 0-6 s", "Peak\n(0–6 s)"), ("mean change 0-1 s", "Early\n(0–1 s)"),
             ("mean change 1-3 s", "Mid\n(1–3 s)"), ("mean change 3-6 s", "Late\n(3–6 s)"),
             ("mean change 6-13 s", "Sustained\n(6–13 s)"), ("peak change / resting level", "Peak /\nrest")]
    rows = [tests[tests.measure == f"{lab} {k}"].iloc[0] for k, _ in names]
    y = np.arange(len(rows))[::-1]
    dv = np.array([r.d for r in rows])
    c.errorbar(dv, y, xerr=[dv - np.array([r.d_lo for r in rows]), np.array([r.d_hi for r in rows]) - dv],
               fmt="o", color=DIFF_C, capsize=4, lw=1.4)
    c.axvline(0, color="0.4", lw=1)
    c.set_yticks(y)
    c.set_yticklabels([n for _, n in names])
    c.set_xlim(-1.3, 1.3)
    c.set_xlabel("Cohen's d with 95% CI, participant level\nDark Triad − Servant Leader")
    c.set_title("c", loc="left", fontweight="bold", fontsize=13)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{muscle}: d = " + ", ".join(f"{n.split()[0]} {v:+.2f}" for (_, n), v in zip(names, dv)) + f"; clusters drawn: {spans}")


def main():
    os.makedirs(FIGS, exist_ok=True)
    wv = dict(np.load(f"{RES}/physio_participant_waveforms.npz", allow_pickle=True))
    cl = pd.read_csv(f"{RES}/physio_participant_clusters.csv")
    tests = pd.read_csv(f"{RES}/physio_participant_tests.csv")
    eda_figure(wv, cl)
    emg_figure(wv, cl, tests, "corrugator", "corrugator supercilii", os.path.join(FIGS, "suppfig9_corrugator_event_related.png"))
    emg_figure(wv, cl, tests, "zygomaticus", "zygomaticus major", os.path.join(FIGS, "suppfig10_zygomaticus_event_related.png"))


if __name__ == "__main__":
    main()
