#!/usr/bin/env python
"""
Participant-level event-related analysis, step 2 of 3: statistics.

Every test uses the participant as the unit of analysis: epochs enter only through each
participant's mean waveform, or, in the mixed models, with a participant random intercept.

  EDA   peak and AUC (0-18 s) with BCa bootstrap CIs for the difference and the ratio;
        window means (0-3, 3-6, 6-12, 12-18 s); a whole-window permutation test and a
        cluster-based permutation test (Maris & Oostenveld, 2007) with condition labels
        permuted across participants; a linear mixed model of single-epoch peaks with a
        participant random intercept; sensitivity analyses (non-overlapping epochs,
        reply-stream duration, epoch count); skin conductance level in microsiemens;
        tonic/phasic decomposition.
  fEMG  the same waveform tests for corrugator and zygomaticus; absolute pre-onset level;
        resting level from the eyes-closed recording; responses relative to rest; ANCOVA
        of the absolute post-onset level on the absolute pre-onset level. Epochs whose mean
        envelope exceeds the participant's median by more than 5 MAD are excluded (primary
        analysis); every fEMG result is also computed with all epochs.

Outputs in $STUDY_DATA_ROOT/event_related_participant_level/results/: physio_participant_tests.csv,
physio_participant_clusters.csv, physio_participant_waveforms.npz, physio_participant_summary.txt.
Seed 20261006; 10,000 permutations; BCa with 10,000 resamples.
"""
import glob
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from scipy.stats import nct
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
ROOT = os.path.join(DATA_ROOT, "event_related_participant_level")
IN = os.path.join(ROOT, "participants")
RES = os.path.join(ROOT, "results")
SEED, NPERM, NBOOT = 20261006, 10000, 10000
FSD = 100.0
LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def taxis(pre, n):
    return (np.arange(n) + 0.5) / FSD - pre


def cohen_d(sl, dt):
    n1, n2 = len(sl), len(dt)
    sp = np.sqrt(((n1 - 1) * np.var(sl, ddof=1) + (n2 - 1) * np.var(dt, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(dt) - np.mean(sl)) / sp


def d_ci(d, n1, n2):
    df, a = n1 + n2 - 2, 1 / n1 + 1 / n2
    t = d / np.sqrt(a)
    from scipy.optimize import brentq
    try:
        lo = brentq(lambda x: nct.cdf(t, df, x) - 0.975, t - 10, t + 10)
        hi = brentq(lambda x: nct.cdf(t, df, x) - 0.025, t - 10, t + 10)
        return lo * np.sqrt(a), hi * np.sqrt(a)
    except ValueError:                       # noncentral t unstable at large |t|: normal approximation
        se = np.sqrt(a + d ** 2 / (2 * (n1 + n2)))
        return d - 1.96 * se, d + 1.96 * se


def bca(sl, dt, fn, rng):
    th = fn(sl, dt)
    bo = np.array([fn(sl[rng.integers(0, len(sl), len(sl))], dt[rng.integers(0, len(dt), len(dt))])
                   for _ in range(NBOOT)])
    z0 = stats.norm.ppf((np.sum(bo < th) + 0.5) / (NBOOT + 1))
    jk = np.array([fn(np.delete(sl, i), dt) for i in range(len(sl))] +
                  [fn(sl, np.delete(dt, j)) for j in range(len(dt))])
    m = jk.mean()
    acc = np.sum((m - jk) ** 3) / (6 * np.sum((m - jk) ** 2) ** 1.5)
    q = [stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))) for z in stats.norm.ppf([0.025, 0.975])]
    return np.quantile(bo, q)


def compare(name, v, is_dt, unit="", rng=None, ci=False):
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    sl, dt = v[ok & ~is_dt], v[ok & is_dt]
    d = cohen_d(sl, dt)
    dlo, dhi = d_ci(d, len(sl), len(dt))
    _, pw = stats.ttest_ind(dt, sl, equal_var=False)
    _, pm = stats.mannwhitneyu(dt, sl, alternative="two-sided")
    row = dict(measure=name, unit=unit, n_SL=len(sl), n_DT=len(dt), M_SL=sl.mean(), SD_SL=sl.std(ddof=1),
               M_DT=dt.mean(), SD_DT=dt.std(ddof=1), d=d, d_lo=dlo, d_hi=dhi, welch_p=pw, mw_p=pm)
    if ci:
        row["diff"] = dt.mean() - sl.mean()
        row["diff_lo"], row["diff_hi"] = bca(sl, dt, lambda x, y: y.mean() - x.mean(), rng)
        row["ratio"] = dt.mean() / sl.mean()
        row["ratio_lo"], row["ratio_hi"] = bca(sl, dt, lambda x, y: y.mean() / x.mean(), rng)
    say(f"  {name:52s} SL {sl.mean():9.4g}  DT {dt.mean():9.4g} {unit:6s} d={d:+.2f} [{dlo:+.2f},{dhi:+.2f}]"
        f"  Welch p={pw:.3f}  MW p={pm:.3f}  (n={len(sl)}/{len(dt)})")
    if ci:
        say(f"  {'':52s} diff {row['diff']:+.4g} [{row['diff_lo']:+.4g}, {row['diff_hi']:+.4g}]  "
            f"ratio {row['ratio']:.3f} [{row['ratio_lo']:.3f}, {row['ratio_hi']:.3f}]")
    return row


def cluster_test(W, t, is_dt, rng, window):
    """Maris & Oostenveld cluster-based permutation over participant mean waveforms.
    Cluster-forming threshold: |t| > t_crit(.975, df); cluster statistic: sum of t;
    null: maximum |cluster mass| under permutation of condition labels across participants."""
    m = (t >= window[0]) & (t <= window[1])
    X, tt = W[:, m], t[m]
    df = len(X) - 2
    tc = stats.t.ppf(0.975, df)

    def clusters(lab):
        tv = stats.ttest_ind(X[lab], X[~lab]).statistic
        out = []
        for sgn in (1, -1):
            above = sgn * tv > tc
            i = 0
            while i < len(tv):
                if above[i]:
                    j = i
                    while j + 1 < len(tv) and above[j + 1]:
                        j += 1
                    out.append((float(tv[i:j + 1].sum()), i, j))
                    i = j + 1
                else:
                    i += 1
        return out

    obs = clusters(is_dt)
    null = np.empty(NPERM)
    for b in range(NPERM):
        c = clusters(rng.permutation(is_dt))
        null[b] = max([abs(x[0]) for x in c], default=0.0)
    rows = [dict(mass=s, start=tt[i], end=tt[j], direction="DT>SL" if s > 0 else "SL>DT",
                 p=(np.sum(null >= abs(s)) + 1) / (NPERM + 1)) for s, i, j in obs]
    return sorted(rows, key=lambda r: r["p"]), tc


def omnibus(W, t, is_dt, rng, window):
    m = (t >= window[0]) & (t <= window[1])
    X = W[:, m]
    stat = lambda lab: np.mean(np.abs(X[lab].mean(0) - X[~lab].mean(0)))
    obs = stat(is_dt)
    null = np.array([stat(rng.permutation(is_dt)) for _ in range(NPERM)])
    return (np.sum(null >= obs) + 1) / (NPERM + 1)


def outlier_mask(E):
    """Epoch-level outliers: epoch mean envelope > participant median + 5 x MAD."""
    mu = E.mean(1)
    med = np.median(mu)
    mad = 1.4826 * np.median(np.abs(mu - med))
    return mu <= med + 5 * mad if mad > 0 else np.ones(len(mu), bool)


def main():
    os.makedirs(RES, exist_ok=True)
    rng = np.random.default_rng(SEED)
    P = [dict(np.load(f, allow_pickle=True)) for f in sorted(glob.glob(f"{IN}/*.npz"))]
    pid = np.array([str(p["pid"]) for p in P])
    is_dt = np.array([str(p["cond"]) == "DT" for p in P])
    say(f"participants: {len(P)} (SL {np.sum(~is_dt)}, DT {np.sum(is_dt)})")
    rows, clus, wave = [], [], {}

    # =============================== EDA =====================================
    nE = P[0]["eda_epochs"].shape[1]
    tE = taxis(2.0, nE)
    post = (tE >= 0) & (tE <= 18)
    W = np.array([p["eda_epochs"].mean(0) for p in P], float)
    n_ep = np.array([len(p["eda_kept"]) for p in P])
    say(f"\nEDA epochs: SL {n_ep[~is_dt].sum()}, DT {n_ep[is_dt].sum()}")
    wave["eda_t"], wave["eda_W"], wave["is_dt"], wave["pid"] = tE, W, is_dt, pid
    peak = W[:, post].max(1)
    auc = np.trapezoid(W[:, post], tE[post], axis=1)
    say("\n[EDA] participant-level event-related response (baseline-corrected, 0-18 s)")
    rows.append(compare("EDA peak 0-18 s", peak, is_dt, "uS", rng, ci=True))
    rows.append(compare("EDA AUC 0-18 s", auc, is_dt, "uS*s", rng, ci=True))
    for a, b in [(0, 3), (3, 6), (6, 12), (12, 18)]:
        m = (tE >= a) & (tE < b)
        rows.append(compare(f"EDA mean {a}-{b} s", W[:, m].mean(1), is_dt, "uS"))
    pk_lat = tE[post][W[:, post].argmax(1)]
    rows.append(compare("EDA peak latency (participant waveform)", pk_lat, is_dt, "s"))

    p_omni = omnibus(W, tE, is_dt, rng, (0, 18))
    cl, tc = cluster_test(W, tE, is_dt, rng, (0, 18))
    say(f"  participant-level omnibus permutation (0-18 s): p = {p_omni:.4f}")
    say(f"  participant-level cluster permutation (0-18 s, |t|>{tc:.3f}): "
        + ("no supra-threshold cluster" if not cl else
           "; ".join(f"{r['start']:.2f}-{r['end']:.2f} s {r['direction']} mass={r['mass']:.1f} p={r['p']:.4f}" for r in cl)))
    rows.append(dict(measure="EDA omnibus permutation 0-18 s (participant level)", p_perm=p_omni))
    clus += [dict(channel="EDA", **r) for r in cl] or [dict(channel="EDA", mass=0, start=np.nan, end=np.nan,
                                                         direction="none", p=np.nan)]

    # mixed model on epoch peaks (participant random intercept)
    rec = []
    for p, dt_ in zip(P, is_dt):
        for v, nx, pv, sd in zip(p["eda_epoch_peak"], p["eda_next_s"], p["eda_prev_s"], p["eda_stream_s"]):
            rec.append(dict(pid=str(p["pid"]), DT=int(dt_), peak=v, overlap=int(nx < 18 or pv < 20), stream=sd))
    ep = pd.DataFrame(rec)
    mm = smf.mixedlm("peak ~ DT", ep, groups=ep["pid"]).fit(reml=True)
    say(f"  mixed model, epoch peak ~ DT + (1|participant): b = {mm.params['DT']:+.4f} uS, "
        f"SE {mm.bse['DT']:.4f}, z = {mm.tvalues['DT']:.2f}, p = {mm.pvalues['DT']:.4f}  ({len(ep)} epochs)")
    rows.append(dict(measure="EDA epoch peak, mixed model (DT effect, uS)", diff=mm.params["DT"],
                     se=mm.bse["DT"], welch_p=mm.pvalues["DT"]))
    mm2 = smf.mixedlm("peak ~ DT + stream", ep, groups=ep["pid"]).fit(reml=True)
    say(f"  mixed model + response-stream duration: b_DT = {mm2.params['DT']:+.4f}, p = {mm2.pvalues['DT']:.4f}; "
        f"b_stream = {mm2.params['stream']:+.4f}/s, p = {mm2.pvalues['stream']:.4f}")
    rows.append(dict(measure="EDA epoch peak, mixed model adjusted for stream duration (DT effect, uS)",
                     diff=mm2.params["DT"], se=mm2.bse["DT"], welch_p=mm2.pvalues["DT"]))

    say("\n[EDA] sensitivity analyses")
    ovl = [((p["eda_next_s"] < 18) | (p["eda_prev_s"] < 20)) for p in P]
    o_sl = np.concatenate([o for o, d in zip(ovl, is_dt) if not d]).mean()
    o_dt = np.concatenate([o for o, d in zip(ovl, is_dt) if d]).mean()
    say(f"  overlapping epochs (next onset < 18 s or previous < 20 s): SL {100*o_sl:.1f}%, DT {100*o_dt:.1f}%")
    rows.append(dict(measure="EDA overlapping epochs (%)", M_SL=100 * o_sl, M_DT=100 * o_dt))
    Wn = np.array([p["eda_epochs"][~o].mean(0) if (~o).sum() else np.full(nE, np.nan) for p, o in zip(P, ovl)])
    rows.append(compare("EDA peak 0-18 s, non-overlapping epochs only", Wn[:, post].max(1), is_dt, "uS", rng, ci=True))
    stream = np.array([p["eda_stream_s"].mean() for p in P])
    rows.append(compare("Response-stream duration per participant", stream, is_dt, "s"))
    df = pd.DataFrame(dict(peak=peak, DT=is_dt.astype(int), stream=stream, n=n_ep))
    for nm, f in (("unadjusted", "peak ~ DT"), ("adjusted for stream duration", "peak ~ DT + stream"),
                  ("adjusted for epoch count", "peak ~ DT + n")):
        m = smf.ols(f, df).fit()
        say(f"  OLS participant peak {nm:30s}: b_DT = {m.params['DT']:+.4f} uS, p = {m.pvalues['DT']:.4f}, "
            f"partial d = {m.params['DT']/np.sqrt(m.mse_resid):+.2f}")
        rows.append(dict(measure=f"EDA participant peak, OLS {nm}", diff=m.params["DT"], welch_p=m.pvalues["DT"],
                         d=m.params["DT"] / np.sqrt(m.mse_resid)))
    r, pr = stats.pearsonr(n_ep, peak)
    rs, ps = stats.spearmanr(n_ep, peak)
    say(f"  peak vs epoch count: r = {r:+.2f} (p = {pr:.3f}); rho = {rs:+.2f} (p = {ps:.3f})")
    rows.append(compare("EDA epochs per participant", n_ep, is_dt, "n"))

    say("\n[EDA] level and tonic/phasic decomposition (absolute microsiemens)")
    rows.append(compare("SCL, session mean", [p["eda_session_scl"] for p in P], is_dt, "uS"))
    rows.append(compare("SCL, pre-onset windows", [p["eda_pre_scl"].mean() for p in P], is_dt, "uS"))
    for k in ("eda_tonic_mean", "eda_tonic_sd", "eda_phasic_rms", "eda_phasic_var_share"):
        rows.append(compare(k, [float(p[k]) for p in P], is_dt))
    share = np.array([float(p["eda_phasic_var_share"]) for p in P])
    say(f"  phasic share of variance: median {np.median(share):.3f} (IQR {np.quantile(share,.25):.3f}-{np.quantile(share,.75):.3f})")

    # =============================== fEMG ====================================
    for lab in ("corrugator", "zygomaticus"):
        nM = P[0][f"{lab}_epochs_abs"].shape[1]
        tM = taxis(2.0, nM)
        base = tM < 0
        say(f"\n[{lab}] (envelope in uV)")
        for variant in ("outlier epochs removed", "all epochs"):        # primary first
            Wa, Wc, pre, nrej = [], [], [], []
            for p in P:
                E = p[f"{lab}_epochs_abs"].astype(float) * 1e3                       # mV -> uV
                keep = outlier_mask(E) if variant == "outlier epochs removed" else np.ones(len(E), bool)
                nrej.append((len(E) - keep.sum(), len(E)))
                E = E[keep]
                b = E[:, base].mean(1)
                Wa.append(E.mean(0))
                Wc.append((E - b[:, None]).mean(0))
                pre.append(b.mean())
            Wa, Wc, pre = np.array(Wa), np.array(Wc), np.array(pre)
            tag = "" if variant == "outlier epochs removed" else " [all epochs]"
            if variant == "outlier epochs removed":
                r_sl = sum(a for (a, n), d in zip(nrej, is_dt) if not d) / sum(n for (a, n), d in zip(nrej, is_dt) if not d)
                r_dt = sum(a for (a, n), d in zip(nrej, is_dt) if d) / sum(n for (a, n), d in zip(nrej, is_dt) if d)
                say(f"  epochs rejected as outliers: SL {100*r_sl:.1f}%, DT {100*r_dt:.1f}%")
                rows.append(dict(measure=f"{lab} outlier epochs rejected (%)", M_SL=100 * r_sl, M_DT=100 * r_dt))
                ne = [n for (a, n) in nrej]
                say(f"  epochs before rejection: SL {sum(n for n, d in zip(ne, is_dt) if not d)}, DT {sum(n for n, d in zip(ne, is_dt) if d)}")
                wave[f"{lab}_t"], wave[f"{lab}_W"] = tM, Wc
            w06 = (tM >= 0) & (tM <= 6)
            pk = Wc[:, w06].max(1)
            rest = np.array([float(p[f"{lab}_rest"]) for p in P]) * 1e3
            rows.append(compare(f"{lab} peak change 0-6 s{tag}", pk, is_dt, "uV"))
            for a, b2 in [(0, 1), (1, 3), (3, 6), (6, 13)]:
                m = (tM >= a) & (tM < b2)
                rows.append(compare(f"{lab} mean change {a}-{b2} s{tag}", Wc[:, m].mean(1), is_dt, "uV"))
            rows.append(compare(f"{lab} absolute pre-onset level{tag}", pre, is_dt, "uV"))
            rows.append(compare(f"{lab} peak change / resting level{tag}", pk / rest, is_dt, "ratio"))
            rows.append(compare(f"{lab} mean change 0-6 s / resting level{tag}", Wc[:, w06].mean(1) / rest, is_dt, "ratio"))
            post_abs = Wa[:, w06].mean(1)
            m = smf.ols("post ~ DT + pre", pd.DataFrame(dict(post=post_abs, pre=pre, DT=is_dt.astype(int)))).fit()
            say(f"  ANCOVA absolute 0-6 s level ~ DT + absolute pre-onset level{tag}: b_DT = {m.params['DT']:+.3f} uV, "
                f"p = {m.pvalues['DT']:.3f}, partial d = {m.params['DT']/np.sqrt(m.mse_resid):+.2f}, b_pre = {m.params['pre']:.3f}")
            rows.append(dict(measure=f"{lab} ANCOVA abs 0-6 s ~ DT + abs pre{tag}", diff=m.params["DT"],
                             welch_p=m.pvalues["DT"], d=m.params["DT"] / np.sqrt(m.mse_resid)))
            if variant == "outlier epochs removed":
                rows.append(compare(f"{lab} resting level (eyes-closed recording)", rest, is_dt, "uV"))
                rows.append(compare(f"{lab} pre-onset level / resting level", pre / rest, is_dt, "ratio"))
                n_m = np.array([len(p[f"{lab}_kept"]) for p in P])
                r, pr = stats.pearsonr(n_m, pk / rest)
                say(f"  standardised peak vs epoch count: r = {r:+.2f} (p = {pr:.3f})")
                rows.append(dict(measure=f"{lab} standardised peak vs epoch count (Pearson r)", d=r, welch_p=pr))
                po = omnibus(Wc, tM, is_dt, rng, (0, 13))
                cl, tc = cluster_test(Wc, tM, is_dt, rng, (0, 13))
                say(f"  participant-level omnibus permutation (0-13 s): p = {po:.4f}")
                say(f"  participant-level cluster permutation (0-13 s): " + ("no supra-threshold cluster" if not cl else
                    "; ".join(f"{r['start']:.2f}-{r['end']:.2f} s {r['direction']} p={r['p']:.4f}" for r in cl)))
                rows.append(dict(measure=f"{lab} omnibus permutation 0-13 s (participant level)", p_perm=po))
                clus += [dict(channel=lab, **r) for r in cl] or [dict(channel=lab, mass=0, start=np.nan, end=np.nan,
                                                                     direction="none", p=np.nan)]

    pd.DataFrame(rows).to_csv(f"{RES}/physio_participant_tests.csv", index=False)
    pd.DataFrame(clus).to_csv(f"{RES}/physio_participant_clusters.csv", index=False)
    np.savez_compressed(f"{RES}/physio_participant_waveforms.npz", **wave)
    open(f"{RES}/physio_participant_summary.txt", "w").write("\n".join(LINES) + "\n")
    print(f"\nwrote {RES}/physio_participant_*.csv|npz|txt")


if __name__ == "__main__":
    main()
