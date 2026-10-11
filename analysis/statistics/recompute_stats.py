#!/usr/bin/env python
"""Shared statistical helpers, plus reconstruction of the survey scale scores.

Running this module directly rebuilds the scored survey table from the raw
Qualtrics export and writes it to STATS_OUT/survey_scored.pkl, which
compute_survey.py and regenerate_manuscript_figures.py consume. It only ever
reads the source data.
"""
import json, os, warnings

import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import nct, ncf, mannwhitneyu, shapiro, levene, pearsonr
from scipy.optimize import brentq
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
np.random.seed(20260704)
RNG = np.random.default_rng(20260704)

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
STATS_OUT = os.path.join(DATA_ROOT, "statistics_output")
QUALTRICS = os.path.join(DATA_ROOT, "SURVEY_data", "AI_boss_qualtrics_quality.csv")
os.makedirs(STATS_OUT, exist_ok=True)
NBOOT = 5000

# ----------------------------------------------------------------------------
# effect-size CI helpers
# ----------------------------------------------------------------------------

def round_rating_means(df):
    """The Qualtrics export stores the expert-rating means (Average_*, *_avg) with floating-point
    noise of about 1e-15, which splits tied participants in rank tests; the archived data table
    stores them to nine decimals. Rounding to nine decimals gives both the exact ties."""
    cols = [c for c in df.columns
            if c.startswith("Average_") or c in ("Completeness_avg", "Originality_avg", "Strategic_fit_avg")]
    df[cols] = df[cols].round(9)
    return df


def anonymise_raters(df):
    """The Qualtrics export names the three expert raters' columns by the raters' first names
    (e.g. Average_<name>). Refer to them as Judge1-3, numbered in alphabetical order of the names."""
    names = sorted(c.split("_", 1)[1] for c in df.columns if c.startswith("Average_") and c != "Average_all")
    ren = {}
    for c in df.columns:
        stem, _, last = c.rpartition("_")
        if last in names and stem:
            ren[c] = f"{stem}_Judge{names.index(last) + 1}"
    return df.rename(columns=ren)

def d_ci_nct(d, n1, n2, conf=0.95):
    """95% CI for Cohen's d (independent groups) via noncentral t."""
    df = n1 + n2 - 2
    a = 1.0/n1 + 1.0/n2
    t = d / np.sqrt(a)
    alpha = 1 - conf
    def solve(target):
        f = lambda ncp: nct.cdf(t, df, ncp) - target
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

def eta2_ci(F, df1, df2, conf=0.90):
    """CI for partial eta^2 via noncentral F (default 90%)."""
    alpha = 1 - conf
    def ncp_ci():
        def solve(target):
            try:
                return brentq(lambda ncp: ncf.cdf(F, df1, df2, ncp) - target,
                              0 if target>0.5 else 0, 10000, maxiter=500)
            except Exception:
                return 0.0
        lo = solve(1 - alpha/2)
        hi = solve(alpha/2)
        return lo, hi
    # if F small, lower ncp may be 0
    try:
        lo = brentq(lambda ncp: ncf.cdf(F, df1, df2, ncp) - (1 - alpha/2), 0, 100000, maxiter=500)
    except Exception:
        lo = 0.0
    try:
        hi = brentq(lambda ncp: ncf.cdf(F, df1, df2, ncp) - (alpha/2), 0, 100000, maxiter=500)
    except Exception:
        hi = 0.0
    def ncp_to_eta(ncp):
        return ncp / (ncp + df1 + df2 + 1)
    return ncp_to_eta(lo), ncp_to_eta(hi)

def fisher_z_ci(r, n, conf=0.95):
    if n < 4 or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r)
    se = 1/np.sqrt(n-3)
    zc = stats.norm.ppf(1-(1-conf)/2)
    return np.tanh(z-zc*se), np.tanh(z+zc*se)

def mw_z(x, y):
    """Standardized Z for Mann-Whitney with tie correction (normal approx)."""
    x = np.asarray(x); y = np.asarray(y)
    n1, n2 = len(x), len(y)
    U, p = mannwhitneyu(x, y, alternative="two-sided")
    all_v = np.concatenate([x, y])
    ranks = stats.rankdata(all_v)
    n = n1 + n2
    mu = n1*n2/2.0
    # tie correction
    _, counts = np.unique(all_v, return_counts=True)
    tie = np.sum(counts**3 - counts)
    sigma = np.sqrt((n1*n2/12.0)*((n+1) - tie/(n*(n-1))))
    z = (U - mu)/sigma if sigma>0 else np.nan
    return U, p, z

def rank_biserial(x, y):
    """Rank-biserial r = 1 - 2U/(n1*n2), sign: positive => x>y (group1 larger)."""
    x = np.asarray(x); y = np.asarray(y)
    n1, n2 = len(x), len(y)
    U1, _ = mannwhitneyu(x, y, alternative="two-sided")  # U for x
    return (2.0*U1/(n1*n2)) - 1.0

def hodges_lehmann(x, y, conf=0.95):
    """HL median difference (x - y) with distribution-free CI."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    diffs = np.sort((x[:,None] - y[None,:]).ravel())
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
    # acceleration via jackknife (leave-one-out over pooled? use both groups)
    jack = []
    for i in range(n1):
        xj = np.delete(x, i)
        jack.append(stat_fn(xj, y))
    for j in range(n2):
        yj = np.delete(y, j)
        jack.append(stat_fn(x, yj))
    jack = np.array(jack); jbar = jack.mean()
    denom = 6*(np.sum((jbar-jack)**2))**1.5
    a = np.sum((jbar-jack)**3)/denom if denom!=0 else 0.0
    alpha = 1-conf
    def adj(q):
        zq = stats.norm.ppf(q)
        return stats.norm.cdf(z0 + (z0+zq)/(1-a*(z0+zq)))
    lo_q, hi_q = adj(alpha/2), adj(1-alpha/2)
    return np.nanpercentile(boots, 100*lo_q), np.nanpercentile(boots, 100*hi_q)

def bayes_ttest(t, n1, n2, r=0.707):
    """JZS Bayes factor (BF10) for two-sample t via pingouin."""
    try:
        import pingouin as pg
        bf = float(pg.bayesfactor_ttest(t, n1, n2, paired=False, r=r))
        return bf
    except Exception as e:
        return np.nan

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

# ============================================================================
# SURVEY: reconstruct per-participant scale scores (mirror survey_analysis.py)
# ============================================================================
def reverse_score(s, mx): return mx+1-s
def find_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    for col in df.columns:
        if cands[-1].replace("_","") in col.replace("_",""): return col
    return None

def build_survey_table():
    """Reconstruct the per-participant scale scores from the raw Qualtrics export.

    Mirrors the scoring in survey_analysis.py; validated against the stored
    descriptive_statistics.csv group means.
    """
    df = pd.read_csv(QUALTRICS); df.columns = df.columns.str.strip()
    df = round_rating_means(anonymise_raters(df))
    TASKS = ["Persona","Brief","Communication","Content"]
    NEO = {
     "Neuroticism":{"items":[1,6,11,16,21,26,31,36,41,46,51,56],"reverse":[1,16,31,46]},
     "Extraversion":{"items":[2,7,12,17,22,27,32,37,42,47,52,57],"reverse":[12,27,42,57]},
     "Openness":{"items":[3,8,13,18,23,28,33,38,43,48,53,58],"reverse":[3,8,18,23,33,38,48]},
     "Agreeableness":{"items":[4,9,14,19,24,29,34,39,44,49,54,59],"reverse":[9,14,24,29,39,44,54,59]},
     "Conscientiousness":{"items":[5,10,15,20,25,30,35,40,45,50,55,60],"reverse":[15,30,45,55]}}
    for tr,inf in NEO.items():
        sc=[]
        for i in inf["items"]:
            c=f"Personality_{i}"
            if i in inf["reverse"]:
                df[f"Personality_{i}_R"]=reverse_score(df[c],5); sc.append(f"Personality_{i}_R")
            else: sc.append(c)
        df[tr]=df[sc].mean(axis=1)
    df["AI_Experience"]=df[["Experience_with_AI_1","Experience_with_AI_2","Experience_with_AI_3"]].mean(axis=1)
    df["Technology_Trust"]=df[["Technology_Trust_1","Technology_Trust_2","Technology_Trust_3"]].mean(axis=1)
    df["AI_fear_2_R"]=reverse_score(df["AI_fear_2"],5); df["AI_fear_4_R"]=reverse_score(df["AI_fear_4"],5)
    df["AI_Fear_composite"]=df[["AI_fear_1","AI_fear_2_R","AI_fear_3","AI_fear_4_R","AI_fear_5"]].mean(axis=1)
    df["AI_Collaborators_pre"]=df[["AI_collaborators_1","AI_collaborators_2","AI_collaborators_3"]].mean(axis=1)
    df["AI_Willingness_pre"]=df[[f"AI_willingness_to_collaborate_{i}" for i in range(1,5)]].mean(axis=1)
    for task in TASKS:
        fa=[find_col(df,[f"Focused_Attention_{task}_{i}",f"Focused_Attention_{task[:6]}_{i}",f"Focused_Attention_{task[:7]}_{i}"]) for i in range(1,4)]
        fa=[c for c in fa if c]
        if fa: df[f"Focused_Attention_{task}"]=df[fa].mean(axis=1)
        enj=[find_col(df,[f"Enjoyment_{task}_{i}"]) for i in range(1,4)]; enj=[c for c in enj if c]
        if enj: df[f"Enjoyment_{task}"]=df[enj].mean(axis=1)
        sat=[find_col(df,[f"Satisfaction_{task}_{i}"]) for i in [1,2]]; sat=[c for c in sat if c]
        rek=find_col(df,[f"Satisfaction_{task}_3_rek"])
        if rek: sat.append(rek)
        if sat: df[f"Satisfaction_{task}"]=df[sat].mean(axis=1)
    df["AI_Collaboration_post"]=df[["AI_collab_post_1","AI_collab_post_2","AI_collab_post_3"]].mean(axis=1)
    df["AI_Reliability"]=df[["AI_collab_post_4","AI_collab_post_5","AI_collab_post_6"]].mean(axis=1)
    df["Behavioral_Intention"]=df[["AI_collab_post_7","AI_collab_post_8","AI_collab_post_9"]].mean(axis=1)
    df["AI_Willingness_post"]=df[[f"AI_collab_post_{i}" for i in range(10,14)]].mean(axis=1)
    df["AI_collab_post_15_R"]=reverse_score(df["AI_collab_post_15"],5); df["AI_collab_post_17_R"]=reverse_score(df["AI_collab_post_17"],5)
    df["AI_Fear_post"]=df[["AI_collab_post_14","AI_collab_post_15_R","AI_collab_post_16","AI_collab_post_17_R","AI_collab_post_18"]].mean(axis=1)
    df["Human_AI_Fit"]=df[["AI_collab_post_19","AI_collab_post_20","AI_collab_post_21"]].mean(axis=1)
    df["Anthropomorphism"]=df[["Lifelike","Humanlike","Natural","Conscious"]].mean(axis=1)
    for m in ["Focused_Attention","Enjoyment","Satisfaction"]:
        cols=[f"{m}_{t}" for t in TASKS if f"{m}_{t}" in df.columns]
        if cols: df[f"{m}_avg"]=df[cols].mean(axis=1)

    print("Scale reconstruction done. Group counts:", df["Group"].value_counts().to_dict())
    df.to_pickle(os.path.join(STATS_OUT, "survey_scored.pkl"))
    print("saved survey_scored.pkl with", df.shape[1], "cols")
    return df


if __name__ == "__main__":
    build_survey_table()
