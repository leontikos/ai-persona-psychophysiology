#!/usr/bin/env python
"""Shared helpers for the exploratory self-report and performance analyses
(Supplementary Notes 8 and 11).

The statistical helpers are copied verbatim (logic unchanged) from
analysis/statistics/recompute_stats.py so that numbers are
comparable with the manuscript. They are copied rather than imported because importing
recompute_stats.py runs its scale reconstruction at import time. Scale scoring mirrors recompute_stats.py / survey_analysis.py exactly.
Reads the source data only; outputs go to $EXPLORATORY_OUT.
"""
import os, numpy as np, pandas as pd, json, glob, warnings
from scipy import stats
from scipy.stats import nct, ncf, mannwhitneyu
from scipy.optimize import brentq
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
BASE = os.environ.get("STUDY_DATA_ROOT", "data")
QUALTRICS = f"{BASE}/SURVEY_data/AI_boss_qualtrics_quality.csv"
LOGS = f"{BASE}/conversation_logs"
CODED = f"{BASE}/CONVERSATION_ANALYSES/successful_analyses"
OUT = os.environ.get("EXPLORATORY_OUT", f"{BASE}/SURVEY_data/analysis_output/exploratory")
os.makedirs(OUT, exist_ok=True)
NBOOT = 5000
SEED = 20260704
TASKS = ["Persona", "Brief", "Communication", "Content"]


def new_rng():
    return np.random.default_rng(SEED)


RNG = new_rng()  # module-level RNG, as in recompute_stats.py


# --------------------------------------------------------------------------- helpers
def d_ci_nct(d, n1, n2, conf=0.95):
    df = n1 + n2 - 2
    a = 1.0 / n1 + 1.0 / n2
    t = d / np.sqrt(a)
    alpha = 1 - conf

    def solve(target):
        f = lambda ncp: nct.cdf(t, df, ncp) - target
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
    return solve(1 - alpha / 2) * np.sqrt(a), solve(alpha / 2) * np.sqrt(a)


def d_ci_nct_robust(d, n1, n2, conf=0.95):
    """Same estimand as d_ci_nct, but brackets each root on a fine grid so it never
    returns NaN (the original's +/-20 span bracket fails for some t values because
    nct.cdf returns NaN far in the tails)."""
    df = n1 + n2 - 2
    a = 1.0 / n1 + 1.0 / n2
    t = d / np.sqrt(a)
    alpha = 1 - conf

    def solve(target):
        f = lambda ncp: nct.cdf(t, df, ncp) - target
        grid = np.linspace(t - 10, t + 10, 401)
        vals = np.array([f(g) for g in grid])
        for i in range(len(grid) - 1):
            if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
                return brentq(f, grid[i], grid[i + 1])
        return np.nan
    return solve(1 - alpha / 2) * np.sqrt(a), solve(alpha / 2) * np.sqrt(a)


def eta2_ci(F, df1, df2, conf=0.90):
    alpha = 1 - conf
    try:
        lo = brentq(lambda ncp: ncf.cdf(F, df1, df2, ncp) - (1 - alpha / 2), 0, 100000, maxiter=500)
    except Exception:
        lo = 0.0
    try:
        hi = brentq(lambda ncp: ncf.cdf(F, df1, df2, ncp) - (alpha / 2), 0, 100000, maxiter=500)
    except Exception:
        hi = 0.0
    to_eta = lambda ncp: ncp / (ncp + df1 + df2 + 1)
    return to_eta(lo), to_eta(hi)


def eta2_ci_robust(F, df1, df2, conf=0.90):
    """Same estimand/convention as eta2_ci (lambda/(lambda+df1+df2+1)), but with a bracket
    limited to [0, 1000]: with the original [0, 1e5] bracket brentq evaluates ncf.cdf at large
    non-centrality values where SciPy returns NaN, the exception is caught and the limit is
    silently set to 0."""
    from scipy.stats import f as fdist
    alpha = 1 - conf
    lo = hi = 0.0
    if fdist.cdf(F, df1, df2) > 1 - alpha / 2:
        lo = brentq(lambda l: ncf.cdf(F, df1, df2, l) - (1 - alpha / 2), 1e-9, 1000)
    if fdist.cdf(F, df1, df2) > alpha / 2:
        hi = brentq(lambda l: ncf.cdf(F, df1, df2, l) - (alpha / 2), 1e-9, 1000)
    N = df1 + df2 + 1
    return lo / (lo + N), hi / (hi + N)


def fisher_z_ci(r, n, conf=0.95):
    if n < 4 or abs(r) >= 1:
        return (np.nan, np.nan)
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); zc = stats.norm.ppf(1 - (1 - conf) / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def mw_z(x, y):
    x = np.asarray(x); y = np.asarray(y)
    n1, n2 = len(x), len(y)
    U, p = mannwhitneyu(x, y, alternative="two-sided")
    all_v = np.concatenate([x, y]); n = n1 + n2; mu = n1 * n2 / 2.0
    _, counts = np.unique(all_v, return_counts=True)
    tie = np.sum(counts ** 3 - counts)
    sigma = np.sqrt((n1 * n2 / 12.0) * ((n + 1) - tie / (n * (n - 1))))
    z = (U - mu) / sigma if sigma > 0 else np.nan
    return U, p, z


def rank_biserial(x, y):
    x = np.asarray(x); y = np.asarray(y)
    U1, _ = mannwhitneyu(x, y, alternative="two-sided")
    return (2.0 * U1 / (len(x) * len(y))) - 1.0


def hodges_lehmann(x, y, conf=0.95):
    x = np.asarray(x, float); y = np.asarray(y, float)
    diffs = np.sort((x[:, None] - y[None, :]).ravel()); K = len(diffs)
    hl = np.median(diffs); n1, n2 = len(x), len(y)
    z = stats.norm.ppf(1 - (1 - conf) / 2); sd = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    k = max(1, int(np.round(n1 * n2 / 2.0 - z * sd)))
    return hl, diffs[k - 1], diffs[K - k]


def bca_ci(data_tuple, stat_fn, nboot=NBOOT, conf=0.95, rng=None):
    rng = RNG if rng is None else rng
    x, y = [np.asarray(a) for a in data_tuple]
    theta_hat = stat_fn(x, y)
    boots = np.empty(nboot); n1, n2 = len(x), len(y)
    for b in range(nboot):
        xb = x[rng.integers(0, n1, n1)]; yb = y[rng.integers(0, n2, n2)]
        try:
            boots[b] = stat_fn(xb, yb)
        except Exception:
            boots[b] = np.nan
    boots = boots[~np.isnan(boots)]
    if len(boots) < 100:
        return (np.nan, np.nan)
    z0 = stats.norm.ppf((np.sum(boots < theta_hat) + 0.5) / (len(boots) + 1))
    jack = [stat_fn(np.delete(x, i), y) for i in range(n1)] + [stat_fn(x, np.delete(y, j)) for j in range(n2)]
    jack = np.array(jack); jbar = jack.mean()
    denom = 6 * (np.sum((jbar - jack) ** 2)) ** 1.5
    a = np.sum((jbar - jack) ** 3) / denom if denom != 0 else 0.0
    alpha = 1 - conf

    def adj(q):
        zq = stats.norm.ppf(q)
        return stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
    return np.nanpercentile(boots, 100 * adj(alpha / 2)), np.nanpercentile(boots, 100 * adj(1 - alpha / 2))


def bayes_ttest(t, n1, n2, r=0.707):
    import pingouin as pg
    try:
        return float(pg.bayesfactor_ttest(t, n1, n2, paired=False, r=r))
    except Exception:
        return np.nan


def cohens_d(g1, g2):
    g1 = pd.Series(g1); g2 = pd.Series(g2); n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1 - 1) * g1.std(ddof=1) ** 2 + (n2 - 1) * g2.std(ddof=1) ** 2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0.0


def cronbach(items_df):
    items_df = items_df.dropna(); k = items_df.shape[1]
    if k < 2:
        return np.nan
    iv = items_df.var(axis=0, ddof=1); tv = items_df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - iv.sum() / tv) if tv > 0 else np.nan


def full_compare(df, col, g1_code=1, g2_code=2, bf=False, rng=None, robust_ci=False):
    """Mirror of compute_survey.full_compare. g1 = SL (code 1), g2 = DT (code 2);
    d and r are positive when SL > DT."""
    from scipy.stats import shapiro, levene, ttest_ind
    G = df["Group"]
    g1 = df.loc[G == g1_code, col].dropna().values.astype(float)
    g2 = df.loc[G == g2_code, col].dropna().values.astype(float)
    t, tp = ttest_ind(g1, g2, equal_var=False)
    v1, v2 = g1.var(ddof=1) / len(g1), g2.var(ddof=1) / len(g2)
    dfw = (v1 + v2) ** 2 / (v1 ** 2 / (len(g1) - 1) + v2 ** 2 / (len(g2) - 1))
    d = cohens_d(g1, g2)
    dlo, dhi = (d_ci_nct_robust if robust_ci else d_ci_nct)(d, len(g1), len(g2))
    U, up, z = mw_z(g1, g2)
    rrb = rank_biserial(g1, g2)
    rlo, rhi = bca_ci((g1, g2), rank_biserial, rng=rng)
    hl, hllo, hlhi = hodges_lehmann(g1, g2)
    sw1 = shapiro(g1); sw2 = shapiro(g2); lev = levene(g1, g2, center="median")
    res = {"Variable": col, "n_SL": len(g1), "n_DT": len(g2),
           "M_SL": g1.mean(), "SD_SL": g1.std(ddof=1), "Mdn_SL": np.median(g1), "min_SL": g1.min(), "max_SL": g1.max(),
           "M_DT": g2.mean(), "SD_DT": g2.std(ddof=1), "Mdn_DT": np.median(g2), "min_DT": g2.min(), "max_DT": g2.max(),
           "Welch_t": t, "Welch_df": dfw, "t_p": tp, "Cohen_d": d, "d_lo": dlo, "d_hi": dhi,
           "U": U, "Z": z, "U_p": up, "rrb": rrb, "rrb_lo": rlo, "rrb_hi": rhi,
           "HL_SLminusDT": hl, "HL_lo": hllo, "HL_hi": hlhi,
           "SW_SL_W": sw1.statistic, "SW_SL_p": sw1.pvalue, "SW_DT_W": sw2.statistic, "SW_DT_p": sw2.pvalue,
           "Levene_W": lev.statistic, "Levene_p": lev.pvalue, "VarRatio_DT_SL": g2.var(ddof=1) / g1.var(ddof=1)}
    if bf:
        res["BF10"] = bayes_ttest(t, len(g1), len(g2))
    return res


# --------------------------------------------------------------------------- scoring
def reverse_score(s, mx):
    return mx + 1 - s


def find_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    for col in df.columns:
        if cands[-1].replace("_", "") in col.replace("_", ""):
            return col
    return None


NEO = {
    "Neuroticism": {"items": [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56], "reverse": [1, 16, 31, 46]},
    "Extraversion": {"items": [2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57], "reverse": [12, 27, 42, 57]},
    "Openness": {"items": [3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58], "reverse": [3, 8, 18, 23, 33, 38, 48]},
    "Agreeableness": {"items": [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59], "reverse": [9, 14, 24, 29, 39, 44, 54, 59]},
    "Conscientiousness": {"items": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], "reverse": [15, 30, 45, 55]}}


def round_rating_means(df):
    """The Qualtrics export stores the expert-rating means (Average_*, *_avg) with floating-point
    noise of about 1e-15, which splits tied participants in rank tests; the archived data table
    stores them to nine decimals. Rounding to nine decimals gives both the exact ties."""
    cols = [c for c in df.columns
            if c.startswith("Average_") or c in ("Completeness_avg", "Originality_avg", "Strategic_fit_avg")]
    df[cols] = df[cols].round(9)
    return df


def load_scored():
    df = pd.read_csv(QUALTRICS); df.columns = df.columns.str.strip()
    df = round_rating_means(df)
    new = {}
    for tr, inf in NEO.items():
        sc = []
        for i in inf["items"]:
            c = f"Personality_{i}"
            if i in inf["reverse"]:
                new[f"Personality_{i}_R"] = reverse_score(df[c], 5); sc.append(new[f"Personality_{i}_R"])
            else:
                sc.append(df[c])
        new[tr] = pd.concat(sc, axis=1).mean(axis=1)
    df = pd.concat([df, pd.DataFrame(new)], axis=1); new = {}
    new["AI_Experience"] = df[["Experience_with_AI_1", "Experience_with_AI_2", "Experience_with_AI_3"]].mean(axis=1)
    new["Technology_Trust"] = df[["Technology_Trust_1", "Technology_Trust_2", "Technology_Trust_3"]].mean(axis=1)
    new["AI_fear_2_R"] = reverse_score(df["AI_fear_2"], 5); new["AI_fear_4_R"] = reverse_score(df["AI_fear_4"], 5)
    new["AI_Fear_composite"] = pd.concat([df["AI_fear_1"], new["AI_fear_2_R"], df["AI_fear_3"], new["AI_fear_4_R"], df["AI_fear_5"]], axis=1).mean(axis=1)
    new["AI_Collaborators_pre"] = df[["AI_collaborators_1", "AI_collaborators_2", "AI_collaborators_3"]].mean(axis=1)
    new["AI_Willingness_pre"] = df[[f"AI_willingness_to_collaborate_{i}" for i in range(1, 5)]].mean(axis=1)
    for task in TASKS:
        fa = [find_col(df, [f"Focused_Attention_{task}_{i}", f"Focused_Attention_{task[:6]}_{i}", f"Focused_Attention_{task[:7]}_{i}"]) for i in range(1, 4)]
        new[f"Focused_Attention_{task}"] = df[[c for c in fa if c]].mean(axis=1)
        new[f"Enjoyment_{task}"] = df[[f"Enjoyment_{task}_{i}" for i in range(1, 4)]].mean(axis=1)
        new[f"Satisfaction_{task}"] = df[[f"Satisfaction_{task}_1", f"Satisfaction_{task}_2", f"Satisfaction_{task}_3_rek"]].mean(axis=1)
        # 2-item satisfaction (satisfied + pleased) and the raw frustration item
        new[f"Satisfaction2_{task}"] = df[[f"Satisfaction_{task}_1", f"Satisfaction_{task}_2"]].mean(axis=1)
        new[f"Frustration_{task}"] = df[f"Satisfaction_{task}_3"]
    new["AI_Collaboration_post"] = df[["AI_collab_post_1", "AI_collab_post_2", "AI_collab_post_3"]].mean(axis=1)
    new["AI_Reliability"] = df[["AI_collab_post_4", "AI_collab_post_5", "AI_collab_post_6"]].mean(axis=1)
    new["Behavioral_Intention"] = df[["AI_collab_post_7", "AI_collab_post_8", "AI_collab_post_9"]].mean(axis=1)
    new["AI_Willingness_post"] = df[[f"AI_collab_post_{i}" for i in range(10, 14)]].mean(axis=1)
    new["AI_collab_post_15_R"] = reverse_score(df["AI_collab_post_15"], 5); new["AI_collab_post_17_R"] = reverse_score(df["AI_collab_post_17"], 5)
    new["AI_Fear_post"] = pd.concat([df["AI_collab_post_14"], new["AI_collab_post_15_R"], df["AI_collab_post_16"], new["AI_collab_post_17_R"], df["AI_collab_post_18"]], axis=1).mean(axis=1)
    new["Human_AI_Fit"] = df[["AI_collab_post_19", "AI_collab_post_20", "AI_collab_post_21"]].mean(axis=1)
    new["Anthropomorphism"] = df[["Lifelike", "Humanlike", "Natural", "Conscious"]].mean(axis=1)
    df = pd.concat([df, pd.DataFrame(new)], axis=1); new = {}
    for m in ["Focused_Attention", "Enjoyment", "Satisfaction", "Satisfaction2", "Frustration"]:
        new[f"{m}_avg"] = df[[f"{m}_{t}" for t in TASKS]].mean(axis=1)
    df = pd.concat([df, pd.DataFrame(new)], axis=1)
    return df.copy()


def fmt_p(p):
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"
