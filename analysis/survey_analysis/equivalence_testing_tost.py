# %% equivalence_testing_tost.py

"""
Equivalence Testing (TOST Procedure) for Self-Report Measures
==============================================================
Tests whether group differences in self-report measures fall within
a region of practical equivalence, strengthening the argument that
users are unaware of persona manipulation.

Method: Two One-Sided Tests (TOST) procedure (Lakens, 2017)
  - H₀: |μ₁ - μ₂| ≥ Δ  (groups differ by at least Δ)
  - H₁: |μ₁ - μ₂| < Δ  (groups are equivalent within ±Δ)

Equivalence bounds:
  - Primary: Δ = ±0.5 SD (smallest effect size of interest, SESOI)
  - Secondary: Δ = ±MDES (from sensitivity power analysis)

Reference: Lakens, D. (2017). Equivalence tests: A practical primer
for t tests, correlations, and meta-analyses. Social Psychological
and Personality Science, 8(4), 355-362.

Uses: statsmodels, scipy, numpy, pandas, pingouin, matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
from dotenv import load_dotenv

# ============================================================
# 0. CONFIGURATION
# ============================================================

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT = os.environ.get("STUDY_DATA_ROOT", "data")
DATA_PATH = os.path.join(DATA_ROOT, "SURVEY_data", "AI_boss_qualtrics_quality.csv")
OUTPUT_DIR = os.path.join(DATA_ROOT, "SURVEY_data", "analysis_output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

GROUP_LABELS = {1: "Servant Leader", 2: "Dark Triad"}
ALPHA = 0.05
TASKS = ["Persona", "Brief", "Communication", "Content"]
N_SL = 31
N_DT = 27

# SESOI bounds in Cohen's d units
SESOI_D = 0.5  # smallest effect size of interest (medium effect)

# Compute MDES from sensitivity analysis for secondary bounds
power_analysis = TTestIndPower()
MDES = power_analysis.solve_power(
    effect_size=None, nobs1=N_SL, ratio=N_DT / N_SL,
    alpha=ALPHA, power=0.80, alternative='two-sided'
)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLOR_SL = '#64B5F6'
COLOR_DT = '#E57373'

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reverse_score(series, max_val):
    return max_val + 1 - series

def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    for col in df.columns:
        if candidates[-1].replace("_", "") in col.replace("_", ""):
            return col
    return None

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1 - 1) * g1.std()**2 + (n2 - 1) * g2.std()**2) / (n1 + n2 - 2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0.0


def tost_ind(g1, g2, bound_low, bound_high, alpha=0.05):
    """
    Two One-Sided Tests (TOST) for independent samples.

    Tests equivalence: H₁: bound_low < (μ₁ - μ₂) < bound_high

    Parameters:
    -----------
    g1, g2 : array-like — two group samples
    bound_low, bound_high : float — equivalence bounds (raw units)
    alpha : float — significance level

    Returns:
    --------
    dict with TOST results including the two one-sided t-tests,
    overall TOST p-value, and confidence interval.
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    g1 = g1[~np.isnan(g1)]
    g2 = g2[~np.isnan(g2)]

    n1, n2 = len(g1), len(g2)
    mean_diff = g1.mean() - g2.mean()
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    se = np.sqrt(s1**2 / n1 + s2**2 / n2)

    # Welch-Satterthwaite degrees of freedom
    df_welch = ((s1**2 / n1 + s2**2 / n2)**2 /
                ((s1**2 / n1)**2 / (n1 - 1) + (s2**2 / n2)**2 / (n2 - 1)))

    # Test 1: H₀: μ₁ - μ₂ ≤ bound_low  vs  H₁: μ₁ - μ₂ > bound_low
    t1 = (mean_diff - bound_low) / se
    p1 = stats.t.sf(t1, df_welch)  # upper tail

    # Test 2: H₀: μ₁ - μ₂ ≥ bound_high  vs  H₁: μ₁ - μ₂ < bound_high
    t2 = (mean_diff - bound_high) / se
    p2 = stats.t.cdf(t2, df_welch)  # lower tail

    # TOST p-value: max of the two one-sided p-values
    p_tost = max(p1, p2)

    # 90% CI for equivalence testing (= 1 - 2α when α = 0.05)
    ci_alpha = alpha  # for 90% CI
    t_crit = stats.t.ppf(1 - ci_alpha, df_welch)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    # Also compute standard NHST t-test for comparison
    t_nhst, p_nhst = stats.ttest_ind(g1, g2, equal_var=False)

    # Pooled SD for d calculation
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))

    return {
        "mean_diff": mean_diff,
        "se": se,
        "df": df_welch,
        "t1": t1, "p1": p1,
        "t2": t2, "p2": p2,
        "p_tost": p_tost,
        "ci_90_lower": ci_lower,
        "ci_90_upper": ci_upper,
        "t_nhst": t_nhst,
        "p_nhst": p_nhst,
        "n1": n1, "n2": n2,
        "m1": g1.mean(), "m2": g2.mean(),
        "sd1": s1, "sd2": s2,
        "pooled_sd": pooled_sd,
        "d": mean_diff / pooled_sd if pooled_sd > 0 else 0.0,
        "bound_low": bound_low,
        "bound_high": bound_high,
        "equivalent": p_tost < alpha,
    }


def interpret_tost_nhst(p_tost, p_nhst, alpha=0.05):
    """
    Joint interpretation of TOST + NHST (Lakens, 2017, Figure 1).
    Returns one of four outcomes.
    """
    eq = p_tost < alpha
    sig = p_nhst < alpha
    if eq and not sig:
        return "Equivalent (statistically equivalent, not different)"
    elif not eq and sig:
        return "Different (statistically different, not equivalent)"
    elif eq and sig:
        return "Trivially different (statistically different but within equivalence bounds)"
    else:
        return "Undetermined (neither equivalent nor different)"


# ============================================================
# DATA LOADING & SCALE CONSTRUCTION
# ============================================================

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Group_label"] = df["Group"].map(GROUP_LABELS)

# --- Big Five ---
NEO_SCORING = {
    "Neuroticism": {"items": [1,6,11,16,21,26,31,36,41,46,51,56], "reverse": [1,16,31,46]},
    "Extraversion": {"items": [2,7,12,17,22,27,32,37,42,47,52,57], "reverse": [12,27,42,57]},
    "Openness": {"items": [3,8,13,18,23,28,33,38,43,48,53,58], "reverse": [3,8,18,23,33,38,48]},
    "Agreeableness": {"items": [4,9,14,19,24,29,34,39,44,49,54,59], "reverse": [9,14,24,29,39,44,54,59]},
    "Conscientiousness": {"items": [5,10,15,20,25,30,35,40,45,50,55,60], "reverse": [15,30,45,55]}
}
for trait, info in NEO_SCORING.items():
    scored = []
    for i in info["items"]:
        col = f"Personality_{i}"
        if i in info["reverse"]:
            rc = f"Personality_{i}_R"
            df[rc] = reverse_score(df[col], 5)
            scored.append(rc)
        else:
            scored.append(col)
    df[trait] = df[scored].mean(axis=1)

# --- Pre-task ---
df["AI_Experience"] = df[["Experience_with_AI_1","Experience_with_AI_2","Experience_with_AI_3"]].mean(axis=1)
df["Technology_Trust"] = df[["Technology_Trust_1","Technology_Trust_2","Technology_Trust_3"]].mean(axis=1)
df["AI_fear_2_R"] = reverse_score(df["AI_fear_2"], 5)
df["AI_fear_4_R"] = reverse_score(df["AI_fear_4"], 5)
df["AI_Fear_composite"] = df[["AI_fear_1","AI_fear_2_R","AI_fear_3","AI_fear_4_R","AI_fear_5"]].mean(axis=1)
df["AI_Collaborators_pre"] = df[["AI_collaborators_1","AI_collaborators_2","AI_collaborators_3"]].mean(axis=1)
df["AI_Willingness_pre"] = df[["AI_willingness_to_collaborate_1","AI_willingness_to_collaborate_2",
                                "AI_willingness_to_collaborate_3","AI_willingness_to_collaborate_4"]].mean(axis=1)

# --- Task-specific ---
for task in TASKS:
    fa_cols = []
    for i in range(1, 4):
        col = find_col(df, [f"Focused_Attention_{task}_{i}",
                            f"Focused_Attention_{task[:6]}_{i}",
                            f"Focused_Attention_{task[:7]}_{i}"])
        if col:
            fa_cols.append(col)
    if fa_cols:
        df[f"Focused_Attention_{task}"] = df[fa_cols].mean(axis=1)
    enj_cols = [c for c in [f"Enjoyment_{task}_{i}" for i in range(1, 4)] if c in df.columns]
    if enj_cols:
        df[f"Enjoyment_{task}"] = df[enj_cols].mean(axis=1)
    sat_cols = [c for c in [f"Satisfaction_{task}_{i}" for i in [1, 2]] if c in df.columns]
    rek_col = find_col(df, [f"Satisfaction_{task}_3_rek"])
    if rek_col:
        sat_cols.append(rek_col)
    if sat_cols:
        df[f"Satisfaction_{task}"] = df[sat_cols].mean(axis=1)

# --- Post-task ---
df["AI_Collaboration_post"] = df[["AI_collab_post_1","AI_collab_post_2","AI_collab_post_3"]].mean(axis=1)
df["AI_Reliability"] = df[["AI_collab_post_4","AI_collab_post_5","AI_collab_post_6"]].mean(axis=1)
df["Behavioral_Intention"] = df[["AI_collab_post_7","AI_collab_post_8","AI_collab_post_9"]].mean(axis=1)
df["AI_Willingness_post"] = df[[f"AI_collab_post_{i}" for i in range(10, 14)]].mean(axis=1)
df["AI_collab_post_15_R"] = reverse_score(df["AI_collab_post_15"], 5)
df["AI_collab_post_17_R"] = reverse_score(df["AI_collab_post_17"], 5)
df["AI_Fear_post"] = df[["AI_collab_post_14","AI_collab_post_15_R","AI_collab_post_16",
                          "AI_collab_post_17_R","AI_collab_post_18"]].mean(axis=1)
df["Human_AI_Fit"] = df[["AI_collab_post_19","AI_collab_post_20","AI_collab_post_21"]].mean(axis=1)
df["Anthropomorphism"] = df[["Lifelike","Humanlike","Natural","Conscious"]].mean(axis=1)

# --- Averages ---
for m in ["Focused_Attention", "Enjoyment", "Satisfaction"]:
    cols = [f"{m}_{t}" for t in TASKS if f"{m}_{t}" in df.columns]
    if cols:
        df[f"{m}_avg"] = df[cols].mean(axis=1)

print("=" * 80)
print("  EQUIVALENCE TESTING (TOST) — SELF-REPORT MEASURES")
print("=" * 80)
print()
print(f"  Method: Two One-Sided Tests (TOST; Lakens, 2017)")
print(f"  α = {ALPHA}")
print(f"  SESOI (primary bounds): d = ±{SESOI_D}")
print(f"  MDES (secondary bounds): d = ±{MDES:.3f}")
print()

# ============================================================
# 1. TOST ON KEY SELF-REPORT MEASURES
# ============================================================

# All self-report measures from condition comparisons
self_report_measures = [
    ("AI_Collaboration_post", "Perceived AI Collaboration"),
    ("AI_Reliability", "AI Reliability"),
    ("Behavioral_Intention", "Behavioral Intention to Use"),
    ("AI_Willingness_post", "Intention to Collaborate (post)"),
    ("AI_Fear_post", "AI Fear/Attitudes (post)"),
    ("Human_AI_Fit", "Human-AI Fit"),
    ("Anthropomorphism", "Anthropomorphism"),
    ("Focused_Attention_avg", "Focused Attention (avg)"),
    ("Enjoyment_avg", "Enjoyment (avg)"),
    ("Satisfaction_avg", "Satisfaction (avg)"),
]

all_results = []

for var, label in self_report_measures:
    if var not in df.columns:
        continue

    g1 = df.loc[df["Group"] == 1, var].dropna().values
    g2 = df.loc[df["Group"] == 2, var].dropna().values

    if len(g1) < 3 or len(g2) < 3:
        continue

    # Compute pooled SD for converting d bounds to raw units
    n1, n2 = len(g1), len(g2)
    pooled_sd = np.sqrt(((n1 - 1) * g1.std(ddof=1)**2 + (n2 - 1) * g2.std(ddof=1)**2) / (n1 + n2 - 2))

    # --- TOST with SESOI bounds (d = ±0.5) ---
    bound_raw_sesoi = SESOI_D * pooled_sd
    res_sesoi = tost_ind(g1, g2, -bound_raw_sesoi, bound_raw_sesoi, alpha=ALPHA)

    # --- TOST with MDES bounds (d = ±MDES) ---
    bound_raw_mdes = MDES * pooled_sd
    res_mdes = tost_ind(g1, g2, -bound_raw_mdes, bound_raw_mdes, alpha=ALPHA)

    interpretation_sesoi = interpret_tost_nhst(res_sesoi["p_tost"], res_sesoi["p_nhst"], ALPHA)
    interpretation_mdes = interpret_tost_nhst(res_mdes["p_tost"], res_mdes["p_nhst"], ALPHA)

    all_results.append({
        "Variable": var,
        "Label": label,
        "n_SL": n1, "n_DT": n2,
        "M_SL": res_sesoi["m1"], "SD_SL": res_sesoi["sd1"],
        "M_DT": res_sesoi["m2"], "SD_DT": res_sesoi["sd2"],
        "Mean_Diff": res_sesoi["mean_diff"],
        "Pooled_SD": pooled_sd,
        "Cohen_d": res_sesoi["d"],
        # NHST
        "t_NHST": res_sesoi["t_nhst"],
        "p_NHST": res_sesoi["p_nhst"],
        "NHST_sig": res_sesoi["p_nhst"] < ALPHA,
        # TOST with SESOI bounds
        "SESOI_d": SESOI_D,
        "SESOI_bound_raw": bound_raw_sesoi,
        "TOST_t1_SESOI": res_sesoi["t1"],
        "TOST_p1_SESOI": res_sesoi["p1"],
        "TOST_t2_SESOI": res_sesoi["t2"],
        "TOST_p2_SESOI": res_sesoi["p2"],
        "TOST_p_SESOI": res_sesoi["p_tost"],
        "Equivalent_SESOI": res_sesoi["equivalent"],
        "CI90_lower_SESOI": res_sesoi["ci_90_lower"],
        "CI90_upper_SESOI": res_sesoi["ci_90_upper"],
        "Interpretation_SESOI": interpretation_sesoi,
        # TOST with MDES bounds
        "MDES_d": MDES,
        "MDES_bound_raw": bound_raw_mdes,
        "TOST_p_MDES": res_mdes["p_tost"],
        "Equivalent_MDES": res_mdes["equivalent"],
        "CI90_lower_MDES": res_mdes["ci_90_lower"],
        "CI90_upper_MDES": res_mdes["ci_90_upper"],
        "Interpretation_MDES": interpretation_mdes,
    })

# ============================================================
# 2. FDR CORRECTION ACROSS TOST P-VALUES
# ============================================================

if all_results:
    # FDR on SESOI TOST p-values
    p_sesoi = [r["TOST_p_SESOI"] for r in all_results]
    reject_sesoi, p_adj_sesoi = pg.multicomp(p_sesoi, method="fdr_bh")
    for i, r in enumerate(all_results):
        r["TOST_p_SESOI_fdr"] = p_adj_sesoi[i]
        r["Equivalent_SESOI_fdr"] = reject_sesoi[i]

    # FDR on MDES TOST p-values
    p_mdes = [r["TOST_p_MDES"] for r in all_results]
    reject_mdes, p_adj_mdes = pg.multicomp(p_mdes, method="fdr_bh")
    for i, r in enumerate(all_results):
        r["TOST_p_MDES_fdr"] = p_adj_mdes[i]
        r["Equivalent_MDES_fdr"] = reject_mdes[i]

# ============================================================
# 3. PRINT DETAILED RESULTS
# ============================================================

print("=" * 80)
print("  TOST RESULTS WITH SESOI BOUNDS (d = ±{:.1f})".format(SESOI_D))
print("=" * 80)
print()

for r in all_results:
    print(f"  {r['Label']}:")
    print(f"    SL (n={r['n_SL']}): M = {r['M_SL']:.3f}, SD = {r['SD_SL']:.3f}")
    print(f"    DT (n={r['n_DT']}): M = {r['M_DT']:.3f}, SD = {r['SD_DT']:.3f}")
    print(f"    Mean difference = {r['Mean_Diff']:.3f}, Cohen's d = {r['Cohen_d']:.3f}")
    print(f"    Equivalence bounds (raw): [{-r['SESOI_bound_raw']:.3f}, {r['SESOI_bound_raw']:.3f}]")
    print(f"    90% CI: [{r['CI90_lower_SESOI']:.3f}, {r['CI90_upper_SESOI']:.3f}]")
    print(f"    NHST: t = {r['t_NHST']:.3f}, p = {r['p_NHST']:.4f} {'*' if r['NHST_sig'] else ''}")
    print(f"    TOST: p₁ = {r['TOST_p1_SESOI']:.4f}, p₂ = {r['TOST_p2_SESOI']:.4f}, "
          f"p_TOST = {r['TOST_p_SESOI']:.4f} (FDR: {r['TOST_p_SESOI_fdr']:.4f})")
    equiv = "YES " if r['Equivalent_SESOI'] else "NO"
    equiv_fdr = "YES " if r['Equivalent_SESOI_fdr'] else "NO"
    print(f"    Equivalent (uncorrected): {equiv}")
    print(f"    Equivalent (FDR-corrected): {equiv_fdr}")
    print(f"    Interpretation: {r['Interpretation_SESOI']}")
    print()

# Summary table
print("=" * 80)
print("  SUMMARY TABLE — TOST with SESOI bounds (d = ±{:.1f})".format(SESOI_D))
print("=" * 80)
print()
print(f"  {'Measure':<35} {'d':<8} {'p_NHST':<10} {'p_TOST':<10} {'p_fdr':<10} {'Equiv?':<8} {'Verdict'}")
print(f"  {'-'*105}")
for r in all_results:
    eq = "YES" if r['Equivalent_SESOI_fdr'] else "no"
    # Short verdict
    if r['Equivalent_SESOI_fdr'] and not r['NHST_sig']:
        verdict = "Equivalent"
    elif not r['Equivalent_SESOI_fdr'] and r['NHST_sig']:
        verdict = "Different"
    elif r['Equivalent_SESOI_fdr'] and r['NHST_sig']:
        verdict = "Trivial diff."
    else:
        verdict = "Undetermined"
    print(f"  {r['Label']:<35} {r['Cohen_d']:<8.3f} {r['p_NHST']:<10.4f} "
          f"{r['TOST_p_SESOI']:<10.4f} {r['TOST_p_SESOI_fdr']:<10.4f} {eq:<8} {verdict}")

print()

# MDES bounds summary
print("=" * 80)
print("  SUMMARY TABLE — TOST with MDES bounds (d = ±{:.3f})".format(MDES))
print("=" * 80)
print()
print(f"  {'Measure':<35} {'d':<8} {'p_NHST':<10} {'p_TOST':<10} {'p_fdr':<10} {'Equiv?':<8} {'Verdict'}")
print(f"  {'-'*105}")
for r in all_results:
    eq = "YES" if r['Equivalent_MDES_fdr'] else "no"
    if r['Equivalent_MDES_fdr'] and not r['NHST_sig']:
        verdict = "Equivalent"
    elif not r['Equivalent_MDES_fdr'] and r['NHST_sig']:
        verdict = "Different"
    elif r['Equivalent_MDES_fdr'] and r['NHST_sig']:
        verdict = "Trivial diff."
    else:
        verdict = "Undetermined"
    print(f"  {r['Label']:<35} {r['Cohen_d']:<8.3f} {r['p_NHST']:<10.4f} "
          f"{r['TOST_p_MDES']:<10.4f} {r['TOST_p_MDES_fdr']:<10.4f} {eq:<8} {verdict}")

print()

# Overall interpretation
n_equiv_sesoi = sum(1 for r in all_results if r['Equivalent_SESOI_fdr'])
n_equiv_mdes = sum(1 for r in all_results if r['Equivalent_MDES_fdr'])
n_undetermined = sum(1 for r in all_results
                     if not r['Equivalent_SESOI_fdr'] and not r['NHST_sig'])
n_total = len(all_results)

print("=" * 80)
print("  OVERALL INTERPRETATION")
print("=" * 80)
print()
print(f"  With SESOI bounds (d = ±{SESOI_D}):")
print(f"    {n_equiv_sesoi}/{n_total} measures demonstrate statistical equivalence (FDR-corrected)")
print(f"    {n_undetermined}/{n_total} measures are undetermined (neither equivalent nor different)")
print()
print(f"  With MDES bounds (d = ±{MDES:.3f}):")
print(f"    {n_equiv_mdes}/{n_total} measures demonstrate statistical equivalence (FDR-corrected)")
print()

if n_equiv_sesoi >= n_total * 0.7:
    print("  → STRONG support for the dissociation argument:")
    print("    The majority of self-report measures show statistical equivalence,")
    print("    supporting the claim that users are largely unaware of persona effects.")
elif n_equiv_sesoi >= n_total * 0.5:
    print("  → MODERATE support for the dissociation argument:")
    print("    A majority of self-report measures show statistical equivalence,")
    print("    but some remain undetermined, warranting cautious interpretation.")
else:
    print("  → WEAK support for the dissociation argument:")
    print("    Many self-report measures remain undetermined. The non-significant")
    print("    NHST results may reflect insufficient power rather than true equivalence.")
    print("    The claim that 'users are unaware' should be qualified.")

if n_undetermined > 0:
    undetermined_labels = [r['Label'] for r in all_results
                           if not r['Equivalent_SESOI_fdr'] and not r['NHST_sig']]
    print()
    print(f"  Undetermined measures (require larger N to resolve):")
    for lbl in undetermined_labels:
        print(f"    - {lbl}")

print()

# ============================================================
# 4. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "equivalence_testing_tost.csv"), index=False)
print(f"  → Saved: equivalence_testing_tost.csv")

# ============================================================
# 5. PUBLICATION-QUALITY FIGURES
# ============================================================

# --- Figure 1: TOST Equivalence Plot (SESOI bounds) ---
fig, ax = plt.subplots(figsize=(12, 8))

sorted_results = sorted(all_results, key=lambda x: abs(x["Cohen_d"]))
n_measures = len(sorted_results)
y_positions = np.arange(n_measures)

for i, r in enumerate(sorted_results):
    # Convert 90% CI to d units
    ci_low_d = r["CI90_lower_SESOI"] / r["Pooled_SD"] if r["Pooled_SD"] > 0 else 0
    ci_high_d = r["CI90_upper_SESOI"] / r["Pooled_SD"] if r["Pooled_SD"] > 0 else 0
    d = r["Cohen_d"]

    # Color based on equivalence verdict
    if r["Equivalent_SESOI_fdr"]:
        color = '#43A047'  # green = equivalent
        marker = 's'
    elif r["NHST_sig"]:
        color = '#E53935'  # red = different
        marker = 'D'
    else:
        color = '#FF8F00'  # amber = undetermined
        marker = 'o'

    ax.plot([ci_low_d, ci_high_d], [i, i], color=color, linewidth=2.5, solid_capstyle='round')
    ax.plot(d, i, marker=marker, color=color, markersize=8, zorder=5, markeredgecolor='white',
            markeredgewidth=0.8)

# Equivalence bounds
ax.axvline(x=-SESOI_D, color='#1565C0', linestyle='--', linewidth=1.5, alpha=0.6)
ax.axvline(x=SESOI_D, color='#1565C0', linestyle='--', linewidth=1.5, alpha=0.6)
ax.axvspan(-SESOI_D, SESOI_D, alpha=0.08, color='#1565C0')
ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.4)

ax.set_yticks(y_positions)
ax.set_yticklabels([r["Label"] for r in sorted_results], fontsize=10)
ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
ax.set_title(f"Equivalence Testing (TOST) — Self-Report Measures\n"
             f"Equivalence bounds: d = ±{SESOI_D} (SESOI), α = {ALPHA}, 90% CIs",
             fontsize=14, fontweight='bold')

legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#43A047', markersize=10,
           label='Equivalent (within bounds)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8F00', markersize=10,
           label='Undetermined'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#E53935', markersize=10,
           label='Different (outside bounds)'),
    Line2D([0], [0], color='#1565C0', linestyle='--', linewidth=1.5,
           label=f'Equivalence bounds (d = ±{SESOI_D})'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, axis='x', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "tost_equivalence_plot_sesoi.png"),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  → Figure saved: tost_equivalence_plot_sesoi.png")

# --- Figure 2: TOST with MDES bounds ---
fig, ax = plt.subplots(figsize=(12, 8))

for i, r in enumerate(sorted_results):
    ci_low_d = r["CI90_lower_MDES"] / r["Pooled_SD"] if r["Pooled_SD"] > 0 else 0
    ci_high_d = r["CI90_upper_MDES"] / r["Pooled_SD"] if r["Pooled_SD"] > 0 else 0
    d = r["Cohen_d"]

    if r["Equivalent_MDES_fdr"]:
        color = '#43A047'
        marker = 's'
    elif r["NHST_sig"]:
        color = '#E53935'
        marker = 'D'
    else:
        color = '#FF8F00'
        marker = 'o'

    ax.plot([ci_low_d, ci_high_d], [i, i], color=color, linewidth=2.5, solid_capstyle='round')
    ax.plot(d, i, marker=marker, color=color, markersize=8, zorder=5, markeredgecolor='white',
            markeredgewidth=0.8)

ax.axvline(x=-MDES, color='#1565C0', linestyle='--', linewidth=1.5, alpha=0.6)
ax.axvline(x=MDES, color='#1565C0', linestyle='--', linewidth=1.5, alpha=0.6)
ax.axvspan(-MDES, MDES, alpha=0.08, color='#1565C0')
ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.4)

ax.set_yticks(y_positions)
ax.set_yticklabels([r["Label"] for r in sorted_results], fontsize=10)
ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
ax.set_title(f"Equivalence Testing (TOST) — Self-Report Measures\n"
             f"Equivalence bounds: d = ±{MDES:.3f} (MDES at 80% power), α = {ALPHA}, 90% CIs",
             fontsize=14, fontweight='bold')

legend_elements_mdes = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#43A047', markersize=10,
           label='Equivalent (within bounds)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF8F00', markersize=10,
           label='Undetermined'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#E53935', markersize=10,
           label='Different (outside bounds)'),
    Line2D([0], [0], color='#1565C0', linestyle='--', linewidth=1.5,
           label=f'Equivalence bounds (d = ±{MDES:.3f})'),
]
ax.legend(handles=legend_elements_mdes, loc='lower right', fontsize=9, framealpha=0.9)
ax.grid(True, axis='x', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "tost_equivalence_plot_mdes.png"),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  → Figure saved: tost_equivalence_plot_mdes.png")


# --- Figure 3: Combined NHST + TOST Decision Plot ---
fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

for ax_idx, (bound_label, bound_d, p_key, eq_key) in enumerate([
    (f"SESOI (d = ±{SESOI_D})", SESOI_D, "TOST_p_SESOI", "Equivalent_SESOI_fdr"),
    (f"MDES (d = ±{MDES:.3f})", MDES, "TOST_p_MDES", "Equivalent_MDES_fdr"),
]):
    ax = axes[ax_idx]
    for i, r in enumerate(sorted_results):
        # Quadrant coloring
        if r[eq_key] and not r["NHST_sig"]:
            color, label_txt = '#43A047', 'Equivalent'
        elif not r[eq_key] and r["NHST_sig"]:
            color, label_txt = '#E53935', 'Different'
        elif r[eq_key] and r["NHST_sig"]:
            color, label_txt = '#7B1FA2', 'Trivial diff.'
        else:
            color, label_txt = '#FF8F00', 'Undetermined'

        ax.barh(i, abs(r["Cohen_d"]), color=color, alpha=0.7, edgecolor='white', height=0.6)
        ax.text(abs(r["Cohen_d"]) + 0.02, i, f'{label_txt}', va='center', fontsize=8, color=color)

    ax.axvline(x=bound_d, color='#1565C0', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'Bound = {bound_d:.3f}')
    ax.set_xlabel("|Cohen's d|", fontsize=11)
    ax.set_title(f"Bounds: {bound_label}", fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.3)

axes[0].set_yticks(y_positions)
axes[0].set_yticklabels([r["Label"] for r in sorted_results], fontsize=10)

# Shared legend
legend_elements_combined = [
    mpatches.Patch(facecolor='#43A047', alpha=0.7, label='Equivalent'),
    mpatches.Patch(facecolor='#FF8F00', alpha=0.7, label='Undetermined'),
    mpatches.Patch(facecolor='#E53935', alpha=0.7, label='Different'),
    mpatches.Patch(facecolor='#7B1FA2', alpha=0.7, label='Trivially different'),
]
fig.legend(handles=legend_elements_combined, loc='lower center', ncol=4,
           fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("NHST + TOST Decision Framework (Lakens, 2017)\nSelf-Report Measures: Servant Leader vs. Dark Triad",
             fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "tost_nhst_decision_plot.png"),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  → Figure saved: tost_nhst_decision_plot.png")

print()
print("=" * 80)
print("  EQUIVALENCE TESTING COMPLETE")
print("=" * 80)
