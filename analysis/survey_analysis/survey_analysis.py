# %% survey_analysis.py

"""
Psychophysiological Study: Self-Report Analysis Pipeline
=========================================================
Analyses of Qualtrics data for the AI Chatbot Leadership Styles study.
Enhanced with paper-ready console output and publication-quality visualizations.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind, chi2_contingency, pearsonr, spearmanr
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import os
from dotenv import load_dotenv

# ============================================================
# 0. SETUP & CONFIGURATION
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

# Visualization parameters
VIZ_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 300,
    'confidence_alpha': 0.3,
    'line_width': 2.0
}
COLOR_SL = '#64B5F6'   # Soft blue (Servant Leader)
COLOR_DT = '#E57373'   # Soft red (Dark Triad)
COLOR_GREEN = '#81C784'
COLOR_ORANGE = '#FFB74D'
PALETTE = [COLOR_SL, COLOR_DT, COLOR_GREEN, COLOR_ORANGE]

# Global matplotlib settings
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

GROUP_LABELS = {1: "Servant Leader", 2: "Dark Triad"}
ALPHA = 0.05
TASKS = ["Persona", "Brief", "Communication", "Content"]

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

def cronbachs_alpha(items_df):
    items_df = items_df.dropna()
    k = items_df.shape[1]
    if k < 2:
        return np.nan
    item_vars = items_df.var(axis=0, ddof=1)
    total_var = items_df.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan
    return (k / (k - 1)) * (1 - item_vars.sum() / total_var)

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.std()**2 + (n2-1)*g2.std()**2) / (n1+n2-2))
    return (g1.mean() - g2.mean()) / pooled if pooled > 0 else 0

def interpret_d(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"

def sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    else: return ""

def group_comparison(data, var, group_col="Group"):
    g1 = data.loc[data[group_col] == 1, var].dropna()
    g2 = data.loc[data[group_col] == 2, var].dropna()
    if len(g1) < 3 or len(g2) < 3:
        return None
    u_stat, u_p = mannwhitneyu(g1, g2, alternative="two-sided")
    t_stat, t_p = ttest_ind(g1, g2, equal_var=False)
    df_welch = ((g1.var()/len(g1) + g2.var()/len(g2))**2 /
                ((g1.var()/len(g1))**2/(len(g1)-1) + (g2.var()/len(g2))**2/(len(g2)-1)))
    d = cohens_d(g1, g2)
    return {
        "Variable": var,
        "n_SL": len(g1), "M_SL": g1.mean(), "SD_SL": g1.std(),
        "n_DT": len(g2), "M_DT": g2.mean(), "SD_DT": g2.std(),
        "U": u_stat, "U_p": u_p,
        "t": t_stat, "df": df_welch, "t_p": t_p,
        "Cohen_d": d, "d_interp": interpret_d(d),
    }

def print_comparison(res, label=None):
    name = label or res["Variable"]
    s = sig_stars(res["U_p"])
    print(f"  {name}:")
    print(f"    Servant Leader (n={res['n_SL']}): M = {res['M_SL']:.2f}, SD = {res['SD_SL']:.2f}")
    print(f"    Dark Triad     (n={res['n_DT']}): M = {res['M_DT']:.2f}, SD = {res['SD_DT']:.2f}")
    print(f"    Mann-Whitney U = {res['U']:.1f}, p = {res['U_p']:.4f}{s}")
    print(f"    Welch's t({res['df']:.1f}) = {res['t']:.3f}, p = {res['t_p']:.4f}")
    print(f"    Cohen's d = {res['Cohen_d']:.3f} ({res['d_interp']})")
    print()

def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    → Figure saved: {name}")


# ============================================================
# DATA LOADING
# ============================================================

print("=" * 80)
print("  SELF-REPORT ANALYSIS: AI CHATBOT LEADERSHIP STYLES STUDY")
print("=" * 80)
print()

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Group_label"] = df["Group"].map(GROUP_LABELS)

n_total = len(df)
n_sl = (df['Group'] == 1).sum()
n_dt = (df['Group'] == 2).sum()
print(f"  Sample: N = {n_total} (Servant Leader: n = {n_sl}, Dark Triad: n = {n_dt})")
print()


# ============================================================
# 1. SCALE CONSTRUCTION
# ============================================================

print("=" * 80)
print("  1. SCALE CONSTRUCTION")
print("=" * 80)
print()

# --- NEO-60 ---
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

print("  Big Five personality traits computed (IPIP-NEO-60, 12 items each).")

# --- Pre-task scales ---
ai_exp_cols = ["Experience_with_AI_1", "Experience_with_AI_2", "Experience_with_AI_3"]
df["AI_Experience"] = df[ai_exp_cols].mean(axis=1)

tech_trust_cols = ["Technology_Trust_1", "Technology_Trust_2", "Technology_Trust_3"]
df["Technology_Trust"] = df[tech_trust_cols].mean(axis=1)

ai_fear_cols_raw = ["AI_fear_1", "AI_fear_2", "AI_fear_3", "AI_fear_4", "AI_fear_5"]
df["AI_fear_2_R"] = reverse_score(df["AI_fear_2"], 5)
df["AI_fear_4_R"] = reverse_score(df["AI_fear_4"], 5)
ai_fear_scored = ["AI_fear_1", "AI_fear_2_R", "AI_fear_3", "AI_fear_4_R", "AI_fear_5"]
df["AI_Fear_composite"] = df[ai_fear_scored].mean(axis=1)

ai_collab_pre_cols = ["AI_collaborators_1", "AI_collaborators_2", "AI_collaborators_3"]
df["AI_Collaborators_pre"] = df[ai_collab_pre_cols].mean(axis=1)

ai_will_pre_cols = ["AI_willingness_to_collaborate_1", "AI_willingness_to_collaborate_2",
                    "AI_willingness_to_collaborate_3", "AI_willingness_to_collaborate_4"]
df["AI_Willingness_pre"] = df[ai_will_pre_cols].mean(axis=1)

print("  Pre-task scales: AI_Experience, Technology_Trust, AI_Fear_composite,")
print("    AI_Collaborators_pre, AI_Willingness_pre")

# --- Task-specific composites ---
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

    enj_cols = []
    for i in range(1, 4):
        col = find_col(df, [f"Enjoyment_{task}_{i}"])
        if col:
            enj_cols.append(col)
    if enj_cols:
        df[f"Enjoyment_{task}"] = df[enj_cols].mean(axis=1)

    sat_cols = []
    for i in [1, 2]:
        col = find_col(df, [f"Satisfaction_{task}_{i}"])
        if col:
            sat_cols.append(col)
    rek_col = find_col(df, [f"Satisfaction_{task}_3_rek"])
    if rek_col:
        sat_cols.append(rek_col)
    if sat_cols:
        df[f"Satisfaction_{task}"] = df[sat_cols].mean(axis=1)

print("  Task composites: Focused_Attention, Enjoyment, Satisfaction (×4 tasks)")

# --- Post-task scales ---
df["AI_Collaboration_post"] = df[["AI_collab_post_1","AI_collab_post_2","AI_collab_post_3"]].mean(axis=1)
df["AI_Reliability"] = df[["AI_collab_post_4","AI_collab_post_5","AI_collab_post_6"]].mean(axis=1)
df["Behavioral_Intention"] = df[["AI_collab_post_7","AI_collab_post_8","AI_collab_post_9"]].mean(axis=1)
df["AI_Willingness_post"] = df[[f"AI_collab_post_{i}" for i in range(10,14)]].mean(axis=1)

df["AI_collab_post_15_R"] = reverse_score(df["AI_collab_post_15"], 5)
df["AI_collab_post_17_R"] = reverse_score(df["AI_collab_post_17"], 5)
ai_att_post_scored = ["AI_collab_post_14","AI_collab_post_15_R","AI_collab_post_16",
                      "AI_collab_post_17_R","AI_collab_post_18"]
df["AI_Fear_post"] = df[ai_att_post_scored].mean(axis=1)

df["Human_AI_Fit"] = df[["AI_collab_post_19","AI_collab_post_20","AI_collab_post_21"]].mean(axis=1)
anthro_cols = ["Lifelike", "Humanlike", "Natural", "Conscious"]
df["Anthropomorphism"] = df[anthro_cols].mean(axis=1)

# Task averages
for m in ["Focused_Attention", "Enjoyment", "Satisfaction"]:
    cols = [f"{m}_{t}" for t in TASKS if f"{m}_{t}" in df.columns]
    if cols:
        df[f"{m}_avg"] = df[cols].mean(axis=1)

print("  Post-task scales: AI_Collaboration_post, AI_Reliability, Behavioral_Intention,")
print("    AI_Willingness_post, AI_Fear_post, Human_AI_Fit, Anthropomorphism")
print()


# ============================================================
# 2. SCALE RELIABILITY
# ============================================================

print("=" * 80)
print("  2. SCALE RELIABILITY (Cronbach's Alpha)")
print("=" * 80)
print()

reliability_results = []

def compute_alpha(name, cols, use_reversed=None):
    actual = []
    for c in cols:
        if use_reversed and c in use_reversed:
            actual.append(use_reversed[c])
        else:
            actual.append(c)
    valid = [c for c in actual if c in df.columns]
    if len(valid) < 2:
        return
    a = cronbachs_alpha(df[valid])
    flag = "" if a >= 0.70 else " [below .70]"
    reliability_results.append({"Scale": name, "N_items": len(valid), "Alpha": round(a, 3)})
    print(f"  {name}: α = {a:.3f} ({len(valid)} items){flag}")

compute_alpha("AI Experience", ai_exp_cols)
compute_alpha("Technology Trust", tech_trust_cols)
compute_alpha("AI Fear (pre)", ai_fear_cols_raw,
              {"AI_fear_2": "AI_fear_2_R", "AI_fear_4": "AI_fear_4_R"})
compute_alpha("AI Collaborators (pre)", ai_collab_pre_cols)
compute_alpha("AI Willingness (pre)", ai_will_pre_cols)

for trait, info in NEO_SCORING.items():
    scored = [f"Personality_{i}_R" if i in info["reverse"] else f"Personality_{i}" for i in info["items"]]
    compute_alpha(f"NEO: {trait}", scored)

for task in TASKS:
    fa_alpha_cols = []
    for i in range(1, 4):
        col = find_col(df, [f"Focused_Attention_{task}_{i}",
                            f"Focused_Attention_{task[:6]}_{i}",
                            f"Focused_Attention_{task[:7]}_{i}"])
        if col:
            fa_alpha_cols.append(col)
    compute_alpha(f"Focused Attention ({task})", fa_alpha_cols)
    compute_alpha(f"Enjoyment ({task})", [f"Enjoyment_{task}_{i}" for i in range(1,4)])
    compute_alpha(f"Satisfaction ({task})",
                  [f"Satisfaction_{task}_1", f"Satisfaction_{task}_2", f"Satisfaction_{task}_3_rek"])

compute_alpha("AI Collaboration (post)", [f"AI_collab_post_{i}" for i in range(1,4)])
compute_alpha("AI Reliability", [f"AI_collab_post_{i}" for i in range(4,7)])
compute_alpha("Behavioral Intention", [f"AI_collab_post_{i}" for i in range(7,10)])
compute_alpha("AI Willingness (post)", [f"AI_collab_post_{i}" for i in range(10,14)])
compute_alpha("AI Attitudes (post)", ai_att_post_scored)
compute_alpha("Human-AI Fit", [f"AI_collab_post_{i}" for i in range(19,22)])
compute_alpha("Anthropomorphism", anthro_cols)

reliability_df = pd.DataFrame(reliability_results)
reliability_df.to_csv(os.path.join(OUTPUT_DIR, "reliability_cronbach_alpha.csv"), index=False)

acceptable = sum(1 for r in reliability_results if r["Alpha"] >= 0.70)
print(f"\n  Summary: {acceptable}/{len(reliability_results)} scales meet α ≥ .70 threshold.")
print(f"  → Saved: reliability_cronbach_alpha.csv")
print()


# ============================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================

print("=" * 80)
print("  3. DESCRIPTIVE STATISTICS")
print("=" * 80)
print()

# Demographics
print("  --- Demographics ---")
print()
age_vals = df["Age"].dropna()
# Check if Age is categorical
if age_vals.max() <= 10:
    print(f"  NOTE: Age appears to be categorically coded (range: {int(age_vals.min())}-{int(age_vals.max())}).")
    print(f"    Distribution:")
    for val, cnt in age_vals.value_counts().sort_index().items():
        print(f"      Category {int(val)}: n = {cnt} ({cnt/len(age_vals)*100:.1f}%)")
else:
    print(f"  Age: M = {age_vals.mean():.1f}, SD = {age_vals.std():.1f}, "
          f"Range = {age_vals.min():.0f}-{age_vals.max():.0f}")

print()
if "Gender" in df.columns:
    gc = df["Gender"].value_counts()
    print("  Gender distribution:")
    for g, c in gc.items():
        print(f"    Code {g}: n = {c} ({c/len(df)*100:.1f}%)")
    print()

print(f"  Current Mood (1-10): M = {df['Mood_1'].mean():.2f}, SD = {df['Mood_1'].std():.2f}")
print()

# Build full descriptives table
print("  --- Scale Descriptives by Condition ---")
print()
desc_scales = {
    "AI_Experience": "AI Experience (pre)", "Technology_Trust": "Technology Trust (pre)",
    "AI_Fear_composite": "AI Fear (pre)", "AI_Collaborators_pre": "AI Collaborators (pre)",
    "AI_Willingness_pre": "AI Willingness (pre)",
    "Neuroticism": "Neuroticism", "Extraversion": "Extraversion", "Openness": "Openness",
    "Agreeableness": "Agreeableness", "Conscientiousness": "Conscientiousness", "Mood_1": "Mood",
    "AI_Collaboration_post": "AI Collaboration (post)", "AI_Reliability": "AI Reliability",
    "Behavioral_Intention": "Behavioral Intention", "AI_Willingness_post": "AI Willingness (post)",
    "AI_Fear_post": "AI Fear (post)", "Human_AI_Fit": "Human-AI Fit",
    "Anthropomorphism": "Anthropomorphism",
}
for task in TASKS:
    for m in ["Focused_Attention", "Enjoyment", "Satisfaction"]:
        k = f"{m}_{task}"
        if k in df.columns:
            desc_scales[k] = f"{m.replace('_',' ')} ({task})"

desc_rows = []
for col, label in desc_scales.items():
    if col not in df.columns:
        continue
    ov = df[col].dropna()
    g1 = df.loc[df["Group"]==1, col].dropna()
    g2 = df.loc[df["Group"]==2, col].dropna()
    desc_rows.append({
        "Variable": label, "N": len(ov),
        "M_total": f"{ov.mean():.2f}", "SD_total": f"{ov.std():.2f}",
        "M_SL": f"{g1.mean():.2f}", "SD_SL": f"{g1.std():.2f}",
        "M_DT": f"{g2.mean():.2f}", "SD_DT": f"{g2.std():.2f}",
    })

desc_df = pd.DataFrame(desc_rows)
print(desc_df.to_string(index=False))
desc_df.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"), index=False)
print(f"\n  → Saved: descriptive_statistics.csv")
print()


# ============================================================
# 4. BASELINE EQUIVALENCE
# ============================================================

print("=" * 80)
print("  4. BASELINE EQUIVALENCE (Randomization Check)")
print("=" * 80)
print()

baseline_vars = [
    ("Age", "Age"), ("Mood_1", "Current Mood"),
    ("AI_Experience", "AI Experience"), ("Technology_Trust", "Technology Trust"),
    ("AI_Fear_composite", "AI Fear"), ("AI_Collaborators_pre", "AI Collaborators"),
    ("AI_Willingness_pre", "AI Willingness"),
    ("Neuroticism", "Neuroticism"), ("Extraversion", "Extraversion"),
    ("Openness", "Openness"), ("Agreeableness", "Agreeableness"),
    ("Conscientiousness", "Conscientiousness"),
]

baseline_results = []
sig_baseline = []
for var, label in baseline_vars:
    res = group_comparison(df, var)
    if res:
        baseline_results.append(res)
        print_comparison(res, label)
        if res["U_p"] < 0.05:
            sig_baseline.append(label)

# Gender
if "Gender" in df.columns:
    ct = pd.crosstab(df["Group"], df["Gender"])
    chi2, chi_p, dof, _ = chi2_contingency(ct)
    n = ct.sum().sum()
    cv = np.sqrt(chi2 / (n * (min(ct.shape)-1))) if min(ct.shape) > 1 else 0
    print(f"  Gender: χ²({dof}) = {chi2:.2f}, p = {chi_p:.4f}, Cramér's V = {cv:.3f}")
    print()

if sig_baseline:
    print(f"WARNING: Significant baseline differences found for: {', '.join(sig_baseline)}")
    print(f"    These variables should be considered as covariates in further analyses.")
else:
    print(f"  No significant baseline differences detected (all p > .05).")
    print(f"    Randomization appears successful.")

baseline_df = pd.DataFrame(baseline_results)
baseline_df.to_csv(os.path.join(OUTPUT_DIR, "baseline_equivalence.csv"), index=False)
print(f"\n  → Saved: baseline_equivalence.csv")
print()


# ============================================================
# 5. CONDITION COMPARISONS (Post-Task)
# ============================================================

print("=" * 80)
print("  5. CONDITION COMPARISONS (Post-Task Measures)")
print("=" * 80)
print()

posttask_defs = [
    ("AI_Collaboration_post", "Perceived AI Collaboration"),
    ("AI_Reliability", "AI Reliability"),
    ("Behavioral_Intention", "Behavioral Intention to Use"),
    ("AI_Willingness_post", "Intention to Collaborate (post)"),
    ("AI_Fear_post", "AI Fear/Attitudes (post)"),
    ("Human_AI_Fit", "Human-AI Fit"),
    ("Anthropomorphism", "Anthropomorphism"),
]

condition_results = []
for var, label in posttask_defs:
    if var not in df.columns:
        continue
    res = group_comparison(df, var)
    if res:
        res["Label"] = label
        condition_results.append(res)
        print_comparison(res, label)

# FDR correction
if condition_results:
    pvals = [r["U_p"] for r in condition_results]
    reject, p_adj = pg.multicomp(pvals, method="fdr_bh")
    for i, r in enumerate(condition_results):
        r["U_p_fdr"] = p_adj[i]
        r["Sig_fdr"] = reject[i]

    print("  --- FDR-Corrected Summary (Benjamini-Hochberg) ---")
    print(f"  {'Measure':<40} {'p_raw':<10} {'p_fdr':<10} {'d':<8} {'Sig?':<5}")
    print(f"  {'-'*75}")
    for r in condition_results:
        s = "YES" if r["Sig_fdr"] else "no"
        print(f"  {r['Label']:<40} {r['U_p']:<10.4f} {r['U_p_fdr']:<10.4f} "
              f"{r['Cohen_d']:<8.3f} {s}")

cond_df = pd.DataFrame(condition_results)
cond_df.to_csv(os.path.join(OUTPUT_DIR, "condition_comparisons_posttask.csv"), index=False)
print(f"\n  → Saved: condition_comparisons_posttask.csv")
print()

# --- Task-specific single items ---
print("  --- Task-Specific Single Items by Condition ---")
print()
task_single_results = []
for task in TASKS:
    for prefix_options in [["Familarity", "Familiarity"], ["Difficulty"], ["Creativity"]]:
        for prefix in prefix_options:
            col = f"{prefix}_{task}"
            if col in df.columns:
                res = group_comparison(df, col)
                if res:
                    label = f"{prefix} ({task})"
                    res["Label"] = label
                    task_single_results.append(res)
                    s = sig_stars(res["U_p"])
                    if s:
                        print_comparison(res, label)
                    else:
                        print(f"  {label}: SL={res['M_SL']:.2f}±{res['SD_SL']:.2f}, "
                              f"DT={res['M_DT']:.2f}±{res['SD_DT']:.2f}, "
                              f"U={res['U']:.0f}, p={res['U_p']:.4f}, d={res['Cohen_d']:.3f}")
                break

task_single_df = pd.DataFrame(task_single_results)
task_single_df.to_csv(os.path.join(OUTPUT_DIR, "condition_comparisons_task_items.csv"), index=False)
print(f"\n  → Saved: condition_comparisons_task_items.csv")
print()


# ============================================================
# 6. PRE-POST COMPARISONS
# ============================================================

print("=" * 80)
print("  6. PRE-POST COMPARISONS (Within-Subject Changes)")
print("=" * 80)
print()

prepost_pairs = [
    ("AI_Collaborators_pre", "AI_Collaboration_post", "Perceived AI Collaboration"),
    ("AI_Willingness_pre", "AI_Willingness_post", "Intention to Collaborate"),
    ("AI_Fear_composite", "AI_Fear_post", "AI Fear/Attitudes"),
]

prepost_results = []
for pre_col, post_col, label in prepost_pairs:
    if pre_col not in df.columns or post_col not in df.columns:
        continue
    valid = df[[pre_col, post_col, "Group"]].dropna()
    diff = valid[post_col] - valid[pre_col]
    t_all, p_all = stats.ttest_rel(valid[pre_col], valid[post_col])
    d_all = diff.mean() / diff.std() if diff.std() > 0 else 0

    print(f"  {label}:")
    print(f"    Overall (N={len(valid)}): Pre M={valid[pre_col].mean():.2f}, "
          f"Post M={valid[post_col].mean():.2f}, "
          f"Δ = {diff.mean():.2f}")
    print(f"    Paired t({len(valid)-1}) = {t_all:.3f}, p = {p_all:.4f}, d = {d_all:.3f}")
    print()

    for grp in [1, 2]:
        g = valid[valid["Group"] == grp]
        if len(g) < 3:
            continue
        dg = g[post_col] - g[pre_col]
        tg, pg_ = stats.ttest_rel(g[pre_col], g[post_col])
        dg_d = dg.mean() / dg.std() if dg.std() > 0 else 0
        gl = GROUP_LABELS[grp]
        s = sig_stars(pg_)
        print(f"    {gl} (n={len(g)}): Pre={g[pre_col].mean():.2f}, "
              f"Post={g[post_col].mean():.2f}, Δ={dg.mean():.2f}, "
              f"t({len(g)-1})={tg:.3f}, p={pg_:.4f}{s}, d={dg_d:.3f}")
        prepost_results.append({
            "Measure": label, "Group": gl,
            "n": len(g),
            "M_pre": round(g[pre_col].mean(), 2), "SD_pre": round(g[pre_col].std(), 2),
            "M_post": round(g[post_col].mean(), 2), "SD_post": round(g[post_col].std(), 2),
            "Delta": round(dg.mean(), 2),
            "t": round(tg, 3), "p": round(pg_, 4), "d": round(dg_d, 3)
        })
    print()

prepost_df = pd.DataFrame(prepost_results)
prepost_df.to_csv(os.path.join(OUTPUT_DIR, "pre_post_comparisons.csv"), index=False)
print(f"  → Saved: pre_post_comparisons.csv")
print()


# ============================================================
# 7. TASK-SPECIFIC ANALYSES (Condition × Task)
# ============================================================

print("=" * 80)
print("  7. TASK-SPECIFIC ANALYSES (Mixed ANOVA: Group × Task)")
print("=" * 80)
print()

task_measures = ["Focused_Attention", "Enjoyment", "Satisfaction"]
task_anova_results = []

for measure in task_measures:
    print(f"  --- {measure.replace('_', ' ')} ---")
    long_rows = []
    for task in TASKS:
        col = f"{measure}_{task}"
        if col not in df.columns:
            continue
        for _, row in df.iterrows():
            if pd.notna(row[col]):
                long_rows.append({
                    "participantID": row["participantID"],
                    "Group": row["Group"],
                    "Task": task,
                    "Score": row[col]
                })
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        continue

    try:
        aov = pg.mixed_anova(data=long_df, dv="Score", within="Task",
                             between="Group", subject="participantID")
        aov["Measure"] = measure
        task_anova_results.append(aov)

        for _, r in aov.iterrows():
            src = r["Source"]
            f_val = r["F"]
            p_val = r["p-unc"]
            np2 = r["np2"]
            s = sig_stars(p_val)
            gg = f", p_GG = {r['p-GG-corr']:.4f}" if pd.notna(r.get("p-GG-corr")) else ""
            print(f"    {src}: F({r['DF1']:.0f},{r['DF2']:.0f}) = {f_val:.3f}, "
                  f"p = {p_val:.4f}{s}, ηp² = {np2:.3f}{gg}")
        print()

        # Post-hoc per task
        for task in TASKS:
            col = f"{measure}_{task}"
            if col not in df.columns:
                continue
            res = group_comparison(df, col)
            if res:
                s = sig_stars(res["U_p"])
                print(f"    {task}: SL={res['M_SL']:.2f}±{res['SD_SL']:.2f}, "
                      f"DT={res['M_DT']:.2f}±{res['SD_DT']:.2f}, "
                      f"d={res['Cohen_d']:.3f}{s}")
        print()
    except Exception as e:
        print(f"    Mixed ANOVA failed: {e}\n")

if task_anova_results:
    anova_df = pd.concat(task_anova_results, ignore_index=True)
    anova_df.to_csv(os.path.join(OUTPUT_DIR, "task_mixed_anova.csv"), index=False)
    print(f"  → Saved: task_mixed_anova.csv")
print()


# ============================================================
# 8. CORRELATIONAL ANALYSES
# ============================================================

print("=" * 80)
print("  8. CORRELATIONAL ANALYSES")
print("=" * 80)
print()

corr_vars = [
    "Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness",
    "AI_Experience", "Technology_Trust", "AI_Fear_composite",
    "AI_Collaborators_pre", "AI_Willingness_pre",
    "AI_Collaboration_post", "AI_Reliability", "Behavioral_Intention",
    "AI_Willingness_post", "Human_AI_Fit", "Anthropomorphism",
    "Average_all", "Focused_Attention_avg", "Enjoyment_avg", "Satisfaction_avg"
]
corr_vars = [v for v in corr_vars if v in df.columns]

corr_matrix = df[corr_vars].corr(method="pearson")
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_matrix_pearson.csv"))

# P-value matrix
n_cv = len(corr_vars)
pval_matrix = pd.DataFrame(np.ones((n_cv, n_cv)), index=corr_vars, columns=corr_vars)
for i in range(n_cv):
    for j in range(i+1, n_cv):
        pair = df[[corr_vars[i], corr_vars[j]]].dropna()
        if len(pair) > 3:
            r, p = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
            pval_matrix.iloc[i, j] = p
            pval_matrix.iloc[j, i] = p

pval_matrix.to_csv(os.path.join(OUTPUT_DIR, "correlation_pvalues.csv"))

# Print noteworthy correlations
print("  Noteworthy correlations (p < .05, |r| ≥ .30):\n")
sig_corrs = []
for i in range(n_cv):
    for j in range(i+1, n_cv):
        r = corr_matrix.iloc[i, j]
        p = pval_matrix.iloc[i, j]
        if p < 0.05 and abs(r) >= 0.30:
            sig_corrs.append({"Var1": corr_vars[i], "Var2": corr_vars[j],
                              "r": round(r, 3), "p": round(p, 4)})
            s = sig_stars(p)
            print(f"    {corr_vars[i]} ↔ {corr_vars[j]}: r = {r:.3f}, p = {p:.4f}{s}")

pd.DataFrame(sig_corrs).to_csv(os.path.join(OUTPUT_DIR, "significant_correlations.csv"), index=False)
print(f"\n  Total significant correlations (|r|≥.30, p<.05): {len(sig_corrs)}")
print(f"  → Saved: correlation_matrix_pearson.csv, significant_correlations.csv")
print()


# ============================================================
# 9. INDIVIDUAL DIFFERENCE MODERATION
# ============================================================

print("=" * 80)
print("  9. INDIVIDUAL DIFFERENCE MODERATION (Group × Moderator Interactions)")
print("=" * 80)
print()

moderators = ["Neuroticism", "Extraversion", "Openness", "Agreeableness",
              "Conscientiousness", "AI_Experience", "Technology_Trust", "AI_Fear_composite"]
outcomes = ["Enjoyment_avg", "Satisfaction_avg", "AI_Collaboration_post",
            "AI_Reliability", "Human_AI_Fit", "Anthropomorphism", "Average_all"]

mod_results = []
for outcome in outcomes:
    if outcome not in df.columns:
        continue
    for mod in moderators:
        if mod not in df.columns:
            continue
        temp = df[["Group", mod, outcome]].dropna()
        if len(temp) < 10:
            continue
        temp["mod_c"] = temp[mod] - temp[mod].mean()
        temp["group_c"] = temp["Group"] - temp["Group"].mean()
        temp["interaction"] = temp["group_c"] * temp["mod_c"]
        try:
            result = pg.linear_regression(temp[["group_c","mod_c","interaction"]], temp[outcome])
            ir = result[result["names"] == "interaction"]
            if not ir.empty:
                b = ir["coef"].values[0]
                se = ir["se"].values[0]
                t = ir["T"].values[0]
                p = ir["pval"].values[0]
                r2 = result["r2"].values[0]
                mod_results.append({
                    "Outcome": outcome, "Moderator": mod,
                    "b_interaction": round(b, 3), "SE": round(se, 3),
                    "t": round(t, 3), "p": round(p, 4),
                    "R2_model": round(r2, 3), "Significant": p < 0.05
                })
        except Exception:
            pass

mod_df = pd.DataFrame(mod_results)
if not mod_df.empty:
    reject, p_adj = pg.multicomp(mod_df["p"].values, method="fdr_bh")
    mod_df["p_fdr"] = np.round(p_adj, 4)
    mod_df["Sig_fdr"] = reject

    # Print significant
    sig_mod = mod_df[mod_df["Significant"]]
    if len(sig_mod) > 0:
        print("  Significant interactions (uncorrected p < .05):\n")
        for _, r in sig_mod.iterrows():
            s = sig_stars(r["p"])
            fdr_note = " [survives FDR]" if r["Sig_fdr"] else " [does NOT survive FDR]"
            print(f"    {r['Outcome']} ~ Group × {r['Moderator']}:")
            print(f"      b = {r['b_interaction']:.3f}, SE = {r['SE']:.3f}, "
                  f"t = {r['t']:.3f}, p = {r['p']:.4f}{s}{fdr_note}")
            print()
    else:
        print("  No significant Group × Moderator interactions detected.\n")

    print(f"  Total tests: {len(mod_df)}")
    print(f"  Significant (uncorrected): {mod_df['Significant'].sum()}")
    print(f"  Significant (FDR-corrected): {mod_df['Sig_fdr'].sum()}")

mod_df.to_csv(os.path.join(OUTPUT_DIR, "moderation_results.csv"), index=False)
print(f"\n  → Saved: moderation_results.csv")
print()


# ============================================================
# 10. OPEN-ENDED KEYWORD ANALYSIS
# ============================================================

print("=" * 80)
print("  10. OPEN-ENDED KEYWORD ANALYSIS")
print("=" * 80)
print()

pos_kw = ["good","great","helpful","nice","like","enjoy","easy","useful","interesting","fun","well","appreciate","love"]
neg_kw = ["bad","difficult","hard","annoying","frustrating","irritating","confusing","unhelpful","brat","whining","judgment","predictable"]

if "Open_ended" in df.columns:
    oe = df[["participantID","Group","Group_label","Open_ended"]].dropna(subset=["Open_ended"]).copy()
    oe["text_lower"] = oe["Open_ended"].str.lower()

    for kw in pos_kw:
        oe[f"pos_{kw}"] = oe["text_lower"].str.contains(rf"\b{kw}\b", regex=True).astype(int)
    for kw in neg_kw:
        oe[f"neg_{kw}"] = oe["text_lower"].str.contains(rf"\b{kw}\b", regex=True).astype(int)

    pos_c = [f"pos_{k}" for k in pos_kw]
    neg_c = [f"neg_{k}" for k in neg_kw]
    oe["n_positive"] = oe[pos_c].sum(axis=1)
    oe["n_negative"] = oe[neg_c].sum(axis=1)
    oe["sentiment_ratio"] = oe["n_positive"] / (oe["n_positive"] + oe["n_negative"]).replace(0, np.nan)

    for grp in [1, 2]:
        g = oe[oe["Group"] == grp]
        gl = GROUP_LABELS[grp]
        print(f"  {gl} (n = {len(g)}):")
        print(f"    Mean positive keywords: {g['n_positive'].mean():.2f} (SD = {g['n_positive'].std():.2f})")
        print(f"    Mean negative keywords: {g['n_negative'].mean():.2f} (SD = {g['n_negative'].std():.2f})")
        top_p = [(c.replace("pos_",""), int(v)) for c,v in g[pos_c].sum().sort_values(ascending=False).head(5).items() if v>0]
        top_n = [(c.replace("neg_",""), int(v)) for c,v in g[neg_c].sum().sort_values(ascending=False).head(5).items() if v>0]
        print(f"    Top positive: {top_p}")
        print(f"    Top negative: {top_n}")
        print()

    for metric in ["n_positive", "n_negative"]:
        g1 = oe.loc[oe["Group"]==1, metric]
        g2 = oe.loc[oe["Group"]==2, metric]
        u, p = mannwhitneyu(g1, g2, alternative="two-sided")
        s = sig_stars(p)
        print(f"  Group comparison ({metric}): U = {u:.1f}, p = {p:.4f}{s}")

    oe[["participantID","Group_label","Open_ended","n_positive","n_negative"]].to_csv(
        os.path.join(OUTPUT_DIR, "open_ended_responses.csv"), index=False)
    print(f"\n  → Saved: open_ended_responses.csv")
print()


# ============================================================
# 11. TASK PERFORMANCE
# ============================================================

print("=" * 80)
print("  11. TASK PERFORMANCE ANALYSIS")
print("=" * 80)
print()

# --- 11a. Condition comparisons ---
print("  --- 11a. Task Performance by Condition ---")
print()

perf_defs = [
    ("Average_all", "Overall Quality (all judges, all dimensions)"),
    ("Completeness_avg", "Completeness"), ("Originality_avg", "Originality/Creativity"),
    ("Strategic_fit_avg", "Strategic Fit"),
    ("Average_Agata", "Judge 1 (Ag)"), ("Average_Alicja", "Judge 2 (Al)"),
    ("Average_Natalia", "Judge 3 (N)"),
]

perf_results = []
for col, label in perf_defs:
    if col not in df.columns:
        continue
    res = group_comparison(df, col)
    if res:
        res["Label"] = label
        perf_results.append(res)
        print_comparison(res, label)

perf_df = pd.DataFrame(perf_results)
perf_df.to_csv(os.path.join(OUTPUT_DIR, "task_performance_comparisons.csv"), index=False)
print(f"  → Saved: task_performance_comparisons.csv")
print()

# --- 11b. Inter-rater reliability ---
print("  --- 11b. Inter-Rater Reliability (ICC) ---")
print()

icc_dims = {
    "Completeness": ["Completeness_Agata","Completeness_Alicja","Completeness_Natalia"],
    "Originality": ["Originality_and_creativity_Agata","Originality_and_creativity_Alicja",
                     "Originality_and_creativity_Natalia"],
    "Strategic Fit": ["Strategic_fit_Agata","Strategic_fit_Alicja","Strategic_fit_Natalia"],
    "Overall Average": ["Average_Agata","Average_Alicja","Average_Natalia"],
}

icc_results = []
for dim, cols in icc_dims.items():
    valid_cols = [c for c in cols if c in df.columns]
    if len(valid_cols) < 2:
        continue
    icc_rows = []
    for _, row in df.iterrows():
        for j, c in enumerate(valid_cols):
            if pd.notna(row[c]):
                icc_rows.append({"target": row["participantID"], "rater": f"Judge_{j+1}", "rating": row[c]})
    icc_long = pd.DataFrame(icc_rows)
    if icc_long.empty:
        continue
    try:
        icc_out = pg.intraclass_corr(data=icc_long, targets="target", raters="rater", ratings="rating")
        icc2k = icc_out[icc_out["Type"] == "ICC2k"]
        if not icc2k.empty:
            val = icc2k["ICC"].values[0]
            ci = icc2k["CI95%"].values[0]
            pv = icc2k["pval"].values[0]
            flag = "" if val >= 0.70 else "  [below 0.70]"
            print(f"  {dim}: ICC(2,k) = {val:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}], p = {pv:.4f}{flag}")
            icc_results.append({"Dimension": dim, "ICC_2k": round(val,3),
                                "CI_low": round(ci[0],3), "CI_high": round(ci[1],3),
                                "p": round(pv,4), "Acceptable": val >= 0.70})
    except Exception as e:
        print(f"  {dim}: ICC failed: {e}")

pd.DataFrame(icc_results).to_csv(os.path.join(OUTPUT_DIR, "icc_interrater_reliability.csv"), index=False)
print(f"\n  → Saved: icc_interrater_reliability.csv")
print()

# --- 11c. Performance correlates ---
print("  --- 11c. Performance Correlates ---")
print()

perf_corr_vars = [
    "Enjoyment_avg", "Satisfaction_avg", "Focused_Attention_avg",
    "AI_Collaboration_post", "Human_AI_Fit", "AI_Reliability",
    "Anthropomorphism", "AI_Experience", "Neuroticism", "Extraversion",
    "Openness", "Agreeableness", "Conscientiousness"
]

pcr = []
for var in perf_corr_vars:
    if var not in df.columns or "Average_all" not in df.columns:
        continue
    pair = df[[var, "Average_all"]].dropna()
    if len(pair) < 5:
        continue
    r, p = pearsonr(pair[var], pair["Average_all"])
    pcr.append({"Variable": var, "r": round(r,3), "p": round(p,4)})
    s = sig_stars(p)
    print(f"  Overall Quality ~ {var}: r = {r:.3f}, p = {p:.4f}{s}")

pd.DataFrame(pcr).to_csv(os.path.join(OUTPUT_DIR, "performance_correlates.csv"), index=False)
print(f"\n  → Saved: performance_correlates.csv")
print()


# ============================================================
# ============================================================
#  VISUALIZATIONS
# ============================================================
# ============================================================

print("=" * 80)
print("  GENERATING VISUALIZATIONS")
print("=" * 80)
print()

# ------------------------------------------------------------------
# VIZ 1: Effect Size Forest Plot (Post-Task Condition Comparisons)
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

all_comp = condition_results + perf_results
labels = [r.get("Label", r["Variable"]) for r in all_comp]
ds = [r["Cohen_d"] for r in all_comp]
ps = [r["U_p"] for r in all_comp]

y_pos = np.arange(len(labels))
colors = [COLOR_SL if d > 0 else COLOR_DT for d in ds]
edge_colors = ['#333333' if p < 0.05 else '#aaaaaa' for p in ps]

bars = ax.barh(y_pos, ds, color=colors, edgecolor=edge_colors, linewidth=1.5, height=0.6, alpha=0.85)
ax.axvline(x=0, color='grey', linestyle='-', linewidth=0.8)
ax.axvline(x=0.5, color='grey', linestyle=':', linewidth=0.6, alpha=0.5)
ax.axvline(x=-0.5, color='grey', linestyle=':', linewidth=0.6, alpha=0.5)
ax.axvline(x=0.8, color='grey', linestyle='--', linewidth=0.6, alpha=0.5)
ax.axvline(x=-0.8, color='grey', linestyle='--', linewidth=0.6, alpha=0.5)

for i, (d, p) in enumerate(zip(ds, ps)):
    s = sig_stars(p)
    ax.text(d + 0.03 if d >= 0 else d - 0.03, i, f"{d:.2f}{s}",
            va='center', ha='left' if d >= 0 else 'right', fontsize=9,
            fontweight='bold' if p < 0.05 else 'normal')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Cohen's d (positive = Servant Leader > Dark Triad)", fontsize=12)
ax.set_title("Effect Sizes: Condition Comparisons\n(Post-Task Measures & Task Performance)", fontsize=14, fontweight='bold')
ax.invert_yaxis()

legend_elements = [
    mpatches.Patch(facecolor=COLOR_SL, label='Favors Servant Leader'),
    mpatches.Patch(facecolor=COLOR_DT, label='Favors Dark Triad'),
    Line2D([0],[0], color='#333', linewidth=1.5, label='p < .05'),
    Line2D([0],[0], color='#aaa', linewidth=1.5, label='p ≥ .05'),
]
ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9, fontsize=9)
fig.tight_layout()
save_fig(fig, "01_effect_size_forest_plot.png")


# ------------------------------------------------------------------
# VIZ 2: Post-Task Measures Grouped Bar Chart
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 6))
pt_labels = [r.get("Label", r["Variable"]) for r in condition_results]
m_sl = [r["M_SL"] for r in condition_results]
sd_sl = [r["SD_SL"] for r in condition_results]
m_dt = [r["M_DT"] for r in condition_results]
sd_dt = [r["SD_DT"] for r in condition_results]

x = np.arange(len(pt_labels))
w = 0.35
bars1 = ax.bar(x - w/2, m_sl, w, yerr=sd_sl, label='Servant Leader', color=COLOR_SL,
               capsize=3, edgecolor='white', linewidth=0.5, alpha=0.9, error_kw={'linewidth': 1})
bars2 = ax.bar(x + w/2, m_dt, w, yerr=sd_dt, label='Dark Triad', color=COLOR_DT,
               capsize=3, edgecolor='white', linewidth=0.5, alpha=0.9, error_kw={'linewidth': 1})

for i, r in enumerate(condition_results):
    if r["U_p"] < 0.05:
        max_h = max(m_sl[i]+sd_sl[i], m_dt[i]+sd_dt[i])
        ax.text(i, max_h + 0.15, sig_stars(r["U_p"]), ha='center', fontsize=14, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([l.replace("(post)","").strip() for l in pt_labels], rotation=25, ha='right', fontsize=10)
ax.set_ylabel("Mean Score (±SD)", fontsize=12)
ax.set_title("Post-Task Self-Report Measures by Condition", fontsize=14, fontweight='bold')
ax.legend(fontsize=11, framealpha=0.9)
ax.set_ylim(0, max(max(m_sl), max(m_dt)) + 1.2)
fig.tight_layout()
save_fig(fig, "02_posttask_measures_barplot.png")


# ------------------------------------------------------------------
# VIZ 3: Task Performance by Condition
# ------------------------------------------------------------------

perf_plot_defs = [("Average_all","Overall"),("Completeness_avg","Completeness"),
                  ("Originality_avg","Originality"),("Strategic_fit_avg","Strategic Fit")]

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
for idx, (col, label) in enumerate(perf_plot_defs):
    ax = axes[idx]
    g1 = df.loc[df["Group"]==1, col].dropna()
    g2 = df.loc[df["Group"]==2, col].dropna()

    parts1 = ax.violinplot([g1.values], positions=[0], showmeans=True, showmedians=False, widths=0.6)
    parts2 = ax.violinplot([g2.values], positions=[1], showmeans=True, showmedians=False, widths=0.6)

    for pc in parts1['bodies']:
        pc.set_facecolor(COLOR_SL)
        pc.set_alpha(0.6)
    for pc in parts2['bodies']:
        pc.set_facecolor(COLOR_DT)
        pc.set_alpha(0.6)
    for key in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
        if key in parts1:
            parts1[key].set_color(COLOR_SL)
        if key in parts2:
            parts2[key].set_color(COLOR_DT)

    # Overlay individual points
    ax.scatter(np.random.normal(0, 0.04, len(g1)), g1, color=COLOR_SL, alpha=0.5, s=20, zorder=3)
    ax.scatter(np.random.normal(1, 0.04, len(g2)), g2, color=COLOR_DT, alpha=0.5, s=20, zorder=3)

    # Stats annotation
    res = group_comparison(df, col)
    if res:
        s = sig_stars(res["U_p"])
        ax.set_title(f"{label}\nd = {res['Cohen_d']:.2f}{s}", fontsize=12, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Servant\nLeader", "Dark\nTriad"], fontsize=10)
    if idx == 0:
        ax.set_ylabel("Expert Rating (1-7)", fontsize=12)

fig.suptitle("Task Performance: Expert Ratings by Condition", fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, "03_task_performance_violin.png")


# ------------------------------------------------------------------
# VIZ 4: Task × Condition Line Plots (Enjoyment, Satisfaction, Focus)
# ------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

for idx, measure in enumerate(["Focused_Attention", "Enjoyment", "Satisfaction"]):
    ax = axes[idx]
    for grp, color, marker, lbl in [(1, COLOR_SL, 'o', 'Servant Leader'),
                                     (2, COLOR_DT, 's', 'Dark Triad')]:
        means, sds, ns = [], [], []
        for task in TASKS:
            col = f"{measure}_{task}"
            if col in df.columns:
                vals = df.loc[df["Group"]==grp, col].dropna()
                means.append(vals.mean())
                sds.append(vals.std() / np.sqrt(len(vals)))  # SEM
                ns.append(len(vals))
        x = np.arange(len(TASKS))
        ax.errorbar(x, means, yerr=sds, marker=marker, color=color, linewidth=2,
                    markersize=8, capsize=4, label=lbl, alpha=0.9)
        ax.fill_between(x, np.array(means)-np.array(sds), np.array(means)+np.array(sds),
                        color=color, alpha=VIZ_CONFIG['confidence_alpha'])

    # Add significance asterisks for group differences at each task
    for t_idx, task in enumerate(TASKS):
        col = f"{measure}_{task}"
        if col in df.columns:
            res = group_comparison(df, col)
            if res:
                stars = sig_stars(res["U_p"])
                if stars:
                    # Place asterisk above the higher mean + SEM
                    g1_vals = df.loc[df["Group"]==1, col].dropna()
                    g2_vals = df.loc[df["Group"]==2, col].dropna()
                    m1, m2 = g1_vals.mean(), g2_vals.mean()
                    sem1 = g1_vals.std() / np.sqrt(len(g1_vals))
                    sem2 = g2_vals.std() / np.sqrt(len(g2_vals))
                    y_max = max(m1 + sem1, m2 + sem2)
                    ax.text(t_idx, y_max + 0.08, stars,
                            ha='center', va='bottom', fontsize=13, fontweight='bold', color='black')

    ax.set_xticks(np.arange(len(TASKS)))
    ax.set_xticklabels(TASKS, fontsize=10)
    ax.set_title(measure.replace("_", " "), fontsize=13, fontweight='bold')
    if idx == 0:
        ax.set_ylabel("Mean Score (±SEM)", fontsize=12)
    if idx == 2:
        ax.legend(fontsize=10, framealpha=0.9)

fig.suptitle("Task Experience Across Tasks by Condition", fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, "04_task_experience_lineplot.png")


# ------------------------------------------------------------------
# VIZ 5: Correlation Heatmap (Key Variables)
# ------------------------------------------------------------------

# Select a readable subset
hm_vars = ["Neuroticism", "Extraversion", "Conscientiousness",
           "AI_Experience", "Technology_Trust", "AI_Fear_composite",
           "AI_Collaboration_post", "AI_Reliability", "Behavioral_Intention",
           "Human_AI_Fit", "Anthropomorphism", "Enjoyment_avg", "Satisfaction_avg",
           "Average_all"]
hm_vars = [v for v in hm_vars if v in df.columns]

hm_labels = {
    "Neuroticism": "Neuroticism", "Extraversion": "Extraversion",
    "Conscientiousness": "Conscient.", "AI_Experience": "AI Exper.",
    "Technology_Trust": "Tech Trust", "AI_Fear_composite": "AI Fear",
    "AI_Collaboration_post": "AI Collab (post)", "AI_Reliability": "AI Reliab.",
    "Behavioral_Intention": "Behav. Intent.", "Human_AI_Fit": "H-AI Fit",
    "Anthropomorphism": "Anthropom.", "Enjoyment_avg": "Enjoyment",
    "Satisfaction_avg": "Satisfaction", "Average_all": "Task Perf."
}

hm_data = df[hm_vars].corr()
hm_pvals = pd.DataFrame(np.ones((len(hm_vars), len(hm_vars))),
                         index=hm_vars, columns=hm_vars)
for i in range(len(hm_vars)):
    for j in range(i+1, len(hm_vars)):
        pair = df[[hm_vars[i], hm_vars[j]]].dropna()
        if len(pair) > 3:
            _, p = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
            hm_pvals.iloc[i, j] = p
            hm_pvals.iloc[j, i] = p

# Create mask for upper triangle
mask = np.triu(np.ones_like(hm_data, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(12, 10))
display_labels = [hm_labels.get(v, v) for v in hm_vars]

cmap = sns.diverging_palette(10, 220, as_cmap=True)
sns.heatmap(hm_data, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
            annot=True, fmt='.2f', square=True, linewidths=0.5,
            xticklabels=display_labels, yticklabels=display_labels,
            cbar_kws={"shrink": 0.8, "label": "Pearson r"}, ax=ax,
            annot_kws={"fontsize": 9})

# Bold significant cells
for i in range(len(hm_vars)):
    for j in range(i):
        if hm_pvals.iloc[i, j] < 0.05 and abs(hm_data.iloc[i, j]) >= 0.30:
            ax.text(j + 0.5, i + 0.5, f"{hm_data.iloc[i,j]:.2f}",
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    color='white' if abs(hm_data.iloc[i,j]) > 0.5 else 'black')

ax.set_title("Correlation Matrix: Key Study Variables", fontsize=14, fontweight='bold', pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
fig.tight_layout()
save_fig(fig, "05_correlation_heatmap.png")


# ------------------------------------------------------------------
# VIZ 6: Pre-Post Comparison (Paired Changes)
# ------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, (pre, post, label) in enumerate(prepost_pairs):
    ax = axes[idx]
    for grp, color, offset in [(1, COLOR_SL, -0.15), (2, COLOR_DT, 0.15)]:
        g = df[[pre, post, "Group"]].dropna()
        g = g[g["Group"]==grp]
        pre_m, post_m = g[pre].mean(), g[post].mean()
        pre_se = g[pre].std() / np.sqrt(len(g))
        post_se = g[post].std() / np.sqrt(len(g))

        x_pts = np.array([0 + offset, 1 + offset])
        ax.errorbar(x_pts, [pre_m, post_m], yerr=[pre_se, post_se],
                    marker='o', color=color, linewidth=2.5, markersize=10,
                    capsize=5, label=GROUP_LABELS[grp], alpha=0.9)

        # Individual lines (subtle)
        for _, row in g.iterrows():
            ax.plot(x_pts, [row[pre], row[post]], color=color, alpha=0.08, linewidth=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pre-Task", "Post-Task"], fontsize=11)
    ax.set_title(label, fontsize=12, fontweight='bold')
    if idx == 0:
        ax.set_ylabel("Mean Score (±SEM)", fontsize=12)
    if idx == 2:
        ax.legend(fontsize=10, framealpha=0.9)

fig.suptitle("Pre-Post Changes by Condition", fontsize=15, fontweight='bold', y=1.02)
fig.tight_layout()
save_fig(fig, "06_pre_post_comparison.png")


# ------------------------------------------------------------------
# VIZ 7: ICC Inter-Rater Reliability Plot
# ------------------------------------------------------------------

if icc_results:
    fig, ax = plt.subplots(figsize=(8, 5))
    dims = [r["Dimension"] for r in icc_results]
    iccs = [r["ICC_2k"] for r in icc_results]
    ci_lo = [r["CI_low"] for r in icc_results]
    ci_hi = [r["CI_high"] for r in icc_results]

    y = np.arange(len(dims))
    errors = [[icc - lo for icc, lo in zip(iccs, ci_lo)],
              [hi - icc for icc, hi in zip(iccs, ci_hi)]]

    ax.errorbar(iccs, y, xerr=errors, fmt='o', color=COLOR_SL, markersize=10,
                capsize=6, linewidth=2, markeredgecolor='white', markeredgewidth=1.5)

    ax.axvline(x=0.70, color=COLOR_DT, linestyle='--', linewidth=1.5, alpha=0.7, label='Good (≥.70)')
    ax.axvline(x=0.50, color=COLOR_ORANGE, linestyle=':', linewidth=1.5, alpha=0.7, label='Moderate (≥.50)')

    for i, (icc, lo, hi) in enumerate(zip(iccs, ci_lo, ci_hi)):
        ax.text(icc + 0.02, i + 0.15, f"{icc:.3f}", fontsize=10, fontweight='bold')

    ax.set_yticks(y)
    ax.set_yticklabels(dims, fontsize=11)
    ax.set_xlabel("ICC(2,k)", fontsize=12)
    ax.set_title("Inter-Rater Reliability", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower left')
    ax.set_xlim(0.3, 1.0)
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, "07_icc_reliability.png")


# ------------------------------------------------------------------
# VIZ 8: Reliability (Cronbach's Alpha) Summary
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 8))
rel_names = [r["Scale"] for r in reliability_results]
rel_alphas = [r["Alpha"] for r in reliability_results]

y = np.arange(len(rel_names))
colors_rel = [COLOR_SL if a >= 0.70 else COLOR_ORANGE if a >= 0.60 else COLOR_DT for a in rel_alphas]

bars = ax.barh(y, rel_alphas, color=colors_rel, edgecolor='white', linewidth=0.5, height=0.7, alpha=0.85)
ax.axvline(x=0.70, color='#333', linestyle='--', linewidth=1.2, label='α = .70 threshold')

for i, a in enumerate(rel_alphas):
    ax.text(a + 0.01, i, f"{a:.3f}", va='center', fontsize=9)

ax.set_yticks(y)
ax.set_yticklabels(rel_names, fontsize=9)
ax.set_xlabel("Cronbach's α", fontsize=12)
ax.set_title("Scale Reliability Summary", fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.05)
ax.legend(fontsize=10)
ax.invert_yaxis()
fig.tight_layout()
save_fig(fig, "08_reliability_alpha.png")


# ------------------------------------------------------------------
# VIZ 9: Open-Ended Keyword Comparison
# ------------------------------------------------------------------

if "Open_ended" in df.columns:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (kw_list, prefix, title, cmap_colors) in enumerate([
        (pos_kw, "pos_", "Positive Keywords", [COLOR_SL, COLOR_DT]),
        (neg_kw, "neg_", "Negative Keywords", [COLOR_SL, COLOR_DT])
    ]):
        ax = axes[idx]
        sl_counts = oe[oe["Group"]==1][[f"{prefix}{k}" for k in kw_list]].sum()
        dt_counts = oe[oe["Group"]==2][[f"{prefix}{k}" for k in kw_list]].sum()

        # Only show keywords that appear at least once
        total = sl_counts + dt_counts
        mask_kw = total > 0
        kws_show = [k for k, m in zip(kw_list, mask_kw) if m]
        sl_show = [sl_counts[f"{prefix}{k}"] for k in kws_show]
        dt_show = [dt_counts[f"{prefix}{k}"] for k in kws_show]

        x = np.arange(len(kws_show))
        w = 0.35
        ax.bar(x - w/2, sl_show, w, color=COLOR_SL, label='Servant Leader', alpha=0.85)
        ax.bar(x + w/2, dt_show, w, color=COLOR_DT, label='Dark Triad', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(kws_show, rotation=45, ha='right', fontsize=10)
        ax.set_ylabel("Frequency", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        if idx == 0:
            ax.legend(fontsize=10)

    fig.suptitle("Open-Ended Response: Keyword Frequencies by Condition",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save_fig(fig, "09_open_ended_keywords.png")


# ------------------------------------------------------------------
# VIZ 10: Baseline Equivalence Dot Plot
# ------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 7))

bl_labels = [r.get("Label", r["Variable"]) for _, r in
             zip([l for _, l in baseline_vars], baseline_results)]
bl_labels = [l for _, l in baseline_vars if _ in [r["Variable"] for r in baseline_results]]
bl_ds = [r["Cohen_d"] for r in baseline_results]
bl_ps = [r["U_p"] for r in baseline_results]

y = np.arange(len(bl_labels))
colors_bl = ['#333333' if p < 0.05 else '#888888' for p in bl_ps]
sizes = [120 if p < 0.05 else 60 for p in bl_ps]

ax.scatter(bl_ds, y, c=colors_bl, s=sizes, zorder=3, edgecolors='white', linewidth=1)
ax.axvline(x=0, color='grey', linestyle='-', linewidth=0.8)

# Reference bands
ax.axvspan(-0.2, 0.2, alpha=0.08, color='green', label='Negligible (|d| < 0.2)')
ax.axvspan(-0.5, -0.2, alpha=0.05, color='orange')
ax.axvspan(0.2, 0.5, alpha=0.05, color='orange')

for i, (d, p) in enumerate(zip(bl_ds, bl_ps)):
    s = sig_stars(p)
    ax.text(d + 0.03, i, f"d={d:.2f}{s}", va='center', fontsize=9,
            fontweight='bold' if p < 0.05 else 'normal')

ax.set_yticks(y)
ax.set_yticklabels(bl_labels, fontsize=10)
ax.set_xlabel("Cohen's d (positive = Servant Leader > Dark Triad)", fontsize=11)
ax.set_title("Baseline Equivalence: Group Differences on Pre-Task Measures",
             fontsize=13, fontweight='bold')
ax.invert_yaxis()
fig.tight_layout()
save_fig(fig, "10_baseline_equivalence.png")


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 80)
print("  ANALYSIS COMPLETE — SUMMARY OF KEY FINDINGS")
print("=" * 80)
print()
print(f"  Sample: N = {n_total} (Servant Leader: {n_sl}, Dark Triad: {n_dt})")
print()

# Baseline
print("  BASELINE EQUIVALENCE:")
if sig_baseline:
    print(f"WARNING: Groups differed on: {', '.join(sig_baseline)}")
else:
    print(f"    Groups equivalent on all pre-task measures")
print()

# Main effects
print("  SIGNIFICANT CONDITION EFFECTS (p < .05, uncorrected):")
for r in condition_results:
    if r["U_p"] < 0.05:
        direction = "SL > DT" if r["Cohen_d"] > 0 else "DT > SL"
        print(f"    • {r['Label']}: d = {r['Cohen_d']:.3f} ({r['d_interp']}), "
              f"p = {r['U_p']:.4f}, {direction}")
for r in task_single_results:
    if r.get("U_p", 1) < 0.05:
        direction = "SL > DT" if r["Cohen_d"] > 0 else "DT > SL"
        print(f"    • {r.get('Label', r['Variable'])}: d = {r['Cohen_d']:.3f} ({r['d_interp']}), "
              f"p = {r['U_p']:.4f}, {direction}")
print()

# Task performance
print("  TASK PERFORMANCE (Expert Ratings):")
for r in perf_results:
    if r["U_p"] < 0.05:
        print(f"    • {r['Label']}: SL M={r['M_SL']:.2f} vs DT M={r['M_DT']:.2f}, "
              f"d = {r['Cohen_d']:.3f} ({r['d_interp']}), p = {r['U_p']:.4f}")
print()

# ICC
print("  INTER-RATER RELIABILITY:")
for r in icc_results:
    print(f"    • {r['Dimension']}: ICC(2,k) = {r['ICC_2k']:.3f} "
          f"[{r['CI_low']:.3f}, {r['CI_high']:.3f}]{'' if r['Acceptable'] else '  [below 0.70]'}")
print()

# Pre-post
print("  PRE-POST CHANGES:")
any_sig_pp = any(r["p"] < 0.05 for r in prepost_results)
if any_sig_pp:
    for r in prepost_results:
        if r["p"] < 0.05:
            print(f"    • {r['Measure']} ({r['Group']}): Δ={r['Delta']}, p={r['p']:.4f}")
else:
    print("    No significant pre-post changes detected.")
print()

# Files
print("  OUTPUT FILES:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if not os.path.isdir(os.path.join(OUTPUT_DIR, f)):
        sz = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f} ({sz:,} bytes)")
print()
print("  FIGURES:")
for f in sorted(os.listdir(FIG_DIR)):
    sz = os.path.getsize(os.path.join(FIG_DIR, f))
    print(f"    {f} ({sz:,} bytes)")
print()
print("  Done.")

# %%
