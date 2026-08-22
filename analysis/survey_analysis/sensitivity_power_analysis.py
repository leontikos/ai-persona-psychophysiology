# %% sensitivity_power_analysis.py

"""
Sensitivity Power Analysis
===========================
Determines the minimum detectable effect size (MDES) at 80% power
for the between-group comparisons in our study (N=58: n1=31, n2=27).

Also provides:
  - Power curves across a range of effect sizes
  - Post-hoc power for observed effect sizes on key self-report measures
  - A summary table suitable for inclusion in the paper / supplementary materials

Uses: statsmodels.stats.power, scipy, numpy, pandas, matplotlib
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.power import TTestIndPower
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from dotenv import load_dotenv

# ============================================================
# 0. CONFIGURATION
# ============================================================

N_SL = 31          # Servant Leader group
N_DT = 27          # Dark Triad group
N_TOTAL = N_SL + N_DT
ALPHA = 0.05       # two-tailed
TARGET_POWER = 0.80
RATIO = N_DT / N_SL  # n2/n1 ratio for unequal groups

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


# ============================================================
# 1. SENSITIVITY ANALYSIS — Minimum Detectable Effect Size
# ============================================================

print("=" * 80)
print("  SENSITIVITY POWER ANALYSIS")
print("=" * 80)
print()

power_analysis = TTestIndPower()

# Solve for minimum detectable effect size at 80% power
mdes = power_analysis.solve_power(
    effect_size=None,
    nobs1=N_SL,
    ratio=RATIO,
    alpha=ALPHA,
    power=TARGET_POWER,
    alternative='two-sided'
)

print(f"  Study parameters:")
print(f"    n1 (Servant Leader) = {N_SL}")
print(f"    n2 (Dark Triad)     = {N_DT}")
print(f"    N total             = {N_TOTAL}")
print(f"    α (two-tailed)      = {ALPHA}")
print(f"    Target power        = {TARGET_POWER}")
print(f"    Group ratio (n2/n1) = {RATIO:.3f}")
print()
print(f"  ┌─────────────────────────────────────────────────────────────┐")
print(f"  │  Minimum Detectable Effect Size (Cohen's d) = {mdes:.3f}       │")
print(f"  │  at 80% power, α = .05, two-tailed                        │")
print(f"  └─────────────────────────────────────────────────────────────┘")
print()

# Additional power thresholds
for pwr in [0.70, 0.80, 0.90, 0.95]:
    d = power_analysis.solve_power(
        effect_size=None, nobs1=N_SL, ratio=RATIO,
        alpha=ALPHA, power=pwr, alternative='two-sided'
    )
    print(f"  MDES at {int(pwr*100)}% power: d = {d:.3f}")

print()

# ============================================================
# 2. POWER CURVE — Power as a function of effect size
# ============================================================

print("  --- Power Curve ---")
print()

effect_sizes = np.arange(0.0, 1.51, 0.01)
powers = power_analysis.power(
    effect_size=effect_sizes,
    nobs1=N_SL,
    ratio=RATIO,
    alpha=ALPHA,
    alternative='two-sided'
)

# Print key data points
print(f"  {'Effect size (d)':<20} {'Power':<10}")
print(f"  {'-'*30}")
for d_val in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 1.0, 1.2]:
    idx = int(d_val * 100)
    if idx < len(powers):
        print(f"  {d_val:<20.2f} {powers[idx]:<10.3f}")

print()

# ============================================================
# 3. DATA LOADING & SCALE CONSTRUCTION
#    (mirrors survey_analysis.py)
# ============================================================

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df["Group_label"] = df["Group"].map(GROUP_LABELS)

TASKS = ["Persona", "Brief", "Communication", "Content"]

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

# ============================================================
# 4. POST-HOC POWER FOR OBSERVED EFFECT SIZES
# ============================================================

print("=" * 80)
print("  POST-HOC POWER FOR OBSERVED EFFECTS")
print("=" * 80)
print()

# All self-report outcome measures tested in condition comparisons
posttask_vars = [
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

posthoc_results = []

print(f"  {'Measure':<35} {'|d|':<8} {'Power':<8} {'Detectable?':<12}")
print(f"  {'-'*65}")

for var, label in posttask_vars:
    if var not in df.columns:
        continue
    g1 = df.loc[df["Group"] == 1, var].dropna()
    g2 = df.loc[df["Group"] == 2, var].dropna()
    if len(g1) < 3 or len(g2) < 3:
        continue
    d = cohens_d(g1, g2)
    abs_d = abs(d)

    pwr = power_analysis.power(
        effect_size=abs_d,
        nobs1=len(g1),
        ratio=len(g2) / len(g1),
        alpha=ALPHA,
        alternative='two-sided'
    )

    detectable = "Yes" if abs_d >= mdes else "No"
    posthoc_results.append({
        "Variable": var,
        "Label": label,
        "M_SL": g1.mean(),
        "SD_SL": g1.std(),
        "M_DT": g2.mean(),
        "SD_DT": g2.std(),
        "Cohen_d": d,
        "abs_d": abs_d,
        "Observed_Power": pwr,
        "Above_MDES": detectable,
        "MDES_80": mdes
    })
    print(f"  {label:<35} {abs_d:<8.3f} {pwr:<8.3f} {detectable:<12}")

print()
print(f"  Note: MDES at 80% power = {mdes:.3f}")
print(f"  Effects below this threshold are undetectable with our sample size.")
print()

# Interpretation summary
n_above = sum(1 for r in posthoc_results if r["Above_MDES"] == "Yes")
n_below = sum(1 for r in posthoc_results if r["Above_MDES"] == "No")
print(f"  Summary: {n_above} measures above MDES, {n_below} below MDES")
print(f"  → Null self-report findings only rule out effects ≥ d = {mdes:.2f}")
print(f"  → Small-to-medium effects (d < {mdes:.2f}) would go undetected")
print()

# ============================================================
# 5. SAMPLE SIZE REQUIRED FOR MEDIUM EFFECTS
# ============================================================

print("=" * 80)
print("  REQUIRED SAMPLE SIZES FOR VARIOUS EFFECT SIZES")
print("=" * 80)
print()

print(f"  {'Target d':<12} {'n per group':<15} {'N total':<12} {'Power':<8}")
print(f"  {'-'*50}")
for target_d in [0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 1.00]:
    n_needed = power_analysis.solve_power(
        effect_size=target_d,
        nobs1=None,
        ratio=1.0,  # equal groups
        alpha=ALPHA,
        power=TARGET_POWER,
        alternative='two-sided'
    )
    n_needed = int(np.ceil(n_needed))
    print(f"  {target_d:<12.2f} {n_needed:<15d} {2*n_needed:<12d} {TARGET_POWER:<8.2f}")

print()

# ============================================================
# 6. SAVE RESULTS
# ============================================================

posthoc_df = pd.DataFrame(posthoc_results)
posthoc_df.to_csv(os.path.join(OUTPUT_DIR, "sensitivity_power_analysis.csv"), index=False)
print(f"  → Saved: sensitivity_power_analysis.csv")

# ============================================================
# 7. PUBLICATION-QUALITY FIGURES
# ============================================================

# --- Figure 1: Power Curve ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(effect_sizes, powers, color='#1565C0', linewidth=2.5, label='Power curve')
ax.axhline(y=0.80, color='#E53935', linestyle='--', linewidth=1.2, alpha=0.7, label='80% power')
ax.axvline(x=mdes, color='#43A047', linestyle='--', linewidth=1.2, alpha=0.7,
           label=f'MDES = {mdes:.3f}')

# Mark observed effect sizes on the curve
for r in posthoc_results:
    idx = min(int(r["abs_d"] * 100), len(powers) - 1)
    if idx >= 0:
        ax.plot(r["abs_d"], powers[idx], 'o', markersize=7, color='#FF8F00', zorder=5)
        # Only label measures with d > 0.3 to avoid clutter
        if r["abs_d"] > 0.3:
            ax.annotate(r["Label"], xy=(r["abs_d"], powers[idx]),
                        xytext=(5, 10), textcoords='offset points',
                        fontsize=7.5, color='#424242',
                        arrowprops=dict(arrowstyle='->', color='#9E9E9E', lw=0.8))

ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
ax.set_ylabel("Statistical Power (1 − β)", fontsize=12)
ax.set_title(f"Sensitivity Power Analysis\n(n₁ = {N_SL}, n₂ = {N_DT}, α = {ALPHA}, two-tailed)",
             fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.5)
ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "power_curve_sensitivity.png"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  → Figure saved: power_curve_sensitivity.png")

# --- Figure 2: Observed Effects vs MDES ---
fig, ax = plt.subplots(figsize=(10, 6))

sorted_results = sorted(posthoc_results, key=lambda x: x["abs_d"], reverse=True)
labels = [r["Label"] for r in sorted_results]
abs_ds = [r["abs_d"] for r in sorted_results]
observed_powers = [r["Observed_Power"] for r in sorted_results]
colors = ['#43A047' if r["Above_MDES"] == "Yes" else '#E53935' for r in sorted_results]

y_pos = range(len(labels))
bars = ax.barh(y_pos, abs_ds, color=colors, alpha=0.75, edgecolor='white', height=0.6)

ax.axvline(x=mdes, color='#1565C0', linestyle='--', linewidth=1.5, alpha=0.8,
           label=f'MDES = {mdes:.3f}')

for i, (d_val, pwr) in enumerate(zip(abs_ds, observed_powers)):
    ax.text(d_val + 0.02, i, f'd={d_val:.2f}, π={pwr:.2f}',
            va='center', fontsize=8.5, color='#424242')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Effect Size |Cohen's d|", fontsize=12)
ax.set_title("Observed Effect Sizes vs. Minimum Detectable Effect Size\n(Self-Report Measures)",
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.invert_yaxis()
ax.grid(True, axis='x', alpha=0.3)

# Add legend for bar colors
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#43A047', alpha=0.75, label='Detectable (|d| ≥ MDES)'),
    Patch(facecolor='#E53935', alpha=0.75, label='Undetectable (|d| < MDES)'),
    plt.Line2D([0], [0], color='#1565C0', linestyle='--', linewidth=1.5, label=f'MDES = {mdes:.3f}')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "observed_effects_vs_mdes.png"), dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"  → Figure saved: observed_effects_vs_mdes.png")

print()
print("=" * 80)
print("  SENSITIVITY POWER ANALYSIS COMPLETE")
print("=" * 80)
