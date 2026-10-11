#!/usr/bin/env python
"""
Equivalence-testing (TOST) figures with an FDR-consistent NHST decision.

Redraws, from the archived results table equivalence_testing_tost.csv (written by
analysis/survey_analysis/equivalence_testing_tost.py), the three TOST figures of the paper:

  figure20.png  main Fig. 6       TOST, MDES bounds (d = +-0.750)
  figure19.png  Supp. Fig. 14     TOST, SESOI bounds (d = +-0.50)
  figure21.png  Supp. Fig. 15     joint NHST + TOST decision framework (both bounds)

Only the classification rule changes. In the original, "equivalent" required the
Benjamini-Hochberg-corrected TOST p < .05, but "different" required only the UNCORRECTED
Welch p < .05. Here "different" requires the Welch p to be significant after
Benjamini-Hochberg correction across the same 10 self-report measures:

  equivalent           TOST p (BH) < .05 and NHST p (BH) >= .05
  different            NHST p (BH) < .05 and TOST p (BH) >= .05
  trivially different  both < .05
  undetermined         neither

Measures with a nominal Welch p < .05 that does not survive the correction are marked with a
dagger. Effect sizes, 90% CIs, bounds, TOST p-values, layout and colours are those of the
original script; the CSV is read, not recomputed.

Outputs (FIGS, default: alongside this script): the three PNGs (dpi 300) and
tost_bh_classification.csv.
"""
import os

from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches                                # noqa: E402
import matplotlib.pyplot as plt                                      # noqa: E402
import numpy as np                                                   # noqa: E402
import pandas as pd                                                  # noqa: E402
from matplotlib.lines import Line2D                                  # noqa: E402
from statsmodels.stats.multitest import multipletests                # noqa: E402

# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA = os.environ.get("STUDY_DATA_ROOT", "data")
TOST_CSV = f"{DATA}/SURVEY_data/analysis_output/equivalence_testing_tost.csv"
FIGS = os.environ.get("FIGS", os.path.join(DATA, "figures"))
ALPHA = 0.05
DAGGER = "†"
DAGGER_TXT = f"{DAGGER} nominal p < 0.05, not significant after FDR correction"

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
})
GREEN, AMBER, RED, PURPLE, BLUE = '#43A047', '#FF8F00', '#E53935', '#7B1FA2', '#1565C0'


def classify(eq, diff):
    if eq and not diff:
        return "Equivalent"
    if diff and not eq:
        return "Different"
    if eq and diff:
        return "Trivial diff."
    return "Undetermined"


def load():
    t = pd.read_csv(TOST_CSV)
    t["p_NHST_BH"] = multipletests(t.p_NHST, method="fdr_bh")[1]
    t["NHST_sig_BH"] = t.p_NHST_BH < ALPHA
    t["nominal_only"] = (t.p_NHST < ALPHA) & ~t.NHST_sig_BH
    for b in ["SESOI", "MDES"]:
        t[f"Class_{b}_original"] = [classify(e, s) for e, s in zip(t[f"Equivalent_{b}_fdr"], t.NHST_sig)]
        t[f"Class_{b}"] = [classify(e, s) for e, s in zip(t[f"Equivalent_{b}_fdr"], t.NHST_sig_BH)]
    t["CI90_lower_d"] = t.CI90_lower_SESOI / t.Pooled_SD
    t["CI90_upper_d"] = t.CI90_upper_SESOI / t.Pooled_SD
    t["Label_fig"] = [f"{lab} {DAGGER}" if nom else lab for lab, nom in zip(t.Label, t.nominal_only)]
    return t


def tost_plot(t, bound, btxt, cls_key, title_line2, fname):
    s = t.reindex(t.Cohen_d.abs().sort_values().index).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, r in s.iterrows():
        cls = r[cls_key]
        color, marker = {"Equivalent": (GREEN, 's'), "Different": (RED, 'D'),
                         "Trivial diff.": (PURPLE, 's'), "Undetermined": (AMBER, 'o')}[cls]
        ax.plot([r.CI90_lower_d, r.CI90_upper_d], [i, i], color=color, linewidth=2.5, solid_capstyle='round')
        ax.plot(r.Cohen_d, i, marker=marker, color=color, markersize=8, zorder=5, markeredgecolor='white',
                markeredgewidth=0.8)
    ax.axvline(x=-bound, color=BLUE, linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axvline(x=bound, color=BLUE, linestyle='--', linewidth=1.5, alpha=0.6)
    ax.axvspan(-bound, bound, alpha=0.08, color=BLUE)
    ax.axvline(x=0, color='black', linewidth=0.8, alpha=0.4)
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(s.Label_fig, fontsize=10)
    ax.set_xlabel("Effect Size (Cohen's d)", fontsize=12)
    ax.set_title(f"Equivalence Testing (TOST) — Self-Report Measures\n{title_line2}", fontsize=14, fontweight='bold')
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=GREEN, markersize=10, label='Equivalent (within bounds)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=AMBER, markersize=10, label='Undetermined'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor=RED, markersize=10,
               label='Different (FDR-corrected NHST p < 0.05)'),
        Line2D([0], [0], color=BLUE, linestyle='--', linewidth=1.5, label=f'Equivalence bounds (d = ±{btxt})'),
    ]
    if s.nominal_only.any():
        handles.append(Line2D([0], [0], linestyle='none', marker='None', label=DAGGER_TXT))
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9,
              framealpha=0.9)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, fname), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("wrote", os.path.join(FIGS, fname))


def decision_plot(t, sesoi, mdes):
    s = t.reindex(t.Cohen_d.abs().sort_values().index).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    colors = {"Equivalent": GREEN, "Different": RED, "Trivial diff.": PURPLE, "Undetermined": AMBER}
    for ax, (blab, bd, ck) in zip(axes, [(f"SESOI (d = ±{sesoi})", sesoi, "Class_SESOI"),
                                          (f"MDES (d = ±{mdes:.3f})", mdes, "Class_MDES")]):
        for i, r in s.iterrows():
            c = colors[r[ck]]
            ax.barh(i, abs(r.Cohen_d), color=c, alpha=0.7, edgecolor='white', height=0.6)
            ax.text(abs(r.Cohen_d) + 0.02, i, r[ck] + (f" {DAGGER}" if r.nominal_only else ""), va='center',
                    fontsize=8, color=c)
        ax.axvline(x=bd, color=BLUE, linestyle='--', linewidth=1.5, alpha=0.7, label=f'Bound = {bd:.3f}')
        ax.set_xlabel("|Cohen's d|", fontsize=11)
        ax.set_title(f"Bounds: {blab}", fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
    axes[0].set_yticks(np.arange(len(s)))
    axes[0].set_yticklabels(s.Label_fig, fontsize=10)
    handles = [mpatches.Patch(facecolor=GREEN, alpha=0.7, label='Equivalent'),
               mpatches.Patch(facecolor=AMBER, alpha=0.7, label='Undetermined'),
               mpatches.Patch(facecolor=RED, alpha=0.7, label='Different'),
               mpatches.Patch(facecolor=PURPLE, alpha=0.7, label='Trivially different')]
    if s.nominal_only.any():
        handles.append(Line2D([0], [0], linestyle='none', marker='None', label=DAGGER_TXT))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("NHST + TOST Decision Framework (Lakens, 2017)\nSelf-Report Measures: Servant Leader vs. Dark Triad",
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "figure21.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("wrote", os.path.join(FIGS, "figure21.png"))


def main():
    os.makedirs(FIGS, exist_ok=True)
    t = load()
    sesoi, mdes = float(t.SESOI_d.iloc[0]), float(t.MDES_d.iloc[0])
    tost_plot(t, mdes, f"{mdes:.3f}", "Class_MDES",
              f"Equivalence bounds: d = ±{mdes:.3f} (MDES at 80% power), α = {ALPHA}, 90% CIs", "figure20.png")
    tost_plot(t, sesoi, f"{sesoi}", "Class_SESOI",
              f"Equivalence bounds: d = ±{sesoi} (SESOI), α = {ALPHA}, 90% CIs", "figure19.png")
    decision_plot(t, sesoi, mdes)
    cols = ["Label", "n_SL", "n_DT", "Cohen_d", "CI90_lower_d", "CI90_upper_d", "t_NHST", "p_NHST", "p_NHST_BH",
            "TOST_p_SESOI", "TOST_p_SESOI_fdr", "TOST_p_MDES", "TOST_p_MDES_fdr",
            "Class_SESOI_original", "Class_SESOI", "Class_MDES_original", "Class_MDES", "nominal_only"]
    t[cols].to_csv(os.path.join(DATA, "SURVEY_data", "analysis_output", "tost_bh_classification.csv"), index=False)
    pd.set_option("display.width", 250)
    print(t[cols].round(4).to_string())
    for b in ["SESOI", "MDES"]:
        print(f"{b}: original {t[f'Class_{b}_original'].value_counts().to_dict()}  ->  "
              f"BH rule {t[f'Class_{b}'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
