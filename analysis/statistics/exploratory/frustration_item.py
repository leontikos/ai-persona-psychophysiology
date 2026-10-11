"""Self-reported frustration (Supplementary Note 8): the single item (Satisfaction_<Task>_3, raw: 1 = strongly disagree ...
5 = strongly agree with 'I felt frustrated working on the task'; higher = more frustrated) and the
2-item satisfaction composite (satisfied + pleased).
Sign convention for THIS analysis: group 1 = Dark Triad, group 2 = Servant Leader, so U = U_DT,
Z > 0, r > 0 and d > 0 all mean DT > SL. RNG: fresh default_rng(20260704), tests run in the order
written below (frustration Persona..Content, mean, complete-case mean; then satisfaction-2)."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
import numpy as np, pandas as pd, pingouin as pg
from scipy import stats
from statsmodels.stats.multitest import multipletests
import common as C

df = C.load_scored(); OUT = C.OUT
# item direction check
for t in C.TASKS:
    raw, rek = df[f"Satisfaction_{t}_3"], df[f"Satisfaction_{t}_3_rek"]
    ok = bool(((6 - raw) == rek)[raw.notna()].all())
    print(f"{t}: rek == 6 - raw for all {raw.notna().sum()} non-missing: {ok}; r(raw, satisfied) = {raw.corr(df[f'Satisfaction_{t}_1']):.3f}")

# swap group codes so DT is the first group in full_compare
dd = df.copy(); dd["Group"] = dd["Group"].map({2: 1, 1: 2})
cc = df[[f"Frustration_{t}" for t in C.TASKS]].notna().all(axis=1)
dd["Frustration_cc"] = np.where(cc, df["Frustration_avg"], np.nan)
dd["Satisfaction2_cc"] = np.where(cc, df["Satisfaction2_avg"], np.nan)
rng = C.new_rng()
rows = []
for var in [f"Frustration_{t}" for t in C.TASKS] + ["Frustration_avg", "Frustration_cc"] + \
           [f"Satisfaction2_{t}" for t in C.TASKS] + ["Satisfaction2_avg", "Satisfaction2_cc"]:
    r = C.full_compare(dd, var, bf=True, rng=rng, robust_ci=True)
    r = {("DT" + k[2:] if k.endswith("_SL") and False else k): v for k, v in r.items()}
    # relabel: in dd, code 1 = DT -> full_compare's "_SL" fields are DT, "_DT" fields are SL
    fix = {}
    for k, v in r.items():
        if k.endswith("_SL"): fix[k[:-3] + "_DTgrp"] = v
        elif k.endswith("_DT"): fix[k[:-3] + "_SLgrp"] = v
        else: fix[k] = v
    rows.append(fix)
res = pd.DataFrame(rows).rename(columns=lambda c: c.replace("_DTgrp", "_DT").replace("_SLgrp", "_SL"))
res = res.rename(columns={"HL_SLminusDT": "HL_DTminusSL", "VarRatio_DT_SL": "VarRatio_SL_over_DT"})
# BH-FDR across the four per-task tests (MW and Welch separately)
for fam in ["Frustration", "Satisfaction2"]:
    m = res.Variable.isin([f"{fam}_{t}" for t in C.TASKS])
    res.loc[m, "U_p_FDR4"] = multipletests(res.loc[m, "U_p"], method="fdr_bh")[1]
    res.loc[m, "t_p_FDR4"] = multipletests(res.loc[m, "t_p"], method="fdr_bh")[1]
res.to_csv(f"{OUT}/a_frustration_between_group.csv", index=False)
pd.set_option("display.width", 260); pd.set_option("display.max_columns", 50)
print(res[["Variable", "n_SL", "M_SL", "SD_SL", "n_DT", "M_DT", "SD_DT", "U", "Z", "U_p", "U_p_FDR4", "rrb", "rrb_lo", "rrb_hi",
           "Cohen_d", "d_lo", "d_hi", "t_p", "t_p_FDR4", "BF10"]].round(3).to_string())

# mixed ANOVAs (complete cases) -- frustration and 2-item satisfaction
arows = []
for m in ["Frustration", "Satisfaction2", "Satisfaction"]:
    cols = [f"{m}_{t}" for t in C.TASKS]
    sub = df[["participantID", "Group"] + cols].dropna()
    long = sub.melt(id_vars=["participantID", "Group"], value_vars=cols, var_name="Task", value_name="Score")
    aov = pg.mixed_anova(data=long, dv="Score", within="Task", between="Group", subject="participantID")
    # pingouin < 0.6 used hyphenated column names
    aov = aov.rename(columns={"p-unc": "p_unc", "p-GG-corr": "p_GG_corr", "W-spher": "W_spher", "p-spher": "p_spher"})
    eps = float(aov.loc[aov.Source == "Task", "eps"].iloc[0])
    for _, r in aov.iterrows():
        lo, hi = C.eta2_ci_robust(r.F, r.DF1, r.DF2)
        within = r.Source != "Group"
        arows.append({"Measure": m, "n": len(sub), "n_SL": int((sub.Group == 1).sum()), "n_DT": int((sub.Group == 2).sum()),
                      "Source": r.Source, "F": r.F, "df1": r.DF1, "df2": r.DF2, "p": r.p_unc, "np2": r.np2,
                      "np2_lo90": lo, "np2_hi90": hi, "GG_eps": eps if within else np.nan,
                      "p_GG": stats.f.sf(r.F, eps * r.DF1, eps * r.DF2) if within else np.nan,
                      "p_GG_pingouin": r.p_GG_corr, "Mauchly_W": r.W_spher, "Mauchly_p": r.p_spher})
    for t, c in zip(C.TASKS, cols):
        arows.append({"Measure": m, "Source": f"cell means {t}", "M_SL": sub.loc[sub.Group == 1, c].mean(),
                      "M_DT": sub.loc[sub.Group == 2, c].mean(), "SD_SL": sub.loc[sub.Group == 1, c].std(), "SD_DT": sub.loc[sub.Group == 2, c].std()})
an = pd.DataFrame(arows); an.to_csv(f"{OUT}/a_mixed_anova.csv", index=False)
print(an.round(4).to_string())

# simple-effects style follow-up: Group x Task interaction contrast Persona vs rest is NOT done (not pre-specified)
# distribution of the raw frustration item by group and task
dist = []
for t in C.TASKS:
    for g, lab in [(1, "SL"), (2, "DT")]:
        s = df.loc[df.Group == g, f"Frustration_{t}"].dropna()
        dist.append({"Task": t, "Group": lab, "n": len(s), **{f"resp_{k}": int((s == k).sum()) for k in range(1, 6)},
                     "pct_agree_4or5": (s >= 4).mean() * 100})
pd.DataFrame(dist).to_csv(f"{OUT}/a_frustration_distribution.csv", index=False)
print(pd.DataFrame(dist).round(1).to_string())
