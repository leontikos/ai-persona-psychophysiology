"""Task completion (Supplementary Note 11): deliverable submission per task, by condition; expert-rating differences among
participants who submitted all four deliverables. No free text is printed or saved."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
import json, glob, numpy as np, pandas as pd
from scipy import stats
import common as C

df = C.load_scored(); OUT = C.OUT
T = ["Task_1_Persona", "Task_2_Brief", "Task_3_Communication", "Task_4_Content"]
sub = pd.DataFrame({t: df[t].notna() for t in T}); df["n_deliv"] = sub.sum(axis=1)
df["pattern"] = sub.astype(int).astype(str).agg("".join, axis=1)
df["all4"] = df.pattern == "1111"
# Non-substantive entry: one Brief field contains a note about the chat limit instead of a brief
# (identified by keyword; text not reproduced)
nonsub = df["Task_2_Brief"].fillna("").str.contains("chat limit", case=False)
print("non-substantive Brief entries:", int(nonsub.sum()), "group:", df.loc[nonsub, "Group"].tolist(),
      "pattern:", df.loc[nonsub, "pattern"].tolist(), "Average_all:", df.loc[nonsub, "Average_all"].round(2).tolist())
df["all4_substantive"] = df.all4 & ~nonsub

# 58th participant (in logs, absent from survey)
logs = {json.load(open(f))["metadata"]["user_id"]: json.load(open(f))["metadata"]["chatbot_version"] for f in sorted(glob.glob(C.LOGS + "/*.json"))}
missing = [u for u in logs if u not in set(df.participantID)]
print("in logs but not in survey:", len(missing), "chatbot_version:", [logs[u] for u in missing])

rows = []
for g, lab in [(1, "SL"), (2, "DT")]:
    s = df[df.Group == g]
    rows.append({"Group": lab, "n_in_survey": len(s), **{t: int(s[t].notna().sum()) for t in T},
                 "all4": int(s.all4.sum()), "pattern_1110": int((s.pattern == "1110").sum()),
                 "pattern_1000": int((s.pattern == "1000").sum()), "all4_substantive": int(s.all4_substantive.sum())})
tab = pd.DataFrame(rows); print(tab.to_string(index=False))
tab.to_csv(f"{OUT}/b_completion_table.csv", index=False)

fisher = []
for t in T + ["all4", "all4_substantive"]:
    a = df.loc[df.Group == 1, t].notna().sum() if t in T else df.loc[df.Group == 1, t].sum()
    b = df.loc[df.Group == 2, t].notna().sum() if t in T else df.loc[df.Group == 2, t].sum()
    n1, n2 = (df.Group == 1).sum(), (df.Group == 2).sum()
    p = stats.fisher_exact([[a, n1 - a], [b, n2 - b]])[1]
    p58 = stats.fisher_exact([[a, n1 - a], [b, n2 + 1 - b]])[1]  # counting the DT participant with no survey as non-submitter
    fisher.append({"Outcome": t, "SL": f"{a}/{n1}", "DT": f"{b}/{n2}", "Fisher_p": p, "DT_incl_58th": f"{b}/{n2+1}", "Fisher_p_N58": p58})
fisher = pd.DataFrame(fisher); print(fisher.round(3).to_string(index=False)); fisher.to_csv(f"{OUT}/b_completion_fisher.csv", index=False)

# ratings of incomplete submitters (numbers only)
print(df.loc[~df.all4, ["Group", "pattern", "Average_all", "Completeness_avg", "Originality_avg", "Strategic_fit_avg"]].round(2).to_string(index=False))
rho, p = stats.spearmanr(df.n_deliv, df.Average_all); print(f"Spearman n_deliverables vs Overall Quality: rho = {rho:.3f}, p = {p:.4f}")
df.loc[df.Group == 2].sort_values("Average_all")[["pattern", "Average_all"]].head(5).round(2).to_csv(f"{OUT}/b_lowest_DT_scores.csv", index=False)

# expert ratings among complete submitters
rng = C.new_rng()
res = []
dims = ["Average_all", "Completeness_avg", "Originality_avg", "Strategic_fit_avg"]
for label, mask in [("All 57 (paper)", np.ones(len(df), bool)), ("Submitted all four (52)", df.all4.values),
                    ("All four, substantive (51)", df.all4_substantive.values),
                    ("Excl. 1-deliverable participant (56)", (df.pattern != "1000").values)]:
    for v in dims:
        r = C.full_compare(df[mask], v, rng=rng, robust_ci=True); r["Sample"] = label; res.append(r)
res = pd.DataFrame(res)
res.to_csv(f"{OUT}/b_ratings_by_sample.csv", index=False)
pd.set_option("display.width", 250)
print(res[["Sample", "Variable", "n_SL", "n_DT", "M_SL", "SD_SL", "M_DT", "SD_DT", "U", "Z", "U_p", "rrb", "rrb_lo", "rrb_hi",
           "Cohen_d", "d_lo", "d_hi", "t_p", "Levene_p", "VarRatio_DT_SL"]].round(3).to_string(index=False))

# deliverable length (characters, all submitted tasks) by group and vs quality -- a participant-side volume index
df["deliv_chars"] = sum(df[t].fillna("").str.strip().str.len() for t in T)   # ignore leading/trailing whitespace
r1 = C.full_compare(df, "deliv_chars", rng=rng, robust_ci=True)
print(f"Deliverable total chars: SL {r1['M_SL']:.0f} ({r1['SD_SL']:.0f}) vs DT {r1['M_DT']:.0f} ({r1['SD_DT']:.0f}), U p = {r1['U_p']:.3f}, d = {r1['Cohen_d']:.2f}")
rr, pp = stats.pearsonr(df.deliv_chars, df.Average_all); rs, ps = stats.spearmanr(df.deliv_chars, df.Average_all)
print(f"deliv_chars vs Overall Quality: r = {rr:.3f} (p = {pp:.4f}); rho = {rs:.3f} (p = {ps:.4f})")
df[["participantID", "Group", "pattern", "n_deliv", "all4", "all4_substantive", "deliv_chars"]].to_csv(f"{OUT}/b_participant_completion.csv", index=False)
