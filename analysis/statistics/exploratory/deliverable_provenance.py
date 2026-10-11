"""Deliverable provenance (Supplementary Note 11): did the chatbot draft the deliverables? For each submitted deliverable, the share
of its word 6-grams whose FIRST appearance in the conversation log is in a chatbot message
('chatbot-originated'), in a participant message, or nowhere in the conversation. No text is
saved or printed. Both personas were instructed never to solve tasks (Methods)."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
import json, glob, re, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import common as C

N = 6
tok = lambda s: re.findall(r"\w+", s.lower())
def grams(words, n=N): return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

first = {}
for f in sorted(glob.glob(C.LOGS + "/*.json")):
    d = json.load(open(f)); fo = {}
    for msg in d["messages"]:
        if msg["role"] not in ("user", "assistant"): continue
        for g in grams(tok(msg["content"])):
            fo.setdefault(g, msg["role"])
    first[d["metadata"]["user_id"]] = fo

df = C.load_scored()
T = {"Persona": "Task_1_Persona", "Brief": "Task_2_Brief", "Communication": "Task_3_Communication", "Content": "Task_4_Content"}
rows = []
for _, r in df.iterrows():
    fo = first[r.participantID]; tot = {"bot": 0, "user": 0, "none": 0}
    rec = {"participantID": r.participantID, "Group": r.Group, "Average_all": r.Average_all}
    for t, col in T.items():
        if pd.isna(r[col]): rec[f"bot_share_{t}"] = np.nan; continue
        gs = grams(tok(r[col])); c = {"bot": 0, "user": 0, "none": 0}
        for g in gs:
            role = fo.get(g); c["bot" if role == "assistant" else "user" if role == "user" else "none"] += 1
        n = max(1, sum(c.values())); rec[f"bot_share_{t}"] = c["bot"] / n
        for k in c: tot[k] += c[k]
    n = max(1, sum(tot.values()))
    rec.update({"bot_share_all": tot["bot"] / n, "user_share_all": tot["user"] / n, "none_share_all": tot["none"] / n})
    rows.append(rec)
p = pd.DataFrame(rows)
p["SL"] = (p.Group == 1).astype(int)
p.drop(columns=["participantID"]).to_csv(C.OUT + "/c2_deliverable_provenance.csv", index=False)

rng = C.new_rng(); out = []
for v in ["bot_share_all", "user_share_all", "none_share_all"] + [f"bot_share_{t}" for t in T]:
    out.append(C.full_compare(p, v, rng=rng, robust_ci=True))
out = pd.DataFrame(out); out.to_csv(C.OUT + "/c2_provenance_by_condition.csv", index=False)
pd.set_option("display.width", 250)
print(out[["Variable", "n_SL", "M_SL", "SD_SL", "Mdn_SL", "n_DT", "M_DT", "SD_DT", "Mdn_DT", "U_p", "Cohen_d"]].round(3).to_string(index=False))
for g, lab in [(1, "SL"), (2, "DT")]:
    s = p[p.Group == g]
    print(lab, "participants with >=50% of deliverable 6-grams chatbot-originated:", int((s.bot_share_all >= .5).sum()), "/", len(s),
          "| >=25%:", int((s.bot_share_all >= .25).sum()))
r, pv = stats.pearsonr(p.bot_share_all, p.Average_all); rho, ps = stats.spearmanr(p.bot_share_all, p.Average_all)
print(f"pooled r(bot_share_all, Overall Quality) = {r:.3f} (p = {pv:.4f}); rho = {rho:.3f} (p = {ps:.4f})")
for g, lab in [(1, "SL"), (2, "DT")]:
    s = p[p.Group == g]; r, pv = stats.pearsonr(s.bot_share_all, s.Average_all); rho, ps = stats.spearmanr(s.bot_share_all, s.Average_all)
    print(f"  within {lab}: r = {r:.3f} (p = {pv:.4f}); rho = {rho:.3f} (p = {ps:.4f})")
p["bot_share_z"] = (p.bot_share_all - p.bot_share_all.mean()) / p.bot_share_all.std()
b = smf.ols("Average_all ~ SL", data=p).fit(); f = smf.ols("Average_all ~ SL + bot_share_z", data=p).fit()
print(f"OLS: B_SL alone = {b.params.SL:.3f} (SE {b.bse.SL:.3f}, p = {b.pvalues.SL:.4f}); "
      f"+ bot_share_z: B_SL = {f.params.SL:.3f} (SE {f.bse.SL:.3f}, p = {f.pvalues.SL:.4f}), B_cov = {f.params.bot_share_z:.3f} (p = {f.pvalues.bot_share_z:.4f}); "
      f"partial d {b.params.SL/np.sqrt(b.mse_resid):.2f} -> {f.params.SL/np.sqrt(f.mse_resid):.2f}; r(SL, cov) = {np.corrcoef(p.SL, p.bot_share_z)[0,1]:.3f}")

# combined exploratory model: completion + chatbot-originated share
comp = pd.read_csv(C.OUT + "/b_participant_completion.csv")[["participantID", "all4"]]
q = p.merge(comp, on="participantID")
q["all4"] = q.all4.astype(int)
f2 = smf.ols("Average_all ~ SL + all4 + bot_share_z", data=q).fit()
print(f"OLS Average_all ~ SL + all4 + bot_share_z: B_SL = {f2.params.SL:.3f} (SE {f2.bse.SL:.3f}, p = {f2.pvalues.SL:.4f}), "
      f"B_all4 = {f2.params.all4:.3f} (p = {f2.pvalues.all4:.4f}), B_bot = {f2.params.bot_share_z:.3f} (p = {f2.pvalues.bot_share_z:.4f}); partial d = {f2.params.SL/np.sqrt(f2.mse_resid):.2f}")
s4 = q[q.all4 == 1]
f3 = smf.ols("Average_all ~ SL + bot_share_z", data=s4).fit(); f3b = smf.ols("Average_all ~ SL", data=s4).fit()
print(f"Complete submitters only (n={len(s4)}): B_SL alone = {f3b.params.SL:.3f} (p = {f3b.pvalues.SL:.4f}); + bot_share_z: B_SL = {f3.params.SL:.3f} (p = {f3.pvalues.SL:.4f}), "
      f"partial d {f3b.params.SL/np.sqrt(f3b.mse_resid):.2f} -> {f3.params.SL/np.sqrt(f3.mse_resid):.2f}")

# Near-verbatim adoption: per deliverable, the largest share of its 6-grams contained in a single
# chatbot message (or a single participant message). Deliverables too short to contain a 6-gram
# count in the denominator as not meeting the criterion.
conv = {}
for f in sorted(glob.glob(C.LOGS + "/*.json")):
    d = json.load(open(f))
    conv[d["metadata"]["user_id"]] = [(m["role"], grams(tok(m["content"]))) for m in d["messages"]
                                      if m["role"] in ("user", "assistant")]
per = []
for _, r in df.iterrows():
    msgs = conv[r.participantID]
    for t, col in T.items():
        if pd.isna(r[col]):
            continue
        gs = grams(tok(r[col]))
        mb = max((len(gs & g) / len(gs) for role, g in msgs if role == "assistant"), default=0) if gs else np.nan
        mu = max((len(gs & g) / len(gs) for role, g in msgs if role == "user"), default=0) if gs else np.nan
        per.append({"Group": r.Group, "Task": t, "max_single_bot": mb, "max_single_user": mu})
per = pd.DataFrame(per)
per.to_csv(C.OUT + "/c3_provenance_per_deliverable.csv", index=False)
for g, lab in [(1, "SL"), (2, "DT")]:
    s = per[per.Group == g]
    print(f"{lab}: deliverables {len(s)}; >=80% of 6-grams inside one chatbot message: "
          f"{int((s.max_single_bot >= .8).sum())}; inside one participant message: {int((s.max_single_user >= .8).sum())}; "
          f"too short for a 6-gram: {int(s.max_single_bot.isna().sum())}")
