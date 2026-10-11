"""Feedback volume (Supplementary Note 11): chatbot feedback volume as an alternative explanation (or mediator) of the performance
effect. Per participant from conversation_logs: chatbot messages (excluding the opening greeting
'How may I help you?' -- two in the merged file -- and the merge marker 'system' message),
mean characters per chatbot message, total chatbot characters, participant messages; from the
archived coder: count/proportion of chatbot messages tagged CONSTRUCTIVE_FEEDBACK (same exclusions).
Condition coded SL = 1, DT = 0 so B_condition > 0 means SL > DT (paper's sign convention)."""
import sys; sys.path.insert(0, __file__.rsplit('/', 1)[0])
import json, glob, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import common as C

GREET = "How may I help you?"
rows = []
for f in sorted(glob.glob(C.LOGS + "/*.json")):
    d = json.load(open(f)); uid = d["metadata"]["user_id"]; ver = d["metadata"]["chatbot_version"]
    bot = [m["content"] for m in d["messages"] if m["role"] == "assistant" and m["content"].strip() != GREET]
    usr = [m["content"] for m in d["messages"] if m["role"] == "user"]
    a = json.load(open(f.replace(C.LOGS, C.CODED).replace(".json", "_analyzed.json")))
    assert a["original_metadata"]["user_id"] == uid
    coded = [c for c in a["chatbot_message_analyses"]
             if c["chatbot_message"].strip() != GREET and not c["chatbot_message"].startswith("[SESSION CONTINUATION")]
    cf = [("CONSTRUCTIVE_FEEDBACK" in (c.get("leadership_tactics") or [])) for c in coded]
    rows.append({"participantID": uid, "chatbot_version": ver, "n_bot": len(bot), "bot_chars_mean": np.mean([len(b) for b in bot]),
                 "bot_chars_total": sum(len(b) for b in bot), "n_user": len(usr), "user_chars_total": sum(len(u) for u in usr),
                 "n_bot_coded": len(coded), "n_constructive_fb": int(sum(cf)), "prop_constructive_fb": np.mean(cf)})
conv = pd.DataFrame(rows)
print("bot messages excl. greeting: total", conv.n_bot.sum(), "| coded excl. greeting/marker:", conv.n_bot_coded.sum())
print("participants where log and coder counts differ:", int((conv.n_bot != conv.n_bot_coded).sum()))

df = C.load_scored()
m = df[["participantID", "Group", "Average_all", "Completeness_avg", "Originality_avg", "Strategic_fit_avg"]].merge(conv, on="participantID", how="left")
assert m.n_bot.notna().all() and ((m.Group == 1) == (m.chatbot_version == "1")).all()
m["SL"] = (m.Group == 1).astype(int)
for v in ["bot_chars_mean", "bot_chars_total", "prop_constructive_fb", "n_constructive_fb", "n_bot", "n_user"]:
    m[v + "_z"] = (m[v] - m[v].mean()) / m[v].std(ddof=1)
m.drop(columns=["participantID"]).to_csv(C.OUT + "/c_participant_covariates.csv", index=False)  # no IDs, no text

# descriptives by condition
rng = C.new_rng()
desc = []
for v in ["n_bot", "bot_chars_mean", "bot_chars_total", "n_user", "n_constructive_fb", "prop_constructive_fb"]:
    r = C.full_compare(m, v, rng=rng, robust_ci=True); desc.append(r)
desc = pd.DataFrame(desc); desc.to_csv(C.OUT + "/c_covariate_by_condition.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(desc[["Variable", "n_SL", "M_SL", "SD_SL", "n_DT", "M_DT", "SD_DT", "U_p", "Cohen_d", "d_lo", "d_hi"]].round(3).to_string(index=False))

# OLS models
DV = {"Average_all": "Overall Quality", "Completeness_avg": "Completeness", "Originality_avg": "Originality", "Strategic_fit_avg": "Strategic Fit"}
COV = {"bot_chars_mean_z": "mean chatbot chars/message (z)", "bot_chars_total_z": "total chatbot chars (z)",
       "prop_constructive_fb_z": "constructive-feedback proportion (z)"}
out = []
for dv, dl in DV.items():
    base = smf.ols(f"{dv} ~ SL", data=m).fit()
    for cov, cl in COV.items():
        full = smf.ols(f"{dv} ~ SL + {cov}", data=m).fit()
        hc3 = smf.ols(f"{dv} ~ SL + {cov}", data=m).fit(cov_type="HC3")
        X = full.model.exog
        vif = variance_inflation_factor(X, 1)
        r_cond_cov = np.corrcoef(m.SL, m[cov])[0, 1]
        out.append({"DV": dl, "Covariate": cl,
                    "B_cond_alone": base.params.SL, "SE_cond_alone": base.bse.SL, "p_cond_alone": base.pvalues.SL,
                    "partial_d_alone": base.params.SL / np.sqrt(base.mse_resid),
                    "B_cond_adj": full.params.SL, "SE_cond_adj": full.bse.SL, "p_cond_adj": full.pvalues.SL,
                    "p_cond_adj_HC3": hc3.pvalues.SL, "partial_d_adj": full.params.SL / np.sqrt(full.mse_resid),
                    "B_cov": full.params[cov], "SE_cov": full.bse[cov], "p_cov": full.pvalues[cov], "p_cov_HC3": hc3.pvalues[cov],
                    "pct_change_B_cond": 100 * (full.params.SL - base.params.SL) / base.params.SL,
                    "r_condition_covariate": r_cond_cov, "VIF": vif, "R2_alone": base.rsquared, "R2_adj_model": full.rsquared, "n": int(full.nobs)})
out = pd.DataFrame(out); out.to_csv(C.OUT + "/c_ols_models.csv", index=False)
print(out.round(3).to_string(index=False))

# within-condition correlations with Overall Quality
wc = []
for cov in ["bot_chars_mean", "bot_chars_total", "prop_constructive_fb", "n_constructive_fb", "n_bot", "n_user"]:
    for g, lab in [(1, "SL"), (0, "DT")]:
        s = m[m.SL == g]
        r, p = stats.pearsonr(s[cov], s.Average_all); rho, ps = stats.spearmanr(s[cov], s.Average_all)
        lo, hi = C.fisher_z_ci(r, len(s))
        wc.append({"Covariate": cov, "Condition": lab, "n": len(s), "pearson_r": r, "r_lo": lo, "r_hi": hi, "p": p, "spearman_rho": rho, "p_rho": ps})
    r, p = stats.pearsonr(m[cov], m.Average_all); rho, ps = stats.spearmanr(m[cov], m.Average_all)
    wc.append({"Covariate": cov, "Condition": "pooled", "n": len(m), "pearson_r": r, "r_lo": C.fisher_z_ci(r, len(m))[0], "r_hi": C.fisher_z_ci(r, len(m))[1], "p": p, "spearman_rho": rho, "p_rho": ps})
wc = pd.DataFrame(wc); wc.to_csv(C.OUT + "/c_within_condition_correlations.csv", index=False)
print(wc.round(3).to_string(index=False))

# exploratory mediation: condition -> covariate -> Overall Quality, percentile bootstrap (5,000; seed 20260704)
rngm = C.new_rng()
med = []
for cov in ["bot_chars_mean_z", "bot_chars_total_z", "prop_constructive_fb_z"]:
    def ab(d):
        a = np.polyfit(d.SL, d[cov], 1)[0]
        X = np.column_stack([np.ones(len(d)), d.SL, d[cov]]); beta = np.linalg.lstsq(X, d.Average_all.values, rcond=None)[0]
        c = np.polyfit(d.SL, d.Average_all, 1)[0]
        return a * beta[2], beta[1], c
    ind, direct, total = ab(m)
    boots = []
    idx1 = np.where(m.SL == 1)[0]; idx0 = np.where(m.SL == 0)[0]
    for b in range(C.NBOOT):  # stratified by condition
        ii = np.concatenate([rngm.choice(idx1, len(idx1)), rngm.choice(idx0, len(idx0))])
        boots.append(ab(m.iloc[ii]))
    boots = np.array(boots)
    med.append({"Covariate": cov, "total_c": total, "direct_c_prime": direct, "indirect_ab": ind,
                "ab_lo": np.percentile(boots[:, 0], 2.5), "ab_hi": np.percentile(boots[:, 0], 97.5),
                "direct_lo": np.percentile(boots[:, 1], 2.5), "direct_hi": np.percentile(boots[:, 1], 97.5),
                "prop_mediated": ind / total})
med = pd.DataFrame(med); med.to_csv(C.OUT + "/c_mediation_exploratory.csv", index=False)
print(med.round(3).to_string(index=False))
