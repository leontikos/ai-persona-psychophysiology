"""Open-ended feedback (Supplementary Note 8): agreement between the human and the language-model
coding, and mentions of the chatbot's treatment of the participant by condition.

Inputs, in $STUDY_DATA_ROOT/SURVEY_data/analysis_output/:
  open_ended_coding_for_verification.csv  optional: responses keyed by a random code, coded 0/1 by a
                                          human coder who did not see the condition or the model's codes
  open_ended_coding_ai.csv                the language model's codes for the same responses
  open_ended_coding_key.csv               random code -> participantID (open only after coding)
The condition comes from the Qualtrics export (Group 1 = Servant Leader, 2 = Dark Triad).

Prints Cohen's kappa per code, the responses on which the two coders disagree (codes only), and,
using the human codes, the number of participants per condition with each code, Fisher's exact p
and the difference in proportions with a Newcombe 95% CI. Until the human coding is complete, the
table is computed from the language-model codes instead (the values in Supplementary Note 8).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd                                                      # noqa: E402
from scipy.stats import fisher_exact                                     # noqa: E402
from sklearn.metrics import cohen_kappa_score                            # noqa: E402
from statsmodels.stats.proportion import confint_proportions_2indep      # noqa: E402

import common as C                                                       # noqa: E402

AO = f"{C.BASE}/SURVEY_data/analysis_output"
CODES = ["NEG_TREATMENT", "POS_TREATMENT", "NEG_AFFECT_SELF", "POS_AFFECT_SELF",
         "FUNCTIONAL_CRITIQUE", "SUSPECTS_MANIPULATION", "EMPTY_OR_IRRELEVANT"]

ai = pd.read_csv(f"{AO}/open_ended_coding_ai.csv")
key = pd.read_csv(f"{AO}/open_ended_coding_key.csv")
human_path = f"{AO}/open_ended_coding_for_verification.csv"
human = pd.read_csv(human_path) if os.path.exists(human_path) else ai[["code"]].assign(**{c: float("nan") for c in CODES})
missing = human[CODES].isna().any(axis=1)
if missing.any():
    print(f"{int(missing.sum())} response(s) without human codes; "
          "the table below uses the language-model codes.\n")
    final, source = ai, "Language-model codes"
else:
    human[CODES] = human[CODES].astype(int)
    m = human[["code"] + CODES].merge(ai, on="code", suffixes=("_human", "_ai"))
    print("Agreement, human vs language model (n = %d responses)" % len(m))
    for c in CODES:
        h, a = m[c + "_human"], m[c + "_ai"]
        k = cohen_kappa_score(h, a) if (h.nunique() > 1 or a.nunique() > 1) else float("nan")
        print(f"  {c:22s} agreement {100 * (h == a).mean():5.1f}%  kappa {k:.2f}  (human {h.sum()}, model {a.sum()})")
    dis = m[[any(r[c + "_human"] != r[c + "_ai"] for c in CODES) for _, r in m.iterrows()]]
    print(f"\nResponses with any disagreement ({len(dis)}): " + ", ".join(dis.code))
    print("Disagreements are listed for adjudication by the coders.\n")
    final, source = human, "Human codes"

q = pd.read_csv(C.QUALTRICS, usecols=["participantID", "Group"])
d = final.merge(key, on="code").merge(q, on="participantID")
d["cond"] = d.Group.map({1: "SL", 2: "DT"})
rows = []
for c in CODES:
    a, na = int(d.loc[d.cond == "DT", c].sum()), int((d.cond == "DT").sum())
    b, nb = int(d.loc[d.cond == "SL", c].sum()), int((d.cond == "SL").sum())
    p = fisher_exact([[a, na - a], [b, nb - b]])[1]
    lo, hi = confint_proportions_2indep(a, na, b, nb, method="newcomb")
    rows.append({"code": c, "DT": f"{a}/{na}", "SL": f"{b}/{nb}", "DT_pct": 100 * a / na, "SL_pct": 100 * b / nb,
                 "fisher_p": p, "diff_pp": 100 * (a / na - b / nb), "diff_lo": 100 * lo, "diff_hi": 100 * hi})
res = pd.DataFrame(rows)
res.to_csv(os.path.join(C.OUT, "open_ended_by_condition.csv"), index=False)
pd.set_option("display.width", 200)
print(f"{source} by condition (Dark Triad vs Servant Leader):")
print(res.round(3).to_string(index=False))
