#!/usr/bin/env python
"""Behavioral statistics: Mann-Whitney effect-size CIs, Cramer's V cluster-bootstrap CIs,
correlation Fisher-z CIs, and ANOVA partial eta^2 90% CIs."""
import os, warnings

import numpy as np, pandas as pd
from dotenv import load_dotenv
from scipy import stats
from scipy.optimize import brentq
from recompute_stats import (STATS_OUT, mw_z, rank_biserial, hodges_lehmann,
                             bca_ci, d_ci_nct, fisher_z_ci, eta2_ci, cramers_v)

warnings.filterwarnings("ignore")
RNG=np.random.default_rng(7)

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT=os.environ.get("STUDY_DATA_ROOT","data")
CONV=os.path.join(DATA_ROOT,"CONVERSATION_ANALYSES")
OUT=os.path.join(DATA_ROOT,"SURVEY_data","analysis_output")

cv=pd.read_csv(f"{CONV}/conversations_dataframe.csv")
cv["cond"]=cv["condition"].map({"servant_leader":1,"dark_triad":2})
def cohens_d(a,b):
    n1,n2=len(a),len(b)
    p=np.sqrt(((n1-1)*a.std(ddof=1)**2+(n2-1)*b.std(ddof=1)**2)/(n1+n2-2))
    return (a.mean()-b.mean())/p if p>0 else 0.0

BEHAV=["strategy_diversity","num_turning_points","total_participant_messages","total_chatbot_messages",
       "avg_participant_msg_length","avg_chatbot_msg_length","duration_minutes",
       "prop_compliance","prop_resistance","prop_jailbreaking","prop_negotiation","prop_help_seeking","prop_coping",
       "prop_emotion_frustrated","prop_emotion_anxious","prop_emotion_confident","prop_emotion_neutral",
       "prop_emotion_resigned","prop_emotion_defensive","prop_engagement_high","prop_engagement_medium","prop_engagement_low"]
rows=[]
for v in BEHAV:
    g1=cv.loc[cv.cond==1,v].dropna().values  # SL
    g2=cv.loc[cv.cond==2,v].dropna().values  # DT
    U,up,z=mw_z(g1,g2)
    rrb=rank_biserial(g1,g2); rlo,rhi=bca_ci((g1,g2),rank_biserial)
    hl,hllo,hlhi=hodges_lehmann(g1,g2)
    d=cohens_d(pd.Series(g1),pd.Series(g2)); dlo,dhi=d_ci_nct(d,len(g1),len(g2))
    rows.append(dict(Variable=v,n_SL=len(g1),n_DT=len(g2),M_SL=g1.mean(),SD_SL=g1.std(ddof=1),
        Mdn_SL=np.median(g1),M_DT=g2.mean(),SD_DT=g2.std(ddof=1),Mdn_DT=np.median(g2),
        U=U,Z=z,p=up,rrb=rrb,rrb_lo=rlo,rrb_hi=rhi,HL=hl,HL_lo=hllo,HL_hi=hlhi,d=d,d_lo=dlo,d_hi=dhi))
beh=pd.DataFrame(rows); beh.to_csv(f"{STATS_OUT}/behavioral_enhanced.csv",index=False)
pd.set_option('display.width',260);pd.set_option('display.max_columns',40)
print("=== BEHAVIORAL MW (conversation-level) ===")
print(beh[["Variable","n_SL","n_DT","U","Z","p","rrb","rrb_lo","rrb_hi","HL","HL_lo","HL_hi","d","d_lo","d_hi"]].round(3).to_string())

# ---------- Chi-square Cramer's V with cluster (conversation) bootstrap ----------
pm=pd.read_csv(f"{CONV}/participant_messages_dataframe.csv")
def cluster_boot_V(long, cat_col, nboot=5000):
    """long: df with conversation_id, condition, cat_col (single label per row).
    Cluster (conversation) bootstrap: resample conversations, sum their per-category
    count vectors into the appropriate condition row, recompute Cramer's V."""
    cats=sorted(long[cat_col].dropna().unique())
    conds=sorted(long["condition"].unique())
    cat_idx={c:i for i,c in enumerate(cats)}
    cond_idx={c:i for i,c in enumerate(conds)}
    ct0=np.zeros((len(conds),len(cats)))
    conv_vecs={}   # conversation_id -> (cond_row, count_vector)
    for cid,sub in long.groupby("conversation_id"):
        row=cond_idx[sub["condition"].iloc[0]]
        vec=np.zeros(len(cats))
        for c in sub[cat_col].dropna(): vec[cat_idx[c]]+=1
        conv_vecs[cid]=(row,vec); ct0[row]+=vec
    v0=cramers_v(ct0)
    convs=np.array(list(conv_vecs.keys()))
    boots=[]
    for _ in range(nboot):
        samp=RNG.choice(convs,len(convs),replace=True)
        ct=np.zeros_like(ct0)
        for cid in samp:
            r,vec=conv_vecs[cid]; ct[r]+=vec
        if (ct.sum(1)>0).all() and (ct.sum(0)>0).sum()>=2:
            b=cramers_v(ct)
            if np.isfinite(b): boots.append(b)
    lo,hi=np.percentile(boots,[2.5,97.5]) if boots else (np.nan,np.nan)
    return v0,lo,hi,len(long)

vres=[]
# strategy_category
d=pm.dropna(subset=["strategy_category"])
vres.append(("strategy_category x condition","chi2(5)",)+cluster_boot_V(d,"strategy_category"))
# engagement
d=pm.dropna(subset=["task_engagement"])
vres.append(("task_engagement x condition","chi2(2)",)+cluster_boot_V(d,"task_engagement"))
# jailbreaking subtype
d=pm[pm["primary_strategy"].astype(str).str.startswith("JAILBREAKING")].copy()
vres.append(("jailbreaking subtype x condition","chi2(2)",)+cluster_boot_V(d,"primary_strategy"))
# emotion (multi-label melt)
emo_cols=[c for c in pm.columns if c.startswith("emotion_")]
mrows=[]
for _,r in pm.iterrows():
    for c in emo_cols:
        if r[c]==1: mrows.append({"conversation_id":r["conversation_id"],"condition":r["condition"],"emotion":c.replace("emotion_","")})
emo=pd.DataFrame(mrows)
vres.append(("emotion x condition (multi-label)","chi2(5)",)+cluster_boot_V(emo,"emotion"))
vdf=pd.DataFrame(vres,columns=["Test","df","CramersV","V_lo","V_hi","N_msgs"])
vdf.to_csv(f"{STATS_OUT}/cramersv_enhanced.csv",index=False)
print("\n=== CRAMER'S V + cluster(conversation) bootstrap 95% CI ===")
print(vdf.round(3).to_string())

# ---------- Correlation Fisher-z CIs ----------
def recover_n(r,p):
    """recover n from Pearson r and two-sided p."""
    if p<=0 or p>=1 or abs(r)>=1: return np.nan
    def f(nn):
        dfree=nn-2
        t=abs(r)*np.sqrt(dfree/(1-r**2))
        return 2*stats.t.sf(t,dfree)-p
    try: return brentq(f,4,10000)
    except Exception: return np.nan
pc=pd.read_csv(f"{OUT}/performance_correlates.csv")
prows=[]
for _,row in pc.iterrows():
    n=recover_n(row["r"],row["p"]); n=int(round(n)) if n==n else np.nan
    lo,hi=fisher_z_ci(row["r"],n) if n==n else (np.nan,np.nan)
    prows.append(dict(Variable=row["Variable"],r=row["r"],p=row["p"],n=n,ci_lo=lo,ci_hi=hi))
pcr=pd.DataFrame(prows); pcr.to_csv(f"{STATS_OUT}/perf_correlates_ci.csv",index=False)
print("\n=== PERFORMANCE CORRELATES (Overall Quality ~ trait/measure), Fisher-z 95% CI ===")
print(pcr.round(3).to_string())

sc=pd.read_csv(f"{OUT}/significant_correlations.csv")
srows=[]
for _,row in sc.iterrows():
    n=recover_n(row["r"],row["p"]); n=int(round(n)) if n==n else np.nan
    lo,hi=fisher_z_ci(row["r"],n) if n==n else (np.nan,np.nan)
    srows.append(dict(Var1=row["Var1"],Var2=row["Var2"],r=row["r"],p=row["p"],n=n,ci_lo=lo,ci_hi=hi))
scr=pd.DataFrame(srows); scr.to_csv(f"{STATS_OUT}/sig_correlations_ci.csv",index=False)
print("\n=== SIGNIFICANT CORRELATIONS w/ Fisher-z CI (n recovered from r,p) ===")
print(scr.round(3).to_string())

# ---------- ANOVA partial eta^2 90% CI ----------
an=pd.read_csv(f"{OUT}/task_mixed_anova.csv")
arows=[]
for _,row in an.iterrows():
    lo,hi=eta2_ci(row["F"],int(row["DF1"]),int(row["DF2"]),conf=0.90)
    arows.append(dict(Measure=row["Measure"],Source=row["Source"],F=row["F"],df1=int(row["DF1"]),df2=int(row["DF2"]),
        p_unc=row["p-unc"],p_GG=row["p-GG-corr"],np2=row["np2"],eps=row["eps"],np2_lo=lo,np2_hi=hi))
anr=pd.DataFrame(arows); anr.to_csv(f"{STATS_OUT}/anova_eta2_ci.csv",index=False)
print("\n=== MIXED ANOVA partial eta^2 with 90% CI ===")
print(anr.round(4).to_string())
