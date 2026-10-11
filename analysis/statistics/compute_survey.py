#!/usr/bin/env python
"""Survey and performance between-group statistics with CIs and Bayes factors."""
import os, warnings

import numpy as np, pandas as pd
from dotenv import load_dotenv
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind
from recompute_stats import (STATS_OUT, d_ci_nct, mw_z, rank_biserial,
                             hodges_lehmann, bca_ci, bayes_ttest)

warnings.filterwarnings("ignore")

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT=os.environ.get("STUDY_DATA_ROOT","data")
OUT=os.path.join(DATA_ROOT,"SURVEY_data","analysis_output")
df=pd.read_pickle(f"{STATS_OUT}/survey_scored.pkl")
G=df["Group"]

def cohens_d(g1,g2):
    n1,n2=len(g1),len(g2)
    pooled=np.sqrt(((n1-1)*g1.std(ddof=1)**2+(n2-1)*g2.std(ddof=1)**2)/(n1+n2-2))
    return (g1.mean()-g2.mean())/pooled if pooled>0 else 0.0

def full_compare(var, col=None, bf=False):
    col = col or var
    if col not in df.columns:
        return {"Variable":var,"ERROR":f"col {col} missing"}
    g1=df.loc[G==1,col].dropna().values  # Servant Leader
    g2=df.loc[G==2,col].dropna().values  # Dark Triad
    if len(g1)<3 or len(g2)<3: return {"Variable":var,"ERROR":"n<3"}
    # parametric
    t,tp=ttest_ind(g1,g2,equal_var=False)
    dfw=((g1.var(ddof=1)/len(g1)+g2.var(ddof=1)/len(g2))**2/
         ((g1.var(ddof=1)/len(g1))**2/(len(g1)-1)+(g2.var(ddof=1)/len(g2))**2/(len(g2)-1)))
    d=cohens_d(pd.Series(g1),pd.Series(g2))
    dlo,dhi=d_ci_nct(d,len(g1),len(g2))
    # nonparametric
    U,up,z=mw_z(g1,g2)
    rrb=rank_biserial(g1,g2)  # sign: + => SL>DT
    rlo,rhi=bca_ci((g1,g2),rank_biserial)
    hl,hllo,hlhi=hodges_lehmann(g1,g2)  # SL - DT
    # assumptions
    sw1=shapiro(g1); sw2=shapiro(g2)
    lev=levene(g1,g2,center="median")
    vratio=(g2.var(ddof=1)/g1.var(ddof=1))  # DT/SL
    res={"Variable":var,"n_SL":len(g1),"n_DT":len(g2),
         "M_SL":g1.mean(),"SD_SL":g1.std(ddof=1),"Mdn_SL":np.median(g1),
         "min_SL":g1.min(),"max_SL":g1.max(),
         "M_DT":g2.mean(),"SD_DT":g2.std(ddof=1),"Mdn_DT":np.median(g2),
         "min_DT":g2.min(),"max_DT":g2.max(),
         "Welch_t":t,"Welch_df":dfw,"t_p":tp,
         "Cohen_d":d,"d_lo":dlo,"d_hi":dhi,
         "U":U,"Z":z,"U_p":up,"rrb":rrb,"rrb_lo":rlo,"rrb_hi":rhi,
         "HL_SLminusDT":hl,"HL_lo":hllo,"HL_hi":hlhi,
         "SW_SL_W":sw1.statistic,"SW_SL_p":sw1.pvalue,
         "SW_DT_W":sw2.statistic,"SW_DT_p":sw2.pvalue,
         "Levene_W":lev.statistic,"Levene_p":lev.pvalue,"VarRatio_DT_SL":vratio}
    if bf:
        res["BF10"]=bayes_ttest(t,len(g1),len(g2))
        res["BF01"]=1.0/res["BF10"] if res["BF10"] and res["BF10"]>0 else np.nan
    return res

FAMILIES={
 "BASELINE":["Age","Mood_1","AI_Experience","Technology_Trust","AI_Fear_composite",
     "AI_Collaborators_pre","AI_Willingness_pre","Neuroticism","Extraversion","Openness",
     "Agreeableness","Conscientiousness"],
 "TASK_ITEMS":["Familarity_Persona","Difficulty_Persona","Creativity_Persona",
     "Familiarity_Brief","Difficulty_Brief","Creativity_Brief",
     "Familiarity_Communication","Difficulty_Communication","Creativity_Communication",
     "Familarity_Content","Difficulty_Content","Creativity_Content"],
 "POST_TASK":["AI_Collaboration_post","AI_Reliability","Behavioral_Intention",
     "AI_Willingness_post","AI_Fear_post","Human_AI_Fit","Anthropomorphism"],
 "PERSONA_TASK_EXPERIENCE":["Focused_Attention_Persona","Enjoyment_Persona","Satisfaction_Persona"],
 "PERFORMANCE":["Average_all","Completeness_avg","Originality_avg","Strategic_fit_avg",
     "Average_Judge1","Average_Judge2","Average_Judge3"],
}
BF_VARS=set(FAMILIES["POST_TASK"]+FAMILIES["PERSONA_TASK_EXPERIENCE"])

rows=[]
for fam,vs in FAMILIES.items():
    for v in vs:
        r=full_compare(v, bf=(v in BF_VARS))
        r["Family"]=fam
        rows.append(r)
res=pd.DataFrame(rows)
res.to_csv(f"{STATS_OUT}/survey_enhanced.csv",index=False)

# validation against descriptive_statistics.csv
desc=pd.read_csv(f"{OUT}/descriptive_statistics.csv")
print("=== VALIDATION (reconstructed vs stored group means) ===")
checks=[("AI_Collaboration_post","AI Collaboration (post)"),("Human_AI_Fit","Human-AI Fit"),
        ("Conscientiousness","Conscientiousness"),("Enjoyment_Persona","Enjoyment (Persona)")]
for col,lbl in checks:
    rr=res[res.Variable==col]
    if len(rr):
        rr=rr.iloc[0]
        d0=desc[desc.Variable==lbl]
        if len(d0):
            d0=d0.iloc[0]
            print(f"  {col}: recon M_SL={rr['M_SL']:.2f}/M_DT={rr['M_DT']:.2f}  stored M_SL={d0['M_SL']}/M_DT={d0['M_DT']}")

pd.set_option('display.width',260); pd.set_option('display.max_columns',60)
print("\n=== ENHANCED SURVEY STATS (key cols) ===")
show=["Family","Variable","n_SL","n_DT","Welch_t","Welch_df","t_p","Cohen_d","d_lo","d_hi",
      "U","Z","U_p","rrb","rrb_lo","rrb_hi","HL_SLminusDT","HL_lo","HL_hi",
      "SW_SL_p","SW_DT_p","Levene_p","VarRatio_DT_SL"]
print(res[show].round(3).to_string())
print("\n=== BAYES FACTORS (post-task + persona nulls) ===")
print(res.loc[res.BF10.notna(),["Variable","Cohen_d","t_p","BF10","BF01"]].round(3).to_string())
