#!/usr/bin/env python
"""Build Supplementary Figs. 11 and 12 (figure15.png, figure16.png: task experience and pre-post
attitudes) on the full 1-5 scale, written to $STUDY_DATA_ROOT/figures/.
The box-plot figures (main-text Figs. 2-3, Supplementary Fig. 3, and figure02.png, an overview of
all six strategy families not shown in the paper) are drawn by
analysis/conversations_analysis/behavior_corrected.py.
Requires survey_scored.pkl from recompute_stats.py.
"""
import os

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT=os.environ.get("STUDY_DATA_ROOT","data")
FIG=os.path.join(DATA_ROOT,"figures"); os.makedirs(FIG,exist_ok=True)
PKL=os.path.join(DATA_ROOT,"statistics_output","survey_scored.pkl")
SLc="#4C9BE0"; DTc="#E4695B"
df=pd.read_pickle(PKL)

# ---- Fig 15 task experience: line + SEM, FULL 1-5 axis ----
TASKS=["Persona","Brief","Communication","Content"]
def sem(a): a=a[~np.isnan(a)]; return np.std(a,ddof=1)/np.sqrt(len(a)) if len(a)>1 else 0
fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharey=True)
for ax,meas in zip(axes,["Focused_Attention","Enjoyment","Satisfaction"]):
    for g,color,lab in [(1,SLc,"Servant Leader"),(2,DTc,"Dark Triad")]:
        sub=df[df.Group==g]
        m=[sub[f"{meas}_{t}"].mean() for t in TASKS]
        e=[sem(sub[f"{meas}_{t}"].values) for t in TASKS]
        ax.errorbar(range(4),m,yerr=e,marker="o",color=color,label=lab,capsize=3,lw=2)
        ax.fill_between(range(4),np.array(m)-np.array(e),np.array(m)+np.array(e),color=color,alpha=.15)
    ax.set_xticks(range(4)); ax.set_xticklabels(TASKS,rotation=20,fontsize=9)
    ax.set_title(meas.replace("_"," "),fontsize=11,fontweight="bold")
    ax.set_ylim(1,5); ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel("Mean score (1–5) ±SEM"); axes[2].legend(fontsize=9,loc="lower right")
fig.suptitle("Task Experience Across Tasks by Condition",fontsize=13,fontweight="bold",y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/figure15.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
print("wrote figure15.png (y-axis 1-5)")

# ---- Fig 16 pre-post: mean +/- SEM, consistent 1-5 axis, faint individual lines ----
prepost=[("Perceived AI Collaboration","AI_Collaborators_pre","AI_Collaboration_post"),
         ("Intention to Collaborate","AI_Willingness_pre","AI_Willingness_post"),
         ("AI Fear/Attitudes","AI_Fear_composite","AI_Fear_post")]
fig,axes=plt.subplots(1,3,figsize=(15,4.6),sharey=True)
for ax,(lab,pre,post) in zip(axes,prepost):
    for g,color in [(1,SLc),(2,DTc)]:
        sub=df[df.Group==g][[pre,post]].dropna()
        for _,r in sub.iterrows(): ax.plot([0,1],[r[pre],r[post]],color=color,alpha=.12,lw=.7)
        m=[sub[pre].mean(),sub[post].mean()]; e=[sem(sub[pre].values),sem(sub[post].values)]
        ax.errorbar([0,1],m,yerr=e,marker="o",color=color,capsize=4,lw=2.5,ms=9,
                    label=("Servant Leader" if g==1 else "Dark Triad"),zorder=5)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Pre-Task","Post-Task"]); ax.set_xlim(-0.2,1.2)
    ax.set_ylim(1,5); ax.set_title(lab,fontsize=11,fontweight="bold")
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel("Mean score (1–5) ±SEM"); axes[0].legend(fontsize=9,loc="lower right")
fig.suptitle("Pre-Post Changes by Condition",fontsize=13,fontweight="bold",y=1.02)
fig.tight_layout(); fig.savefig(f"{FIG}/figure16.png",dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
print("wrote figure16.png (y-axis 1-5)")
