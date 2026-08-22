#!/usr/bin/env python
"""Build the supplementary figures, written to $STUDY_DATA_ROOT/figures/:
 - Fig 2 (communication strategy), Fig 5 (emotional tone), Fig 6 (task engagement):
   box plots with individual conversation points, all categories.
 - Fig 15 (task experience) and Fig 16 (pre-post attitudes) on the full 1-5 scale.
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
CONV=os.path.join(DATA_ROOT,"CONVERSATION_ANALYSES","conversations_dataframe.csv")
PKL=os.path.join(DATA_ROOT,"statistics_output","survey_scored.pkl")
SLc="#4C9BE0"; DTc="#E4695B"; rng=np.random.default_rng(3)
cv=pd.read_csv(CONV)
df=pd.read_pickle(PKL)

def boxpoints(ax, sl, dt):
    for i,(data,color) in enumerate([(sl,SLc),(dt,DTc)]):
        ax.boxplot(data,positions=[i],widths=0.55,patch_artist=True,showfliers=False,
                   medianprops=dict(color="black",lw=1.4),
                   boxprops=dict(facecolor=color,alpha=.35,edgecolor=color),
                   whiskerprops=dict(color=color),capprops=dict(color=color))
        x=np.full(len(data),i)+rng.uniform(-.11,.11,len(data))
        ax.scatter(x,data,s=14,color=color,alpha=.75,edgecolor="white",linewidth=.4,zorder=3)
        ax.scatter([i],[np.mean(data)],marker="D",s=42,color="black",zorder=4)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Servant\nLeader","Dark\nTriad"],fontsize=9)
    ax.spines[['top','right']].set_visible(False)

def grid_fig(specs, title, fname, ncol, ylabel="Proportion of messages", ymax=None):
    n=len(specs); nrow=int(np.ceil(n/ncol))
    fig,axes=plt.subplots(nrow,ncol,figsize=(2.7*ncol,4.0*nrow),sharey=True)
    axes=np.array(axes).reshape(-1)
    for ax,(col,lab) in zip(axes,specs):
        sl=cv.loc[cv.condition=="servant_leader",col].dropna().values
        dt=cv.loc[cv.condition=="dark_triad",col].dropna().values
        boxpoints(ax,sl,dt); ax.set_title(lab,fontsize=11,fontweight="bold")
        if ymax: ax.set_ylim(-0.02,ymax)
    for ax in axes[n:]: ax.set_visible(False)
    for r in range(nrow): axes[r*ncol].set_ylabel(ylabel)
    fig.suptitle(title,fontsize=13,fontweight="bold",y=1.005)
    fig.tight_layout()
    fig.savefig(f"{FIG}/{fname}",dpi=300,bbox_inches="tight",facecolor="white"); plt.close(fig)
    print("wrote",fname)

# ---- Fig 2 strategy (6), Fig 5 emotion (6), Fig 6 engagement (3) ----
grid_fig([("prop_compliance","Compliance"),("prop_negotiation","Negotiation"),
          ("prop_help_seeking","Help-seeking"),("prop_resistance","Resistance"),
          ("prop_coping","Coping"),("prop_jailbreaking","Jailbreaking")],
         "Communication strategy: conversation-level proportions by condition",
         "figure02.png",ncol=3)
grid_fig([("prop_emotion_neutral","Neutral"),("prop_emotion_confident","Confident"),
          ("prop_emotion_frustrated","Frustrated"),("prop_emotion_anxious","Anxious"),
          ("prop_emotion_defensive","Defensive"),("prop_emotion_resigned","Resigned")],
         "Participant emotional tone: conversation-level proportions by condition",
         "figure05.png",ncol=3)
grid_fig([("prop_engagement_high","High engagement"),("prop_engagement_medium","Medium engagement"),
          ("prop_engagement_low","Low engagement")],
         "Task engagement: conversation-level proportions by condition",
         "figure06.png",ncol=3)

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
print("SL n=%d DT n=%d (conversations)"%((cv.condition=='servant_leader').sum(),(cv.condition=='dark_triad').sum()))
