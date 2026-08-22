#!/usr/bin/env python
"""Build the consolidated main-text Figure 2 (behavioral effects overview):
 panel a: communication strategies (compliance / resistance / help-seeking)
 panel b: emotional tone (frustrated / anxious / defensive)
 panel c: strategy diversity (Shannon entropy)
 panel d: turning points (strategy switches)
Box-and-whisker + individual conversation points + group-mean diamond, SL vs DT.
Written to $STUDY_DATA_ROOT/figures/figure_main02_behavior.png.
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
SLc="#4C9BE0"; DTc="#E4695B"; rng=np.random.default_rng(3)
cv=pd.read_csv(CONV)

def boxpoints(ax, xpos, col, ylab_prop=True):
    sl=cv.loc[cv.condition=="servant_leader",col].dropna().values
    dt=cv.loc[cv.condition=="dark_triad",col].dropna().values
    for j,(data,color) in enumerate([(sl,SLc),(dt,DTc)]):
        x0=xpos+j
        ax.boxplot(data,positions=[x0],widths=0.55,patch_artist=True,showfliers=False,
                   medianprops=dict(color="black",lw=1.4),
                   boxprops=dict(facecolor=color,alpha=.35,edgecolor=color),
                   whiskerprops=dict(color=color),capprops=dict(color=color))
        xx=np.full(len(data),x0)+rng.uniform(-.11,.11,len(data))
        ax.scatter(xx,data,s=13,color=color,alpha=.75,edgecolor="white",linewidth=.4,zorder=3)
        ax.scatter([x0],[np.mean(data)],marker="D",s=40,color="black",zorder=4)
    return sl,dt

fig,axes=plt.subplots(1,4,figsize=(15,4.6))

# Panel a: strategies (grouped)
ax=axes[0]
cats_a=[("prop_compliance","Compliance"),("prop_resistance","Resistance"),("prop_help_seeking","Help-\nseeking")]
ticks=[]
for i,(col,lab) in enumerate(cats_a):
    boxpoints(ax,i*2.5,col); ticks.append((i*2.5+0.5,lab))
ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks],fontsize=9)
ax.set_ylim(-0.03,1.03); ax.set_ylabel("Proportion of messages")
ax.set_title("a  Communication strategy",fontsize=11,fontweight="bold",loc="left")

# Panel b: emotions (grouped)
ax=axes[1]
cats_b=[("prop_emotion_frustrated","Frustrated"),("prop_emotion_anxious","Anxious"),("prop_emotion_defensive","Defensive")]
ticks=[]
for i,(col,lab) in enumerate(cats_b):
    boxpoints(ax,i*2.5,col); ticks.append((i*2.5+0.5,lab))
ax.set_xticks([t[0] for t in ticks]); ax.set_xticklabels([t[1] for t in ticks],fontsize=9)
ax.set_ylim(-0.03,0.75); ax.set_ylabel("Proportion of messages")
ax.set_title("b  Emotional tone",fontsize=11,fontweight="bold",loc="left")

# Panel c: strategy diversity
ax=axes[2]
boxpoints(ax,0,"strategy_diversity")
ax.set_xticks([0,1]); ax.set_xticklabels(["Servant\nLeader","Dark\nTriad"],fontsize=9)
ax.set_ylabel("Shannon entropy (bits)")
ax.set_title("c  Strategy diversity",fontsize=11,fontweight="bold",loc="left")

# Panel d: turning points
ax=axes[3]
boxpoints(ax,0,"num_turning_points")
ax.set_xticks([0,1]); ax.set_xticklabels(["Servant\nLeader","Dark\nTriad"],fontsize=9)
ax.set_ylabel("Turning points (count)")
ax.set_title("d  Turning points",fontsize=11,fontweight="bold",loc="left")

for ax in axes: ax.spines[['top','right']].set_visible(False)

# shared legend
from matplotlib.patches import Patch
handles=[Patch(facecolor=SLc,alpha=.5,label="Servant Leader (n = 31)"),
         Patch(facecolor=DTc,alpha=.5,label="Dark Triad (n = 27)")]
fig.legend(handles=handles,loc="upper center",ncol=2,fontsize=10,frameon=False,bbox_to_anchor=(0.5,1.06))
fig.tight_layout()
fig.savefig(f"{FIG}/figure_main02_behavior.png",dpi=300,bbox_inches="tight",facecolor="white")
plt.close(fig)
print("wrote figure_main02_behavior.png")
print("SL n=%d DT n=%d"%((cv.condition=='servant_leader').sum(),(cv.condition=='dark_triad').sum()))
