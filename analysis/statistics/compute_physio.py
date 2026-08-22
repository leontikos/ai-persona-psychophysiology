#!/usr/bin/env python
"""Physiology: participant-level bootstrap CIs for the EDA peak and AUC difference and
ratio, and the zygomaticus (channel 228) overall-window effect size with CI."""
import json, os

import numpy as np, pandas as pd
from scipy import stats
from recompute_stats import STATS_OUT, bca_ci, d_ci_nct, mw_z, rank_biserial
from dotenv import load_dotenv

RNG=np.random.default_rng(11)

# ===== CONFIGURATION =====
# All paths derive from STUDY_DATA_ROOT; see README.md, "Data layout".
load_dotenv()
DATA_ROOT=os.environ.get("STUDY_DATA_ROOT","data")
d=json.load(open(os.path.join(DATA_ROOT,"event_related_analysis","extracted_data","participant_epochs.json")))

# group_1 = Servant Leader, group_2 = Dark Triad
def per_participant(channel, window=(0.0,18.0)):
    sl_peak,sl_auc,dt_peak,dt_auc=[],[],[],[]
    for pid,pd_ in d.items():
        grp=pd_["experimental_group"]
        if channel not in pd_["channels"]: continue
        ch=pd_["channels"][channel]
        t=np.array(ch["time_axis"]); ep=np.array(ch["epochs"])
        if ep.ndim!=2 or ep.shape[0]==0: continue
        mean_ep=np.nanmean(ep,axis=0)
        mask=(t>=window[0])&(t<=window[1])
        peak=np.nanmax(mean_ep[mask])
        auc=np.trapz(mean_ep[mask],t[mask])
        if grp=="group_1": sl_peak.append(peak); sl_auc.append(auc)
        else: dt_peak.append(peak); dt_auc.append(auc)
    return (np.array(sl_peak),np.array(dt_peak),np.array(sl_auc),np.array(dt_auc),t)

def summarize(name, sl, dt, unit=""):
    diff=np.mean(dt)-np.mean(sl); ratio=np.mean(dt)/np.mean(sl)
    dlo,dhi=bca_ci((sl,dt), lambda x,y: np.mean(y)-np.mean(x))
    rlo,rhi=bca_ci((sl,dt), lambda x,y: np.mean(y)/np.mean(x))
    print(f"{name}: SL M={np.mean(sl):.4f} (n={len(sl)}), DT M={np.mean(dt):.4f} (n={len(dt)}) {unit}")
    print(f"   DT-SL diff = {diff:.4f}, 95% BCa CI [{dlo:.4f}, {dhi:.4f}]")
    print(f"   DT/SL ratio = {ratio:.3f}  (i.e. {100*(ratio-1):+.1f}%), 95% BCa CI [{rlo:.3f}, {rhi:.3f}]")
    return dict(measure=name,M_SL=np.mean(sl),M_DT=np.mean(dt),diff=diff,diff_lo=dlo,diff_hi=dhi,
                ratio=ratio,pct=100*(ratio-1),ratio_lo=rlo,ratio_hi=rhi,n_SL=len(sl),n_DT=len(dt))

print("="*70); print("EDA (channel 305) participant-level, window 0-18s"); print("="*70)
sl_p,dt_p,sl_a,dt_a,t=per_participant("main_EDA2111000305",(0,18))
print("time axis range:",round(t.min(),2),"to",round(t.max(),2))
rows=[]
rows.append(summarize("EDA peak amplitude", sl_p, dt_p, "uS"))
rows.append(summarize("EDA AUC (0-18s)", sl_a, dt_a, "uS*s"))

def cohend(a,b):
    n1,n2=len(a),len(b)
    p=np.sqrt(((n1-1)*np.var(a,ddof=1)+(n2-1)*np.var(b,ddof=1))/(n1+n2-2))
    return (np.mean(b)-np.mean(a))/p if p>0 else 0.0

print("\n"+"="*70); print("ZYGOMATICUS (channel 228) overall window 0-6s, participant-level"); print("="*70)
slz,dtz,_,_,tz=per_participant("main_fEMG2212000228",(0,6))
dz=cohend(slz,dtz); dlo,dhi=d_ci_nct(dz,len(slz),len(dtz))
U,up,z=mw_z(slz,dtz); rrb=rank_biserial(slz,dtz); rlo,rhi=bca_ci((slz,dtz),rank_biserial)
print(f"peak: SL M={np.mean(slz):.5f}, DT M={np.mean(dtz):.5f}")
print(f"   Cohen d (DT-SL) = {dz:.3f}, 95% CI [{dlo:.3f},{dhi:.3f}]; rank-biserial r={rrb:.3f} [{rlo:.3f},{rhi:.3f}]; MW p(participant)={up:.3f}")
rows.append(dict(measure="Zygomaticus peak (0-6s), participant-level",M_SL=np.mean(slz),M_DT=np.mean(dtz),
                 d=dz,d_lo=dlo,d_hi=dhi,rrb=rrb,rrb_lo=rlo,rrb_hi=rhi,mw_p=up,n_SL=len(slz),n_DT=len(dtz)))

print("\n"+"="*70); print("CORRUGATOR (channel 226) overall window 0-6s, participant-level"); print("="*70)
slc,dtc,_,_,tc=per_participant("main_fEMG2212000226",(0,6))
dc=cohend(slc,dtc); dclo,dchi=d_ci_nct(dc,len(slc),len(dtc))
Uc,upc,zc=mw_z(slc,dtc)
print(f"peak: SL M={np.mean(slc):.5f}, DT M={np.mean(dtc):.5f}, Cohen d (DT-SL)={dc:.3f} [{dclo:.3f},{dchi:.3f}], MW p={upc:.3f}")
rows.append(dict(measure="Corrugator peak (0-6s), participant-level",M_SL=np.mean(slc),M_DT=np.mean(dtc),
                 d=dc,d_lo=dclo,d_hi=dchi,mw_p=upc,n_SL=len(slc),n_DT=len(dtc)))

pd.DataFrame(rows).to_csv(f"{STATS_OUT}/physio_enhanced.csv",index=False)
print("\nsaved physio_enhanced.csv")
print("\nNOTE: original permutation tests used n_permutations=1000; p-values: EDA=0.017, zygomaticus(228)=0.010, corrugator(226)=0.307")
