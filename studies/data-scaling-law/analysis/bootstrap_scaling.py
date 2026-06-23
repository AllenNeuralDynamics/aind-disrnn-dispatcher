#!/usr/bin/env python
"""Bootstrap the held-out scaling curve over the ~149 held-out mice (the generalization
unit) to put CIs on: per-D held-out LL, the late-D gain (saturation), the fraction of gain
captured by D=100, and the power-law exponent/asymptote. Uses the DEDUPED offline
per-subject data (one run per variant,ratio,seed). Run with the wrapper venv."""
from __future__ import annotations
import json
from collections import defaultdict
import numpy as np
import wandb
from scipy.optimize import curve_fit

PROJECT = "AIND-disRNN/mice_data_scaling"
PREFIXES = ("heldout-rerun-v1@", "heldout-rerun-v1-retry@", "heldout-rerun-v2@", "heldout-rerun-v2-retry@")
RATIO_D = {0.016:10, 0.049:30, 0.163:100, 0.489:300, 1.0:614}
NBOOT = 1000
RNG = np.random.default_rng(0)

def _variant(m):
    v=(m or {}).get("variant",""); return "v1" if v.startswith("v1") else ("v2" if v.startswith("v2") else None)

def collect():
    api=wandb.Api()
    runs=[r for r in api.runs(PROJECT) if any((r.group or "").startswith(p) for p in PREFIXES) and r.state=="finished"]
    by_cell={}
    for r in runs:
        m=r.config.get("meta",{}) or {}; var=_variant(m); ratio=m.get("source_subject_ratio"); seed=m.get("source_seed")
        if var is None or ratio is None: continue
        k=(var,round(float(ratio),3),seed)
        if k not in by_cell or str(getattr(r,"created_at","")) > str(getattr(by_cell[k],"created_at","")):
            by_cell[k]=r
    # (variant, D, subject) -> list of per-seed LL
    cell=defaultdict(lambda: defaultdict(list))  # (var,D) -> subject -> [ll over seeds]
    for (var,ratio,seed),r in by_cell.items():
        D=RATIO_D[ratio]
        art=next(a for a in r.logged_artifacts() if a.type=="run_table")
        df=art.get(next(iter(art.manifest.entries))).get_dataframe()
        for _,row in df.iterrows():
            cell[(var,D)][str(row["heldout_subject_id"])].append(float(row["eval_likelihood"]))
    # mean over seeds -> (var,D) -> {subject: ll}
    out=defaultdict(dict)
    for (var,D),subs in cell.items():
        for s,lls in subs.items(): out[(var,D)][s]=float(np.mean(lls))
    return out

def powerlaw(D,E,Dc,a): return E+(Dc/D)**a

def main():
    data=collect()
    Ds=sorted({D for (_,D) in data});
    # common held-out subjects (intersection across all cells)
    subj_sets=[set(data[k]) for k in data]; common=sorted(set.intersection(*subj_sets))
    print(f"D points: {Ds}; common held-out subjects: {len(common)}")
    def curve(var, subjects):
        return np.array([np.mean([data[(var,D)][s] for s in subjects]) for D in Ds])
    res={"Ds":Ds,"n_subjects":len(common),"nboot":NBOOT,"variants":{}}
    for var in ("v1","v2"):
        boots={"perD":[], "late_gain":[], "frac_by100":[], "alpha":[], "E":[]}
        for b in range(NBOOT):
            samp=list(RNG.choice(common,size=len(common),replace=True))
            y=curve(var,samp); boots["perD"].append(y)
            i100=Ds.index(100); i10=Ds.index(10); i614=Ds.index(614)
            boots["late_gain"].append(y[i614]-y[i100])
            tot=y[i614]-y[i10]
            boots["frac_by100"].append(((y[i100]-y[i10])/tot) if abs(tot)>1e-9 else np.nan)
            try:
                p,_=curve_fit(powerlaw,np.array(Ds,float),y,p0=[y.max(),Ds[0],0.5],maxfev=5000)
                boots["E"].append(p[0]); boots["alpha"].append(p[2])
            except Exception: pass
        perD=np.array(boots["perD"])
        def ci(a): a=np.array(a,float); a=a[np.isfinite(a)]; return [float(np.percentile(a,2.5)),float(np.percentile(a,97.5))]
        res["variants"][var]={
            "perD_mean":[float(perD[:,i].mean()) for i in range(len(Ds))],
            "perD_CI":[ci(perD[:,i]) for i in range(len(Ds))],
            "late_gain_D100_to_D614":{"mean":float(np.mean(boots["late_gain"])),"CI":ci(boots["late_gain"])},
            "frac_of_gain_by_D100":{"mean":float(np.nanmean(boots["frac_by100"])),"CI":ci(boots["frac_by100"])},
            "powerlaw_E":{"mean":float(np.mean(boots["E"])) if boots["E"] else None,"CI":ci(boots["E"]) if boots["E"] else None},
            "powerlaw_alpha":{"mean":float(np.mean(boots["alpha"])) if boots["alpha"] else None,"CI":ci(boots["alpha"]) if boots["alpha"] else None},
        }
    json.dump(res, open("studies/data-scaling-law/analysis/bootstrap_scaling.json","w"), indent=2)
    for var in ("v1","v2"):
        v=res["variants"][var]
        print(f"\n=== {var} (n={len(common)} mice, {NBOOT} boots) ===")
        for i,D in enumerate(Ds):
            lo,hi=v["perD_CI"][i]; print(f"  D={D:>4}: {v['perD_mean'][i]:.4f}  95%CI[{lo:.4f},{hi:.4f}]")
        lg=v["late_gain_D100_to_D614"]; print(f"  late gain D100->D614: {lg['mean']:+.5f}  95%CI[{lg['CI'][0]:+.5f},{lg['CI'][1]:+.5f}]  {'(excludes 0 -> still scaling)' if lg['CI'][0]>0 else '(includes 0 -> saturated)'}")
        fb=v["frac_of_gain_by_D100"]; print(f"  frac of total gain by D=100: {fb['mean']:.2f}  95%CI[{fb['CI'][0]:.2f},{fb['CI'][1]:.2f}]")
        print(f"  power-law E={v['powerlaw_E']['mean']:.4f} CI{[round(x,4) for x in v['powerlaw_E']['CI']]}  alpha={v['powerlaw_alpha']['mean']:.3f} CI{[round(x,3) for x in v['powerlaw_alpha']['CI']]}")

if __name__=="__main__": main()
