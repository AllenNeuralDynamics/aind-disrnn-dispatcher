#!/usr/bin/env python
"""nxd-h512 grid-progress report generator.

Queries W&B for live per-run progress and renders a job-status board: overall
progress bar, per-job progress + per-job ETA from each run's actual
steps/sec, Seattle/Pacific time on the header, and a grid ETA. Overall counts
(done/running/queued) are derived ENTIRELY from per-cell W&B run state, not
from the parsed SLURM snapshot -- deliberately: SLURM's own COMPLETED status
does not distinguish a crashed run (`wandb agent` can exit 0 on a child crash)
from a genuinely finished one, so W&B is the source of truth for "done" here.
`parse_slurm()` still reads a SLURM snapshot (written by the launching session
via the compute transport into ./slurm_state.txt) for `tasks`/`coll`/`squeue`,
but as of this version that parsed state is not consulted by `build_report()`
-- it is a currently-unused input, not a fallback that "catches queued tasks
W&B can't see" as an earlier draft of this docstring claimed.

Usage:
    WANDB_API_KEY=... python report_nxd_h512.py [handoff/slurm_state.txt]

Writes the report to stdout. The launching session polls SLURM every ~3h,
refreshes slurm_state.txt, and re-runs this.
"""
import requests, json, re, os, sys, datetime
from zoneinfo import ZoneInfo

ENTITY, PROJ = "AIND-disRNN", "mice_data_scaling"
# The H=512 row spans two W&B groups / SLURM arrays:
#   - h200 launch: 9/12 cells (D=10/30/100) finished; the 3 D=614 host-RAM-OOM'd.
#   - d614 resubmit on h200 with --mem=256G: the 3 D=614 seeds.
# The board merges runs across both groups, one row per (H,D,seed) cell.
GROUPS = ["nxd-h512@20260720-195322", "nxd-h512@20260723-125804"]
ARRAYS = ["23263174", "23302804"]
GROUP = GROUPS[0]   # back-compat for footer text
ARRAY = ARRAYS[0]
SWEEP_URLS = {
    "nxd-h512@20260720-195322": "https://wandb.ai/AIND-disRNN/mice_data_scaling/sweeps/ajsw1a8h",
    "nxd-h512@20260723-125804": "https://wandb.ai/AIND-disRNN/mice_data_scaling/sweeps/aw04872p",
}
SWEEP_URL = SWEEP_URLS[GROUPS[0]]
URL = "https://api.wandb.ai/graphql"
EARLYSTOP_STEPS = 90500   # observed median gated-early-stop across the study
N_CELLS = 12
CELLS = [(512, D, s) for D in (10, 30, 100, 614) for s in (0, 1, 2)]
RATIO_D = {0.016: 10, 0.049: 30, 0.163: 100, 1.0: 614}

_sess = requests.Session(); _sess.auth = ("api", os.environ["WANDB_API_KEY"])
_Q = """query Runs($entity:String!,$project:String!,$filters:JSONString,$cursor:String){
 project(name:$project,entityName:$entity){runs(filters:$filters,first:100,after:$cursor){
  edges{node{name state config summaryMetrics}} pageInfo{hasNextPage endCursor}}}}"""

def wb_runs(group):
    out=[]; cur=None
    while True:
        r=_sess.post(URL, json={"query":_Q, "variables":{"entity":ENTITY,"project":PROJ,
              "filters":json.dumps({"group":group}),"cursor":cur}}, timeout=60)
        d=r.json()["data"]["project"]["runs"]
        out+=[e["node"] for e in d["edges"]]
        if d["pageInfo"]["hasNextPage"]: cur=d["pageInfo"]["endCursor"]
        else: break
    return out

def _dkey(x):
    if x is None: return None
    try: x=float(x)
    except: return None
    for rr,dd in RATIO_D.items():
        if abs(x-rr)<1e-4: return dd
    return None
def _arch(cfg):
    m=cfg.get("model")
    if isinstance(m,dict):
        mv=m.get("value",m); return (mv.get("architecture") or {}).get("hidden_size")
    return None
def _ratio(cfg):
    dd=cfg.get("data")
    if isinstance(dd,dict): return (dd.get("value",dd)).get("subject_ratio")
    return None
def _seed(cfg):
    s=cfg.get("seed")
    return s.get("value") if isinstance(s,dict) else s

def parse_slurm(txt):
    sec={}; cur=None
    for line in txt.splitlines():
        if line.startswith("===") and line.endswith("==="):
            cur=line.strip("= "); sec[cur]=[]; continue
        if cur is not None and line.strip(): sec[cur].append(line)
    tasks={}; coll=None
    for row in sec.get("SACCT",[]):
        f=row.split("|"); jid=f[0]
        m=re.match(r"%s_(\d+)$" % ARRAY, jid)
        if m: tasks[int(m.group(1))]=(f[1], f[3] if len(f)>3 else "", f[5] if len(f)>5 else "")
        elif "_[" in jid: coll=f[1] if len(f)>1 else "PENDING"
    squeue={}
    for row in sec.get("SQUEUE",[]):
        f=row.split("|"); squeue[f[0]]=(f[1], f[3] if len(f)>3 else "")
    return tasks, coll, squeue

def bar(frac, width=24):
    frac=max(0.0,min(1.0,frac)); n=int(round(frac*width))
    return "\u2588"*n + "\u2591"*(width-n)
def fmt_hrs(h):
    if h is None: return "  ?"
    if h<0: return "0h"
    if h>=24: return f"{h/24:.1f}d"
    return f"{h:.1f}h"

# cell state priority for merging runs across groups: a finished sibling beats a
# crashed one for the same (D,seed); among live states prefer more progress.
_ST_RANK = {"finished": 4, "running": 3, "crashed": 1, "failed": 1}
def _better(a, b):
    """Return the run dict that best represents a cell (a may be None)."""
    if a is None: return b
    ra, rb = _ST_RANK.get(a["state"], 0), _ST_RANK.get(b["state"], 0)
    if rb != ra: return b if rb > ra else a
    return b if (b["step"] or 0) > (a["step"] or 0) else a

def build_report(slurm_txt):
    now_pt=datetime.datetime.now(ZoneInfo("America/Los_Angeles"))
    tasks, coll, squeue = parse_slurm(slurm_txt)
    # Merge W&B runs across ALL groups, one representative run per (H,D,seed).
    by={}
    for g in GROUPS:
        for n in wb_runs(g):
            cfg=json.loads(n["config"]); sm=json.loads(n["summaryMetrics"] or "{}")
            r=dict(name=n["name"], state=n["state"], group=g, H=_arch(cfg),
                   D=_dkey(_ratio(cfg)), seed=_seed(cfg),
                   step=sm.get("_step"), rt=sm.get("_runtime"))
            k=(r["H"],r["D"],r["seed"])
            if None in k: continue
            by[k]=_better(by.get(k), r)
    # Overall counts derived from per-cell state (W&B is source of truth for done).
    n_done=n_run=n_bad=n_pending=0
    for (H,D,s) in CELLS:
        r=by.get((H,D,s))
        if r is None: n_pending+=1
        elif r["state"]=="finished": n_done+=1
        elif r["state"]=="running": n_run+=1
        elif r["state"] in ("crashed","failed"): n_bad+=1
        else: n_pending+=1
    L=[]
    L.append(f"# nxd-h512 watcher \u2014 {now_pt:%Y-%m-%d %H:%M %Z}")
    L.append(f"groups {' + '.join(GROUPS)}  \u00b7  arrays {' + '.join(ARRAYS)}")
    L.append("")
    L.append(f"OVERALL  [{bar(n_done/N_CELLS)}] {n_done}/{N_CELLS} done   "
             f"({n_run} running, {n_pending} queued"
             + (f", {n_bad} FAILED" if n_bad else "") + ")")
    L.append("")
    L.append("per-job (H=512):")
    L.append(f"  {'cell':<14}{'state':<10}{'progress':<28}{'step':>10}{'ETA':>7}")
    etas=[]
    for (H,D,s) in CELLS:
        r=by.get((H,D,s)); label=f"D={D} s{s}"
        if r is None:
            L.append(f"  {label:<14}{'queued':<10}{bar(0):<28}{'-':>10}{'-':>7}"); continue
        st=r["state"]; step=r["step"] or 0; rt=r["rt"]; tgt=EARLYSTOP_STEPS
        frac=min(step/tgt,1.0) if step else 0.0; eta=None
        if st=="running" and step and rt and step>500:
            eta=max(tgt-step,0)*(rt/step)/3600.0; etas.append(eta)
        elif st=="finished": frac=1.0
        note = " <FAILED, resubmit pending" if st in ("crashed","failed") else ""
        L.append(f"  {label:<14}{st:<10}{bar(frac):<28}{step:>10}{fmt_hrs(eta):>7}{note}")
    L.append("")
    if n_done==N_CELLS:
        L.append(f"ALL {N_CELLS} CELLS FINISHED \u2014 ready to wire H=512 into nxd_scaling.py.")
    elif n_run or n_pending:
        grid_eta=max(etas) if etas else None
        L.append(f"grid ETA (slowest running cell): ~{fmt_hrs(grid_eta)}"
                 + ("  [+queued not yet counted]" if n_pending else ""))
    else:
        L.append(f"{n_bad} cell(s) FAILED and nothing running/queued \u2014 awaiting a fix/resubmit "
                 f"decision (see SLURM sacct for the failure mode).")
    return "\n".join(L)

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "handoff/slurm_state.txt"
    print(build_report(open(path).read()))
