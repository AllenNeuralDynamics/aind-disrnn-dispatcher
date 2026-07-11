import os, json, base64, datetime, urllib.request
from zoneinfo import ZoneInfo

ENTITY="AIND-disRNN"; PROJECT="disrnn_updnet_bottleneck_ratio_100mice"
GROUP="updnet-ratio-100mice@20260703-200122"
GRID_EXP="01KWNH6J6YV382HH35GSDWNJAE"
WARMUP=7500; NMAIN=60000; TOTAL=WARMUP+NMAIN; GRID_N=48
KEY=os.environ["WANDB_API_KEY"]
GQL="https://api.wandb.ai/graphql"

def q(query, variables):
    data=json.dumps({"query":query,"variables":variables}).encode()
    req=urllib.request.Request(GQL, data=data, headers={"Content-Type":"application/json",
        "Authorization":"Basic "+base64.b64encode(f"api:{KEY}".encode()).decode()})
    return json.loads(urllib.request.urlopen(req, timeout=90).read())
def unwrap(d):
    if isinstance(d,dict) and "value" in d and len(d)==1: return d["value"]
    return d
def fmt_eta(s):
    if s is None: return "   -"
    h=int(s//3600); m=int((s%3600)//60)
    return f"{h}h{m:02d}m" if h else f"{m}m"
def bar(frac,width=20,state=""):
    if state=="finished": return "["+"#"*width+"] done"
    f=int(round(max(0,min(1,frac))*width))
    return "["+"#"*f+"-"*(width-f)+f"] {frac*100:3.0f}%"

# ---- Beaker: authoritative 48-task state ----
bk_states=None; n_final=n_run=n_queued=0; cluster="-"
try:
    from beaker import Beaker, Config
    b=Beaker(Config(user_token=os.environ["BEAKER_TOKEN"], default_org="ai1")); b._timeout=120
    g=b.experiment.get(GRID_EXP)
    try:
        spec=b.experiment.spec(g); cls=set()
        for tk in spec.tasks:
            cons=getattr(tk,"constraints",None)
            cl=getattr(cons,"cluster",None) if cons else None
            if cl: cls.update(cl)
        cluster=", ".join(sorted(cls)) if cls else "-"
    except Exception:
        cluster="?"
    from collections import Counter
    bs=Counter()
    for t in b.experiment.tasks(g):
        if not t.jobs: bs["queued"]+=1
        else:
            s=str(getattr(t.jobs[-1].status,'current',None))
            bs[s]+=1
    bk_states=dict(bs)
    n_final=bs.get("finalized",0); n_queued=bs.get("queued",0)
    n_run=sum(v for k,v in bs.items() if k in ("running","idle","scheduled"))
except Exception as e:
    bk_states=f"(beaker unavailable: {str(e)[:60]})"

# ---- W&B: per-run progress + ETA ----
Q="""query($p:String!,$e:String!){project(name:$p,entityName:$e){runs(first:200){edges{node{
  name state group config summaryMetrics}}}}}"""
edges=q(Q,{"p":PROJECT,"e":ENTITY})["data"]["project"]["runs"]["edges"]
rows=[]
for ed in edges:
    n=ed["node"]
    if n.get("group")!=GROUP: continue
    cfg=json.loads(n["config"]) if isinstance(n["config"],str) else (n["config"] or {})
    sm=json.loads(n["summaryMetrics"]) if isinstance(n["summaryMetrics"],str) else (n["summaryMetrics"] or {})
    model=unwrap(cfg.get("model",{})); pen=unwrap(model.get("penalties",{})); tr=unwrap(model.get("training",{}))
    beta=pen.get("beta"); unlp=pen.get("update_net_latent_penalty"); lr=tr.get("lr")
    mult=round(unlp/beta) if isinstance(unlp,(int,float)) and beta else None
    step=sm.get("_step"); runtime=sm.get("_runtime")
    main=((step-WARMUP)/NMAIN) if isinstance(step,(int,float)) and step>WARMUP else (step/TOTAL if isinstance(step,(int,float)) else 0.0)
    main=max(0.0,min(1.0,main))
    rate=(step/runtime) if isinstance(step,(int,float)) and isinstance(runtime,(int,float)) and runtime>0 else None
    eta_s=((TOTAL-step)/rate) if rate and isinstance(step,(int,float)) and step<TOTAL else None
    rows.append(dict(mult=mult,beta=beta,lr=lr,state=n["state"],main=main,eta_s=eta_s,
                     hld=sm.get("heldout/final/eval_likelihood")))

now_utc=datetime.datetime.now(datetime.timezone.utc)
now_sea=now_utc.astimezone(ZoneInfo("America/Los_Angeles"))
print("="*66)
print(f"  GRID PROGRESS  |  Seattle {now_sea:%a %Y-%m-%d %I:%M %p %Z}  (UTC {now_utc:%H:%M})")
print(f"  cluster: {cluster}")
print("="*66)

# overall 48-cell bar
if isinstance(bk_states,dict):
    ov=n_final/GRID_N
    print(f"  OVERALL  {bar(ov,width=30)}  {n_final}/{GRID_N} finalized")
    print(f"           {n_run} running/idle  |  {n_queued} queued  |  states: {bk_states}")
    # grid ETA: remaining cells / concurrency * per-cell wall (~18h at ~0.9 steps/s over 67.5k steps)
    per_cell_h=TOTAL/0.90/3600
    remaining=GRID_N-n_final
    conc=max(n_run,1)
    waves=(remaining+conc-1)//conc
    grid_eta_h=waves*per_cell_h
    print(f"           grid ETA ~{grid_eta_h:.0f}h ({remaining} cells / ~{conc} concurrent x ~{per_cell_h:.0f}h/cell)")
else:
    print(f"  OVERALL  {bk_states}")

rows.sort(key=lambda r:(0 if r["state"]=="finished" else (1 if r["state"]=="running" else 2),
                        r["mult"] or 0, r["beta"] or 0, r["lr"] or 0, -(r["main"] or 0)))
print(f"\n  {'mlt':>3} {'beta':>6} {'lr':>5} {'state':>8}  {'progress':^26} {'ETA':>6} | {'hldLL':>5}")
for r in rows:
    bstr=f"{r['beta']:.0e}" if isinstance(r['beta'],float) else "-"
    lstr=f"{r['lr']:.0e}" if isinstance(r['lr'],float) else "-"
    eta=fmt_eta(r["eta_s"]) if r["state"]=="running" else ("done" if r["state"]=="finished" else "-")
    hld=f"{r['hld']:.3f}" if isinstance(r['hld'],(int,float)) else "  -"
    tag=" <stale;live sibling" if r["state"] in ("crashed","failed") else ""
    print(f"  {str(r['mult']):>3} {bstr:>6} {lstr:>5} {r['state'][:8]:>8}  {bar(r['main'],width=18,state=r['state']):^26} {eta:>6} | {hld:>5}{tag}")
run=[r for r in rows if r["state"]=="running" and r["eta_s"]]
if run: print(f"\n  slowest active job finishes in ~{fmt_eta(max(r['eta_s'] for r in run))}")


# ================= MULT=10 SUPPLEMENT (h200/l40s) =================
SUPP_GROUP="updnet-ratio-100mice-mult10-supp@20260706-093606"
OOMRETRY_EXP="01KWWZ5WXRD8B0AXYC81T1D7SB"  # 3rd attempt, on g6e now
SUPP_EXP="01KWW4K1BVG07223K9SMJAHPP3"
try:
    from beaker import Beaker, Config
    _b=Beaker(Config(user_token=os.environ["BEAKER_TOKEN"], default_org="ai1")); _b._timeout=120
    _g=_b.experiment.get(SUPP_EXP)
    from collections import Counter as _C
    _bs=_C()
    for _t in _b.experiment.tasks(_g):
        _bs["queued" if not _t.jobs else str(getattr(_t.jobs[-1].status,'current',None))]+=1
    _nf=_bs.get("finalized",0)
    _sedges=q(Q,{"p":PROJECT,"e":ENTITY})["data"]["project"]["runs"]["edges"]
    _srows=[]
    for _ed in _sedges:
        _n=_ed["node"]
        if _n.get("group")!=SUPP_GROUP: continue
        _sm=json.loads(_n["summaryMetrics"]) if isinstance(_n["summaryMetrics"],str) else (_n["summaryMetrics"] or {})
        _step=_sm.get("_step"); _rt=_sm.get("_runtime")
        _main=((_step-WARMUP)/NMAIN) if isinstance(_step,(int,float)) and _step>WARMUP else (_step/TOTAL if isinstance(_step,(int,float)) else 0.0)
        _main=max(0.0,min(1.0,_main))
        _rate=(_step/_rt) if isinstance(_step,(int,float)) and isinstance(_rt,(int,float)) and _rt>0 else None
        _eta=((TOTAL-_step)/_rate) if _rate and isinstance(_step,(int,float)) and _step<TOTAL else None
        _srows.append(dict(state=_n["state"],main=_main,eta=_eta,hld=_sm.get("heldout/final/eval_likelihood")))
    print("\n"+"="*66)
    print(f"  MULT=10 SUPPLEMENT (h200/l40s)  exp {SUPP_EXP}")
    print(f"  OVERALL  {bar(_nf/12,width=30)}  {_nf}/12 finalized  | states: {dict(_bs)}")
    _srows.sort(key=lambda r:(0 if r["state"]=="finished" else 1, -(r["main"] or 0)))
    for r in _srows:
        eta=fmt_eta(r["eta"]) if r["state"]=="running" else ("done" if r["state"]=="finished" else "-")
        hld=f"{r['hld']:.3f}" if isinstance(r['hld'],(int,float)) else "  -"
        tag=" <stale" if r["state"] in ("crashed","failed") else ""
        print(f"     mult=10  {r['state'][:8]:>8}  {bar(r['main'],width=18,state=r['state']):^26} {eta:>6} | {hld:>5}{tag}")
except Exception as _e:
    print(f"\n  [supplement query failed: {str(_e)[:80]}]")

# ---- tiny OOM-retry check (g6e) ----
try:
    _og=_b.experiment.get(OOMRETRY_EXP)
    for _t in _b.experiment.tasks(_og):
        _s="created" if not _t.jobs else str(getattr(_t.jobs[-1].status,'current',None))
        print(f"\n  oom-retry-g6e (mult=10 b=3e-4 lr=1e-3 seed=0): {_s}")
except Exception as _e:
    print(f"  [oom-retry query failed: {str(_e)[:60]}]")
