#!/usr/bin/env python
"""Post-hoc distributional bottleneck-sparsity metrics for the beta-scan grid.

WHY: the live grid pins an older WRAPPER_REF, so it logs only the single-threshold
`bottlenecks/*_frac_open` (sigma<0.1), which SATURATES to 0 while channels sit at
sigma~0.85 (mostly-but-not-fully closed) -- and the dashboard `plot_bottlenecks`
figure calls a latent "open" at the far more permissive sigma<0.97, so the same run
reads frac_open=0.0 in the scalar and "almost all open" in the figure. Neither is
wrong; they are different cutoffs on the same sigma distribution. This script
recomputes THRESHOLD-FREE readouts for every finished run, straight from its saved
params artifact, so the multiplier-vs-sparsity comparison is read on a metric that
does not saturate and is comparable across bottleneck families of different size.

Per bottleneck family (latent, update_net_latent [the multiplier's direct target],
choice_net_latent, update_net_obs, update_net_subj, choice_net_subj):
  - n_eff_open       : participation ratio of channel openness w=clip(1-sigma,0,1),
                       (sum w)^2 / sum(w^2). Effective NUMBER of open channels; equals
                       k when k channels are equally open and the rest closed.
  - n_eff_open_frac  : n_eff_open / n_channels, in [1/n, 1]. The headline -- a
                       normalized participation ratio, COMPARABLE across families of
                       different size (5-element latent bottleneck vs the matrix-shaped
                       update-net bottleneck).
  - total_openness   : sum(1-sigma), raw information capacity (un-normalized).
  - sigma_p10/median/p90 : distribution shape (catches a bimodal open/closed split
                           that means/fractions hide).
  - frac_open_s{003,01,05,09,097} : multi-threshold empirical CDF of sigma, subsuming
                           both the strict (sigma<0.1) metric and the figure (sigma<0.97).

Sigma convention (disRNN): small sigma = OPEN (info flows), sigma->1 = CLOSED.
reparameterize_sigma(p) = abs(p) + 1e-5 (disentangled_rnns/library/disrnn.py) -- so
no disrnn library is needed; params.json is plain JSON.

DATA SOURCE: each finished run uploads a W&B training-output artifact
`disrnn-output-<run_id>` containing checkpoints/step_*/params.json. This reads the
highest-step params.json (falls back to train_state.pkl). Requires reachability of
the artifact blob store (storage.googleapis.com) -- runs on HPC / any host with GCS
egress; the Claude Science sandbox blocks that host, so run this on ssh:hpc-code or
locally, NOT in the sandbox.

Multiplier is consumed pre-training by resolve_disrnn_penalties and not logged
directly; recovered as round(update_net_latent_penalty / beta).

Usage:
    python backfill_sparsity_metrics.py \
        --group updnet-ratio-100mice@20260703-200122 \
        --out backfill_sparsity_metrics.csv
"""
import argparse, json, os, pickle, re
import numpy as np
import requests

ENTITY = "AIND-disRNN"
PROJECT = "disrnn_updnet_bottleneck_ratio_100mice"
GQL = "https://api.wandb.ai/graphql"
FAMKEYS = {
    "latent": "latent_sigma_params",
    "update_net_latent": "update_net_latent_sigma_params",
    "choice_net_latent": "choice_net_latent_sigma_params",
    "update_net_obs": "update_net_obs_sigma_params",
    "update_net_subj": "update_net_subj_sigma_params",
    "choice_net_subj": "choice_net_subj_sigma_params",
}
THRESHOLDS = (0.03, 0.1, 0.5, 0.9, 0.97)


def _wandb_key():
    """Prefer WANDB_API_KEY env; fall back to ~/.netrc entry for api.wandb.ai."""
    k = os.environ.get("WANDB_API_KEY")
    if k:
        return k
    try:
        import netrc as _netrc
        auth = _netrc.netrc().authenticators("api.wandb.ai")
        if auth and auth[2]:
            return auth[2]  # password field = the API key
    except Exception:
        pass
    raise RuntimeError("No W&B key: set WANDB_API_KEY or add api.wandb.ai to ~/.netrc")


def _sess():
    s = requests.Session()
    s.auth = ("api", _wandb_key())
    return s


def _unwrap(d):
    return d.get("value", d) if isinstance(d, dict) and "value" in d else d


def reparameterize_sigma(p):
    return np.abs(np.asarray(p, dtype=np.float64)) + 1e-5


def family_metrics(raw_sigma_param):
    sig = reparameterize_sigma(raw_sigma_param).ravel()
    n = int(sig.size)
    w = np.clip(1.0 - sig, 0.0, 1.0)
    sw = float(w.sum()); sw2 = float((w * w).sum())
    n_eff = (sw * sw / sw2) if sw2 > 0 else 0.0
    out = {
        "n": n,
        "mean_sigma": float(sig.mean()),
        "min_sigma": float(sig.min()),
        "n_eff_open": float(n_eff),
        "n_eff_open_frac": float(n_eff) / n,
        "total_openness": sw,
        "sigma_p10": float(np.percentile(sig, 10)),
        "sigma_median": float(np.percentile(sig, 50)),
        "sigma_p90": float(np.percentile(sig, 90)),
    }
    for thr in THRESHOLDS:
        tag = f"{thr:.2f}".rstrip("0").rstrip(".").replace(".", "p")
        out[f"frac_open_s{tag}"] = float(np.sum(sig < thr)) / n
    return out


def _find_ms_params(obj):
    if isinstance(obj, dict):
        if "multisubject_dis_rnn" in obj:
            return obj["multisubject_dis_rnn"]
        for v in obj.values():
            r = _find_ms_params(v)
            if r is not None:
                return r
    return None


def _stepnum(name):
    m = re.search(r"step_(\d+)", name)
    return int(m.group(1)) if m else -1


def list_group_runs(s, group):
    q = """query($p:String!,$e:String!){project(name:$p,entityName:$e){runs(first:200){edges{node{name state group config summaryMetrics}}}}}"""
    edges = s.post(GQL, json={"query": q, "variables": {"p": PROJECT, "e": ENTITY}}, timeout=90).json()["data"]["project"]["runs"]["edges"]
    runs = {}
    for e in edges:
        n = e["node"]
        if n.get("group") != group:
            continue
        cfg = json.loads(n["config"]) if isinstance(n["config"], str) else n["config"]
        sm = json.loads(n["summaryMetrics"]) if isinstance(n["summaryMetrics"], str) else (n["summaryMetrics"] or {})
        m = _unwrap(cfg.get("model", {})); pen = _unwrap(m.get("penalties", {})); tr = _unwrap(m.get("training", {}))
        beta = pen.get("beta"); unlp = pen.get("update_net_latent_penalty"); lr = tr.get("lr")
        runs[n["name"]] = {
            "state": n["state"],
            "mult": round(unlp / beta) if (beta and isinstance(unlp, (int, float))) else None,
            "beta": beta, "lr": lr, "seed": _unwrap(m.get("seed")),
            "in_eval_ll": sm.get("checkpoint/eval_likelihood"),
            "heldout_eval_ll": sm.get("heldout/final/eval_likelihood"),
        }
    return runs


def fetch_params(s, run_name):
    an = f"disrnn-output-{run_name}:latest"
    aid = s.post(GQL, json={"query": """query($p:String!,$e:String!,$an:String!){project(name:$p,entityName:$e){artifact(name:$an){id state}}}""",
                            "variables": {"p": PROJECT, "e": ENTITY, "an": an}}, timeout=90).json()["data"]["project"]["artifact"]
    if not aid or aid.get("state") != "COMMITTED":
        return None, None
    files = [fe["node"] for fe in s.post(GQL, json={"query": """query($id:ID!){artifact(id:$id){files(first:400){edges{node{name directUrl}}}}}""",
                                                    "variables": {"id": aid["id"]}}, timeout=90).json()["data"]["artifact"]["files"]["edges"]]
    pj = sorted([f for f in files if f["name"].endswith("params.json")], key=lambda f: _stepnum(f["name"]))
    ts = sorted([f for f in files if f["name"].endswith("train_state.pkl")], key=lambda f: _stepnum(f["name"]))
    for cand in ([pj[-1]] if pj else []) + ([ts[-1]] if ts else []):
        b = s.get(cand["directUrl"], timeout=300).content
        try:
            o = json.loads(b) if cand["name"].endswith(".json") else pickle.loads(b)
        except Exception:
            continue
        mp = _find_ms_params(o)
        if mp is not None:
            return mp, cand["name"]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--out", default="backfill_sparsity_metrics.csv")
    ap.add_argument("--only-finished", action="store_true", default=True)
    args = ap.parse_args()
    s = _sess()
    runs = list_group_runs(s, args.group)
    finished = {k: v for k, v in runs.items() if (v["state"] == "finished" or not args.only_finished)}
    print(f"group {args.group}: {len(runs)} runs, {len(finished)} finished")
    import csv
    rows = []
    for rn, meta in sorted(finished.items(), key=lambda kv: (kv[1]["mult"] or 0, kv[1]["beta"] or 0, kv[1]["lr"] or 0, kv[1]["seed"] or 0)):
        mp, src = fetch_params(s, rn)
        if mp is None:
            print(f"  SKIP {rn[-8:]}: no params artifact"); continue
        rec = {"run": rn, "src": src, **{k: meta[k] for k in ("mult", "beta", "lr", "seed", "in_eval_ll", "heldout_eval_ll")}}
        for fam, key in FAMKEYS.items():
            if key not in mp:
                continue
            for mk, mv in family_metrics(mp[key]).items():
                rec[f"{fam}.{mk}"] = mv
        rows.append(rec)
        u = {k[len("update_net_latent."):]: v for k, v in rec.items() if k.startswith("update_net_latent.")}
        print(f"  {rn[-8:]} mult={meta['mult']} b={meta['beta']} lr={meta['lr']} seed={meta['seed']} | "
              f"updnet neff_frac={u.get('n_eff_open_frac',float('nan')):.3f} mean_s={u.get('mean_sigma',float('nan')):.3f} "
              f"| inLL={meta['in_eval_ll']} hldLL={meta['heldout_eval_ll']}")
    if rows:
        cols = sorted({k for r in rows for k in r})
        head = [c for c in ("run", "mult", "beta", "lr", "seed", "in_eval_ll", "heldout_eval_ll", "src") if c in cols]
        rest = [c for c in cols if c not in head]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=head + rest); w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {len(rows)} rows x {len(head+rest)} cols -> {args.out}")


if __name__ == "__main__":
    main()
