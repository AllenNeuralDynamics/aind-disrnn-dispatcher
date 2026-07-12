# Launching Beaker jobs from the Claude Science Mac sandbox

The launchers (`code/launch_beaker.py`, `code/beaker_client.py`,
`code/launch_beaker_resumable.py`) run **directly from the Mac sandbox** — no HPC
hop. They are sandbox-safe: `create_wandb_sweep()` hits the W&B GraphQL API
directly (no `wandb-core` subprocess), and `get_beaker_client()` builds
`beaker.Beaker` from `Config(user_token=os.environ["BEAKER_TOKEN"])` directly (no
`~/.beaker/config.yml`).

**Prereqs (one-time):**

- Credentials already in the sandbox env: `BEAKER_TOKEN`, `WANDB_API_KEY`.
- Network: `beaker.org` and `api.wandb.ai` each need a one-time
  `request_network_access` grant.

**PYTHONPATH quirk (sandbox only):** `PYTHONSAFEPATH=1` drops the script dir from
`sys.path`, so the launchers' sibling imports (`beaker_client`, ...) fail with
`ModuleNotFoundError` unless `code/` is on `PYTHONPATH`:

```bash
cd code
PYTHONPATH="$(pwd):$PYTHONPATH" python launch_beaker.py \
  --sweep beaker/sweep_mvp.yaml --experiment beaker/experiment_mvp.yaml \
  --workspace ai1/aind-dynamic-foraging-foundation-model \
  --output-dir ./out --label <label> --note "why this run exists" \
  --no-submit    # SAFE dry-run: creates the W&B sweep (prints SWEEP_ID) and
                 # renders the spec, does NOT submit. Drop it to actually submit.
```

`launch_beaker.py` is two-step: create the W&B sweep (prints `SWEEP_ID` =
`entity/project/id`), then submit the Beaker experiment unless `--no-submit`.
(This quirk is sandbox-only — on HPC/Code Ocean the launchers run without
`PYTHONPATH`.)

## Verify the image name before submitting (#1 stale-fact trap)

Old example specs (`experiment_h100.yaml`, `experiment_h200.yaml`,
`experiment_pack.yaml`) reference `beaker: han-hou/disrnn-wrapper`, which **no
longer exists** -> `ImageNotFound`/404. Current image for the
`ai_hub_pck_integration` line: `han-hou/disrnn-wrapper-pck-integration-20260630` —
it ships the newer `aind-dynamic-foraging-database` with
`select_sessions(snapshot=...)` support; the older `...-pck-integration`
(2026-06-18) fails on the `mice_snapshot_scaling` data path
(`TypeError: ... unexpected keyword argument 'snapshot'`). The authoritative list
is the "Available images" table in `code/beaker/README.md`; cross-check live
images and set the spec's `image.beaker` to one that exists:

```python
# repl cell, via beaker-py
[im.full_name for im in b.workspace.images(
    workspace="ai1/aind-dynamic-foraging-foundation-model")]
# or CLI: beaker workspace images ai1/aind-dynamic-foraging-foundation-model
```

Code is pulled fresh at container startup (`entrypoint.sh` checks out
`WRAPPER_REF`/`DISPATCHER_REF`), so **code/config edits need no image rebuild** —
push the branch and set the refs. Rebuild only when pinned dependencies change (a
code edit that calls a dependency with a *new* signature counts as a dependency
change).

## Transient node failure != code bug

A job can die in ~5 s with `status.message: "no space left on device"` and
`started=None` — the node's NVMe filled while `mkdir`-ing the dataset dir (seen on
`gcp-h100`). This is a per-node infra failure, not your code. Confirm with
`beaker job get <id> --format json` (`status.message` / `status.started`), then
just resubmit — it lands elsewhere.
