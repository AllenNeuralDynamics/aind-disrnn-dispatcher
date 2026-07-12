# Reading W&B from the Claude Science sandbox (GraphQL, no SDK)

Producer scripts run on the HPC node (or any authenticated machine) where the
`wandb` SDK works normally — nothing special needed there. **But** in the Claude
Science Mac sandbox, `wandb.Api()` fails: it tries to spawn a `wandb-core` helper
subprocess (`ServicePollForTokenError`) and write config under `~/.config/wandb`,
both blocked. Working route:

- Store the key as the `WANDB_API_KEY` credential; hit the GraphQL endpoint
  directly with `requests`, `auth=('api', KEY)` — no SDK.
- `https://api.wandb.ai` must be allowlisted (one-time `request_network_access`).
- Entity/login is `houhan` (`hanhou`); team entity `AIND-disRNN`.

```python
import requests, os
KEY = os.environ["WANDB_API_KEY"]
GQL = "https://api.wandb.ai/graphql"
q = """query($e:String!,$p:String!,$s:String!){
  project(name:$p,entityName:$e){ sweep(sweepName:$s){ id state
    runs(first:50){ edges{ node{ name displayName state summaryMetrics config }}}}}}"""
r = requests.post(GQL, auth=("api", KEY),
    json={"query": q, "variables": {"e":"AIND-disRNN","p":"hpc_test","s":"<sweep_id>"}},
    timeout=60).json()
# per-run full history: query run(name:$run){ history(samples:100000) }
```

Prefer running the producer on the HPC node when possible; use the GraphQL route
only for ad-hoc pulls from the sandbox.

Gotcha for run deletion (study wrap-up): the GraphQL `deleteRun` payload has
`clientMutationId` + `numDeleted`, not `success` — selecting `success` returns
HTTP 400.
