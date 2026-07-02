"""Library-only clients for W&B sweep creation and Beaker submission.

WHY THIS MODULE EXISTS: we want ONE launcher code path
(`launch_beaker.py` / `launch_beaker_resumable.py`) that runs unmodified on
BOTH the Allen HPC / Code Ocean side AND the Claude Science sandbox (the Mac
agent environment orchestrating launches per
`docs/claude-science-workflow.md`) -- not two maintained copies. Both
launchers used to shell out to the `wandb` and `beaker` CLIs via `subprocess`,
which works fine on Code Ocean / HPC (both installed by
`environment/postInstall`) but breaks the sandbox in two independent ways:

  1. `wandb sweep <file>` spawns a local `wandb-core` helper process that binds
     a Unix socket and writes to `~/.config/wandb` -- both blocked in the
     Claude Science sandbox, so sweep creation fails outright there.
  2. The `beaker` CLI is a compiled Go binary with no portable install path in
     the sandbox (no apt/brew, no prebuilt wheel) -- it simply can't be put on
     PATH there.

Rather than special-case the sandbox (an `if sandbox: ... else: subprocess...`
branch, or a second sandbox-only launcher), this module swaps BOTH CLI calls
for their underlying HTTP/library equivalents, which work identically
everywhere Python + network access are available -- HPC, Code Ocean, and the
sandbox alike:

  * `create_wandb_sweep` -- the same GraphQL `UpsertSweep` mutation the `wandb`
    CLI issues under the hood (bypasses the `wandb-core` service process
    entirely, so it can't hit the socket/config-file restriction above).
  * `get_beaker_client` / `submit_beaker_experiment` -- the official `beaker-py`
    client, which talks to the Beaker server directly over HTTPS (no CLI
    binary required, so there's nothing to install).

These are used unconditionally by both launchers now, on every platform --
not just when running in the sandbox. That is the point: one code path that
happens to also be more portable than the CLI it replaced, rather than a
sandbox-specific fork. Verified end-to-end (sweep creation, Beaker submission,
run completion) from the Claude Science sandbox, where the old CLI-based code
could not run at all -- see `docs/claude-science-workflow.md` for the broader
Mac/HPC/Beaker architecture this fits into.
"""

from __future__ import annotations

import os
from pathlib import Path

import requests
import yaml

WANDB_GQL_URL = "https://api.wandb.ai/graphql"

_UPSERT_SWEEP_MUTATION = """
mutation UpsertSweep(
    $id: ID,
    $config: String,
    $description: String,
    $entityName: String,
    $projectName: String,
    $controller: JSONString,
    $scheduler: JSONString,
    $state: String,
    $priorRunsFilters: JSONString,
    $displayName: String,
) {
    upsertSweep(input: {
        id: $id,
        config: $config,
        description: $description,
        entityName: $entityName,
        projectName: $projectName,
        controller: $controller,
        scheduler: $scheduler,
        state: $state,
        priorRunsFilters: $priorRunsFilters,
        displayName: $displayName,
    }) {
        sweep {
            name
            project { id name entity { id name } }
        }
        configValidationWarnings
    }
}
"""


def create_wandb_sweep(sweep_file: str, wandb_api_key: str | None = None) -> str:
    """Create a W&B sweep from a sweep YAML file via the GraphQL API directly.

    Equivalent to `wandb sweep <sweep_file>`, but makes one HTTPS POST instead of
    spawning the `wandb` CLI (which starts a local `wandb-core` service process).
    Returns the full sweep path `entity/project/sweep_id`, same format the CLI's
    `wandb agent <SWEEP_ID>` line would have printed.
    """
    api_key = wandb_api_key or os.environ["WANDB_API_KEY"]
    cfg = yaml.safe_load(Path(sweep_file).read_text())
    entity, project = cfg["entity"], cfg["project"]

    resp = requests.post(
        WANDB_GQL_URL,
        auth=("api", api_key),
        json={
            "query": _UPSERT_SWEEP_MUTATION,
            "variables": {
                "config": yaml.safe_dump(cfg),
                "entityName": entity,
                "projectName": project,
            },
        },
        timeout=60,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("errors"):
        raise RuntimeError(f"wandb UpsertSweep failed: {body['errors']}")
    sweep_name = body["data"]["upsertSweep"]["sweep"]["name"]
    return f"{entity}/{project}/{sweep_name}"


def get_beaker_client(workspace: str | None = None):
    """Build an authenticated `beaker.Beaker` client from `BEAKER_TOKEN`.

    Uses `beaker.Config(user_token=...)` directly instead of `Beaker.from_env()` /
    `Beaker.from_config()`, which look for `~/.beaker/config.yml` on disk -- a file
    that may not exist (fresh environment) or may not be readable (sandboxed
    environments that restrict access to dotfiles under `$HOME`). The explicit
    `Config` constructor does no filesystem I/O and works identically everywhere.
    """
    from beaker import Beaker, Config

    token = os.environ["BEAKER_TOKEN"]
    cfg = Config(user_token=token, default_org="ai1", default_workspace=workspace)
    return Beaker(cfg)


def submit_beaker_experiment(spec_path: str, workspace: str, name: str | None = None) -> str:
    """Submit a rendered Beaker experiment spec file. Returns the experiment id."""
    from beaker import ExperimentSpec

    beaker = get_beaker_client(workspace=workspace)
    spec = ExperimentSpec.from_file(spec_path)
    # Pass `name`/`workspace` as keywords only -- ExperimentClient.create()'s
    # positional-arg parser rejects `name=None` passed positionally (it can't tell
    # that apart from a malformed call), so `create(name, spec, ...)` breaks when
    # name is None. `create(spec, name=name, ...)` handles None cleanly.
    experiment = beaker.experiment.create(spec, name=name, workspace=workspace)
    return experiment.id
