"""Library-only clients for W&B sweep creation and Beaker submission.

Both `launch_beaker.py` and `launch_beaker_resumable.py` used to shell out to the
`wandb` and `beaker` CLIs via `subprocess`. That made them depend on two CLI binaries
being installed and reachable on PATH -- fine on Code Ocean / HPC (both installed by
`environment/postInstall`), but a `wandb sweep` invocation also tries to launch a
local `wandb-core` helper process over a Unix socket, and there is no portable way to
install the Beaker CLI (a compiled Go binary) everywhere the dispatcher might run.

This module replaces both CLI calls with their underlying HTTP/library equivalents,
which work identically wherever Python + network access are available:

  * `create_wandb_sweep` -- the same GraphQL `UpsertSweep` mutation the `wandb` CLI
    issues under the hood (bypasses the `wandb-core` service process entirely).
  * `get_beaker_client` / `submit_beaker_experiment` -- the official `beaker-py`
    client, which talks to the Beaker server directly over HTTPS (no CLI required).

Both routes were validated end-to-end (sweep creation, Beaker submission, run
completion) from a sandboxed environment where neither CLI binary is installable
and local sockets/config-file writes are blocked -- see
`docs/claude-science-workflow.md` for that context. They are unconditionally used
here (not sandbox-specific): using them everywhere means one code path instead of
two, and they are strictly more portable than the CLI on HPC/CO as well.
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
    experiment = beaker.experiment.create(name, spec, workspace=workspace)
    return experiment.id
