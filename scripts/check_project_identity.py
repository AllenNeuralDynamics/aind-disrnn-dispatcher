#!/usr/bin/env python3
"""Fail when project-identity ``disrnn`` tokens reappear in live paths.

ADR-0007 splits ``disrnn`` into three kinds of token:

* **project identity** — repo names, the conda envs, the env-var prefix. These
  renamed to ``dynamic-foraging-bfm`` and must not come back. This script
  rejects them.
* **model architecture** — ``DisrnnTrainer``, ``disrnn_network``,
  ``disrnn_config``, ``create_disrnn_dataset``, ``disentangled_rnns``. disRNN
  sits alongside GRU and HB; these are correct and are never matched here.
* **frozen provenance** — ``studies/`` and the ADR itself, which record what
  actually ran. Skipped wholesale.

Deliberate exceptions live in ``project_identity_allowlist.toml`` beside this
file, each with a written reason.

Note what is NOT rejected: ``AIND-disRNN``. The ADR-0007 amendment keeps that
W&B entity permanently -- it is the lab's namespace, holding 2,939 runs across
several people, and W&B has no redirect. It is live identity, not a stale
token.

Usage::

    python scripts/check_project_identity.py            # check the tree
    python scripts/check_project_identity.py --self-test  # prove it can fail
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST = Path(__file__).resolve().parent / "project_identity_allowlist.toml"

# Paths whose content records what actually ran, or *defines* the rule itself.
# The checker and its allowlist necessarily quote the tokens they reject -- in
# the reject patterns, the self-test fixtures, and the written reasons -- so
# scanning them would make the rule fail on its own definition. Same reasoning
# as docs/adr/, which states the boundary in prose.
SKIP_PREFIXES = (
    "studies/",
    "docs/adr/",
    "scripts/check_project_identity.py",
    "scripts/project_identity_allowlist.toml",
)

# Project-identity tokens. Deliberately NOT including AIND-disRNN (see above).
REJECT = {
    "repo name": re.compile(r"aind-disrnn-(?:dispatcher|wrapper)|aind_disrnn_utils"),
    "env-var prefix": re.compile(r"\bDISRNN_[A-Z0-9_]+"),
    "conda env": re.compile(r"\bdisrnn-(?:cpu|gpu)\b"),
}

# Binary-ish suffixes we never scan.
SKIP_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf", ".pkl", ".ico", ".gz", ".zip")


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [p for p in out.stdout.splitlines() if p]


def load_allowlist() -> list[dict]:
    with ALLOWLIST.open("rb") as fh:
        data = tomllib.load(fh)
    rules = data.get("allow", [])
    for i, rule in enumerate(rules):
        missing = {"pattern", "paths", "reason"} - set(rule)
        if missing:
            raise SystemExit(
                f"allowlist entry {i} is missing {sorted(missing)}; every exception "
                f"needs a written reason ({ALLOWLIST.name})"
            )
        rule["_regex"] = re.compile(rule["pattern"])
    return rules


def allowed(path: str, line: str, rules: list[dict]) -> bool:
    for rule in rules:
        if not any(fnmatch.fnmatch(path, g) for g in rule["paths"]):
            continue
        if rule["_regex"].search(line):
            return True
    return False


def scan(rules: list[dict]) -> list[tuple[str, int, str, str]]:
    findings: list[tuple[str, int, str, str]] = []
    for rel in tracked_files():
        if rel.startswith(SKIP_PREFIXES) or rel.endswith(SKIP_SUFFIXES):
            continue
        try:
            text = (REPO_ROOT / rel).read_text(encoding="utf-8", errors="replace")
        except (OSError, IsADirectoryError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for kind, pattern in REJECT.items():
                if pattern.search(line) and not allowed(rel, line, rules):
                    findings.append((rel, lineno, kind, line.strip()[:100]))
                    break
    return findings


def self_test(rules: list[dict]) -> int:
    """Prove the checker can fail, not just pass (ADR-0007 / #84)."""
    cases = [
        ("code/launch_beaker.py", "IMAGE = 'aind-disrnn-wrapper'", "repo name", True),
        ("code/foo.py", 'os.environ["DISRNN_OUTPUT_DIR"]', "env-var prefix", True),
        ("code/hpc/slurm/new.slurm", "conda activate disrnn-gpu", "conda env", True),
        # must NOT trip:
        ("code/x.py", "from models import DisrnnTrainer", None, False),
        ("code/x.py", "import disentangled_rnns", None, False),
        ("code/config/config.yaml", "entity: AIND-disRNN", None, False),
        ("code/config/config.yaml", "${oc.env:BFM_X,${oc.env:DISRNN_X,/results}}", None, False),
        ("code/hpc/slurm/a.slurm", "conda activate dynamic-foraging-bfm-cpu 2>/dev/null || conda activate disrnn-cpu", None, False),
        ("studies/01/notes.md", "conda activate disrnn-cpu", None, False),  # skipped path
        # The rule's own definition quotes what it forbids. Scanning it made the
        # checker fail on itself the first time this ran in CI, because locally
        # the files were still untracked and `git ls-files` never saw them.
        ("scripts/check_project_identity.py", 'r"aind-disrnn-(?:dispatcher|wrapper)"', None, False),
        ("scripts/project_identity_allowlist.toml", "pattern = 'aind_disrnn_utils'", None, False),
    ]
    failures = 0
    for path, line, _kind, should_flag in cases:
        if path.startswith(SKIP_PREFIXES):
            flagged = False
        else:
            flagged = any(
                p.search(line) and not allowed(path, line, rules)
                for p in REJECT.values()
            )
        status = "ok " if flagged == should_flag else "FAIL"
        if flagged != should_flag:
            failures += 1
        print(f"  [{status}] flag={flagged!s:<5} expect={should_flag!s:<5} {path}: {line[:62]}")
    print(f"\nself-test: {len(cases) - failures}/{len(cases)} cases correct")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="verify the checker flags reintroduced tokens and spares legitimate ones")
    args = ap.parse_args()

    rules = load_allowlist()
    if args.self_test:
        return self_test(rules)

    findings = scan(rules)
    if not findings:
        print(f"project identity: clean ({len(tracked_files())} tracked files, "
              f"{len(rules)} allowlist entries)")
        return 0

    print(f"project-identity tokens reintroduced in {len(findings)} place(s):\n")
    for rel, lineno, kind, line in findings:
        print(f"  {rel}:{lineno}  [{kind}]")
        print(f"      {line}")
    print(
        "\nThese renamed to dynamic-foraging-bfm (ADR-0007). If a hit is deliberate, "
        f"add it to {ALLOWLIST.name} with a reason."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
