# Diagrams

Architecture diagrams for the stack as a whole. These describe how the repos fit together
and stay true regardless of which studies exist, which is why they live here rather than in
any one `studies/<study>/figures/` — those hold a study's own results.

Each diagram is a typed JSON source plus generated HTML. Regenerate after editing the source:

    node ~/.claude/skills/archify/bin/archify.mjs deliver architecture \
      docs/diagrams/hb-stack.architecture.json docs/diagrams/hb-stack.html --quality showcase

GitHub renders neither the HTML nor inline SVG, so read them through
[raw.githack](https://raw.githack.com/AllenNeuralDynamics/aind-disrnn-dispatcher/main/docs/diagrams/hb-stack.html).

| diagram | subject |
|---|---|
| `hb-stack` | The hierarchical Bayes baseline across dispatcher, wrapper and models: which repo owns which part of the fit, and where the held-out boundary sits |

**`hb-stack.png` is stale** — it shows the pre-Hydra layout with a `run_hb.py` entrypoint
that no longer exists. Regenerating it needs Chrome, which the HPC host does not have;
`visual-check` reports `skipped` there. The HTML is current.
