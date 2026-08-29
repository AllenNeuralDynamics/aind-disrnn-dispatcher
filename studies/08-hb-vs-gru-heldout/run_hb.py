"""Fit HB-Hattori2019 on study 01's cohort and score held-out subjects.

Reuses study 01's subject selection exactly (`data.subject_ratio` against the ~614 pool,
same snapshot, filters and seed), so the held-out likelihood is directly comparable with the
GRU numbers already in `studies/01-gru-scaling-law/scaling_results.csv` rather than being a
number that compares to nothing.

Both estimators are available; run each as its own job so one failing does not lose the
other.

    python run_hb.py --estimator two_stage --subject-ratio 0.049
"""

import argparse
import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Study 01's data config: code/config/data/mice_snapshot_scaling.yaml
CURRICULA = ["Coupled Baiting", "Uncoupled Baiting", "Uncoupled Without Baiting"]
LOADER_KWARGS = dict(
    curricula=CURRICULA,
    min_sessions=10,
    heldout_every_n=5,
    mature_only=False,
    snapshot="20260603",
)


def _checkpoint(path, payload):
    """Write a partial result so a wall-clock kill does not discard finished work."""
    import numpy as np

    def _plain(value):
        """Make numpy arrays JSON-serialisable."""
        return np.asarray(value).tolist() if hasattr(value, "shape") else value

    with open(path, "w") as handle:
        json.dump({k: _plain(v) for k, v in payload.items()}, handle, indent=2, default=str)


def main():
    """Load the cohort, fit, score, and write a results JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", choices=("two_stage", "one_stage"),
                        default="two_stage")
    parser.add_argument("--subject-ratio", type=float, default=0.049,
                        help="0.049 -> D~30, 0.163 -> D~100, matching study 01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--wrapper", type=str,
                        default="/home/han.hou/code/aind-disrnn-wrapper/code")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--wandb-project", type=str, default="mice_data_scaling")
    parser.add_argument("--wandb-entity", type=str, default="AIND-disRNN")
    parser.add_argument("--launch-id", type=str, default=None,
                        help="group suffix; defaults to a UTC timestamp")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--note", type=str, default=None,
                        help="why this run exists and what it should show; "
                             "required by the study-conventions provenance scheme")
    parser.add_argument("--artifact-dir", type=str, default=None)
    parser.add_argument(
        "--few-shot-k", type=int, nargs="*", default=[0],
        help="k rungs to score in addition to the matched rung. The held-out cohort is "
             "~153 subjects and each rung costs one adaptation fit per subject, so the "
             "full sweep is expensive; production runs need only the matched rung, which "
             "is the number comparable with the GRU and the MLE baseline.",
    )
    args = parser.parse_args()

    sys.path.insert(0, args.wrapper)
    from base.types import DatasetBundle
    from model_trainers.hb_trainer import HBTrainer
    from utils.load_mice_database import load_mice_from_database

    kwargs = dict(LOADER_KWARGS, seed=args.seed, subject_ratio=args.subject_ratio)
    train_df, train_ids = load_mice_from_database(split="train", **kwargs)
    heldout_df, heldout_ids = load_mice_from_database(split="heldout", **kwargs)
    logging.info(
        "train: %d subjects / %d trials | heldout: %d subjects / %d trials",
        len(train_ids), len(train_df), len(heldout_ids), len(heldout_df),
    )

    # W&B, matching the GRU runs' project, group convention and metric namespace so the
    # results drop straight into studies/01's analyze_scaling.py rather than living apart.
    wandb_run = None
    if not args.no_wandb:
        import wandb

        # Seattle time, per AGENTS.md section 7 and the study-conventions skill.
        os.environ.setdefault("TZ", "America/Los_Angeles")
        time.tzset()
        launch_id = args.launch_id or time.strftime("%Y%m%d-%H%M%S")
        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=f"hb-{args.estimator}@{launch_id}",
            name=f"hb-{args.estimator}-D{len(train_ids)}-s{args.seed}",
            config={
                "model_class": f"HB-Hattori2019-{args.estimator}",
                "seed": args.seed,
                # analyze_scaling.py reads D from this key
                "resolved_subject_ids": [str(s) for s in train_ids],
                "resolved_heldout_subject_ids": [str(s) for s in heldout_ids],
                "data": {
                    "subject_ratio": args.subject_ratio,
                    "snapshot": LOADER_KWARGS["snapshot"],
                    "min_sessions": LOADER_KWARGS["min_sessions"],
                    "heldout_every_n": LOADER_KWARGS["heldout_every_n"],
                    "mature_only": LOADER_KWARGS["mature_only"],
                    "curricula": CURRICULA,
                    "eval_every_n": 2,
                    "seed": args.seed,
                },
                "meta": {
                    "study": "08-hb-vs-gru-heldout",
                    "variant": "one-stage-ladder",
                    "launch_id": launch_id,
                    "label": f"D{len(train_ids)}-s{args.seed}",
                    "note": args.note or "",
                    "estimator": args.estimator,
                    "num_warmup": args.num_warmup,
                    "num_samples": args.num_samples,
                    "num_chains": args.num_chains,
                },
            },
        )

    bundle = DatasetBundle(
        raw=train_df,
        train_set=None,
        eval_set=None,
        metadata={"subject_ids": list(train_ids)},
        extras={"heldout_raw": heldout_df},
    )

    trainer = HBTrainer(
        config={
            "estimator": args.estimator,
            "num_warmup": args.num_warmup,
            "num_samples": args.num_samples,
            "num_chains": args.num_chains,
            "eval_every_n": 2,
            "artifact_dir": args.artifact_dir,
            "few_shot_k": tuple(args.few_shot_k),
        },
        seed=args.seed,
    )

    loggers = {"wandb": wandb_run} if wandb_run is not None else None
    started = time.time()
    # These fits run for hours against a hard wall clock, so the fitted population is
    # written as soon as it exists rather than only after held-out scoring completes.
    trainer.on_population_fitted = lambda pop, info: _checkpoint(
        args.output, {"stage": "population_fitted", "population": pop, **info}
    )
    output = trainer.fit(bundle, loggers=loggers)
    output["wall_seconds"] = time.time() - started
    output["subject_ratio"] = args.subject_ratio
    output["n_train_subjects"] = len(train_ids)
    output["n_heldout_subjects"] = len(heldout_ids)

    with open(args.output, "w") as handle:
        json.dump(output, handle, indent=2, default=str)

    print(f"\nestimator={args.estimator}  D={len(train_ids)}  "
          f"wall={output['wall_seconds']:.0f}s")
    for k, value in sorted(output.get("heldout_likelihood", {}).items()):
        print(f"  k={k}: heldout likelihood {value:.5f}")
    print(f"wrote {args.output}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
