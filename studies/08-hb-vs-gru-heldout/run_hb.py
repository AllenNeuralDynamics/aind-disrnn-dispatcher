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
        },
        seed=args.seed,
    )

    started = time.time()
    output = trainer.fit(bundle)
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


if __name__ == "__main__":
    main()
