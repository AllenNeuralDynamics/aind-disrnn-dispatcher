"""Score held-out subjects from an already-fitted population, without refitting.

The expensive part of a run is fitting the population; held-out scoring is cheap by
comparison. When a new conditioning rung is added after a fit has completed, this replays
the scoring against the saved population rather than repeating hours of NUTS.

    python rescore.py --results two_stage_d30.json --output two_stage_d30_matched.json
"""

import argparse
import json
import logging
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CURRICULA = ["Coupled Baiting", "Uncoupled Baiting", "Uncoupled Without Baiting"]
LOADER_KWARGS = dict(
    curricula=CURRICULA, min_sessions=10, heldout_every_n=5,
    mature_only=False, snapshot="20260603",
)


def main():
    """Load a saved population and score the matched-conditioning rung."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True,
                        help="JSON from run_hb.py containing a fitted 'population'")
    parser.add_argument("--subject-ratio", type=float, default=0.049)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every-n", type=int, default=2)
    parser.add_argument("--num-warmup", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--beta-max", type=float, default=10.0)
    parser.add_argument("--wrapper", type=str,
                        default="/home/han.hou/code/aind-disrnn-wrapper/code")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    sys.path.insert(0, args.wrapper)
    import jax

    from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
        POPULATION_SITES, batched_heldout_log_lik, fit_adaptation_batched,
    )
    from model_trainers.hb_trainer import (
        _extract_subject_sessions, _flatten_for_scoring, _normalized_likelihood,
        _pad_context,
    )
    from utils.load_mice_database import load_mice_from_database
    from utils.multisubject import compute_train_eval_session_ids

    with open(args.results) as handle:
        saved = json.load(handle)
    if "population" not in saved:
        raise SystemExit(f"{args.results} has no fitted population; a refit is needed.")
    population = {name: np.asarray(saved["population"][name]) for name in POPULATION_SITES}
    logging.info("loaded population from %s", args.results)

    kwargs = dict(LOADER_KWARGS, seed=args.seed, subject_ratio=args.subject_ratio)
    heldout_df, heldout_ids = load_mice_from_database(split="heldout", **kwargs)
    choices, rewards, session_ids = _extract_subject_sessions(heldout_df)
    logging.info("held-out: %d subjects / %d trials", len(heldout_ids), len(heldout_df))

    started = time.time()
    rng_key = jax.random.PRNGKey(args.seed)

    # Sessions are ragged -- each holds only its own valid trials -- so they are padded
    # and masked, never stacked. Adaptation and scoring both run batched, matching the
    # trainer; fitting subjects one at a time here would defeat the point of the script.
    subjects, context_indices, score_indices = [], [], []
    for subject, ids in session_ids.items():
        try:
            train_ids, eval_ids = compute_train_eval_session_ids(ids, args.eval_every_n)
        except ValueError:
            logging.warning(
                "subject %s: %d sessions cannot split at eval_every_n=%d; skipping",
                subject, len(ids), args.eval_every_n,
            )
            continue
        index_of = {sid: i for i, sid in enumerate(ids)}
        subjects.append(subject)
        context_indices.append([index_of[s] for s in train_ids])
        score_indices.append([index_of[s] for s in eval_ids])

    if not subjects:
        raise SystemExit("No held-out subject could be split; check --eval-every-n.")

    context_c, context_r, ctx_mask, ctx_valid = _pad_context(
        subjects, choices, rewards, context_indices
    )
    key_fit, key_draw = jax.random.split(rng_key)
    samples = fit_adaptation_batched(
        context_c, context_r, population, rng_key=key_fit,
        session_mask=ctx_mask, valid_mask=ctx_valid,
        num_warmup=args.num_warmup, num_samples=args.num_samples,
        beta_max=args.beta_max,
    )

    flat = _flatten_for_scoring(subjects, choices, rewards, score_indices)
    session_log_lik, session_trials = batched_heldout_log_lik(
        samples, flat["subject_indices"], flat["choices"], flat["rewards"],
        valid_mask=flat["valid_mask"], rng_key=key_draw, beta_max=args.beta_max,
    )

    per_subject = {}
    for position, subject in enumerate(subjects):
        rows = flat["rows_by_subject"][position]
        per_subject[str(subject)] = {
            "likelihood": _normalized_likelihood(
                float(np.sum(session_log_lik[rows])), int(np.sum(session_trials[rows]))
            ),
            "n_context": len(context_indices[position]),
            "n_scored": len(score_indices[position]),
            "n_trials": int(np.sum(session_trials[rows])),
        }
    total_log_lik = float(np.sum(session_log_lik))
    total_trials = int(np.sum(session_trials))
    matched = _normalized_likelihood(total_log_lik, total_trials)
    out = {
        "source": args.results,
        "estimator": saved.get("estimator") or saved.get("_meta", {}).get("estimator"),
        "eval_every_n": args.eval_every_n,
        "matched_likelihood": matched,
        "n_heldout_subjects": len(per_subject),
        "n_trials": total_trials,
        "per_subject": per_subject,
        "rescore_seconds": time.time() - started,
    }
    with open(args.output, "w") as handle:
        json.dump(out, handle, indent=2)

    print(f"\nmatched-conditioning likelihood: {matched:.5f}")
    print(f"  {len(per_subject)} held-out subjects, {total_trials} trials scored")
    print("  per-mouse MLE Hattori baseline: 0.71267   GRU at D~30: 0.7248")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
