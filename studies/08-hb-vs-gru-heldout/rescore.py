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
POPULATION_SITES = (
    "population_mean", "population_scale", "log_sigma_mean", "log_sigma_spread",
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
        fit_adaptation, pointwise_log_predictive_density,
        posterior_predictive_choice_prob,
    )
    from model_trainers.hb_trainer import _extract_subject_sessions, _normalized_likelihood
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
    total_log_lik, total_trials, per_subject = 0.0, 0, {}

    for subject, sessions in choices.items():
        ids = session_ids[subject]
        if len(ids) < 2:
            continue
        train_ids, eval_ids = compute_train_eval_session_ids(ids, args.eval_every_n)
        index_of = {sid: i for i, sid in enumerate(ids)}
        context_idx = [index_of[s] for s in train_ids]
        score_idx = [index_of[s] for s in eval_ids]

        key_fit, key_draw, rng_key = jax.random.split(rng_key, 3)
        samples = fit_adaptation(
            np.stack([sessions[i] for i in context_idx]),
            np.stack([rewards[subject][i] for i in context_idx]),
            population, rng_key=key_fit,
            num_warmup=args.num_warmup, num_samples=args.num_samples,
            beta_max=args.beta_max,
        )
        subject_log_lik, subject_trials = 0.0, 0
        for i in score_idx:
            prob = posterior_predictive_choice_prob(
                samples, sessions[i], rewards[subject][i],
                rng_key=key_draw, beta_max=args.beta_max,
            )
            log_lik, n = pointwise_log_predictive_density(prob, sessions[i])
            subject_log_lik += log_lik
            subject_trials += n
        per_subject[str(subject)] = {
            "likelihood": _normalized_likelihood(subject_log_lik, subject_trials),
            "n_context": len(context_idx), "n_scored": len(score_idx),
            "n_trials": subject_trials,
        }
        total_log_lik += subject_log_lik
        total_trials += subject_trials

    matched = _normalized_likelihood(total_log_lik, total_trials)
    out = {
        "source": args.results,
        "estimator": saved.get("estimator"),
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
