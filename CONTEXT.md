# aind-disrnn-dispatcher

Control plane for training and evaluating sequence models (GRU, disRNN) and cognitive-model
baselines on rodent dynamic-foraging behaviour. This glossary fixes the vocabulary shared by
the dispatcher, the wrapper runtime, and the hierarchical-Bayesian baseline.

## Language

### Hierarchy levels

**Population hyperparameter**:
A parameter of the cohort-wide distribution from which subject-level hyperparameters are drawn.
The top level of the three-level hierarchy; what a held-out subject inherits before seeing any of
its own data.
_Avoid_: group-level, global prior

**Subject-level hyperparameter**:
The mean and spread governing one subject's session-level parameters. Written `mu_p` and `sigma`.
_Avoid_: animal-level, mouse-level

**Session-level parameter**:
The cognitive-model parameter values realised for a single behavioural session, drawn from that
subject's distribution.
_Avoid_: trial-level parameter, local parameter

**Subject**:
The experimental animal, and the unit of the middle hierarchy level. Canonical in both code and
prose, matching the existing identifiers (`subject_id`, `test_subject_ids`, subject embedding)
and `aind-dynamic-foraging-models`. Note that the published papers say "mouse-level" for what
this glossary calls subject-level.
_Avoid_: mouse, animal

### Inference and evaluation

**Condition**:
To infer a subject's parameters given its context sessions, holding population hyperparameters
fixed at their training-set posterior. The Bayesian counterpart of the neural models'
gradient fine-tuning, and deliberately not called that: "fine-tune" invites updating the
population level on held-out data, which leaks.
_Avoid_: fine-tune, adapt, refit

**Context session**:
A session of a held-out subject that is used to infer that subject's parameters. Always disjoint
from the sessions being scored.
_Avoid_: support set, adaptation session

**Held-out subject**:
A subject absent from training entirely. Distinct from a held-out session.
_Avoid_: test animal

**Held-out session**:
A session excluded from fitting, belonging to a subject that may or may not itself be held out.
_Avoid_: test session, validation session

**Zero-shot**:
Scoring a held-out subject with no context sessions, drawing its parameters from the population
level alone. Isolates the quality of the population prior.

**Few-shot (k)**:
Scoring a held-out subject after conditioning on k context sessions. Isolates adaptation.

**Partial pooling**:
Estimating each session's parameters under a subject-level prior, so sessions borrow strength
from each other rather than being fit independently or forced to share one value.
_Avoid_: shrinkage, regularisation

**Parameter recovery**:
Fitting simulated data whose true parameters are known, and checking they are returned.
Distinct from model recovery.
_Avoid_: identifiability check

**Model recovery**:
Generating from one model family, fitting several, and checking the generating family wins.

### Model families

**Forager**:
A parameterised generative agent for the foraging task, in the sense used by
`aind-dynamic-foraging-models` (`ForagerQLearning`, `ForagerCompareThreshold`, ...).
_Avoid_: agent, policy

**HB-\<PresetAlias\>**:
The hierarchical-Bayesian counterpart of a named forager preset, using the preset alias
verbatim: `HB-Hattori2019`, `HB-Bari2019`, `HB-CompareToThreshold`. Aliases match
`ForagerCollection.FORAGER_PRESETS` keys so results join against existing MLE fits.
_Avoid_: naming by parameter count (e.g. "5params")

**Two-stage empirical Bayes**:
Fitting each subject independently, then fitting a population distribution to the resulting
subject-level posterior draws. The cheaper alternative to a single joint three-level fit.

### Model parameters

**Forgetting rate**:
The per-trial decay applied to the unchosen option's value, in the sense of
`aind-dynamic-foraging-models`: `forget_rate_unchosen = 0` means no forgetting. The reference
Stan implementation's `aF` is the opposite quantity — a retention factor where `aF = 1` means
no forgetting — and equals `1 - forget_rate_unchosen`.
_Avoid_: retention, decay factor, aF
