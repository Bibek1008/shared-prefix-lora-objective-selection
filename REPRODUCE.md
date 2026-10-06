# Reproduce

## Requirements

- macOS with Apple Silicon for MLX
- Python 3.12
- `mlx==0.31.2`
- `mlx-lm==0.31.3`
- NumPy

No model weights are required for these lightweight checks.

## Historical rule analysis

From this directory:

```sh
python src/evaluate_trajectory.py data/historical_curves.csv
```

Expected classification: `BOUNDED_NEGATIVE`, with discovery agreement 5/5 and
stress agreement 2/3.

## Discovery-only conservative freeze-thaw audit

```sh
python src/evaluate_freeze_thaw.py data/freeze_thaw_discovery.csv
```

Expected classification: `BOUNDED_NEGATIVE`, with three correct decisions at
label 480, two abstentions, 5,472 cohort updates, and 5.0% savings. The input
contains no stress rows.

## Post-result failure diagnosis

```sh
python src/diagnose_failure.py data/historical_curves.csv
```

Expected status: `POST_HOC_DESCRIPTIVE_NOT_A_SELECTOR`, with four direction-
reversal triggers and seed 20260819 as the sole false trigger. This command
only derives facts about the frozen rule; it does not fit or score an
alternative.

## Exact-continuation smoke

```sh
python tests/test_continuation.py
```

Expected deltas are all zero, with equal NumPy state and batch order. The test
creates and removes its own temporary checkpoint under `validation/`.
It also requires rejection of a manifest missing any of the five state hashes
and rejection of a cursor whose schema was altered and rehashed.

These checks reproduce the packaged analyses and tiny-model engineering test;
they do not rerun historical Qwen training.

## Inspect the bounded Qwen equivalence result

The completed Qwen run is preserved in the initiative project rather than
duplicating model artifacts in this repository package. From the initiative
root, verify the immutable comparator output with:

```sh
shasum -a 256 projects/qwen-exact-resume-equivalence/RESULT.json
python3 projects/qwen-exact-resume-equivalence/inspect_stage_progress.py
```

Expected SHA-256:
`4922fdeef3fd76af84d1ab33b50dcbeb0d8bf5de7706690b77466c992dfcff11`.
The progress inspector should report all five receipts `VALID`, the result
present, and no next eligible stage. The exact sealed commands and environment
fingerprints are in that project's `PROTOCOL.md` and `PRECOMPUTE-v4.json`.
Do not rerun the sequence merely to inspect the recorded claim; new execution
is a separate replication, not part of this frozen result.

## Inspect the prospective shared-prefix result

From the initiative root:

```sh
shasum -a 256 projects/shared-prefix-prospective-selection/PRIMARY-LOCK.json \
  projects/shared-prefix-prospective-selection/RESULT.json
python3 projects/shared-prefix-prospective-selection/inspect_progress.py
```

Expected hashes are, respectively,
`c4825fddbd59bf858823a34f552ee052a224113a02db0306184c52b579774db6`
and `5c40c3a1cb64739de3c5a4479468f1d6b04b6dc77a5d4dc22e92c088e81dbc84`.
The inspector should report all 29 stages `VALID`, no hard-invalid state, no
loaded Qwen model, and no next eligible stage. `PRIMARY-LOCK.json` is the
test-isolated primary result; `RESULT.json` contains only the locked diagnostic
extension and must not be used to retune the selector.
