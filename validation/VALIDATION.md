# Local validation receipt

Validated on 2026-08-10 with Python 3.12, MLX 0.31.2, MLX-LM 0.31.3, and
NumPy 2.2.5.

From the artifact root, all four documented commands exited 0:

```sh
PYTHONDONTWRITEBYTECODE=1 python src/evaluate_trajectory.py data/historical_curves.csv
PYTHONDONTWRITEBYTECODE=1 python src/evaluate_freeze_thaw.py data/freeze_thaw_discovery.csv
PYTHONDONTWRITEBYTECODE=1 python src/diagnose_failure.py data/historical_curves.csv
PYTHONDONTWRITEBYTECODE=1 python tests/test_continuation.py
```

Observed analysis: `BOUNDED_NEGATIVE`; discovery final/test agreement 5/5,
stress final/test agreement 2/3, and 25% update savings for every valid row.

Observed conservative freeze-thaw analysis: `BOUNDED_NEGATIVE`; three correct
decisions at label 480, two abstentions, 5,472 cohort updates, 1,094.4 mean
updates, 5.0% savings, and no stress rows in the input.

Observed post-result diagnosis:
`POST_HOC_DESCRIPTIVE_NOT_A_SELECTOR`; all eight arms favored token at label
288, all three eventual example winners triggered, and the four triggers
included the sole false trigger 20260819. A one-of-three fallback has mean cost
960 updates and only 16.7% savings.

Observed continuation smoke: parameter delta 0, optimizer delta 0, MLX RNG
delta 0, NumPy RNG equal. The expanded test also rejected a manifest missing
the cursor hash and rejected a wrong cursor schema even when the cursor file
was rehashed. All Python files parsed successfully. The packaged
eight-row CSV was mechanically cross-checked at labels 96, 192, 288, 384, 480,
and 576 against the canonical curve inventory, with all values and final
winners matching. All local Markdown links resolve.

Source evidence SHA-256:

- trajectory protocol:
  `c975f74e1ab17275f6d5a9fe5023df345b3c915be57bf7ec04efc41ca8f11b5e`
- trajectory discovery:
  `41e9ae7758a470922a104b812744266d01a19b254d7a5b456ff8d4f648faba71`
- one-shot trajectory stress:
  `75097f07b7f5c31d1f7359da5b6111e24267a581882b40924f4314a74e9e84c6`
- Luna resume result:
  `5bf8c07fdc645c3b74009a570fe3f9a1e4397c8919ea957ed7130abc3cd44fee`
- exact-environment continuation smoke result:
  `d4fffac4aaab0cd33d1f4c8d4cde80fc6edb7a383577addedd74faf8593ec9f5`
- hardened continuation sidecar:
  `3d5cd2b313e0ba4b7492d4e429bcb9342ad90f76a141bb4ba80f2460a9cb84d4`
- ES-006 final pre-outcome seal:
  `2f19dd6196befe2f0d221f115f816948b255d19ccfc82fab0269af6267fb46db`
- ES-006 immutable result:
  `4922fdeef3fd76af84d1ab33b50dcbeb0d8bf5de7706690b77466c992dfcff11`
- Luna post-result diagnostic:
  `75c3931d02be95f3fb370285197358037b47abafd7ca8130dcfa3e0b9f9f1474`
- packaged six-checkpoint CSV:
  `6e24ad82e0b7b32a964e5ff96b519a1d61ed5d7383956e8601b32147d12481b8`
- ES-005 frozen protocol:
  `822a38be616fdcb6f94647bfc9432cb6a80d189759eeecc09975a1467e7f21c0`
- ES-005 discovery-only input and packaged copy:
  `937807959c0ee8de8f1a7306d42191d09b673bbb38a1b079fa485cbdf7b78c65`
- ES-005 analysis source:
  `b30eb6ef0b8b01513c60785b28b3a5398c05c602f0ca78441749bf1211dc3648`
- packaged freeze-thaw evaluator:
  `420085f6c6eabbd00451363eb70889bf7ddb4c2900114cf13950f14c0e054bfe`
- ES-005 result JSON:
  `28b2cb86fc31103cb143d9ece6c9d2880a9a3cabb527ee166a30f2e6f8090a10`
- ES-005 midpoint feasibility audit source:
  `719815b26dc15c2994e7ba5f8dc173a03f556b7763fc2d007cc7a4f25c7de4eb`
- ES-005 midpoint exact-branch smoke source:
  `f1595cbdb371d44fa1f3a77b7078301ed15f0845181a321a53285d6fd62bbe0c`

The ES-006 post-result inspection found all five execution receipts valid,
every comparator validity gate true, a fresh resume process, and exact state,
trace, adapter, generation, teacher-forced, and evaluator equality. Maximum
numerical deltas were zero. This is bounded Qwen continuation support, not a
positive early-selection result.

ES-007 terminal inspection additionally found all 29 sealed stages `VALID`, no
hard-invalid state, no next eligible stage, and no loaded model process. The
primary lock predates every test receipt and hashes to
`c4825fddbd59bf858823a34f552ee052a224113a02db0306184c52b579774db6`;
terminal `RESULT.json` hashes to
`5c40c3a1cb64739de3c5a4479468f1d6b04b6dc77a5d4dc22e92c088e81dbc84`.

After packaging ES-007, all four documented lightweight commands were rerun
and exited 0; all Python files compiled, `evidence/result.json` parsed, and all
local Markdown links resolved. The artifact scan found no absolute user paths,
private-key markers, nested `.git` directory, or files over 10 MB. No remote
validation or publication was attempted.
