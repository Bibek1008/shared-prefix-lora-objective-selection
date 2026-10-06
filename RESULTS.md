# Results

## Headline result

One prospectively frozen shared-prefix selector matched the full-control
final-validation winner on 3/3 fresh seeds while meeting the frozen 25%
training-update reduction. This is narrow bounded support. It does not support
a broad test-winner claim, because locked post-primary diagnostics found
validation/test reversals.

The historical baseline trains token-mean and example-mean arms for 576 updates
each: 1,152 updates total.

### Prospective shared-prefix selector

ES-007 trained a fixed arithmetic-mean loss prefix for 288 updates, cloned the
complete continuation state, and trained token-mean and example-mean suffixes
for 288 updates each. The selector therefore used 864 updates per seed, exactly
25% less than two full 576-update arms. Independent full arms were run as
controls and charged separately as experimental validation, not hidden from
the accounting.

Before test access, the immutable primary lock found 3/3 agreement between the
shared-prefix branch winner and the independent-control validation winner on
fresh seeds 20260882–20260884. Every receipt and trace gate passed, including
exact prefix/suffix batch-order checks. Classification was
`SUPPORTED_BOUNDED_SHARED_PREFIX_SELECTION`; the exact 95% Clopper-Pearson
interval is 0.2924–1.0.

The result is deliberately narrow. All three control validation winners were
token, so the experiment did not prospectively demonstrate selection across
both winner classes. Tests were opened only after the primary lock and cannot
change its classification, but they falsify a broader interpretation. Seed
20260883's control validation winner was token while both control test NLL and
strict exact match favored example. Branch-level validation/test reversals also
occurred. The terminal scorer therefore records
`broad_selector_claim_blocked=true`.

### Current-validation rank

On discovery seeds 20260812–20260816, choosing the arm with lower validation
loss at labels 96, 192, or 288 matched the final validation and both test
winners on 3/5 seeds. Label 480 reached 4/5, but cost 1,056 updates and saved
only 8.33%. The miss was seed 20260813.

### Frozen trajectory rule

The primary trajectory rule defined the relative loss gap
`g_k = example_validation_k - token_validation_k` at labels 96, 192, and 288.
If `(g_192-g_96)(g_288-g_192) < 0`, it selected example; otherwise it selected
the lower-loss arm at 288. Invalid or tied evidence fell back to both complete
arms.

It was selected from a predeclared four-rule family and matched 5/5 discovery
winners. Every leave-one-seed-out fold selected the same rule on the other four
seeds and correctly predicted the held-out seed. The exact-continuation
schedule costs 864 updates, saving 25%. Applied once and unchanged to stress
seeds 20260817–20260819, it matched 2/3. On seed 20260819 the relative gap
changed direction and triggered example, but token won final validation, test
NLL, and strict exact match. This counterexample makes the result bounded
negative.

### Midpoint-prefix branch feasibility

A structurally different candidate trains one fixed 50/50 token/example
prefix through update 288, clones complete continuation state, then gives the
token and example objectives 288 suffix updates each. The deployed arithmetic
is exactly `288+288+288=864` updates, and a tiny trainer smoke obtained exact
branch starts before the objectives diverged.

This does not establish selection quality. Historical data contain three full
hybrid-from-initialization runs, not token/example endpoints branched from a
hybrid checkpoint. Full-hybrid final validation loss is below both endpoint
arms on seed 20260815 (0.034 versus 0.126 and 0.691) and above both on seed
20260816 (0.617 versus 0.053 and 0.528), despite lying between them at update
288. It is therefore not a valid behavior-space proxy. With zero historical
shared-prefix branch endpoints, LOOCV accuracy and calibrated path-dependence
uncertainty are not estimable. The candidate is mechanically feasible but did
not earn Qwen.

### Conservative freeze-thaw envelope

A separate continuation used only discovery seeds 20260812–20260816 and never
opened the three stress rows. At rungs 192, 288, 384, and 480, each held-out
seed's gap was compared with the largest absolute rung-to-endpoint movement
among the other four discovery seeds. The rule selected only when this
worst-case envelope excluded zero; otherwise it completed both arms.

It selected token correctly on seeds 20260812, 20260815, and 20260816, all at
update 480, and abstained on both example-winning seeds. Honest fallback
accounting cost 5,472 of 5,760 updates: 1,094.4 per seed and only 5.0% savings.
Selected-only accuracy was 3/3, but the Wilson 95% interval was 0.439–1.000 and
the four-row calibration folds do not create a calibrated confidence claim.
The rule therefore failed both the all-five decision gate and the 25% budget
gate before prospective compute.

### Post-result diagnosis

All eight seeds still had lower token validation loss at label 288. The rule's
direction-reversal trigger fired on all three later example winners, but also
on seed 20260819, which never crossed. Final-validation and both test winner
labels agree on every row, so the failure belongs to the early trajectory
proxy rather than endpoint disagreement. [The diagnosis](DIAGNOSIS.md) is
descriptive only and does not introduce a replacement rule.

## Exact-continuation engineering

MLX-LM 0.31.3's adapter resume restores weights but not AdamW moments/step,
MLX or NumPy RNG state, or the minibatch position. `src/continuation_state.py`
stores those states in safetensors plus JSON, verifies SHA-256, and restores
MLX 0.31.2's live RNG list in place. The packaged test obtains zero maximum
absolute parameter, optimizer, and MLX-RNG differences versus uninterrupted
training, with equal NumPy state and batch order.

A pre-Qwen adversarial audit found two loader-integrity defects in the earlier
sidecar: a manifest could omit one required state hash, and a cursor with the
wrong schema could be accepted if its altered file hash was supplied. The
packaged loader now requires exactly the model, optimizer, MLX RNG, NumPy RNG,
and cursor entries; exact cursor schema plus a nonnegative integer update; and
the exact typed NumPy payload. Both prior fail-open probes now fail closed,
while exact tiny-model continuation remains zero-delta.

ES-006 later tested the same continuation mechanism on Qwen. Its frozen
12-update token-mean Qwen3-0.6B-3bit run compared an uninterrupted control with
a six-update checkpoint and fresh-process resume. All receipts and validity
gates passed. Trainable model, adapter, optimizer arrays, MLX and NumPy RNG,
cursor, batch and loss traces, generations, teacher-forced records, and
evaluator metric sections were exactly equal; all maximum numerical deltas
were zero. The optimizer container-file hashes differed, but decoded keys and
arrays were exactly equal.

This is `SUPPORTED_BOUNDED_EXACT_EQUIVALENCE` for one seed, model,
configuration, short run, dataset order, device, and local MLX environment. It
validates implementation plumbing only. It does not repair the failed
selection rules, prove 25% savings, or establish portability to other stacks.
The earlier selection branches remain bounded negative. ES-007 adds bounded
support for one validation-defined shared-prefix selector, while leaving the
general or test-defined selection claim unsupported.

## Scope

The historical task is synthetic associative recall with held-out prompt
templates but shared facts, Qwen3-0.6B 3-bit, MLX/MLX-LM, one LoRA setup, and
eight relevant paired seeds. The result does not show that early selection is
generally impossible, nor that exact continuation generalizes beyond the one
frozen Qwen configuration.
