# Frozen decision protocol

Historical discovery seeds were 20260812–20260816. Historical stress seeds
were 20260817–20260819. Test NLL and exact match were endpoint audits only; they
did not select a rule.

"Materially less" meant at least 25% fewer optimizer updates than two complete
576-update arms. Exact continuation after a decision at label `c` costs
`576+c` updates. A tie, missing value, or invalid receipt falls back to 1,152
updates and fails the per-seed materiality gate.

The gap-crossover rule in `RESULTS.md` was selected from a predeclared family
of lower-loss-at-96, lower-loss-at-192, lower-loss-at-288, and gap-crossover-at-
288 rules before its stress phase. Family choice used final validation only on
the five discovery seeds; test and stress records were closed. It earned stress
by matching all 5/5 discovery final-validation winners without a fallback,
remaining selected in every leave-one-seed-out fold, and saving 25% on every
seed. Stress required 3/3 agreement
with final validation, lower test NLL, and higher strict exact match, with at
least 25% savings and no fallback. Any failed conjunct was terminal and
forbade fresh Qwen confirmation or historical retuning.

The source trainer logs validation before executing the update named in the
label. Thus label 288 ranks the state after 287 updates, but both arms execute
and checkpoint update 288 before a resumable decision; compute accounting uses
288.

## Midpoint-prefix feasibility gate

The subsequent ES-005 candidate fixed a 50/50 hybrid prefix through update
288, cloned full continuation state, ran token and example suffixes through
576, and selected between the two final-validation endpoints. Its accounting
is exactly 864 updates. Before any prospective protocol, it required observed
historical shared-prefix branch endpoints sufficient to score all-correct
discovery-only LOOCV and calibrate path-dependence uncertainty. The available
full-hybrid-from-initialization trajectories are a different intervention and
were forbidden as branch proxies. Because no matching endpoint exists, this
gate is not estimable and stops before Qwen.

## Independent conservative continuation

After the trajectory branch was terminal, ES-005 froze one different method
without reading stress seeds. In each discovery-only LOOCV fold and rung
`c in {192,288,384,480}`, it formed
`R_c = max |g_576-g_c|` over the other four seeds. It selected token only when
`g_c-R_c>0`, selected example only when `g_c+R_c<0`, and otherwise resumed both
arms. A zero-straddling interval at 480 fell back to two complete arms.

The cohort gate required every held-out winner to be selected correctly and
total cost no greater than `5*864=4,320` updates. An abstention was charged
1,152 updates and failed the all-row selection gate. This deterministic
max-drift envelope was explicitly not called a calibrated confidence
interval. Failure of either accuracy or budget stopped the method before
stress scoring, fresh-seed freezing, or Qwen.
