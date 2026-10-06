# Why the frozen trajectory rule failed

This is a post-result description, not a replacement selection rule. The
bounded-negative decision was locked before this analysis, and no threshold or
fallback was changed after the stress outcomes were opened.

Every one of the eight relevant seeds still favored token mean on validation
at label 288. The rule used a reversal in the direction of the relative gap
`example_loss - token_loss` as a warning of a later crossover. That warning
correctly identified the three observed eventual example winners, but also
fired on stress seed 20260819:

| Seed | Split | Gap 96 | Gap 192 | Gap 288 | Gap 576 | Trigger | Final winner |
| ---: | --- | ---: | ---: | ---: | ---: | --- | --- |
| 20260813 | discovery | 0.204 | 0.198 | 0.238 | -0.133 | yes | example |
| 20260814 | discovery | 0.037 | 0.217 | 0.174 | -0.320 | yes | example |
| 20260817 | stress | 0.008 | 0.265 | 0.197 | -0.472 | yes | example |
| 20260819 | stress | 0.135 | 0.269 | 0.163 | 0.173 | yes | token |

Seed 20260819 narrowed the token advantage before the decision, but relative
post-288 improvement did not produce a crossing. Direction reversal was thus
not sufficient evidence for the mechanism the rule assumed. Its label-288 gap
was also close to successful flagged seed 20260814 (0.163 versus 0.174), so the
decision-boundary magnitude alone does not explain the divergence.

The full seed-20260819 gap path was
`0.135, 0.269, 0.163, 0.211, 0.098, 0.173` at labels 96, 192, 288, 384, 480,
and 576. Its rebound at 384 is only a hindsight warning with many false alarms,
and its 480 gap is almost the same as successful seed 20260813 (0.098 versus
0.100). A fallback on one of three stress seeds would raise mean cost from 864
to 960 updates and reduce savings from 25% to 16.7%, below the materiality
gate.

Final-validation, test-NLL, and strict-exact winner labels agree on all eight
rows. The observed failure is therefore in the early proxy, not a conflict
between validation and test endpoints.

The 5/5 leave-one-seed-out result tested selection stability within the frozen
four-rule family. It was not independent rule-design validation: the
gap-crossover form itself came from the five discovery curves. With only five
design seeds, that internal check could not rule out the stress counterexample.

This evidence does not establish why post-288 dynamics diverged or prove that
all richer early signals fail. Plausible unresolved sources include stochastic
late-training variation, omitted shape between logged checkpoints, and
seed-by-normalization interactions. Testing a new rule on the now-open stress
set would be reward hacking, so the initiative stops here.
