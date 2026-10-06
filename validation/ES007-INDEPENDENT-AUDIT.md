# ES-007 independent final audit

**Verdict: PASS**

A separate Luna owner recomputed the decisive checks from the frozen protocol,
grant, 29 stage records, receipts, primary lock, and terminal result. It did not
load Qwen or modify the experiment evidence.

- All 29 stages were valid: 15 training receipts, one primary lock, 12 held-out
  test receipts, and one terminal result.
- All 15 training receipts predated the primary lock; all 12 test receipts
  followed it.
- `PRIMARY-LOCK.json` hashed to
  `c4825fddbd59bf858823a34f552ee052a224113a02db0306184c52b579774db6`.
- Terminal `RESULT.json` hashed to
  `5c40c3a1cb64739de3c5a4479468f1d6b04b6dc77a5d4dc22e92c088e81dbc84`.
- The selector used 864 rather than 1,152 deployed updates per seed, exactly
  25% savings.
- Validation agreement was 3/3.
- Held-out reversals occurred and correctly block any broad test-winner or
  general selector claim.
- No model process remained active at audit time.

The PASS applies only to the recorded execution, evidence integrity, and
bounded validation-selection claim. It is not independent replication on a
new model, dataset, or seed block.
