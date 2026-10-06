#!/usr/bin/env python3
"""Tiny exact-continuation test; no model download or Qwen load."""

from __future__ import annotations

import json
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from continuation_state import load_continuation, restore_mx_rng_state, save_continuation


def make_model(seed: int):
    mx.random.seed(seed)
    model = nn.Linear(3, 2)
    mx.eval(model.state)
    return model


def batches():
    return [(
        mx.array(np.asarray([[1 + k, .5, -.25], [.2, -.3, .7]], dtype=np.float32)),
        mx.array(np.asarray([k % 2, (k + 1) % 2], dtype=np.int32)),
    ) for k in range(6)]


def run(model, optimizer, data, start, stop):
    vg = nn.value_and_grad(model, lambda m, x, y: nn.losses.cross_entropy(m(x), y).mean())
    for index in range(start, stop):
        mx.eval(mx.random.normal((2,)))
        np.random.permutation(index + 3)
        value, gradients = vg(model, *data[index])
        optimizer.update(model, gradients)
        mx.eval(model.state, optimizer.state, value)


def delta(left, right):
    a, b = dict(tree_flatten(left)), dict(tree_flatten(right))
    assert set(a) == set(b)
    return max(float(mx.max(mx.abs(a[key] - b[key])).item()) for key in a)


def numpy_equal(left, right):
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def main():
    data = batches()
    np.random.seed(23)
    reference = make_model(101)
    reference_optimizer = optim.AdamW(1e-2, weight_decay=.01)
    run(reference, reference_optimizer, data, 0, 6)
    reference_rng = [mx.array(np.array(value, copy=True)) for value in mx.random.state]
    mx.eval(reference_rng)
    reference_numpy = np.random.get_state()

    np.random.seed(23)
    prefix = make_model(101)
    prefix_optimizer = optim.AdamW(1e-2, weight_decay=.01)
    run(prefix, prefix_optimizer, data, 0, 3)
    validation = ROOT / "validation"
    validation.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=validation) as temporary:
        checkpoint = Path(temporary) / "checkpoint"
        save_continuation(
            checkpoint, model_state=prefix.trainable_parameters(),
            optimizer_state=prefix_optimizer.state, mx_rng_state=mx.random.state,
            numpy_rng_state=np.random.get_state(), next_update=3,
        )
        restored = make_model(999)
        restored_optimizer = optim.AdamW(1e-2, weight_decay=.01)
        state = load_continuation(checkpoint)
        restored.update(state["model_state"])
        restored_optimizer.state = state["optimizer_state"]
        restore_mx_rng_state(state["mx_rng_state"])
        np.random.set_state(state["numpy_rng_state"])
        run(restored, restored_optimizer, data, state["next_update"], 6)

    result = {
        "parameter_delta": delta(reference.parameters(), restored.parameters()),
        "optimizer_delta": delta(reference_optimizer.state, restored_optimizer.state),
        "mx_rng_delta": delta(reference_rng, mx.random.state),
        "numpy_rng_equal": numpy_equal(reference_numpy, np.random.get_state()),
    }
    assert result == {
        "parameter_delta": 0.0, "optimizer_delta": 0.0,
        "mx_rng_delta": 0.0, "numpy_rng_equal": True,
    }, result
    with tempfile.TemporaryDirectory(dir=validation) as temporary:
        root = Path(temporary)
        source = root / "source"
        np.random.seed(23)
        probe = make_model(101)
        probe_optimizer = optim.AdamW(1e-2, weight_decay=.01)
        run(probe, probe_optimizer, data, 0, 1)
        save_continuation(
            source, model_state=probe.trainable_parameters(),
            optimizer_state=probe_optimizer.state, mx_rng_state=mx.random.state,
            numpy_rng_state=np.random.get_state(), next_update=1,
        )

        missing_key = root / "missing-key"
        shutil.copytree(source, missing_key)
        manifest = json.loads((missing_key / "manifest.json").read_text())
        del manifest["sha256"]["cursor"]
        (missing_key / "manifest.json").write_text(json.dumps(manifest))
        try:
            load_continuation(missing_key)
        except ValueError:
            pass
        else:
            raise AssertionError("missing manifest key was accepted")

        bad_cursor = root / "bad-cursor"
        shutil.copytree(source, bad_cursor)
        cursor = json.loads((bad_cursor / "cursor.json").read_text())
        cursor["schema"] = "wrong"
        (bad_cursor / "cursor.json").write_text(json.dumps(cursor))
        manifest = json.loads((bad_cursor / "manifest.json").read_text())
        manifest["sha256"]["cursor"] = hashlib.sha256(
            (bad_cursor / "cursor.json").read_bytes()
        ).hexdigest()
        (bad_cursor / "manifest.json").write_text(json.dumps(manifest))
        try:
            load_continuation(bad_cursor)
        except ValueError:
            pass
        else:
            raise AssertionError("bad cursor schema was accepted")

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
