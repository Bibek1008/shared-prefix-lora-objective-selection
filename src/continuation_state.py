"""Safetensors/JSON sidecar for exact local MLX continuation state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


SCHEMA = "lora-early-selection-continuation-v1"
STATE_NAMES = ("model", "optimizer", "mx_rng", "numpy_rng", "cursor")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_tree(path: Path, tree: Any) -> None:
    flat = dict(tree_flatten(tree))
    if not flat or any(not isinstance(value, mx.array) for value in flat.values()):
        raise TypeError("Expected a non-empty MLX array tree")
    mx.save_safetensors(str(path), flat)


def load_tree(path: Path) -> Any:
    loaded = mx.load(str(path))
    if not isinstance(loaded, dict):
        raise TypeError("Expected a safetensors mapping")
    return tree_unflatten(list(loaded.items()))


def save_continuation(
    directory: Path,
    *,
    model_state: Any,
    optimizer_state: Any,
    mx_rng_state: Any,
    numpy_rng_state: tuple,
    next_update: int,
) -> None:
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": directory / "model.safetensors",
        "optimizer": directory / "optimizer.safetensors",
        "mx_rng": directory / "mx_rng.safetensors",
        "numpy_rng": directory / "numpy_rng.json",
        "cursor": directory / "cursor.json",
    }
    save_tree(paths["model"], model_state)
    save_tree(paths["optimizer"], optimizer_state)
    save_tree(paths["mx_rng"], mx_rng_state)
    algorithm, keys, position, has_gauss, cached = numpy_rng_state
    paths["numpy_rng"].write_text(json.dumps({
        "algorithm": algorithm, "keys": keys.tolist(), "position": int(position),
        "has_gauss": int(has_gauss), "cached": float(cached),
    }, sort_keys=True) + "\n")
    paths["cursor"].write_text(json.dumps({
        "schema": SCHEMA, "next_update": next_update,
    }, sort_keys=True) + "\n")
    (directory / "manifest.json").write_text(json.dumps({
        "schema": SCHEMA, "sha256": {name: sha256(path) for name, path in paths.items()},
    }, indent=2, sort_keys=True) + "\n")


def load_continuation(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if set(manifest) != {"schema", "sha256"} or manifest["schema"] != SCHEMA:
        raise ValueError("Unsupported schema")
    if not isinstance(manifest["sha256"], dict) or set(manifest["sha256"]) != set(STATE_NAMES):
        raise ValueError("Continuation manifest must bind every state file exactly once")
    paths = {
        "model": directory / "model.safetensors",
        "optimizer": directory / "optimizer.safetensors",
        "mx_rng": directory / "mx_rng.safetensors",
        "numpy_rng": directory / "numpy_rng.json",
        "cursor": directory / "cursor.json",
    }
    for name in STATE_NAMES:
        expected = manifest["sha256"][name]
        if not isinstance(expected, str) or len(expected) != 64 or not paths[name].is_file():
            raise ValueError(f"Invalid manifest entry: {name}")
        if sha256(paths[name]) != expected:
            raise ValueError(f"Hash mismatch: {name}")
    numpy_payload = json.loads(paths["numpy_rng"].read_text())
    cursor = json.loads(paths["cursor"].read_text())
    if set(numpy_payload) != {"algorithm", "keys", "position", "has_gauss", "cached"}:
        raise ValueError("Invalid NumPy RNG payload")
    if (
        not isinstance(numpy_payload["algorithm"], str)
        or not isinstance(numpy_payload["keys"], list)
        or not all(isinstance(value, int) and 0 <= value <= 0xFFFFFFFF for value in numpy_payload["keys"])
        or not isinstance(numpy_payload["position"], int)
        or not isinstance(numpy_payload["has_gauss"], int)
        or not isinstance(numpy_payload["cached"], (int, float))
    ):
        raise ValueError("Invalid NumPy RNG field types")
    if set(cursor) != {"schema", "next_update"} or cursor["schema"] != SCHEMA:
        raise ValueError("Invalid continuation cursor schema")
    if isinstance(cursor["next_update"], bool) or not isinstance(cursor["next_update"], int) or cursor["next_update"] < 0:
        raise ValueError("Invalid continuation cursor value")
    return {
        "model_state": load_tree(paths["model"]),
        "optimizer_state": load_tree(paths["optimizer"]),
        "mx_rng_state": load_tree(paths["mx_rng"]),
        "numpy_rng_state": (
            numpy_payload["algorithm"], np.asarray(numpy_payload["keys"], dtype=np.uint32),
            numpy_payload["position"], numpy_payload["has_gauss"], numpy_payload["cached"],
        ),
        "next_update": cursor["next_update"],
    }


def restore_mx_rng_state(saved_state: Any) -> None:
    live = mx.random.state
    if not isinstance(live, list) or not isinstance(saved_state, list) or len(live) != len(saved_state):
        raise ValueError("MLX RNG structure mismatch")
    for index, value in enumerate(saved_state):
        live[index] = value
    mx.eval(live)
