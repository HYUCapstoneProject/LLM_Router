from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow


class RuntimeContractError(RuntimeError):
    """Raised when a training runtime differs from its versioned contract."""


@dataclass(frozen=True)
class RuntimeContract:
    runtime_version: str
    allowed_python_minors: tuple[str, ...]
    torch_version: str
    transformers_version: str
    numpy_version: str
    pandas_version: str
    pyarrow_version: str
    cuda_version: str

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RuntimeContract":
        contract = cls(
            runtime_version=str(values["runtime_version"]),
            allowed_python_minors=tuple(
                str(value) for value in values["allowed_python_minors"]
            ),
            torch_version=str(values["torch_version"]),
            transformers_version=str(values["transformers_version"]),
            numpy_version=str(values["numpy_version"]),
            pandas_version=str(values["pandas_version"]),
            pyarrow_version=str(values["pyarrow_version"]),
            cuda_version=str(values["cuda_version"]),
        )
        if not contract.runtime_version:
            raise RuntimeContractError("runtime_version must not be empty")
        if not contract.allowed_python_minors:
            raise RuntimeContractError("allowed_python_minors must not be empty")
        return contract


def load_runtime_contract(path: Path) -> RuntimeContract:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, dict):
        raise RuntimeContractError("Runtime contract must be a JSON object")
    return RuntimeContract.from_mapping(values)


def collect_runtime_versions(torch: Any, transformers: Any) -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_minor": ".".join(platform.python_version_tuple()[:2]),
        "torch_version": str(torch.__version__),
        "torch_base_version": str(torch.__version__).split("+", maxsplit=1)[0],
        "transformers_version": str(transformers.__version__),
        "numpy_version": str(np.__version__),
        "pandas_version": str(pd.__version__),
        "pyarrow_version": str(pyarrow.__version__),
        "cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def validate_runtime_contract(
    contract: RuntimeContract,
    runtime: Mapping[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    mismatches: dict[str, dict[str, Any]] = {}

    def compare(name: str, expected: Any, actual: Any) -> None:
        if actual != expected:
            mismatches[name] = {"expected": expected, "actual": actual}

    python_minor = str(runtime.get("python_minor"))
    if python_minor not in contract.allowed_python_minors:
        mismatches["python_minor"] = {
            "expected": list(contract.allowed_python_minors),
            "actual": python_minor,
        }
    compare("torch_version", contract.torch_version, runtime.get("torch_base_version"))
    compare(
        "transformers_version",
        contract.transformers_version,
        runtime.get("transformers_version"),
    )
    compare("numpy_version", contract.numpy_version, runtime.get("numpy_version"))
    compare("pandas_version", contract.pandas_version, runtime.get("pandas_version"))
    compare("pyarrow_version", contract.pyarrow_version, runtime.get("pyarrow_version"))
    if device.startswith("cuda"):
        compare("cuda_version", contract.cuda_version, runtime.get("cuda_version"))
        compare("cuda_available", True, runtime.get("cuda_available"))

    report = {
        "runtime_version": contract.runtime_version,
        "contract_passed": not mismatches,
        "mismatches": mismatches,
        **dict(runtime),
    }
    if mismatches:
        formatted = ", ".join(
            f"{name}: expected {values['expected']!r}, got {values['actual']!r}"
            for name, values in mismatches.items()
        )
        raise RuntimeContractError(
            f"Runtime does not satisfy {contract.runtime_version}: {formatted}"
        )
    return report
