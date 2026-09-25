from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..training.routerbench_preprocessing import assign_group_splits


def assign_continual_splits(frame: pd.DataFrame, config: Mapping[str, Any]) -> pd.DataFrame:
    """Partition development questions before threshold fitting or anchor training."""
    output = frame.copy()
    development = output["split"].isin(["train", "validation"])
    protected = set(output.loc[~development, "duplicate_group_id"])
    if set(output.loc[development, "duplicate_group_id"]) & protected:
        raise ValueError("Development groups overlap a protected test group")
    working = output.loc[development].copy()
    boundaries = sorted(set(config["thresholds"]["hard"] + config["thresholds"]["easy"]))
    # Stratify against all candidate boundaries before selecting any threshold.
    working["difficulty_label"] = np.searchsorted(boundaries, working["overall_solve_rate"], side="right").astype(str)
    working["domain"] = working["source_dataset"] + "|" + working["category"]
    output.loc[development, "split"] = assign_group_splits(
        working, config["continual_split"]["fractions"], int(config["seed"])
    )
    if output.groupby("duplicate_group_id")["split"].nunique().max() != 1:
        raise ValueError("A duplicate group crosses continual partitions")
    return output
