"""
Patient-level ER/PR aggregation and inter-rater agreement (Cohen's kappa).

Aggregates block-level marker statuses to patient level,
and optionally computes Cohen's kappa against QuPath reference data.
"""

import numpy as np
import pandas as pd
from typing import Optional

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None


def _resolve_patient_id(block_name: str) -> str:
    """
    Default patient↔block mapping: strip trailing block suffix.

    Convention examples:
      'A4_1'  → 'A4'
      'A4'    → 'A4'
      'PAT01_B2' → 'PAT01'

    Override by providing a patient_map dict to aggregate_to_patient().
    """
    parts = block_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return block_name


def aggregate_to_patient(
    df: pd.DataFrame,
    patient_map: Optional[dict] = None,
    min_pos_fraction: float = 0.01,
) -> pd.DataFrame:
    """
    Aggregate block-level ER/PR scores to patient level.

    For each patient and marker:
      - total cells across all blocks
      - total positive cells
      - weighted positive fraction (sum_pos / sum_total)
      - patient call: Positive if weighted fraction >= min_pos_fraction, else Negative
      - per-block breakdown

    Parameters
    ----------
    df : DataFrame
        Cell-level output from batch_run (must have columns: block, ER_status, PR_status).
    patient_map : dict, optional
        Custom {block_name: patient_id} mapping. If None, uses _resolve_patient_id().
    min_pos_fraction : float
        Minimum positive fraction for patient-level "Positive" call.

    Returns
    -------
    DataFrame with one row per (patient, marker), columns:
        patient_id, marker, total_cells, n_positive, positive_fraction,
        patient_status, n_blocks, blocks
    """
    if df.empty:
        return pd.DataFrame()

    if patient_map is None:
        patient_map = {}

    def _get_patient(block):
        if block in patient_map:
            return patient_map[block]
        return _resolve_patient_id(block)

    df = df.copy()
    df["patient_id"] = df["block"].apply(_get_patient)

    rows = []
    for marker in ["ER", "PR"]:
        status_col = f"{marker}_status"
        if status_col not in df.columns:
            continue

        for pid, grp in df.groupby("patient_id"):
            pos_mask = grp[status_col] == "Positive"
            n_total = len(grp)
            n_pos = int(pos_mask.sum())
            frac = n_pos / n_total if n_total > 0 else 0.0
            call = "Positive" if frac >= min_pos_fraction else "Negative"
            blocks = sorted(grp["block"].unique().tolist())

            rows.append({
                "patient_id": pid,
                "marker": marker,
                "total_cells": n_total,
                "n_positive": n_pos,
                "positive_fraction": round(frac, 4),
                "patient_status": call,
                "n_blocks": len(blocks),
                "blocks": "; ".join(blocks),
            })

    return pd.DataFrame(rows)


def compute_kappa_vs_qupath(
    our_labels: pd.Series,
    qupath_labels: pd.Series,
    labels: list = None,
) -> dict:
    """
    Compute Cohen's kappa between our pipeline labels and QuPath reference.

    Parameters
    ----------
    our_labels : Series of str ("Positive" / "Negative")
    qupath_labels : Series of str, same index alignment expected
    labels : list, category order for kappa. Default ["Negative", "Positive"].

    Returns
    -------
    dict with kappa, agreement_rate, n_samples, label_counts
    """
    if labels is None:
        labels = ["Negative", "Positive"]

    # Align & dropna
    combined = pd.DataFrame({"ours": our_labels, "qupath": qupath_labels}).dropna()
    n = len(combined)

    if n == 0:
        return {"kappa": float("nan"), "agreement_rate": float("nan"),
                "n_samples": 0, "label_counts": {}}

    agreement = (combined["ours"] == combined["qupath"]).mean()

    if cohen_kappa_score is not None:
        kappa = float(cohen_kappa_score(
            combined["qupath"], combined["ours"], labels=labels
        ))
    else:
        # Manual fallback: weighted agreement minus chance agreement
        # Simple unweighted kappa
        po = agreement
        pe = sum(
            ((combined["ours"] == lab).mean() * (combined["qupath"] == lab).mean())
            for lab in labels
        )
        kappa = (po - pe) / (1 - pe) if pe < 1.0 else float("nan")

    label_counts = {
        lab: {
            "ours": int((combined["ours"] == lab).sum()),
            "qupath": int((combined["qupath"] == lab).sum()),
        }
        for lab in labels
    }

    return {
        "kappa": round(kappa, 4),
        "agreement_rate": round(agreement, 4),
        "n_samples": n,
        "label_counts": label_counts,
    }


def patient_level_kappa(
    patient_df: pd.DataFrame,
    qupath_patient_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Cohen's kappa at patient level for each marker.

    Parameters
    ----------
    patient_df : output of aggregate_to_patient()
        Must have columns: patient_id, marker, patient_status
    qupath_patient_df : same shape, from QuPath aggregation
        Must have columns: patient_id, marker, patient_status

    Returns
    -------
    DataFrame with one row per marker: marker, kappa, agreement_rate, n_patients
    """
    results = []
    for marker in ["ER", "PR"]:
        ours = patient_df[patient_df["marker"] == marker].set_index("patient_id")["patient_status"]
        qp = qupath_patient_df[qupath_patient_df["marker"] == marker].set_index("patient_id")["patient_status"]
        # Keep only patients present in both
        common = ours.index.intersection(qp.index)
        if len(common) == 0:
            results.append({"marker": marker, "kappa": float("nan"),
                            "agreement_rate": float("nan"), "n_patients": 0})
            continue
        res = compute_kappa_vs_qupath(ours.loc[common], qp.loc[common])
        results.append({
            "marker": marker,
            "kappa": res["kappa"],
            "agreement_rate": res["agreement_rate"],
            "n_patients": res["n_samples"],
        })
    return pd.DataFrame(results)


def block_level_kappa(
    df: pd.DataFrame,
    qupath_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Cohen's kappa at cell level (matched by spatial proximity or global_cell_id).

    This expects qupath_df to have columns: our_status, qupath_status, marker,
    typically produced by compare_qupath.py pipeline extension.

    Returns
    -------
    DataFrame with one row per marker: marker, kappa, agreement_rate, n_cells
    """
    results = []
    for marker in ["ER", "PR"]:
        our_col = f"{marker}_status"
        qp_col = f"qupath_{marker}_status"

        if our_col not in df.columns or qp_col not in qupath_df.columns:
            continue

        # Merge on global_cell_id if available
        if "global_cell_id" in df.columns and "global_cell_id" in qupath_df.columns:
            merged = df[["global_cell_id", our_col]].merge(
                qupath_df[["global_cell_id", qp_col]],
                on="global_cell_id", how="inner"
            )
            ours = merged[our_col]
            qp = merged[qp_col]
        else:
            # Fallback: use index alignment
            ours = df[our_col]
            qp = qupath_df[qp_col]

        res = compute_kappa_vs_qupath(ours, qp)
        res["marker"] = marker
        results.append(res)

    return pd.DataFrame(results)
