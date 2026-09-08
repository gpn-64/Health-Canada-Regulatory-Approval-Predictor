"""SHAP interpretation of the two fitted pipelines.

For each model we compute TreeExplainer values on the encoded test matrix, save a
bar + beeswarm summary figure, and write a CSV of mean |SHAP| per feature — the
raw material for the regulatory reading in ``docs/methodology.md``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from . import config as C
from .features import output_feature_names


def _encoded_frame(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    encoded = pipeline.named_steps["prep"].transform(X)
    return pd.DataFrame(encoded, columns=output_feature_names(), index=X.index)


def explain_model(pipeline, X: pd.DataFrame, name: str, csv_path) -> pd.DataFrame:
    """Save SHAP summary plots + a mean|SHAP| table for one fitted pipeline."""
    X_enc = _encoded_frame(pipeline, X)
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    shap_values = explainer.shap_values(X_enc)

    mean_abs = np.abs(shap_values).mean(axis=0)
    table = (
        pd.DataFrame({"feature": X_enc.columns, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    table.to_csv(csv_path, index=False)

    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for plot_type, suffix in [("bar", "bar"), ("dot", "beeswarm")]:
        plt.figure()
        shap.summary_plot(
            shap_values, X_enc, plot_type=plot_type, show=False, max_display=20
        )
        plt.title(f"SHAP ({suffix}) — {name}")
        plt.tight_layout()
        plt.savefig(C.FIGURES_DIR / f"shap_{name}_{suffix}.png", dpi=120)
        plt.close()

    return table
