import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from uc_timeseries.util_common import load_ts_data
from uc_timeseries.util_visual import hvplot_vlines


def isolation_forest(df, contamination: float) -> Tuple[pd.DataFrame, IsolationForest, np.ndarray]:
    """Perform anomaly detection with IF

    Ensure contamination * df.shape[0] = Z is close to the expected number of anomaly.
    If Z is < 1, you will not get any anomaly points.

    Args:
        df ([type]): Input features dataframe. Exclude label.
        contamination (float): Percentage of anomaly

    Returns:
        Tuple[pd.DataFrame, IsolationForest, np.ndarray]:
        shap: shap values feature importance, low to high
        est: IF estimator
        preds: Prediction of anomaly point. (-1 is anomaly)
    """
    est = IsolationForest(n_estimators=100, random_state=0, contamination=contamination)
    est.fit(df)
    preds = est.predict(df)

    # # Create shap values and plot them
    shap_values = shap.TreeExplainer(est).shap_values(df)
    feature_cols = df.columns[np.argsort(np.abs(shap_values).mean(0))]
    feature_scores = [f"{i:.02f}" for i in sorted(np.abs(shap_values).mean(0))]
    df_shap = pd.DataFrame({"col": feature_cols, "score": feature_scores})

    return df_shap, est, preds


def elliptic_envelope(df, contamination: float) -> Tuple[EllipticEnvelope, np.ndarray, np.ndarray]:
    est = EllipticEnvelope(random_state=0, contamination=contamination)
    est.fit(df)
    preds = est.predict(df)
    mahalanobis_dist = est.dist_
    return est, preds, mahalanobis_dist


def main(args):
    df = load_ts_data(args.input_file)
    print(df.shape)

    # # Load data and train Anomaly Detector as usual
    df_shap, est, preds = isolation_forest(df, contamination=0.005)
    print(f"Num of anomalies: {len(df[preds==-1])}")
    print(df_shap)
    # shap.summary_plot(shap_values, X_explain)

    idx_position = df[preds == -1].index
    hvplot_vlines(
        df,
        title="anomaly_points",
        output_dir=args.output_dir / "multivariate",
        vlines=pd.to_datetime(idx_position),
        save_figure=True,
    )
    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=Path,
        default="refdata/tags_test.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="data/isolation_forest",
    )
    args, _ = parser.parse_known_args()
    print(args)

    main(args)
