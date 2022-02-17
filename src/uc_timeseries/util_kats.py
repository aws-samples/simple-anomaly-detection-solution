import argparse
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
from kats.consts import TimeSeriesData
from kats.detectors.outlier import MultivariateAnomalyDetector, OutlierDetector
from kats.models.var import VARParams

from uc_timeseries.util_common import load_dummy_data, load_pickle, save_pickle
from uc_timeseries.util_visual import hvplot_line, hvplot_vlines


def df_to_timeseries(df, value_col: List[str]) -> TimeSeriesData:
    """Convert dataframe to timeseries.

    Assume the time has been set to index.
    """
    time_data = pd.to_datetime(df.index)
    ts = TimeSeriesData(time=time_data, value=df[value_col])
    return ts


def get_kats_single_outlier(ts: TimeSeriesData, iqr_mult: int) -> Tuple[pd.DatetimeIndex, Any]:
    """Find outlier based on interquantile range (iqr). Pick up pointts that exceed iqr_mult * iqr.
    Increase iqr_mult to reduce anomaly points.

    Method: "additive", "multiplicative"

    Similar to outlier detection in R, decompose ts and identifier in residual value.

    Args:
        ts (TimeSeriesData): ts structure

    Returns:
        pd.DatetimeIndex: dataframe series index
        outliers_idx: dataframe true/false list
    """
    ts_outlierDetection = OutlierDetector(ts, "additive", iqr_mult=iqr_mult)
    # Fit
    ts_outlierDetection.detector()
    outliers = ts_outlierDetection.outliers
    if not outliers[0]:
        outliers = None
    else:
        outliers = pd.to_datetime(outliers[0])

    return outliers


def get_kats_mv_outlier(mv_ts, training_days: float, maxlags: int):
    """Multvariate outlier.

    Ensure there is enough data point (training_data) in mv_ts.
        - AttributeError: 'DataFrame' object has no attribute 'p_value'

    Args:
        mv_ts ([type]): Kats TS structure
        training_days (int): Number if days of data for training. Can be a fraction of day
        maxlags (int): Number of lags (t, t-1, t-2..) that can affect t+1

    Returns:
        [type]: [description]
    """
    params = VARParams(maxlags=maxlags)
    d = MultivariateAnomalyDetector(mv_ts, params, training_days=training_days)
    try:
        anomaly_score_df = d.detector()
    except AttributeError:
        print("Probably training_days (f{training_days}_is too large")
        raise Exception("Probably training_days (f{training_days}_is too large")

    return anomaly_score_df, d


def main(args):

    df = load_dummy_data()
    TS_COLS = list(df.columns)

    if True:
        # Single feature outlier
        for col in TS_COLS:
            print(f"Processing: {col}")
            col_name = [col]
            ts = df_to_timeseries(df, value_col=col_name)
            outliers = get_kats_single_outlier(ts)
            print(outliers)
            hvplot_line(
                df,
                title=col,
                x="time",  # This is index name
                y=col_name,
                vlines=outliers,
                output_dir=args.output_dir / "univariate",
                save_figure=True,
                width=1500,
                height=500,
            )

    if True:
        # Multivariate outlier
        model_name = args.output_dir / "multivariate/multi_var_detector.pkl"
        model_name.parent.mkdir(parents=True, exist_ok=True)
        mv_ts = df_to_timeseries(df, TS_COLS)

        if True:
            anomaly_score_df, detector = get_kats_mv_outlier(mv_ts, training_days=12, maxlags=2)
            # Save model
            anomaly_score_df.to_csv(args.output_dir / "anomaly_score_df.csv", index=True)
            save_pickle(detector, model_name)

        # Load model
        detector = load_pickle(model_name)
        try:
            anomalies = detector.get_anomaly_timepoints(alpha=0.1)
        except:
            raise Exception("Try a smaller alpha value...")
        print(f"Num of anomaly: {len(anomalies)}")

        hvplot_vlines(
            df,
            title="anomaly_points",
            output_dir=args.output_dir / "multivariate",
            vlines=pd.to_datetime(anomalies),
            save_figure=True,
        )


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
        default="data/kats_result",
    )
    args, _ = parser.parse_known_args()
    print(args)

    main(args)
