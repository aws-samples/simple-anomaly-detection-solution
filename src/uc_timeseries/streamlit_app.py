import argparse
from pathlib import Path
from typing import List

import holoviews as hv
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, JsCode

from uc_timeseries.util_common import load_dummy_data, load_ts_data, save_pickle
from uc_timeseries.util_kats import (
    df_to_timeseries,
    get_kats_mv_outlier,
    get_kats_single_outlier,
)
from uc_timeseries.util_sklearn import elliptic_envelope, isolation_forest
from uc_timeseries.util_tsne import AnomalyTSNE
from uc_timeseries.util_visual import (
    hvplot_line,
    hvplot_line_grid,
    hvplot_scatter,
    hvplot_vlines,
)

MODEL_MAP = {
    "uni_kat": "IQR Univariate",
    "multi_kat": "VAR multivariate",
    "iso_forest": "Isolation Forest",
    "mahalanobis": "Mahalanobis Distance",
}
DEFAULT_MODEL_IDX = 2
DEFAUTL_CONFIG = {
    "sidebar": {"num_col": 2},
    "input_file": "refdata/tags_medium.csv",
    "iso_forest": {"contamination": 0.02},
    "uni_kat": {"iqr_mult": 2},
    "multi_kat": {"training_days": 5, "maxlags": 2, "alpha": 0.05},
    "mahalanobis": {"contamination": 0.02},
}

# To support streamlit width constraint
WIDTH = 600

COLS = [
    "external_chiller_condenser_inlet_temp",
    "external_chiller_evaporator_exit_temp",
    "external_chiller_evaporator_heat_load_rt",
    "target_system_efficiency_kw_per_rt",
    "external_E03_thermal_load_kw",
    "total_cooling_tower_thermal_load",
]


def display_editable_df(df, editable: bool = False):
    gb = GridOptionsBuilder.from_dataframe(df.reset_index())

    if editable:
        # make all columns editable
        gb.configure_columns(list(df.columns), editable=True)

        js = JsCode(
            """
        function(e) {
            let api = e.api;
            let rowIndex = e.rowIndex;
            let col = e.column.colId;

            let rowNode = api.getDisplayedRowAtIndex(rowIndex);
            api.flashCells({
            rowNodes: [rowNode],
            columns: [col],
            flashDelay: 10000000000
            });
        };
        """
        )
        gb.configure_grid_options(onCellValueChanged=js)
        gb.configure_pagination(enabled=True)
    go = gb.build()

    update_mode = GridUpdateMode.VALUE_CHANGED
    ag = AgGrid(
        df.reset_index(),
        update_mode=update_mode,
        gridOptions=go,
        allow_unsafe_jscode=True,
        reload_data=False,
    )
    org_index = df.index
    print("Original shape:", df.shape)
    new_df = ag["data"].drop("time", axis=1)
    print("New shape:", new_df.shape)
    new_df = new_df.set_index(org_index)
    assert new_df.shape == df.shape
    return new_df


def plot_hvplot(fig):
    st.bokeh_chart(hv.render(fig, backend="bokeh"))


def display_setup_sidebar(st):
    model = st.sidebar.selectbox(
        "Select model", MODEL_MAP.values(), index=DEFAULT_MODEL_IDX
    )

    cols = st.sidebar.multiselect(
        label="Select columns (1st two by default)",
        options=data_cols,
        default=data_cols[: DEFAUTL_CONFIG["sidebar"]["num_col"]],
    )

    button_retrain = st.sidebar.button("Rerun")
    st.sidebar.write(f"Rerun: {button_retrain}")

    return cols, model, button_retrain


def model_kats_univariate(df, selected_cols: List[str], iqr_mult: int):
    for col in selected_cols:
        print(f"Processing: {col}")
        col_name = [col]
        ts = df_to_timeseries(df, value_col=col_name)
        outliers = get_kats_single_outlier(ts, iqr_mult=iqr_mult)
        print(outliers)
        st.write(f"Num of anomaly: {len(outliers) if outliers is not None else 0}")
        p = hvplot_line(
            df,
            title=f"Column: {col}",
            x="time",  # This is index name
            y=col_name,
            vlines=outliers,
            output_dir=OUTPUT_DIR / "kats/univariate",
            save_figure=True,
            width=WIDTH,
            height=200,
        )
        plot_hvplot(p)


def display_fileuploader():
    data_file = st.file_uploader("Select data file")
    if data_file:
        df = pd.read_csv(data_file)
        df = df.rename(columns={"timestamp": "time"})
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time", drop=True)
        df = df.resample("H").bfill()
        df = df.sort_index()
    else:
        df = load_ts_data(Path("refdata/tags_short.csv"))

    return df


def display_file_selector(folder_path: Path = Path("./refdata")):
    filenames = folder_path.glob("*.csv")
    selected_filename = st.selectbox("Select a file", filenames, index=0)
    return selected_filename


def display_ts_overview(OUTPUT_DIR, df, selected_cols):
    st.write({"Columns": selected_cols})

    with st.beta_expander("Overview", expanded=False):
        p = hvplot_line_grid(
            df,
            title="Summary",
            x="time",
            y=selected_cols,
            ncol=1,
            output_dir=OUTPUT_DIR / "summary",
            save_figure=False,
            width=WIDTH,
            height=200,
        )
        plot_hvplot(p)


def display_model(
    OUTPUT_DIR,
    df,
    selected_cols,
    selected_model,
    button_retrain,
):
    if selected_model == MODEL_MAP["uni_kat"]:
        st.header(f"Anomaly: {selected_model}")
        iqr_mult = float(
            st.text_input("IQR factor", value=DEFAUTL_CONFIG["uni_kat"]["iqr_mult"])
        )
        model_kats_univariate(df, selected_cols, iqr_mult)

    elif selected_model == MODEL_MAP["multi_kat"]:
        st.header(f"Anomaly: {selected_model}")
        training_days = int(
            st.text_input(
                "Training day (exclude from anomaly detection)",
                value=DEFAUTL_CONFIG["multi_kat"]["training_days"],
            )
        )
        maxlags = int(
            st.text_input("Max lag", value=DEFAUTL_CONFIG["multi_kat"]["maxlags"])
        )
        alpha = float(
            st.text_input("Alpha", value=DEFAUTL_CONFIG["multi_kat"]["alpha"])
        )
        # Multivariate outlier
        model_name = OUTPUT_DIR / "kats/multivariate/multi_var_detector.pkl"
        model_name.parent.mkdir(parents=True, exist_ok=True)

        mv_ts = df_to_timeseries(df, selected_cols)

        with st.spinner("Training model..."):
            anomaly_score_df, detector = get_kats_mv_outlier(
                mv_ts, training_days=training_days, maxlags=maxlags
            )
            if hasattr(anomaly_score_df, "p_value"):
                anomaly_score_df["p_value"] = anomaly_score_df["p_value"].apply(
                    lambda x: f"{x:.03f}"
                )
                anomaly_score_df["p_value"] = anomaly_score_df["p_value"].astype(float)

                # Return anomaly_score_df.p_value < alpha
                anomalies = detector.get_anomaly_timepoints(alpha=alpha)
                print(anomalies)
            else:
                anomalies = None

            # Save model
            anomaly_score_df.to_csv(
                OUTPUT_DIR / "kats/multivariate/anomaly_score_df.csv", index=True
            )
            save_pickle(detector, model_name)

        # Load model
        # detector = load_pickle(model_name)
        st.write(f"{selected_model} - Num of anomaly: {len(anomalies)}")

        p_kat_mv = hvplot_vlines(
            df,
            title="anomaly_points",
            output_dir=OUTPUT_DIR / "kats/multivariate",
            vlines=pd.to_datetime(anomalies),
            save_figure=True,
            width=WIDTH,
            height=200,
        )
        plot_hvplot(p_kat_mv)

        with st.spinner("Evaluating model..."):
            preds = anomaly_score_df.index.isin(anomalies).astype(int)
            display_tsne(
                anomaly_score_df.drop(["overall_anomaly_score", "p_value"], axis=1),
                preds,
            )

    elif selected_model == MODEL_MAP["iso_forest"]:
        contamination = float(
            st.text_input(
                "Contamination", value=DEFAUTL_CONFIG["iso_forest"]["contamination"]
            )
        )

        with st.spinner("Training model..."):
            df_shap, est, preds = isolation_forest(df, contamination=contamination)

        model_name = OUTPUT_DIR / "isolation_forest/estimator.pkl"
        save_pickle(est, model_name)

        # detector = load_pickle(model_name)
        num_anomaly = sum(preds == -1)
        anomaly_point_df = df[preds == -1]
        st.write(f"{selected_model} - Num of anomalies: {num_anomaly}")

        p_if = hvplot_vlines(
            df,
            title="anomaly_points",
            output_dir=OUTPUT_DIR / "isolation_forest/multivariate",
            vlines=pd.to_datetime(anomaly_point_df.index),
            save_figure=True,
            width=WIDTH,
            height=200,
        )
        plot_hvplot(p_if)
        st.write("Feature importance")
        st.dataframe(df_shap.sort_values(["score"], ascending=False))

        with st.spinner("Evaluating model..."):
            display_tsne(df, preds)

    elif selected_model == MODEL_MAP["mahalanobis"]:
        contamination = float(
            st.text_input(
                "Contamination (MH distance)",
                value=DEFAUTL_CONFIG["mahalanobis"]["contamination"],
            )
        )
        with st.spinner("Training model..."):
            est, preds, mahalanobis_dist = elliptic_envelope(
                df, contamination=contamination
            )

        model_name = OUTPUT_DIR / "mahalanobis/estimator.pkl"
        save_pickle(est, model_name)

        # detector = load_pickle(model_name)
        num_anomaly = sum(preds == -1)
        anomaly_point_df = df[preds == -1]
        st.write(f"{selected_model} - Num of anomalies: {num_anomaly}")

        p_if = hvplot_vlines(
            df,
            title="anomaly_points",
            output_dir=OUTPUT_DIR / "mahalanobis/multivariate",
            vlines=pd.to_datetime(anomaly_point_df.index),
            save_figure=True,
            width=WIDTH,
            height=200,
        )
        plot_hvplot(p_if)

        with st.spinner("Evaluating model..."):
            display_tsne(df, preds)

    else:
        print("Model not implemented....")


def display_tsne(df, preds):
    tsne = AnomalyTSNE(df, preds)
    result_df = tsne.result_df
    p = hvplot_scatter(
        result_df,
        title="Normalized clustering",
        x="0",
        y="1",
        category="preds",
        output_dir=OUTPUT_DIR / "tsne",
        save_figure=False,
        xlabel="X",
        ylabel="Y",
    )
    plot_hvplot(p)


def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Path to default data",
    )
    args, _ = parser.parse_known_args()
    print(args)
    return args


if __name__ == "__main__":
    args = parser()
    OUTPUT_DIR = Path("./data/streamlit")
    ########################
    # Load data
    ########################
    st.title("Simple Time Series Anomaly Detection")
    dummy_data = st.sidebar.checkbox("Demo Mode", value=False)

    with st.spinner("Loading data.."):

        if dummy_data:
            file = Path("refdata/dummy.csv")
            df = load_dummy_data(str(file))
        else:

            selected_file = display_file_selector(args.data_dir)
            file = Path(selected_file)
            df = load_dummy_data(str(file))

    data_cols = list(df.columns)

    message = f"File: **{file}**\n\n"
    message += f"Time series start: **{df.index.min()}**\n\n"
    message += f"Time series end: **{df.index.max()}**\n\n"
    message += f"Resample data shape: **{df.shape}**"
    st.info(message)

    df = display_editable_df(df, editable=True)

    ########################
    # Side bar
    ########################
    selected_cols, selected_model, button_retrain = display_setup_sidebar(st)

    df = df[selected_cols]

    ########################
    # Main page
    ########################
    st.header("Overview")

    if len(selected_cols) <= 1:
        st.write("Please select columns (at least 2) and model")
        st.stop()

    display_ts_overview(OUTPUT_DIR, df, selected_cols)

    display_model(
        OUTPUT_DIR,
        df,
        selected_cols,
        selected_model,
        button_retrain,
    )

    print("Done")
