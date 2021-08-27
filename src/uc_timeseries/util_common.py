import pickle
from pathlib import Path

import pandas as pd


def load_dummy_data(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, index_col=0, parse_dates=["time"])
    df = df.drop("label", axis=1)
    return df


def load_ts_data(input_file: Path) -> pd.DataFrame:
    """Load data.

    Perform imputation as needed. Kats requires no missing values

    """
    raise Exception(
        "Please implement your custom time series loading here. "
        "Then remove this exception."
    )

    df = pd.read_csv(
        input_file,
        parse_dates=["timestamp"],
        date_parser=lambda x: pd.datetime.strptime(x, "%d/%m/%y %H:%M"),
    )
    # df["timestamp"] = df["timestamp"].apply(lambda x: pd.datetime.strptime(x, "%d/%m/%y %H:%M"))
    df = df.rename(columns={"timestamp": "time"})
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time", drop=True)
    df = df.resample("H").mean()
    df = df.sort_index()

    # aggrid does not support column name with '.'
    cols = [i.replace(".", "_") for i in df.columns]
    df.columns = cols
    # Kats requires no missing values
    index_diff = df.index - df.index.shift()
    assert len(index_diff.unique()) == 1

    print(f"Data shape: {df.shape}")
    df.to_csv(input_file.parent.parent / "data/processed.csv", index=True)
    return df


def save_pickle(obj, output_file: Path):
    """Save object to file."""
    Path(output_file).parent.mkdir(exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filename: Path):
    """Load object from file."""
    with open(filename, "rb") as f:
        obj = pickle.load(f)

    return obj


if __name__ == "__main__":
    df = load_dummy_data("refdata/dummy.csv")
    print(df)
