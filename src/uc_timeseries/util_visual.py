from pathlib import Path
from typing import List

import holoviews as hv
import hvplot.pandas
import pandas as pd


def hvplot_line_grid(
    df, title, x, y: List[str], output_dir: Path, ncol: int = 2, save_figure=True, **kwargs
):
    """Draw line splot with multiple column."""

    output_dir.mkdir(parents=True, exist_ok=True)

    p = df.hvplot(
        x=x,
        y=y,
        kind="line",
        xlabel="Time",
        ylabel="Value",
        subplots=True,
        shared_axes=False,
        grid=True,
        # legend=True,
        # fontsize=10,
        rot=45,
        **kwargs,
    ).cols(ncol)

    if save_figure:
        hvplot.save(p, output_dir / f"{title}.html")

    return p


def hvplot_line(
    df, title, x, y: List[str], output_dir: Path, vlines=None, save_figure=True, **kwargs
):
    """Draw line splot with optional vertical lines.

    Example:
            hvplot_line(
                df,
                title=col,
                x="time",  # This is index name
                y=col_name,
                vlines=outliers,
                output_dir=args.output_dir / "single",
                save_figure=True,
                width=1500,
                height=500,
                # by="timestamp.month",
                # groupby=["timestamp.year", "timestamp.month"],
            )

    Args:
        df ([type]): Input dataframe
        title ([type]): Graph title
        x ([type]): Column name for x-axis, can be index's name
        y (List[str]): Column name for y-axis
        output_dir (Path): Output dir for html files
        vlines ([type], optional): Vertiline of interest. Defaults to None.
        save_figure (bool, optional): True to save html file. Defaults to True.

    Returns:
        [type]: [description]
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    p = df.hvplot(
        x=x,
        y=y,
        title=title,
        kind="line",
        xlabel="Time",
        ylabel="Value",
        # size=10,
        grid=True,
        legend=True,
        # fontsize=15,
        rot=45,
        **kwargs,
    )

    if vlines is not None:
        for x in vlines:
            p = p * hv.VLine(pd.to_datetime(x)).opts(color="red", alpha=0.3)

    if save_figure:
        hvplot.save(p, output_dir / f"{title}.html")

    return p


def hvplot_vlines(df, title, output_dir: Path, vlines=None, save_figure=True, **kwargs):
    """Plot vertical line over input dataframe index."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dummy_df = pd.DataFrame(data=[0] * len(df), index=df.index, columns=["Value"])
    dummy_df = dummy_df.set_index(df.index.rename("time"))
    p = dummy_df.hvplot(
        x="time",
        y="Value",
        kind="line",
        # size=10,
        alpha=0.3,
        xlabel="Time",
        ylabel=None,
        grid=True,
        subplots=False,
        legend=True,
        # fontsize=15,
        rot=45,
        **kwargs,
    ).opts(toolbar="above")

    for x in vlines:
        p = p * hv.VLine(pd.to_datetime(x)).opts(color="red", alpha=0.3)

    if save_figure:
        hvplot.save(p, output_dir / f"{title}.html")

    return p


def hvplot_scatter(
    df, title, x, y: List[str], category: str, output_dir: Path, save_figure=True, **kwargs
):
    p = df.hvplot.scatter(
        x=x,
        y=y,
        by=category,
        title=title,
        **kwargs,
    )

    if save_figure:
        hvplot.save(p, output_dir / f"{title}.html")

    return p
