from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def prepare_df(
    file: str, col_x: str, col_y: str, need_total: bool = False
) -> pd.DataFrame:
    """
    prepare a dataframe for column x vs. column y.
    """
    df = pd.read_csv(file, usecols=[col_x, col_y]).fillna("No Answer")
    g_df = df.groupby([col_x, col_y]).size().reset_index(name="count")
    p_df = g_df.pivot(index=col_x, columns=col_y).fillna(0)
    if need_total:
        p_df = p_df.append(p_df.sum().rename("total"))
    return p_df


def add_text_in_bar(
    ax: mpl.axes.Axes, f_size: tuple, p_df: pd.DataFrame
) -> Tuple[list, mpl.axes.Axes]:
    """
    add the percentage and other text content into each bar.
    """
    data, r_sum, c_data, h = list(), list(), list(), list()
    for _, row in p_df.iterrows():
        r_sum.append(sum(list(row)))
    for i, (_, col) in enumerate(p_df.iteritems()):
        c_data.append(np.asarray(col) / np.asarray(r_sum))
    r = len(col)
    data = np.asarray(c_data).flatten()

    accumu_height = list()
    for i, p in enumerate(ax.patches):
        h.append(p.get_height())
        if i < r:
            accumu_height.append(p.get_height())
        if data[i] >= 0.05:
            percent = f"{100 * data[i]:.0f}%"
            accumu_height, height = text_height(h, i, r, accumu_height)
            ax.annotate(
                percent,
                (p.get_x() + p.get_width() * 0.3, p.get_y() + p.get_height() * 0.4),
                fontsize=f_size[0] * 0.8,
                fontweight="heavy",
                color="white",
            )
    return r_sum, ax


def text_height(
    h: List[float], i: int, r: int, accumu_height: List[float]
) -> Tuple[List[float], float]:
    """
    calculate the position of the text in every patch of each bar
    """
    height = 0
    if i < r:
        height = h[i] * 0.5
    else:
        accumu_height[i % r] += h[i]
        height = accumu_height[i % r] - h[i] * 0.36

    return (accumu_height, height)


def bar_plot_settings(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    f_size: tuple,
    title: str = None,
    ylabel: str = None,
    xlabel: str = None,
    legend_loc: str = None,
    legend_col: int = 2,
    save_fig: bool = False,
    no_frame: bool = True,
    row_sum: int = None,
    num_plots: int = 1,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    some figure settings in simple bar plot.
    """
    # lable font
    label_font = {
        "fontsize": f_size[0] * 1.2,
        "fontfamily": "DejaVu Sans",
        "fontstyle": "oblique",
    }

    # title font
    title_font = {
        "fontsize": f_size[0] * 1.5,
        "fontfamily": "DejaVu Sans",
        "fontstyle": "oblique",
    }

    # ticks
    ax.tick_params(direction="out", labelsize=f_size[0] * 1.2)
    # when use below set_xticks, then in multi_bar_compare(),
    # the "width" option should not include dim_x.
    # ax.set_xticks(np.linspace(-1, 7, len(row_sum)))
    # ax.xaxis.set_major_locator(ticker.FixedLocator(np.linspace(-1, 7, len(row_sum))))
    # or maybe use MultipleLocator(1) or AutoLocator()

    # label
    ax.set_ylabel(ylabel, **label_font)
    ax.set_xlabel(xlabel, **label_font)

    # ticklabels
    if row_sum:
        tic_label = list(ax.get_xticklabels())
        # print(ax.get_xticks())
        new_tic_label = set_ticklabels(labels=tic_label, r_sum=row_sum)
        ax.set_xticklabels(new_tic_label, **label_font)

    # title
    if title:
        ax.set_title(title, y=1.1, **title_font)

    # frame: if you use old version of matplotlib maybe this will work for you:
    # mpl.rcParams['axes.spines.bottom'] = False
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    if no_frame:
        ax.spines.left.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.tick_params(left=False, labelleft=False)
    else:
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)

    # legend
    if legend_loc is None:
        # works in 3.4.3 not work some previous version
        ax.get_legend().remove()
    else:
        # if num_plots == 1:
        anchor = (0.5, 1.1)
        # else:
        #     anchor = ()
        handles, leg_labels = ax.get_legend_handles_labels()
        new_leg_labels = list()
        for name in leg_labels:
            # _, tu = name.strip("()").split(", ")
            tu = name
            new_leg_labels.append(tu)
        ax.legend(
            handles,
            new_leg_labels,
            bbox_to_anchor=anchor,
            fontsize=f_size[0] * 1.5,
            loc=legend_loc,
            ncol=legend_col,
            frameon=False,
        )

    # scale
    # ax.autoscale()
    # ax.set_autoscale_on(True)

    # figure settings
    # fig.tight_layout()
    fig.autofmt_xdate()
    if save_fig:
        fig.savefig(save_fig)
    return fig, ax


def set_ticklabels(labels: list, r_sum: int) -> List[str]:
    """
    deal with the xtick label text
    """
    tic_l = list()
    tic_l = [
        i.get_text() + "(" + str(int(j)) + ")" if i.get_text() != "-oth-" else "Other"
        for i, j in zip(labels, r_sum)
    ]
    return tic_l


def percent_df(p_df: pd.DataFrame) -> pd.DataFrame:
    df_dic = dict()
    for label, row in p_df.iterrows():

        row = np.asarray(row)
        new_row = row / row.sum()
        df_dic[label] = new_row
    new_df = pd.DataFrame.from_dict(df_dic)
    col_name = [name for _, name in p_df.columns]
    new_df.index = col_name
    return new_df.transpose()


def compare_bar_plot(
    fig: mpl.figure.Figure,
    ax: mpl.axes.Axes,
    filepath: str,
    col_x: str,
    col_y: str,
    save_figure: str,
    give_title: str = None,
    legend_location: str = "upper center",
    color: list = None,
    display_noanswer: bool = True,
    fig_size: tuple = (21, 11),
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    make comparision plot between different columns.

    note: when dealing with groupby, method refer to
    https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count-mean-etc-using-pandas-groupby
    """
    # fig, ax = plt.subplots(figsize=fig_size)
    p_df = prepare_df(filepath, col_x, col_y)
    if color is not None and len(color) != len(p_df.columns):
        raise Exception(
            f"lau shir, there need {len(p_df.columns)} kinds of colors as a list"
        )
    ax = percent_df(p_df).plot.bar(
        ax=ax, stacked=True, width=0.06 * fig_size[1], color=color
    )
    r_sum, ax = add_text_in_bar(ax, fig_size, p_df)
    bar_plot_settings(
        fig,
        ax,
        f_size=fig_size,
        title=give_title,
        legend_loc=legend_location,
        row_sum=r_sum,
        save_fig=save_figure,
        legend_col=int(len(p_df.columns)),
    )
    return fig, ax


def multi_bar_compare(
    filepath: str,
    col_xs: List[str],
    col_y: str,
    save_figure: str = None,
    give_title: str = None,
    legend_location: str = "upper center",
    color: list = None,
    display_noanswer: bool = True,
    fig_size: tuple = (10, 5),
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    if len(col_xs) == 1:
        axs = [0] * len(col_xs)
        fig, axs[0] = plt.subplots(len(col_xs), figsize=fig_size)
    else:
        # legend_location = 'lower right'

        axs = [] * len(col_xs)
        fig, axs = plt.subplots(1, len(col_xs), figsize=fig_size)
    for i, col_x in enumerate(col_xs):
        # fig, axs[i] = compare_bar_plot(fig,axs[i],filepath,col_x,col_y,color,fig_size)
        p_df = prepare_df(filepath, col_x, col_y)
        dim_x = len(p_df.index)
        if color is not None and len(color) != len(p_df.columns):
            raise Exception(
                f"lau shir, there need {len(p_df.columns)} kinds of colors as a list"
            )
        axs[i] = percent_df(p_df).plot.bar(
            ax=axs[i], stacked=True, width=0.02 * fig_size[1] * dim_x, color=color
        )
        r_sum, axs[i] = add_text_in_bar(axs[i], fig_size, p_df)

        if i >= 0:
            give_title = None
            legend_location = None

        bar_plot_settings(
            fig,
            axs[i],
            f_size=fig_size,
            title=give_title,
            legend_loc=legend_location,
            row_sum=r_sum,
            # save_fig=save_figure)
            # legend_col=int(len(p_df.columns))
        )
    # very good explain of how to share legend
    # refer to:
    # https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
    anchor = (0.5, 0.96)
    dim_y = len(p_df.columns)
    tuple_labels = [ax.get_legend_handles_labels() for ax in axs]
    handles, new_leg_labels = [sum(lol, []) for lol in zip(*tuple_labels)]
    # new_leg_labels = list()
    # for name in leg_labels:
    #     new_leg_labels.append(name)
    # print()

    fig.legend(
        handles[0:dim_y],
        new_leg_labels[0:dim_y],
        ncol=dim_y,
        bbox_to_anchor=anchor,
        fontsize=fig_size[0] * 1.5,
        loc="upper center",
        frameon=False,
    )
    fig.suptitle("give_title", x=0.5, y=1, fontsize=fig_size[0] * 2)
    fig.tight_layout(pad=1.2)
    fig.savefig("../playground/tmp/multiple_compare_bar_plot.png")
    return fig, axs


# final version I will remove it, keep it just for easy testing, else I have to find the test file and others.
if __name__ == "__main__":
    multi_bar_compare(
        "../data_set/test_Oct08.csv",
        ["A7", "A00"],
        "A6",  # save_figure='../test/tmp/multiple_compare_bar_plot.png',
        give_title="lau shir, bai lee nar, ku tree zhe, drong",
        legend_location="upper center",
        color=[
            "royalblue",
            "steelblue",
            "cornflowerblue",
            "dodgerblue",
            "deepskyblue",
            "mediumslateblue",
        ],
    )
