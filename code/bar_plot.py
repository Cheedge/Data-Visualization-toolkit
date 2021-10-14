from collections import Counter
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker


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
    no_frame: bool = False,
    row_sum: int = None,
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

    # label
    ax.set_ylabel(ylabel, **label_font)
    ax.set_xlabel(xlabel, **label_font)

    # title
    if title:
        ax.set_title(title, **title_font)

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

    # legend: remove()
    if legend_loc is None:
        ax.get_legend()
    else:
        handles, leg_labels = ax.get_legend_handles_labels()
        new_leg_labels = list()
        for name in leg_labels:
            _, tu = name.strip("()").split(", ")
            new_leg_labels.append(tu)
        ax.legend(
            handles,
            new_leg_labels,
            fontsize=f_size[0] * 1.5,
            loc=legend_loc,
            ncol=legend_col,
            frameon=False,
        )

    # scale
    ax.autoscale()
    ax.set_autoscale_on(True)

    # ticker: can be deleted later.
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

    # figure settings
    fig.tight_layout()
    fig.autofmt_xdate()
    if save_fig:
        fig.savefig(save_fig)
    return fig, ax


def set_ticklabels(labels: list) -> List[str]:
    """
    deal with the xtick label text
    """
    tic_l = list()
    tic_l = [i.get_text() if i.get_text() != "-oth-" else "Other" for i in labels]
    return tic_l


def extract_data(file: str, col_name: str) -> pd.Series:
    """
    return a dataframe which include bar plot data.
    """
    df = pd.read_csv(file)
    ser = df.fillna("No Answer")
    return ser[col_name]


def extract_question_choice(xml_res_dict: dict, col_name: str) -> list:
    question = xml_res_dict["questions"]
    name_list = question["choices"]
    return name_list


def add_text_to_patch(
    ax: mpl.axes.Axes, font_size: float, values: list
) -> mpl.axes.Axes:
    for i, p in enumerate(ax.patches):
        percent = f"{100 * p.get_height()/len(values):.1f}%"
        num_i = f"{values[i]}"
        ax.annotate(
            percent,
            (p.get_x(), p.get_height() + 0.2),
            fontsize=font_size * 1.1,
            color="dimgrey",
        )
        ax.annotate(
            num_i,
            (p.get_x() + p.get_width() * 0.45, p.get_y() + 0.2),
            fontsize=font_size * 1.1,
            fontweight="heavy",
            color="white",
        )
    return ax


def fading_colors(interval: int, color_name: str) -> list:
    """
    normalize color to rgba
    """
    try:
        r, g, b, _ = mcolors.to_rgba(color_name)
    except ValueError:
        print(
            "Wrong name, please refer to https://matplotlib.org/stable/gallery/color/named_colors.html"
        )
        raise
    a = np.linspace(0.2, 1, interval)
    grad_colors = [(r, g, b, i) for i in a]
    return grad_colors


def bar_plot(
    ser: pd.Series,
    color: str,
    display_noanswer: bool,
    save_fig: str,
    f_size: tuple = (12, 6),
    give_title: str = None,
    give_x_label: str = None,
    give_y_label: str = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    plot single choice bar graph
    """
    cnt = Counter(ser)
    if display_noanswer:
        x = cnt.keys()
    else:
        cnt.pop("No Answer")
        x = cnt.keys()
    y = list(cnt.values())
    fig, ax = plt.subplots(figsize=f_size)
    ax.bar(x, y, color=fading_colors(len(y), color))
    add_text_to_patch(ax, font_size=f_size[0], values=y)
    bar_plot_settings(
        fig, ax, f_size, give_title, give_y_label, give_x_label, save_fig=save_fig
    )
    plt.text(len(y) * 0.9, max(y), f"Total: {sum(y)}", size=f_size[0] * 1.2)
    return fig, ax


# final version I will remove it, keep it just for testing else I have to find the test file and others.
if __name__ == "__main__":
    ser = extract_data("../../../DataScience/Matplotlib/test_Oct08.csv", "A3")
    bar_plot(
        ser,
        "orange",
        display_noanswer=True,
        save_fig="../playground/tmp/bar_img.png",
        give_title="title",
        give_y_label="y label",
    )
    print("finished")
