from collections import Counter
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lime import read_lime_questionnaire_structure
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


def extract_question(xml_dict: dict, col_name: str) -> dict:
    questions = xml_dict["questions"]
    for ques in questions:
        if ques["name"] == col_name:
            name_dict = ques["choices"]
    return name_dict


def add_text_to_patch(
    ax: mpl.axes.Axes, font_size: float, values: list
) -> mpl.axes.Axes:
    for i, p in enumerate(ax.patches):
        percent = f"{100 * p.get_height()/sum(values):.1f}%"
        num_i = f"{values[i]}"
        ax.annotate(
            percent,
            (p.get_x() + p.get_width() * 0.3, p.get_height() + 0.2),
            fontsize=font_size * 1.1,
            color="dimgrey",
        )
        if values[i] != 0:
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
            "Laushir, Wrong name, please refer to https://matplotlib.org/stable/gallery/color/named_colors.html"
        )
        raise
    a = np.linspace(0.2, 1, interval)
    grad_colors = [(r, g, b, i) for i in a]
    return grad_colors


def produce_xy(
    ser: pd.Series,
    org_name_dict: dict,
    display_noanswer: bool,
    replace_name_dict: dict = None,
) -> Tuple[list]:
    count = Counter(ser)
    print(
        f"Laushir, if name is tai long and you want to change, here provides the {org_name_dict.values() = },\
        zayang."
    )
    if replace_name_dict:
        name_dict = dict()
        for key, val in org_name_dict.items():
            if val in replace_name_dict.keys():
                name_dict[key] = replace_name_dict[val]
            else:
                name_dict[key] = val
    else:
        name_dict = org_name_dict

    # reorder dict
    cnt = dict()
    # for i, _ in name_dict.items():
    #     if i in count.keys():
    #         cnt[name_dict[i]] = count[i]
    #     else:
    #         cnt[name_dict[i]] = 0

    cnt = {
        name_dict[i]: count[i] if i in count.keys() else 0 for i, _ in name_dict.items()
    }
    cnt["No Answer"] = count["No Answer"]

    if display_noanswer:
        x = cnt.keys()
    else:
        cnt.pop("No Answer")
        x = cnt.keys()

    y = list(cnt.values())
    return x, y


def bar_plot(
    csv_file_path: str,
    col_name: str,
    color: str,
    structure_dict: dict,  # read from xml
    display_noanswer: bool,
    save_fig: str,
    f_size: tuple = (12, 6),
    give_title: str = None,
    give_x_label: str = None,
    give_y_label: str = None,
    replace_name_dict: dict = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    plot single choice bar graph
    """
    ser = extract_data(csv_file_path, col_name)
    name_dict = extract_question(structure_dict, col_name)
    x, y = produce_xy(ser, name_dict, display_noanswer, replace_name_dict)
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
    structure_dict = read_lime_questionnaire_structure("../data_set/test_Oct.xml")
    bar_plot(
        csv_file_path="../../../DataScience/Matplotlib/test_Oct08.csv",
        col_name="C2",
        structure_dict=structure_dict,
        color="orange",
        display_noanswer=True,
        save_fig="../playground/tmp/bar_img.png",
        replace_name_dict={"I don’t want to answer this question": "No Comment"},
        #                     'Gender diverse (Gender-fluid)':"Gender diverse",
        #                     'Other gender representations:': 'Other'},\
        give_title="DeGene",
        give_y_label="y label",
    )
    print(f"{'finished'}")
