import warnings
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from lime import read_lime_questionnaire_structure

warnings.filterwarnings("error")


def prepare_df(
    file: str,
    col_x: str,
    col_y: str,
    ques: dict,
    replace_name_dict_ind: dict = None,
    replace_name_dict_col: dict = None,
    need_total: bool = False,
    need_x_null: bool = True,
    need_y_null: bool = True,
) -> pd.DataFrame:
    """
    prepare a dataframe for column x vs. column y.
    """
    name_dict_x = extract_question(ques, col_x)
    name_dict_y = extract_question(ques, col_y)
    # x, y = produce_xy(ser, name_dict, display_noanswer, replace_name_dict)

    df = pd.read_csv(file, usecols=[col_x, col_y]).fillna("No Answer")
    # print(f"{df['B7']=} ")
    g_df = df.groupby([col_x, col_y]).size().reset_index(name="count")
    # print(g_df, col_x, col_y)
    p_df = g_df.pivot(index=col_x, columns=col_y).fillna(0)
    # print(p_df)
    # print(p_df.index, p_df.columns)
    # print(f'before, {col_x=}, {col_y=}, {p_df.columns=}, {p_df.index=}')
    p_df.columns = [name for _, name in p_df.columns]
    p_df = add_empty_cols(p_df.T, name_dict_x)
    p_df = add_empty_cols(p_df.T, name_dict_y)
    # print(f'{p_df=}, {name_dict_x=}, {name_dict_y=} ')
    new_index = find_correspond_names(
        list(p_df.index),
        name_dict_x,
        replace_name_dict_ind,
    )
    new_columns = find_correspond_names(
        list(p_df.columns),
        name_dict_y,
        replace_name_dict_col,
    )
    # print(f'{p_df.index=}, {p_df.columns=}, {new_index}, {new_columns} ')
    p_df.index, p_df.columns = new_index, new_columns
    # print(f'after, {p_df.columns=}, {p_df.index=}')
    if need_x_null is False:
        p_df.drop("No Answer", inplace=True)
    if need_y_null is False:
        p_df.drop(("count", "No Answer"), axis=1, inplace=True)
    if need_total:
        p_df = p_df.append(p_df.sum().rename("Total"))
        # p_df.sum().rename("total").append(p_df)
        # p_df[-1] = p_df.sum().rename('total')
        # p_df.index = p_df.index + 1  # shifting index
        # p_df.sort_index(inplace=True)
    # print(p_df.index, p_df.columns)

    return p_df


def add_empty_cols(df: pd.DataFrame, name_dict: dict) -> pd.DataFrame:
    # print(f'{df=}, {name_dict} ')
    for i, (k, _) in enumerate(name_dict.items()):
        if k not in df.columns:
            df.insert(i, k, [0] * len(df.index))
    # print(f'{df=}, {name_dict} ')
    return df

    # for k in name_dict.keys():
    #     if k not in p_df.index:
    #         ind.append(k)
    # zero_df = pd.DataFrame(index=ind, columns=p_df.columns)
    # p_df = pd.concat([p_df, zero_df]).fillna(0)

    # #         p_df.insert(i, k, [0]*len(p_df.columns))


def extract_question(questions: dict, col_name: str) -> dict:
    # questions = xml_dict["questions"]
    for ques in questions:
        if ques["name"] == col_name:
            name_dict = ques["choices"]
    return name_dict


def find_correspond_names(
    columns: list,
    org_name_dict: dict,
    replace_name_dict: dict = None,
) -> list:
    # print(f'{df.columns=}, {org_name_dict=}, {df.index=}')
    print(
        f"Laushir, if name is tai long and you want to change, \n \
            here provides the {org_name_dict.values() = }, zayang."
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

    if "No Answer" in columns:
        names = list(name_dict.values()) + list(["No Answer"])
    else:
        names = list(name_dict.values())
    # if "No Answer" in df.index:
    #     df.index = list(name_dict.keys()) + list("No Answer")
    # else:
    #     df.index = list(name_dict.keys())
    return names


def add_text_in_bar(
    ax: mpl.axes.Axes,
    f_size: tuple,
    p_df: pd.DataFrame,
    # dim_xs: List[Tuple[int, bool]],
    dim_y: int,
    interval: float,
    x_ticks_scale: float = 0.5,
    x_ticks_displacement: float = 2,
    set_location: list = None,
) -> Tuple[list, mpl.axes.Axes]:
    """
    add the percentage and other text content into each bar.
    """
    data, r_sum, c_data, h = list(), list(), list(), list()
    for _, row in p_df.iterrows():
        if sum(list(row)) == 0:
            r_sum.append(0.00000001)
        else:
            r_sum.append(sum(list(row)))
    for i, (_, col) in enumerate(p_df.iteritems()):
        c_data.append(np.asarray(col) / np.asarray(r_sum))
    r = len(col)
    # print(i, r)
    data = np.asarray(c_data).flatten()

    accumu_height = list()
    # intra_interval = 0
    if set_location:
        locator_list = set_location * dim_y

    # print(np.asarray(ax.get_xlim()))
    for i, p in enumerate(ax.patches):
        # patch x range [-0.5*width, num_bar*1-0.5*width]
        # x coordinate(left down corner): [-0.5*w, -0.5*w+1, -0.5*w+2, ..., num_bar*1-0.5*width]
        # print(i, p.get_x(), p.get_width())
        # for dim_x, need_tot in dim_xs:
        # print(dim_x, need_tot)
        # if need_tot:
        #     # print(p.get_x(), dim_x-1-p.get_width()/2)
        #     if p.get_x() == dim_x-1-p.get_width()/2:
        #         p.set_x(p.get_x()+0.7)#set_intra_interval

        if set_location:
            p.set_x(
                locator_list[i] * x_ticks_scale
                + x_ticks_displacement
                - p.get_width() / 2
            )
        else:
            p.set_x(
                (p.get_x() + p.get_width() / 2) * x_ticks_scale
                + x_ticks_displacement
                - p.get_width() / 2
            )

        # if p.get_x() > dim_xs[0][0] - 1 - p.get_width() / 2:
        #     p.set_x(p.get_x() + interval)

        # p.set_x(
        #     (p.get_x() + p.get_width() / 2) * x_ticks_scale
        #     + x_ticks_displacement
        #     - p.get_width() / 2
        # )

        h.append(p.get_height())
        if i < r:
            accumu_height.append(p.get_height())
        if data[i] >= 0.05:
            percent = f"{100 * data[i]:.0f}%"
            accumu_height, height = text_height(h, i, r, accumu_height)
            ax.annotate(
                percent,
                (p.get_x() + p.get_width() * 0.1, p.get_y() + p.get_height() * 0.4),
                fontsize=f_size[0] * 0.9,
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
    dim_xs: List[Tuple[int, bool]],
    dim_y: int,
    interval: float,
    title: str = None,
    ylabel: str = None,
    xlabel: str = None,
    legend_location: str = None,
    legend_col: int = 2,
    save_figure: bool = False,
    no_frame: bool = True,
    row_sum: int = None,
    x_ticks_scale: float = 0.5,
    x_ticks_displacement: float = 2,
    num_plots: int = 1,
    set_location: list = None,
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

    # tick_list = ax.get_xticks()
    # tick_list = [i if i<7 else i+1 for i in tick_list]
    # ax.xaxis.set_major_locator(ticker.FixedLocator([0, 0.5, 1, 1.5, 2, 2.5]))
    # for need_tot, dim_x in dim_xs:
    #     if need_tot:
    #         locator_list = [i if i<dim_x[0] else i+0.5 for i in list(ax.get_xticks())]
    # x ticks (center of patch)coordinate: [0, 1, 2, ..., num_bar*1]
    # print(ax.get_xticks())
    # locator_list = [
    #     i if i < dim_xs[0][0] else i + interval for i in list(ax.get_xticks())
    # ]

    if set_location:
        locator_list = set_location
    else:
        locator_list = ax.get_xticks()
    ax.xaxis.set_major_locator(
        ticker.FixedLocator(
            np.asarray(locator_list) * x_ticks_scale + x_ticks_displacement
        )
    )
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
    if legend_location:
        # works in 3.4.3 not work some previous version
        ax.get_legend().remove()
    # else:
    #     # if num_plots == 1:
    #     anchor = (0.5, 1.1)
    #     # else:
    #     #     anchor = ()
    #     handles, leg_labels = ax.get_legend_handles_labels()
    #     new_leg_labels = list()
    #     for name in leg_labels:
    #         # _, tu = name.strip("()").split(", ")
    #         tu = name
    #         new_leg_labels.append(tu)
    #     ax.legend(
    #         handles,
    #         new_leg_labels,
    #         bbox_to_anchor=anchor,
    #         fontsize=f_size[0] * 1.5,
    #         loc=legend_loc,
    #         ncol=legend_col,
    #         frameon=False,
    #     )

    # scale
    ax.autoscale()
    ax.set_autoscale_on(True)

    anchor = (0.5, 0.9)
    handles, leg_labels = ax.get_legend_handles_labels()
    new_leg_labels = list()
    for name in leg_labels:
        # _, tu = name.strip("()").split(", ")
        new_leg_labels.append(name)
    fig.legend(
        handles[0:dim_y],
        new_leg_labels[0:dim_y],
        ncol=dim_y,
        bbox_to_anchor=anchor,
        fontsize=f_size[0] * 1,
        loc=legend_location,
        frameon=False,
        labelspacing=0.1,
        handlelength=1,
        handletextpad=0.7,
        columnspacing=1,
        # prop={'legend.labelspacing':0.25}
    )
    # fig.suptitle("she woa dra niaou", x=0.5, y=1, fontsize=fig_size[0] * 2)
    try:
        fig.tight_layout(pad=1.2)
    except UserWarning:
        print(
            "Laushir, the most possible reason for this is that the question choice is too long\n\
            which lead to not enough bottom space for tight layout, so hai shtrolsha."
        )
    fig.autofmt_xdate()
    fig.savefig(save_figure)
    # figure settings
    # fig.tight_layout()
    # fig.autofmt_xdate()
    # if save_fig:
    #     fig.savefig(save_fig)
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


def percent_df(df: pd.DataFrame) -> pd.DataFrame:
    df_dic = dict()

    for label, row in df.iterrows():

        row = np.asarray(row)
        new_row = row / row.sum()
        df_dic[label] = new_row
    new_df = pd.DataFrame.from_dict(df_dic)
    col_name = [name for _, name in df.columns]
    new_df.index = col_name
    return new_df.transpose()


def percent_from_list_to_df(df: pd.DataFrame) -> pd.DataFrame:
    df_list = list()

    for _, row in df.iterrows():
        row = np.asarray(row)
        if row.sum() == 0:
            new_row = row * 0
        else:
            new_row = row / row.sum()
        df_list.append(new_row)
    new_df = pd.DataFrame(df_list)
    # print(df.columns)
    col_name = [name for name in df.columns]
    # print(df.columns)

    new_df.index = df.index
    new_df.columns = col_name

    return new_df


def recast_multi_to_one(
    filepath: str,
    ques: dict,
    col_xs: List[Tuple[str, bool, bool]],
    col_ys: Tuple[str, bool],
    replace_name_dict_ind: dict,
    replace_name_dict_col: dict,
) -> Tuple[List[Tuple[int, bool]], int, pd.DataFrame]:
    col_y, y_no_answer = col_ys
    p_dfs, dim_xs, dim_y = list(), list(), list()
    for col_x, need_total, x_no_answer in col_xs:
        p_df = prepare_df(
            filepath,
            col_x,
            col_y,
            ques,
            replace_name_dict_col,
            replace_name_dict_ind,
            need_total=need_total,
            need_x_null=x_no_answer,
            need_y_null=y_no_answer,
        )
        dim_xs.append((len(p_df.index), need_total))

        p_dfs.append(p_df)
    recast_df = pd.concat(p_dfs)
    dim_y = len(recast_df.columns)

    return dim_xs, dim_y, recast_df


def multi_bar_to_one_compare(
    filepath: str,
    structure_xml: dict,
    col_xs: List[Tuple[str, bool, bool]],
    col_ys: Tuple[str, bool],
    replace_name_dict_ind: dict = None,
    replace_name_dict_col: dict = None,
    save_figure: str = None,
    give_title: str = None,
    interval: float = 0.5,
    legend_location: str = "upper center",
    color: list = None,
    fig_size: tuple = (10, 5),
    x_ticks_displacement: float = 1,
    x_ticks_scale: float = 0.5,
    set_location: list = None,
) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    fig, ax = plt.subplots(figsize=fig_size)
    # plt.style.use("seaborn")
    _, ques = structure_xml["sections"], structure_xml["questions"]

    dim_xs, dim_y, recast_df = recast_multi_to_one(
        filepath,
        ques,
        col_xs,
        col_ys,
        replace_name_dict_ind,
        replace_name_dict_col,
    )

    if set_location:
        if len(set_location) != sum([i for i, _ in dim_xs]):
            raise Exception(
                f"Laushir, the num of bars ({sum([i for i, _ in dim_xs])}) are not match with \n\
                the num of location ({len(set_location)}) you give. Na hwedron."
            )

    if color is not None and len(color) != dim_y:
        raise Exception(
            f"Laushir, there need ({dim_y}) kinds of colors to make it as a color map list,\n\
            you dra only gives ({len(color)}) colors."
        )
    ax = percent_from_list_to_df(recast_df).plot.bar(
        ax=ax, stacked=True, width=0.03 * fig_size[0], color=color
    )
    # _, need_tot, _ = col_xs
    r_sum, ax = add_text_in_bar(
        ax,
        fig_size,
        recast_df,
        dim_y=dim_y,
        interval=interval,
        set_location=set_location,
        x_ticks_displacement=x_ticks_displacement,
        x_ticks_scale=x_ticks_scale,
    )

    bar_plot_settings(
        fig,
        ax,
        f_size=fig_size,
        title=give_title,
        legend_location=legend_location,
        row_sum=r_sum,
        dim_xs=dim_xs,
        dim_y=dim_y,
        interval=interval,
        x_ticks_displacement=x_ticks_displacement,
        x_ticks_scale=x_ticks_scale,
        set_location=set_location,
        save_figure=save_figure,
        # legend_col=int(len(p_df.columns))
    )
    # very good explain of how to share legend
    # refer to:
    # https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
    # anchor = (0.5, 0.82)
    # dim_y = len(p_df.columns)
    # tuple_labels = [ax.get_legend_handles_labels() for ax in axs]
    # tuple_labels = ax.get_legend_handles_labels()
    # handles, new_leg_labels = [sum(lol, []) for lol in zip(*tuple_labels)]
    # new_leg_labels = list()
    # for name in leg_labels:
    #     new_leg_labels.append(name)
    # print()

    # anchor = (0.5, 0.9)
    # handles, leg_labels = ax.get_legend_handles_labels()
    # new_leg_labels = list()
    # for name in leg_labels:
    #     # _, tu = name.strip("()").split(", ")
    #     new_leg_labels.append(name)
    # fig.legend(
    #     handles[0:dim_y],
    #     new_leg_labels[0:dim_y],
    #     ncol=dim_y,
    #     bbox_to_anchor=anchor,
    #     fontsize=fig_size[0] * 1.5,
    #     loc=legend_location,
    #     frameon=False,
    #     labelspacing=0.1,
    #     handlelength=1,
    #     handletextpad=0.7,
    #     columnspacing=1,
    #     # prop={'legend.labelspacing':0.25}
    # )
    # # fig.suptitle("she woa dra niaou", x=0.5, y=1, fontsize=fig_size[0] * 2)
    # try:
    #     fig.tight_layout(pad=1.2)
    # except UserWarning:
    #     print(f"Laushir, the most possible reason for this is that the question choice is too long\n\
    #         which lead to not enough bottom space for tight layout, so hai shtrolsha.")
    # fig.autofmt_xdate()
    # fig.savefig(save_figure)
    return fig, ax


# final version I will remove it, keep it just for easy testing, else I have to find the test file and others.
if __name__ == "__main__":
    structure_xml = read_lime_questionnaire_structure("../data_set/test_Oct.xml")
    multi_bar_to_one_compare(
        "../data_set/test_Oct.csv",
        structure_xml,
        # [("B7", False, True), ("A00", True, False)],
        [("A7", False, True)],
        ("A6", True),
        save_figure="../playground/tmp/new_only_graph0.png",
        give_title="laushir, bai leener, kutree zhe. dron !",
        legend_location="upper center",
        replace_name_dict_ind={
            "I don't want to answer this question": "NO COMMENT",
            "I don’t want to answer this question": "NO COMMENT",
            "Gender diverse (Gender-fluid)": "GD",
            "Other gender representations:": "Other",
        },
        replace_name_dict_col={
            "I don’t want to answer this question": "No Comment",
            "I don't want to answer this question": "No Comment",
            "Yes, it is okay for me and I agree with the data protection regulations for these sensitive questions": "Yes, dege",
            "No, I prefer not to see them": "No, budegir",
        },
        # set_location=[1, 2, 3, 5.5, 6.5, 7.5, 8.5, 9.5, 11.5, 12.5, 15],
        set_location=[0, 1, 2, 3, 5, 6, 7, 8],
        color=[
            "royalblue",
            "steelblue",
            "cornflowerblue",
            "dodgerblue",
            "purple",
            "lightsteelblue",
            "slategrey",
            # "deepskyblue",
            # "darkmagenta",
        ],
    )


# def compare_bar_plot(
#     fig: mpl.figure.Figure,
#     ax: mpl.axes.Axes,
#     filepath: str,
#     col_x: str,
#     col_y: str,
#     save_figure: str,
#     give_title: str = None,
#     legend_location: str = "upper center",
#     color: list = None,
#     display_noanswer: bool = True,
#     fig_size: tuple = (21, 11),
# ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
#     """
#     make comparision plot between different columns.

#     note: when dealing with groupby, method refer to
#     https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count-mean-etc-using-pandas-groupby
#     """
#     # fig, ax = plt.subplots(figsize=fig_size)
#     p_df = prepare_df(filepath, col_x, col_y)
#     if color is not None and len(color) != len(p_df.columns):
#         raise Exception(
#             f"lau shir, there need {len(p_df.columns)} kinds of colors as a list"
#         )
#     ax = percent_df(p_df).plot.bar(
#         ax=ax, stacked=True, width=0.06 * fig_size[1], color=color
#     )
#     r_sum, ax = add_text_in_bar(ax, fig_size, p_df)
#     bar_plot_settings(
#         fig,
#         ax,
#         f_size=fig_size,
#         title=give_title,
#         legend_loc=legend_location,
#         row_sum=r_sum,
#         save_fig=save_figure,
#         legend_col=int(len(p_df.columns)),
#     )
#     return fig, ax


# def multi_bar_compare(
#     filepath: str,
#     col_xs: List[Tuple[str, bool, bool]],
#     col_ys: Tuple[str, bool],
#     save_figure: str = None,
#     give_title: str = None,
#     legend_location: str = "upper center",
#     color: list = None,
#     display_noanswer: bool = True,
#     fig_size: tuple = (10, 5),
# ) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
#     if len(col_xs) == 1:
#         axs = [0] * len(col_xs)
#         fig, axs[0] = plt.subplots(len(col_xs), figsize=fig_size)
#     else:
#         # legend_location = 'lower right'

#         axs = [] * len(col_xs)
#         fig, axs = plt.subplots(1, len(col_xs), figsize=fig_size)
#     for i, col_x in enumerate(col_xs):
#         # fig, axs[i] = compare_bar_plot(fig,axs[i],filepath,col_x,col_y,color,fig_size)
#         p_df = prepare_df(filepath, col_x, col_ys)
#         dim_x = len(p_df.index)
#         if color is not None and len(color) != len(p_df.columns):
#             raise Exception(
#                 f"lau shir, there need {len(p_df.columns)} kinds of colors as a list"
#             )
#         axs[i] = percent_df(p_df).plot.bar(
#             ax=axs[i], stacked=True, width=0.02 * fig_size[1] * dim_x, color=color
#         )
#         r_sum, axs[i] = add_text_in_bar(axs[i], fig_size, p_df)

#         if i >= 0:
#             give_title = None
#             legend_location = None

#         bar_plot_settings(
#             fig,
#             axs[i],
#             f_size=fig_size,
#             title=give_title,
#             legend_loc=legend_location,
#             row_sum=r_sum,
#             # save_fig=save_figure)
#             # legend_col=int(len(p_df.columns))
#         )
#     # very good explain of how to share legend
#     # refer to:
#     # https://stackoverflow.com/questions/9834452/how-do-i-make-a-single-legend-for-many-subplots-with-matplotlib
#     anchor = (0.5, 0.96)
#     dim_y = len(p_df.columns)
#     tuple_labels = [ax.get_legend_handles_labels() for ax in axs]
#     handles, new_leg_labels = [sum(lol, []) for lol in zip(*tuple_labels)]
#     # new_leg_labels = list()
#     # for name in leg_labels:
#     #     new_leg_labels.append(name)
#     # print()

#     fig.legend(
#         handles[0:dim_y],
#         new_leg_labels[0:dim_y],
#         ncol=dim_y,
#         bbox_to_anchor=anchor,
#         fontsize=fig_size[0] * 1.5,
#         loc="upper center",
#         frameon=False,
#     )
#     fig.suptitle("give_title", x=0.5, y=1, fontsize=fig_size[0] * 2)
#     fig.tight_layout(pad=1.2)
#     fig.savefig("../playground/tmp/multiple_compare_bar_plot.png")
#     return fig, axs
