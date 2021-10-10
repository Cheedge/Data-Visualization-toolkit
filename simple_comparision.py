import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from typing import List, Tuple


def prepare_df(file: str, col_x: str, col_y: str, need_total: bool=False) -> pd.DataFrame:
    """
    prepare a dataframe for column x vs. column y.
    """
    df = pd.read_csv(file, usecols=[col_x, col_y]).fillna('no_answer')
    g_df = df.groupby([col_x, col_y]).size().reset_index(name='count')
    p_df = g_df.pivot(index=col_x, columns=col_y).fillna(0)
    if need_total:
        p_df = p_df.append(p_df.sum().rename('total'))
    return p_df

def add_text_in_bar(ax: mpl.axes.Axes, f_size: tuple, p_df: pd.DataFrame)-> Tuple[list, mpl.axes.Axes]:
    """
    add the percentage and other text content into each bar.
    """
    data, r_sum, c_data, h = list(), list(), list(), list()
    for _, row in p_df.iterrows():
        r_sum.append(sum(list(row)))
    for i, (_, col) in enumerate(p_df.iteritems()):
        c_data.append(np.asarray(col)/np.asarray(r_sum))
    r = len(col)
    data = np.asarray(c_data).flatten()

    accumu_height = list()
    for i, p in enumerate(ax.patches):
        h.append(p.get_height())
        if i<r:
            accumu_height.append(p.get_height())
        if data[i] >= 0.1:
            percent = f'{100 * data[i]:.0f}%'
            accumu_height, height = text_height(h, i, r, accumu_height)
            ax.annotate(percent,
                        (p.get_x(),
                        height),
                        fontsize=f_size[0]*1,
                        fontweight='heavy',
                        color='white')
    return r_sum, ax

def text_height(h: List[float], i: int, r: int, accumu_height: List[float])-> Tuple[List[float], float]:
    """
    calculate the position of the text in every patch of each bar
    """
    height = 0
    if i<r:        
        height = h[i]*0.5
    else:
        accumu_height[i%r] += h[i]
        height = accumu_height[i%r] - h[i]/2.0

    return (accumu_height, height)

def bar_plot_settings(fig: mpl.figure.Figure, ax: mpl.axes.Axes, \
        f_size: tuple, title: str=None, ylabel: str=None, xlabel: str=None, \
        legend_loc: str=None, legend_col: int=2, save_fig: bool=False, no_frame: bool=True, \
        row_sum: int=None) -> Tuple[mpl.figure.Figure, mpl.axes.Axes]:
    """
    some figure settings in simple bar plot.
    """
    # lable font
    label_font = {'fontsize': f_size[0]*1.2, 'fontfamily':'DejaVu Sans', 'fontstyle': 'oblique'}

    # title font
    title_font = {'fontsize': f_size[0]*1.5, 'fontfamily':'DejaVu Sans', 'fontstyle': 'oblique'}

    # ticks
    ax.tick_params(direction='out', labelsize= f_size[0]*1.2)

    # label
    ax.set_ylabel(ylabel, **label_font)
    ax.set_xlabel(xlabel, **label_font)

    # ticklabels
    if row_sum:
        tic_label = list(ax.get_xticklabels())
        new_tic_label = set_ticklabels(labels=tic_label, r_sum=row_sum)
        ax.set_xticklabels(new_tic_label, **label_font)

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

    # legend
    if legend_loc is None:
        # works in 3.4.3 not work some previous version
        ax.get_legend().remove()
    else:
        handles, leg_labels = ax.get_legend_handles_labels()
        new_leg_labels = list()
        for name in leg_labels:
            _, tu = name.strip('()').split(', ')
            new_leg_labels.append(tu)
        ax.legend(handles, new_leg_labels, fontsize=f_size[0]*1.5, \
            loc=legend_loc, ncol=legend_col, frameon=False)
  
    # scale
    ax.autoscale()
    ax.set_autoscale_on(True)

    # ticker: can be deleted later
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # figure settings
    fig.tight_layout()
    fig.autofmt_xdate()
    if save_fig:
        fig.savefig(save_fig)
    return fig, ax

def set_ticklabels(labels: list, r_sum: int)-> List[str]:
    """
    deal with the xtick label text
    """
    tic_l = list()
    tic_l = [i.get_text() + '(' + str(int(j)) + ')' if i.get_text()!='-oth-' else 'Other' for i, j in zip(labels, r_sum)]
    return tic_l


def compare_bar_plot(
        filepath: str,
        col_x: str,
        col_y: str,
        save_figure: str,
        give_title: str=None,
        legend_location: str='upper left',
        color: str=None,
        display_noanswer: bool=True,
        fig_size: tuple=(21, 11))-> mpl.axes.Axes:
        """
        make comparision plot between different columns.

        note: when dealing with groupby, method refer to
        https://stackoverflow.com/questions/19384532/get-statistics-for-each-group-such-as-count-mean-etc-using-pandas-groupby
        """
        fig, ax = plt.subplots(figsize=fig_size)
        p_df = prepare_df(filepath, col_x, col_y)
        ax = p_df.plot.bar(stacked=True, ax=ax)
        r_sum, ax = add_text_in_bar(ax, fig_size, p_df)
        bar_plot_settings(fig, ax, f_size=fig_size, title=give_title, legend_loc=legend_location, row_sum=r_sum, \
            save_fig=save_figure)
        return fig, ax

# final version I will remove it, keep it just for easy testing, else I have to find the test file and others.
if __name__=='__main__':
    compare_bar_plot('../../../DataScience/Matplotlib/test_Oct08.csv', 'A3', 'A6',\
        save_figure='./simple_compare_bar_plot.png')