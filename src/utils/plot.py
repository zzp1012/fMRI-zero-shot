# import basic libs
import os
import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_bars(Y: dict, 
                       xticks: list,  
                       save_path: str, 
                       fname: str,
                       title: str,
                       xlabel: str, 
                       ylabel: str,
                       figsize: tuple = (10, 5),
                       ylim: list = None,
                       width: float = 0.2,
                       rot: int = 45) -> None:
    """
    Create a bar plot with multiple bars and a legend.

    Args:
        Y (dict): the dictionary of the curves.
        xticks (list): the x axis.
        title (str): the title of the figure.
        save_path (str): the path to save the figure.
        fname (str): the file name of the figure.
        xlabel (str): the label of x axis.
        ylabel (str): the label of y axis.
        ylim (list): the range of y axis.
        width (float): the width of the bar.
        rot (int): the rotation of the xticks.
    """

    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(xticks))

    # Plot each data series
    for i, (label, y) in enumerate(Y.items()):
        ax.bar(x + i*width, y, width=width, label=label)

    # set the y axis range
    if ylim is not None:
        ax.set_ylim(ylim)

    # Add labels, title, and legend
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation=rot, ha='right')
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(frameon=True, prop={'size': 10})

    # Automatically adjust the layout to prevent overlapping x-axis ticks
    fig.tight_layout()

    # save the fig
    path = os.path.join(save_path, fname)
    fig.savefig(path)
    plt.close()
