import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import pickle

# import from internal libs
from utils.plot import plot_multiple_bars

def plot_pred_y_means(save_path: str,
                      filename: str,
                      data: dict):
    """plot the pred_y_means.

    Args:
        save_path (str): the path of saving the plot.
        filename (str): the filename of the plot.
        data (dict): the data to plot.
    
    Returns:
        None
    """
    print("plot the pred_y_means.")

    # get the verbs
    with open("../data/verbs.pkl", "rb") as f:
        verbs = pickle.load(f)

    plot_multiple_bars(
        Y=data, 
        xticks=verbs,  
        save_path=save_path, 
        fname=filename,
        title=None,
        xlabel="Verbs", 
        ylabel="Frequency",
        figsize=(15, 5)
    )
    
    print("plot the pred_y_means done.")


def main():
    parser = argparse.ArgumentParser(
        description="plot the metrics of the model.")
    parser.add_argument("--data_path", "-d", default=None, type=str,
                        help="the path of the data.")
    
    args = parser.parse_args()

    # load the data
    assert args.data_path.endswith(".pkl"), \
        "the data must be pkl file."
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)
    
    # get the filename
    filename = os.path.basename(args.data_path)
    # get the save path
    save_path = os.path.dirname(args.data_path)

    if "pred_y_means" in filename:
        plot_pred_y_means(save_path, filename.replace(".pkl", ".png"), data)
    else:
        raise ValueError("unknown filename.")

if __name__ == "__main__":
    main()