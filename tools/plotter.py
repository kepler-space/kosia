"""This script plots I/N or C/I+N for any number of .simdata files input as filepaths"""
import os
from numpy import histogram, cumsum, linspace, load, repeat, append
import matplotlib.pyplot as plt
from matplotlib import rcParams


rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))
# pylint: disable=import-error
from open_source_i_n.config import Config
from open_source_i_n.utils import open_file_dialog

colors = [
    'red', 'blue', 'green', 'goldenrod', 'cyan', 'magenta', 'black', 'salmon', "olive", "teal",
    "cornflowerblue", "mediumvioletred", "mediumaquamarine", "blueviolet", "brown", "burlywood",
    "cadetblue", "chartreuse", "coral", "darkgreen", "darkred", "darkorchid", "darkslateblue",
    "darkseagreen", "deeppink", "forestgreen", "greenyellow", "indianred", "khaki", "lightblue",
    "lightgreen", "lightpink", "lightskyblue", "lightseagreen", "lightsalmon", "mediumvioletred",
    "moccasin", "orange", "orangered", "rebeccapurple", "sienna", "steelblue", "tomato",
    "turquoise", "yellowgreen", "slategray", "silver", "sandybrown", "royalblue", "darkslategray",
    "rosybrown", "thistle", "plum", "maroon", "lightgray"
]



def generate_stats(data,data_amt , args):
    """
    Takes a set of simulation data (e.g. I/N values) and returns information required to plot the
    data as a cumulative density function.

    Args:
        data: Data loaded from a Simdata file.
        data_amt (string): choice between plotting data when visible or at all timsteps
        args: simulation parameters form the batch file

    Returns: Bins and CDFs

    """
    # Calculate stats
    bins_plt = []
    cdfs = []
    if len(data) == 0:
        return bins_plt, cdfs

    if data_amt == 'All':
        duration = 86400*float(args['duration'])
        tstep = float(args['granularity'])
        total_tsteps = int(duration/tstep)
        delta_tsteps = total_tsteps - len(data)
        data = append(data,repeat(-100,delta_tsteps))
    no_bins = 100
    step = (data.max() - data.min()) / (no_bins - 1)
    bins = linspace(data.min(), data.max() + step, no_bins + 1)
    hist = histogram(data, bins)

    vis_percent = (100 / len(data)) * hist[0]
    cdfs = cumsum(vis_percent[::-1])[::-1]
    bins_plt = hist[1][:-1]
    return bins_plt, cdfs


# pylint: disable=too-many-locals
def plot_filepaths(file_paths,data_amt, xlim=None, ylim=None):
    """
    Plots I/N for any number of .npy files input as filepaths.

    Args:
        file_paths (tuple): Tuple of file paths as strings.
        data_amt (string): choice between plotting data when visible or at all timsteps
        xlim (tuple): Tuple setting the plot x limits, e.g. (-54, 30)
        ylim (tuple): Tuple setting the plot y limits, e.g. (0.0001, 100)

    Returns: None

    """

    # Simulation specifications
    names = [path.split("/")[-1].split(".")[0].lstrip("_") for path in file_paths]

    plt_title = ""
    plt.figure(figsize=(10, 5))
    plt.axvline(x=-12.2, linestyle="--", color="purple")
    plt.axvline(x=0, linestyle="--", color="navy")
    names = ["-12.2 dB"] + ["0 dB"] + names

    # Plot each line
    for color_no, file in enumerate(file_paths):
        print("Opening: ", file.split("\\")[-1])
        with open(file, "rb") as f:
            print("Loading data...")
            i_n_data = load(f, allow_pickle=True)
            percents_inst = load(f, allow_pickle=True)
            bins_inst = load(f, allow_pickle=True)
            args = load(f, allow_pickle=True).tolist()
            props = load(f, allow_pickle=True).tolist()
            try:
                funcs = load(f, allow_pickle=True).tolist()
                vic_antenna_file = load(f, allow_pickle=True)
                vic_antenna_file = str(vic_antenna_file)
                inter_antenna_file = load(f, allow_pickle=True)
                inter_antenna_file = str(inter_antenna_file)
            except:
                print("Older version of SimData")
                vic_antenna_file = 'victim antenna file not present in file'
                inter_antenna_file = 'interfering antenna file not present in file'



        cfg = Config(args)
        lbl = (cfg.name, str(cfg.duration), str(cfg.gs.lat), str(cfg.gs.lon), file.split("\\")[-1])
        bins, cdfs = generate_stats(i_n_data,data_amt , args)

        print("Plotting...")
        file_nm = file.split("/")[-1].split(".")[0].lstrip("_")
        bins_plt = bins
        plt.plot(bins_plt, cdfs, color=colors[color_no], label=file_nm)

    # Plot settings
    plt.legend(names)
    plt.minorticks_on()
    plt.xlabel("Ratio (dB)")
    plt.ylabel("Percent of time")
    plt.yscale("log")
    plt.grid(which="both", axis="both")
    plt.title(plt_title)
    if xlim:
        plt.xlim(xlim)
        plt.xticks(range(xlim[0], xlim[1], 2))
    if ylim:
        plt.ylim(ylim)
    plt.show()


if __name__ == '__main__':
    folder = os.getcwd()
    paths = open_file_dialog(folder, file_types=[("Simdata", ".simdata")], plural=True)
    # This data amount decides if the data should be analyzed at all timesteps
    # or just when there are visible satellites
    data_amt = 'Vis' # can be 'Vis' or 'All'

    if paths:
        plot_filepaths(paths,data_amt)
