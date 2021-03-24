"""Script for testing antenna gain pattern functions, or whole KOSIA antenna pattern files."""
from sys import stderr
import numpy as np
from scipy import constants as consts
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator, LinearLocator
# pylint: disable=import-error
from open_source_i_n import antenna
from open_source_i_n.satellite import UlDlPair
from open_source_i_n.antenna.__init__ import BaseAntennaModel

R_EARTH = 6371000  # Radius of Earth (m)


def get_params(func):
    """Return parameters of a function."""
    return func.__code__.co_varnames[:func.__code__.co_argcount]


def slant(alt_km, elev):
    """
    Calculates slant distance of a circular orbit.

    Args:
        alt_km (float): Nominal circular altitude in km
        elev (float/ndarray): Elevation above the horizon

    Returns (float): Slant distance in kilometers.

    """
    r_e = R_EARTH
    return (-r_e * np.sin(elev * np.pi / 180) +
            np.sqrt(0.5) * np.sqrt(r_e**2 + 4 * r_e * (alt_km * 1000) + 2 *
                                   (alt_km * 1000)**2 + r_e**2 * np.cos((90 + elev) *
                                                                        (2 * np.pi / 180)))) / 1000


def plot_array_scatter(x_arr, y_arr, z_arr, title="", x_label="", y_label="", tight=False):  # pylint: disable=too-many-arguments
    """
    Quick 2D colormap plotter.

    Args:
        x_arr (array): Numpy array of x coordinates
        y_arr (array): Numpy array of y coordinates
        z_arr (array): Numpy array of z coordinates
        title (str): Plot title
        x_label (str): X axis label
        y_label (str): Y axis label
        tight (bool): Tight layout flag. Default is False.

    Returns: None

    """
    # Setup
    figure = plt.figure()
    sub_plot = figure.add_subplot(111)

    # Plot
    # pylint: disable=no-member
    surface = sub_plot.scatter(x_arr, y_arr, c=z_arr, cmap=cm.jet, linewidth=0.5)

    # Formatting
    if tight:
        figure.tight_layout()
    figure.colorbar(surface, ticks=LinearLocator())
    sub_plot.xaxis.set_major_locator(MaxNLocator(7))
    sub_plot.yaxis.set_major_locator(MaxNLocator(11))
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_func_as_colormap(func, grid_size=90, resolution=1.):
    """
    Create a colormap scatter plot from a function. Function can take 1 or 2 parameters.
    If function takes 1 parameter, plot will assume radial symmetry in two dimensions.
    If function takes 2 parameters, plot will pass x and y grid coordinates as normal.

    Args:
        func (function): Function to be plotted. Can take one (x) or two (x, y) parameters.
        grid_size (int): Size of coordinate grid (assumed as square).
        resolution (float): Resolution of coordinate points.

    Returns: None

    """
    z_arr = None

    # 3D plot: plot settings
    # pylint: disable=invalid-name
    xx, yy = np.meshgrid(np.arange(-grid_size, grid_size, resolution),
                         np.arange(-grid_size, grid_size, resolution))

    dimensions = len(get_params(func))
    if dimensions == 1:
        z_arr = func(np.sqrt((xx**2 + yy**2)))  # Plot pattern with radial symmetry
    elif dimensions == 2:
        z_arr = func(xx, yy)
    else:
        print(
            f"Error: Cannot produce colormap plot with a function of dimension '{dimensions}'. "
            f"Please specify a function with only 1 or 2 parameters.",
            file=stderr)

    if z_arr:
        plot_array_scatter(xx / HPHBW, yy / HPHBW, z_arr)


def plot_patt_list_2d(ant_patt_list, normalize=False):
    """
    Plots 2D line plot of a list of gain pattern functions. List must contain dictionaries
    specifiying the antenna gain pattern function to be plotted, and a code denoting what
    parameters the function takes.

    Codes:
    'A': Parameters are (gain, diameter, off-axis angle, frequency)
    'B': Parameters are (gain, diameter, off-axis angle, frequency, half power half beam width)
    'C': Parameters are (diameter, off-axis angle, frequency)

    Args:
        ant_patt_list: List of two-member dictionaries as follows:
            [{
            "func": Gain pattern function,
            "type": String, denoting the gain pattern is of type 'A'
            }]
        normalize (bool): If True, translates plot down by highest y value.

    Returns: None

    """
    # Vectorize patterns before numpy calcs
    vec = list(map(np.vectorize, [x["func"] for x in ant_patt_list]))
    for i in range(len(ant_patt_list)):
        ant_patt_list[i]["vec"] = vec[i]

    # Calculate points and plot
    for pattern in ant_patt_list:

        def pattern_func(off_axis_ang, patt):
            """Returns a function with proper inputs, according to the pattern's specified type."""
            # pylint: disable=no-else-return
            if patt["type"] == 'A':
                return patt["vec"](GAIN, DIAM, off_axis_ang, FREQ)
            elif patt["type"] == 'B':
                return patt["vec"](GAIN, DIAM, off_axis_ang, FREQ, HPHBW)
            elif patt["type"] == 'C':
                return patt["vec"](DIAM, off_axis_ang, FREQ)
            elif patt["type"] == 'D':
                return patt["vec"](GAIN, off_axis_ang)
            else:
                raise TypeError("Antenna pattern doesn't have recognizable parameters.")

        # Coordinate bounds
        l_bound = -179
        r_bound = 180
        step = 0.1

        # Set up coordinates
        data = np.array(
            [np.arange(l_bound, r_bound, step),
             np.array([0.0] * int((r_bound - l_bound) / step))])

        # Calculate y coordinates
        try:
            data[1] = pattern_func(data[0], pattern)
        except ValueError:  # For patterns that don't like negative separation angles.
            data[1] = pattern_func(np.abs(data[0]), pattern)

        # Normalize data
        if normalize:
            data[0] = data[0] / HPHBW
            data[1] -= max(data[1])

        plt.plot(data[0], data[1])

    # Plot settings
    plt.legend([pattern["func"].__name__ for pattern in ant_pat])
    plt.grid()
    plt.title("Antenna pattern summary")
    plt.show()


# pylint: disable=too-many-arguments,too-many-locals
def plot_kosia_antenna_file(plt_ax_tx,
                            plt_ax_rx,
                            altitude_km,
                            antenna_name,
                            sep_mode_ss='pointed',
                            nad_mode_ss='pointed',
                            sep_mode_es='pointed',
                            nad_mode_es='pointed',
                            plt_ax_cn=None):
    """
    Plots an entire KOSIA antenna file. Useful for validation. Shown are:
        - Plot of PSD over elevation
        - Plot of G/T over elevation
        - Plot of C/N over elevation

    A simplified calculation is performed assuming a satellite in circular orbit passing directly
    over a ground station from horizon to horizon.

    Pointing modes can be specified as needed to change the behaviour of the satellite.
    The following are permitted:
        - 'sweep': Feeds an array from 90 to 0 to 90 into the corresponding antenna function
        - 'pointed': Feeds an array of zeros into the corresponding antenna function
    This option is provided to help visualize what a pattern would look like if an antenna is
    normally pointed, is instead positioned nadir, and vice versa.

    Example: If you have a satellite TX antenna that is normally modelled as pointed to a ground
    station, you can specify sep_mode_ss='pointed' and/or nad_mode_ss='pointed' (depending on
    whether the satellite TX PSD function is dependent on either separation angle or nadir angle,
    as defined in your antenna pattern). The plots will show the correct expected outcome for the
    downlink PSD over elevation. However, you may want to consider experimenting by selecting
    'sweep' for sep_mode_ss or nad_mode_ss, as this will instruct the antenna pattern to behave
    as if it is pointed straight down. This simplification will let you see a clear
    reconstruction of your antenna off-axis gain pattern on your corresponding PSD plot (with
    free space path loss included), which can be very helpful for validating your pattern
    implementation.

    Args:
        plt_ax_tx: Matplotlib pyplot axis for the PSD plot.
        plt_ax_rx: Matplotlib pyplot axis for the G/T plot.
        altitude_km (float): Altitude (km) of presumed circular orbit.
        antenna_name (str): Name of the antenna file to be plotted.
        sep_mode_ss (str): Space station separation mode.
        nad_mode_ss (str): Space station nadir mode
        sep_mode_es (str): Earth station separation mode.
        nad_mode_es (str): Earth station nadir mode.
        plt_ax_cn: (Optional) Matplotlib pyplot axis for the C/N plot.

    Returns:

    """
    def set_mode(mode):
        """Set pointing modes according to string input."""
        if mode == 'sweep':
            output = np.abs(np.arange(-90., 90., 1.))  # np.repeat(0., len(elev))
        elif mode == 'pointed':
            output = np.repeat(0., len(elev))
        else:
            print(f"Invalid mode '{mode}'. Setting to 'pointed'.")
            output = np.repeat(0., len(elev))
        return output

    # Load antenna models and high level inputs
    frequency = UlDlPair(14.25e9, 12.0e9)
    model = antenna.load(antenna_name)
    ant = model.AntennaModel(frequency)

    # Define separate elevation arrays for calculations and plotting respectively
    elev1 = np.arange(1., 91., 1.)
    elev2 = np.arange(90., 0., -1.)
    elev = np.hstack((elev1, elev2))
    elev_plot = np.arange(0., 180., 1.)

    # Instantiate pointing modes
    sep_ss = set_mode(sep_mode_ss)
    nad_ss = set_mode(nad_mode_ss)
    sep_es = set_mode(sep_mode_es)
    nad_es = set_mode(nad_mode_es)

    # Vectorize the PSD and G/T functions of the antenna pattern
    vec_ul_tx = np.vectorize(ant.interfering_es_ul_psd)
    vec_ul_rx = np.vectorize(ant.victim_sat_ul_g_over_t)
    vec_dl_tx = np.vectorize(ant.interfering_sat_dl_psd)
    vec_dl_rx = np.vectorize(ant.victim_es_dl_g_over_t)

    # Perform prerequisite calculations
    distance = slant(altitude_km, elev) * 1000
    lam_dl = consts.c / frequency.dl
    lam_ul = consts.c / frequency.ul
    a_eff_dl = 10 * np.log10(lam_dl * lam_dl / (4 * np.pi))
    a_eff_ul = 10 * np.log10(lam_ul * lam_ul / (4 * np.pi))

    # Perform antenna pattern calculations
    psd_ul = vec_ul_tx(np.abs(elev), sep_es, nad_es, distance)
    g_t_ul = vec_ul_rx(np.abs(elev), sep_es, nad_es, distance)
    psd_dl = vec_dl_tx(np.abs(elev), sep_ss, nad_ss, distance)
    g_t_dl = vec_dl_rx(np.abs(elev), sep_ss, nad_ss, distance)

    # Plot PSD
    plt_ax_tx.plot(elev_plot, psd_ul, label=f"psd_ul {model.AntennaModel.name}")
    plt_ax_tx.plot(elev_plot, psd_dl, label=f"psd_dl {model.AntennaModel.name}")
    plt_ax_tx.legend()
    plt_ax_tx.set_title("Antenna PSD over elevation")

    # Plot G/T
    plt_ax_rx.plot(elev_plot, g_t_ul, label=f"g_t_ul {model.AntennaModel.name}")
    plt_ax_rx.plot(elev_plot, g_t_dl, label=f"g_t_dl {model.AntennaModel.name}")
    plt_ax_rx.legend()
    plt_ax_rx.set_title("Antenna G/T over elevation")

    # Plot C/N
    if plt_ax_cn:
        # Calculate C/N        FIXME: Atmospheric losses are NOT calculated
        cn_ul = psd_ul + g_t_ul - 10 * np.log10(consts.k) + a_eff_ul
        cn_dl = psd_dl + g_t_dl - 10 * np.log10(consts.k) + a_eff_dl

        # Plot
        plt_ax_cn.plot(elev_plot, cn_ul, label=f"cn_ul {model.AntennaModel.name}")
        plt_ax_cn.plot(elev_plot, cn_dl, label=f"cn_dl {model.AntennaModel.name}")
        plt_ax_cn.legend()
        plt_ax_cn.set_title("Antenna C/N over elevation")

    # Apply grids as a final step.
    plt_ax_tx.grid()
    plt_ax_rx.grid()
    plt_ax_cn.grid()


# pylint: disable=pointless-string-statement
if __name__ == '__main__':
    """Plot gain patterns"""
    # Antenna inputs
    GAIN = 34.1
    FREQ = 14.25e9
    LAM = consts.c / FREQ
    EFF = 0.55
    K = 67.6  # reflector constant
    DIAM = (LAM / consts.pi) * np.sqrt(np.power(10, GAIN / 10) / EFF)
    HPHBW = K * LAM / DIAM / 2
    print("D/lambda = %s" % (DIAM / LAM))

    # Add antenna patterns here
    ant_pat = [{
        "func": BaseAntennaModel.gain_app8_es,
        "type": 'A'
    }, {
        "func": BaseAntennaModel.gain_s1528_sat,
        "type": 'B'
    }, {
        "func": BaseAntennaModel.gain_s1428_es,
        "type": 'C'
    }, {
        "func": BaseAntennaModel.gain_s580_6_es,
        "type": 'D'
    }]

    plot_patt_list_2d(ant_pat, True)
    """Plot entire KOSIA antenna files"""
    # # Option 1: One plot per file
    # tx_fig = plt.figure()
    # tx_ax = tx_fig.add_subplot(111)
    # rx_fig = plt.figure()
    # rx_ax = rx_fig.add_subplot(111)
    # plot_antenna_file(tx_ax, rx_ax, 575, 'template_standard',
    #                   sep_mode_ss='pointed',
    #                   nad_mode_ss='pointed',
    #                   sep_mode_es='pointed',
    #                   nad_mode_es='pointed')
    # plot_antenna_file(tx_ax, rx_ax, 575, 'template_custom',
    #                   sep_mode_ss='sweep',
    #                   nad_mode_ss='pointed',
    #                   sep_mode_es='sweep',
    #                   nad_mode_es='pointed')

    # Option 2: Multiple plots superimposed
    fig, axes = plt.subplots(nrows=1, ncols=3)
    fig.set_size_inches(16, 4)
    plot_kosia_antenna_file(axes[0],
                            axes[1],
                            575,
                            'template_standard',
                            sep_mode_ss='pointed',
                            nad_mode_ss='pointed',
                            sep_mode_es='pointed',
                            nad_mode_es='pointed',
                            plt_ax_cn=axes[2])
    plot_kosia_antenna_file(axes[0],
                            axes[1],
                            575,
                            'template_custom',
                            sep_mode_ss='sweep',
                            nad_mode_ss='pointed',
                            sep_mode_es='sweep',
                            nad_mode_es='pointed',
                            plt_ax_cn=axes[2])

    fig.set_size_inches(30, 8)
    axes[0].grid()
    axes[1].grid()
    axes[2].grid()
