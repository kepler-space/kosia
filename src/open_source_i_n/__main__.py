"""C/I+N and I/N Interference Top Level Module"""
import argparse
from math import ceil
from time import monotonic
from sys import stderr
from os import listdir
from numpy import cumsum, histogram, save as np_save, dstack, vstack, repeat, linspace
from pandas import DataFrame
import matplotlib.pyplot as plt

# pylint: disable=import-error
from skyfield.constants import DAY_S
from open_source_i_n import antenna, geometry, simulator, config
from open_source_i_n.interval import Interval, partition_interval
from open_source_i_n.satellite import Constellation, UlDlPair, VicInterPair
from open_source_i_n.utils import relative_path, read_config_file, check_folder, print_config, \
    get_setting, Path, open_file_dialog, root_path_src, open_folder_dialog

TAGS = ['I_N', 'C_I_N', 'C_I', 'C_N']

# Global parameters
settings_path = relative_path('settings.ini')
settings = read_config_file(relative_path('settings.ini'))
save_folder = relative_path(settings['output_folder'])
cfg = []

# High level checks
check_folder(save_folder)  # Check if output older exists


# Private functions
class Result:
    """
    A Result object stores:
        - Simulation data of a single type (e.g. C/N, C/I, or I/N).
        - Simulation parameters used to generate the data, including all antenna model info,
        TLEs, and config inputs.
        - Plotting scripts for displaying the data.
    """
    def __init__(self, data_uldl_pair, tag, sim, args):     # pylint: disable=redefined-outer-name
        self.data = data_uldl_pair
        self.tag = tag
        self.sim = sim
        self.args = args
        self.bins, self.cdfs = self.__generate_stats()

    def __generate_stats(self):
        """
        Converts simulation data to format necessary to plot as a cumulative distribution function.

        Returns: Bins, CDFs

        """
        # Calculate stats
        bins_plt = UlDlPair()
        cdfs = UlDlPair()
        for uldl in self.data.__fields__:
            if len(self.data[uldl]) == 0:
                cdfs[uldl] = []
                bins_plt[uldl] = []
                break

            no_bins = 100
            step = (self.data[uldl].max() - self.data[uldl].min()) / (no_bins - 1)
            bins = linspace(self.data[uldl].min(), self.data[uldl].max() + step, no_bins + 1)
            binned = histogram(self.data[uldl], bins)

            vis_percent = (100 / len(self.data[uldl])) * binned[0]
            cdfs[uldl] = cumsum(vis_percent[::-1])[::-1]
            bins_plt[uldl] = binned[1][:-1]
        return bins_plt, cdfs

    def plot(self):
        """Plots object data. Uplinks and downlinks are denoted in blue/red respectively."""
        _plot(self.sim.constellations, self.args.duration, self.bins, self.cdfs, self.args.gs,
              UlDlPair('blue', 'red'), self.tag, self.args.name, self.args.save_data)

    def save_data(self):
        """Save results to SIMDATA file."""
        _save_results(self.sim.constellations, self.args.duration, self.data, self.cdfs, self.bins,
                      self.tag, self.args.name)


# pylint: disable=too-many-arguments
def _plot(constels,
          sim_days,
          bins_plt,
          cdfs,
          position,
          colors,
          mode_tag,
          sim_name,
          save_data=False):
    """Plot cumulative distribution graph of interference simulation results.

    Args:
        constels (VicInterPair(T, T)): T is Constellation.
        sim_days (float): Number of days simulated.
        cdfs: CDF data.
        bins_plt: Bins for CDF data.
        position (LatLon): Position of the groundstation.
        colors (UlDlPair(T, T)): T is a string naming a matplotlib color.
        save_data (bool): True to save the graph to disk as a png.
    """
    for uldl in bins_plt.__fields__:
        if len(bins_plt[uldl]) == 0:
            print("Plotting Error. Data array is empty.",
                  mode_tag,
                  uldl,
                  bins_plt[uldl],
                  file=stderr)
            break

        plt.figure()
        plt.semilogy(bins_plt[uldl], cdfs[uldl], color=colors[uldl])
        plt.xlabel(f"{mode_tag} (dB)")
        plt.ylabel("Percent of time")
        plt.yscale("log")
        plt.grid(which="both", axis="both")
        plt.xlim((min(bins_plt[uldl]) - 1, max(bins_plt[uldl]) + 1))
        plt.ylim((0.0001, 200))
        plt.minorticks_on()
        plt.title(f'{uldl.upper()} {mode_tag} of {constels.inter} '
                  f'for {sim_days:.4f} days, GS= ({position})')

        if save_data:
            plt.savefig(save_folder / f'{sim_name}_{uldl}_{constels.inter}_'
                        f'{mode_tag}_{sim_days:.4f}_days.png')


# pylint: disable=too-many-arguments,redefined-outer-name
def _save_results(constels, sim_days, data, cdf, bins_plt, mode_tag, sim_name):
    """Save uplink and downlink interference results to disk.

    Args:
        constels (VicInterPair(T, T)): T is Constellation.
        sim_days (float): Number of days simulated.
        data (VicInterPair(T, T)): T is [dB of interference for timestep].
        cdf: CDF data.
        bins_plt: Bins for CDF data.
        mode_tag: Specification of dataset type (I/N, C/N, C/I, C/I+N)
        sim_name: User name for simulation

    Returns:

    """

    for uldl in data.__fields__:
        filename = save_folder / f'{sim_name}_{uldl}_{constels.inter}_' \
                                 f'{mode_tag}_{sim_days:.4f}_days.simdata'
        with open(filename, "wb") as file:
            np_save(file, data[uldl])
            np_save(file, cdf[uldl])
            np_save(file, bins_plt[uldl])
            np_save(file, cfg.args)
            np_save(
                file, {
                    "vic_props": constels['vic']._antenna_model.props,
                    "inter_props": constels['inter']._antenna_model.props
                })


# pylint: disable=redefined-outer-name
def _configure_sim(args):
    """Construct constellation objects."""
    constels = VicInterPair()
    frequency = UlDlPair(args.freq_ul, args.freq_dl)  # UL/DL frequencies.
    for vicinter in constels.__fields__:
        # Create a constructor using all the CLI arguments starting with vic_ or inter_.
        prefix = vicinter + '_'
        init_args = {
            arg[len(prefix):]: val
            for arg, val in vars(args).items() if arg.startswith(prefix)
        }
        constels[vicinter] = Constellation(frequency=frequency, **init_args)

    # Simulation Parameters
    return simulator.InterferenceSim(frequency, constels, args.gs, args.parallel)


# Runtime functions
# pylint: disable=too-many-locals
def main(args):
    """Main launcher of simulation."""
    print_config(cfg)

    # Set up simulation interval
    sim_interval = Interval(0, args.duration * DAY_S, args.granularity)  # Simulation interval (s).

    # Configure/run simulation
    sim = _configure_sim(args)

    # Time this block of code.
    simulation_start_time = monotonic()

    # Run simulation
    result_list = []
    for interval in partition_interval(*sim_interval, args.sim_block_size, sim_interval.step):
        result_list.append(sim.run(interval))

    # If debug is on, consolidate data, then extract link budget information and print to csv.
    if sim.debug:
        # End timer
        elapsed_s = monotonic() - simulation_start_time
        print("\nElapsed time", elapsed_s)

        # Extract data and link budgets, then separate.
        master_block = []
        budget = UlDlPair([], [])
        for sim_block in result_list:
            for partition in sim_block:
                if partition[0].size:
                    master_block.append(partition[0])
                    budget.ul.append(partition[1].ul)
                    budget.dl.append(partition[1].dl)

        try:
            master_block = dstack(master_block)
        except ValueError as val_err:
            print("""Error: No data detected. If you are using a small number of satellites,
                try extending the simulation duration, and ensure your satellites will be in view 
                of your ground station at some point throughout the simulation.""", file=stderr)
            raise SystemExit from val_err
        budget.ul = vstack(budget.ul)
        budget.dl = vstack(budget.dl)

        # Write link budget to file.
        header = [
            'Time(UTC)', 'Sep. Angle', 'Nad. Angle', 'Rcvd.Freq.', 'Rcvr.Gain', 'Tequiv',
            'g/T(peak)', 'Car Xmtr gain', 'Car EIRP', 'Car Range (km)', 'Car Elev (deg)',
            'Car Azim (deg)', 'Car Rcvd. Iso.', 'Car Carrier Power', 'Car Flux Density',
            'Car Free Space Loss', 'Car Atmos Loss', 'Car g/T (actual, off-axis, with losses)',
            'Car Bandwidth Tx', 'Car Bandwidth Rx', 'Car Rcvd. PSD', 'C/N', 'Int Xmtr gain',
            'Int EIRP', 'Int Range (km)', 'Int Elev (deg)', 'Int Azim (deg)', 'Int Rcvd. Iso.',
            'Int Carrier Power', 'Int Flux Density', 'Int Free Space Loss', 'Int Atmos Loss',
            'Int g/T (actual, off-axis, with losses)', 'Int Bandwidth Tx', 'Int Bandwidth Rx',
            'Int BW Overlap', 'Int Rcvd. PSD', 'I/N', 'C/N+I', 'C/I'
        ]

        ul_budget = DataFrame(budget.ul, columns=header)
        dl_budget = DataFrame(budget.dl, columns=header)

        # Convert any numeric columns to float.
        for col in ul_budget.columns:
            try:
                ul_budget[col] = ul_budget[col].astype(float)
                dl_budget[col] = dl_budget[col].astype(float)
            except ValueError:  # Some columns, such as 'Time(UTC)', cannot be converted to float.
                pass

        # Output to csv (this is much faster than outputting to a .xlsx, especially for large sims)
        ul_budget.to_csv(save_folder / f'{args.name}_ulbudget.csv', chunksize=1000000)
        dl_budget.to_csv(save_folder / f'{args.name}_dlbudget.csv', chunksize=1000000)

    # If debug is off, simply consolidate data.
    else:
        # End timer
        elapsed_s = monotonic() - simulation_start_time

        master_block = dstack(result_list)

        print("\nElapsed time", elapsed_s)

    # Parse the raw data into UlDlPair(s). Filter out any null timesteps.
    outputs = [UlDlPair([], []) for i in range(master_block.shape[0])]
    for i, _ in enumerate(outputs):
        outputs[i].ul = master_block[i, 0, :][master_block[i, 0, :] != 0]
        outputs[i].dl = master_block[i, 1, :][master_block[i, 1, :] != 0]

    # Output results
    results = list(
        map(lambda x, y, z, a: Result(x, y, z, a), outputs, TAGS, repeat(sim, len(outputs)),
            repeat(args, len(outputs))))

    for i in results:
        i.plot()
        i.save_data()

    print("\nTotal time after writing to files", monotonic() - simulation_start_time)
    print(f"Saved to {save_folder.absolute()}.")

    return results


def get_arg_parser():
    """Return an argument parser for command-line arguments.

    Returns:
        ArgumentParser: Argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration",
                        type=float,
                        help="Duration of simulation in days.",
                        required=True)
    # To Seconds
    parser.add_argument("--sim_block_size",
                        type=lambda x: ceil(float(x) * DAY_S),
                        help="Number of days to simulate at once.",
                        required=True)
    parser.add_argument("--granularity",
                        type=int,
                        help="Level of time granularity, in seconds.",
                        default=1)
    parser.add_argument("--save_data", help="Save the data.", default=True, action='store_true')
    parser.add_argument("--no_save_data",
                        help="Don't save output data.",
                        dest="save_data",
                        action='store_false')
    parser.add_argument("--gs",
                        type=geometry.LatLon.from_string,
                        help="GS latitude and longitude as lat,lon.",
                        required=True)
    # To Hz
    parser.add_argument("--freq_dl",
                        type=lambda x: float(x) * 1e9,
                        help="Downlink freq in GHz.",
                        required=True)
    # To Hz
    parser.add_argument("--freq_ul",
                        type=lambda x: float(x) * 1e9,
                        help="Uplink freq in GHz.",
                        required=True)
    parser.add_argument("--vic_tle_file",
                        help="TLE filename for the victim constellation.",
                        type=relative_path,
                        required=True)
    parser.add_argument("--vic_min_el",
                        type=float,
                        help="Minimum elevation for the victim constellation Earth station.",
                        required=True)
    parser.add_argument("--vic_geo_angle",
                        type=float,
                        help="Victim constellation Geo exclusion angle.",
                        required=True)
    parser.add_argument("--vic_tracking_strat",
                        type=Constellation.TrackingStrategy,
                        help="Victim constellation tracking strategy.",
                        required=True)
    parser.add_argument("--vic_module",
                        type=antenna.load,
                        help="Uplink victim satellite pattern.",
                        required=True)
    parser.add_argument("--vic_const_name",
                        help="Victim constellation name (for graphs).",
                        required=True)
    parser.add_argument('--vic_opt_args', help="Optional victim antenna parameters.")
    parser.add_argument("--vic_fixed_params",
                        type=lambda x: tuple(map(float, x.split(','))),
                        help="If the victim antenna will be 'stuck' in a given pointing direction, "
                             "please specify the elevation and azimuth in degrees, separated by "
                             "a comma (e.g. '30,145').",
                        default="")
    parser.add_argument("--inter_tle_file",
                        type=relative_path,
                        help="TLE filename for the interfering constellation.",
                        required=True)
    parser.add_argument("--inter_min_el",
                        type=float,
                        help="Minimum elevation for interfering constellation Earth station.",
                        required=True)
    parser.add_argument("--inter_geo_angle",
                        type=float,
                        help="Interfering constellation geo exclusion angle.",
                        required=True)
    parser.add_argument("--inter_tracking_strat",
                        type=Constellation.TrackingStrategy,
                        help="Interfering constellation tracking strategy.",
                        required=True)
    parser.add_argument("--inter_module",
                        type=antenna.load,
                        help="interfering uplink antenna pattern",
                        required=True)
    parser.add_argument("--inter_const_name",
                        help="Interfering constellation name (for graphs).",
                        required=True)
    parser.add_argument('--inter_opt_args', help="Optional interfering antenna parameters.")
    parser.add_argument(
        "--parallel",
        type=int,
        help="Number of threads to use. The value '0' means let the program choose.",
        default=0)
    parser.add_argument(
        "--name",
        type=str,
        help="Optional name for the simulation run. Is added to any output filenames.",
        default="")

    return parser


def run_batch_list(path):
    """
    Runs all batch or shell files in a specified folder path in sequence.

    Args:
        path (Path): Path object specifying folder to scan for batch/shell files.

    Returns: None/

    """
    # Fetch settings
    linux_mode = get_setting(settings, 'input_filetype', ('shell', 'batch'))
    accept_errors = get_setting(settings,
                                'route_errors_to_log', ('True', 'T', '1', 'False', 'F', '0'),
                                is_boolean=True)

    if linux_mode == 'shell':
        file_ext = '.sh'
    elif linux_mode == 'batch':
        file_ext = '.bat'
    else:
        print("Please correct setting 'input_filetype'.", file=stderr)
        raise SystemExit

    # Scan folder for files
    file_list = [i for i in listdir(path) if i.endswith(file_ext)]
    global cfg              # pylint: disable=invalid-name,global-statement

    # Run through files sequentially, routing any errors to a text file if accept_errors is True.
    for file_name in file_list:
        file_path = path / file_name

        if accept_errors:
            try:
                if linux_mode == 'shell':
                    cfg = config.load_shell(file_path)
                else:
                    cfg = config.load_batch(file_path)
                main(cfg)
            except Exception as ex:                     # pylint: disable=broad-except
                print(ex.args)
                log_path = file_path.__str__().rstrip(file_ext) + ' ERROR MESSAGE.txt'

                with open(log_path, 'w') as log_file:
                    log_file.writelines(ex.args)
                log_file.close()
        else:
            if linux_mode == 'shell':
                cfg = config.load_shell(file_path)
            else:
                cfg = config.load_batch(file_path)
            main(cfg)


if __name__ == '__main__':
    # Load run_mode from settings
    try:
        RUN_MODE = settings['run_mode']
    except KeyError("No run_mode specified. Please specify a run_mode in settings.ini"):
        RUN_MODE = None

    # Start program based on run_mode
    if RUN_MODE == 'cli':
        print("RUN MODE:", RUN_MODE)
        # Run from Command Line Interface
        arg_parser = get_arg_parser()
        args = arg_parser.parse_args()
        cfg = config.load_from_args(args)
        main(cfg)
    elif RUN_MODE == 'batch_select':
        # Select a batch or shell file from a file dialog prompt.
        print("RUN MODE:", RUN_MODE)
        batch_path = Path(open_file_dialog(root_path_src.__str__()))
        cfg = config.load_batch(batch_path)
        out = main(cfg)
    elif RUN_MODE == 'batch_list':
        # Specify a folder containing batch or shell files. These will be run in sequence.
        print("RUN MODE:", RUN_MODE)
        folder_specified = get_setting(settings, 'batch_list_folder', optional=True)
        if folder_specified:
            batch_list_folder = relative_path(folder_specified)
        else:
            batch_list_folder = relative_path(open_folder_dialog(root_path_src))
        run_batch_list(batch_list_folder)
    else:
        print(f"Error: run_mode '{RUN_MODE}' not recognized. Accepted values are 'cli', "
              f"'batch_select', or 'batch_list'.",
              file=stderr)
