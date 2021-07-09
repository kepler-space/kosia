"""
Defines the Configuration class (and related functions), used to store inputs for the simulation.
"""
from math import ceil
from sys import argv as sys_argv
from numpy import save, load
from skyfield.constants import DAY_S  # pylint: disable=import-error
from open_source_i_n import antenna, geometry
from open_source_i_n.satellite import Constellation


def save_argv():
    """Saves sys.argv to file."""
    save("argv", sys_argv)


def load_argv_pickle(argv_path):
    """Load an argv file and generate a Config object"""
    argv = load(argv_path)
    argv = argv.tolist()
    args = {name.lstrip("--"): argv[argv.index(name) + 1] for name in argv if name.startswith("--")}
    if args['save_data']:
        args['save_data'] = True
    return Config(args)


def load_argv_list(argv):
    """Generate a Config object from a list of argv parameters"""
    args = {name.lstrip("--"): argv[argv.index(name) + 1] for name in argv if name.startswith("--")}
    try:
        if args['save_data']:
            args['save_data'] = True
    except KeyError:
        args['save_data'] = True
    return Config(args)


def load_from_args(args):
    """Generate a Config object directly from an ArgumentParser NameSpace object (received
    through the command line). """
    args_dict = dict(args._get_kwargs())

    # Since __main__.get_arg_parser() already does all the proper conversions, we pass the
    # dictionary into Config with direct=True (prevents inputs from being converted twice).
    return Config(args_dict, direct=True)


# pylint: disable=unidiomatic-typecheck
def load_batch(bat_path):
    """
    Creates a Config object from a properly formatted batch file.

    Args:
        bat_path (str/Path): Path to batch file.

    Returns: Config object.

    """
    # LOAD BATCH FILE
    with open(bat_path, 'r') as bat_file:
        lines = bat_file.readlines()

    # Clean lines
    for i, _ in enumerate(lines):
        lines[i] = lines[i].strip("\n").strip("^").strip(" \\")

    # Create dictionary
    args = {
        line.split(" ")[0].lstrip("--"): " ".join(line.split()[1:])
        for line in lines if line.startswith("--")
    }
    if args['save_data']:
        args['save_data'] = True
    elif len(args['save_data']) == 0:
        args['save_data'] = True

    for name, val in args.items():
        if type(val) == str:
            args[name] = val.strip("\"")

    return Config(args)


# pylint: disable=unidiomatic-typecheck
def load_shell(sh_path):
    """
    Creates a Config object from a properly formatted shell file.

    Args:
        sh_path (str/Path): Path to shell file.

    Returns: Config object.

    """
    # LOAD SHELL FILE
    with open(sh_path, 'r') as sh_file:
        lines = sh_file.readlines()

    # Clean lines
    for i, _ in enumerate(lines):
        lines[i] = lines[i].strip("\n").strip("\\")

    # Create dictionary
    args = {
        line.lstrip().lstrip('--').split(' ')[0]: " ".join(line.split()[1:])
        for line in lines if line.lstrip().startswith("--")
    }
    if args['save_data']:
        args['save_data'] = True
    elif len(args['save_data']) == 0:
        args['save_data'] = True

    for name, val in args.items():
        if type(val) == str:
            args[name] = val.strip("\"")

    return Config(args)


def open_file_simdata(file):
    """
    Opens a SIMDATA file from disk.

    Args:
        file (str): Filepath to SIMDATA file.

    Returns (dict): Dictionary containing saved SIMDATA information.

    """

    print("Opening: ", file.split("\\")[-1])
    with open(file, "rb") as sim_file:
        print("Loading data...")
        i_n_data = load(sim_file, allow_pickle=True)
        percents_inst = load(sim_file, allow_pickle=True)
        bins_inst = load(sim_file, allow_pickle=True)
        argv = load(sim_file, allow_pickle=True)
        props = load(sim_file, allow_pickle=True)
    return {
        'I_N_data': i_n_data,
        "percents_inst": percents_inst,
        "bins_inst": bins_inst,
        "argv": argv.tolist(),
        "props": props.tolist()
    }


# pylint: disable=too-many-statements,too-many-branches,too-few-public-methods,too-many-instance-attributes
class Config:
    """
    A class to store simulation parameters.

    Parameters can read in from either the command line argument parser, a batch file, or a shell
    file. These parameters are converted to types convenient for relevance, view, and storage.

    Config objects must be passed a dictionary with all required parameter names, including:
    INPUT PARAMETERS:
        duration
        sim_block_size
        save_data OR no_save_data
        gs
        freq_dl
        freq_ul
        vic_tle_file
        vic_min_el
        vic_geo_angle
        vic_tracking_strat
        vic_module
        vic_const_name
        inter_tle_file
        inter_min_el
        inter_geo_angle
        inter_tracking_strat
        inter_module
        inter_const_name
        parallel *
        name *

    * = Optional
    Detail on input parameters can be found in the program ReadMe.


    Config objects instantiated with direct=False (default) will expect all dictionary inputs to
    be read as strings. These will then be converted to appropriate filetypes.

    Config objects instantiated with direct=True will perform no conversions. This is
    typically only done when the arguments are read from the argument parser,
    since main.get_arg_parser() already performs all the correct conversions.

    """
    def __init__(self, args, direct=False):
        self.args = args

        # If not direct, inputs are assumed to be strings. These must be converted to necessary
        # filetypes.
        if not direct:
            self.duration = float(args['duration'])
            self.sim_block_size = ceil(float(args['sim_block_size']) * DAY_S)
            if 'granularity' in args.keys():
                self.granularity = int(args['granularity'])
            else:
                self.granularity = 1
            self.save_data = args['save_data']
            self.no_save_data = None
            # pylint: disable=C0103
            self.gs = geometry.LatLon.from_string(args['gs'])
            self.freq_dl = float(args['freq_dl']) * 1e9
            self.freq_ul = float(args['freq_ul']) * 1e9

            self.vic_tle_file = args['vic_tle_file']
            self.vic_min_el = float(args['vic_min_el'])
            self.vic_geo_angle = float(args['vic_geo_angle'])
            self.vic_tracking_strat = Constellation.TrackingStrategy(args['vic_tracking_strat'])
            self.vic_module = antenna.load(args['vic_module'])
            self.vic_const_name = args['vic_const_name']
            self.vic_opt_args = None

            self.inter_tle_file = args['inter_tle_file']
            self.inter_min_el = float(args['inter_min_el'])
            self.inter_geo_angle = float(args['inter_geo_angle'])
            self.inter_tracking_strat = Constellation.TrackingStrategy(args['inter_tracking_strat'])
            self.inter_module = antenna.load(args['inter_module'])
            self.inter_const_name = args['inter_const_name']
            self.inter_opt_args = None

            try:
                self.vic_fixed_params = tuple(map(float, args['vic_fixed_params'].split(',')))
            except KeyError:
                self.vic_fixed_params = ""

            try:
                self.inter_fixed_params = tuple(map(float, args['inter_fixed_params'].split(',')))
            except KeyError:
                self.inter_fixed_params = ""

            try:
                self.parallel = int(args['parallel'])
            except KeyError:
                self.parallel = 0



            try:
                self.name = args['name']
            except KeyError:
                self.name = ""

        # If direct, arguments are taken as is.
        else:
            self.duration = args['duration']
            self.sim_block_size = args['sim_block_size']
            self.granularity = args['granularity']
            if args['save_data']:
                self.save_data = args['save_data']
            else:
                self.no_save_data = args['no_save_data']
            self.gs = args['gs']
            self.freq_dl = args['freq_dl']
            self.freq_ul = args['freq_ul']

            self.vic_tle_file = args['vic_tle_file']
            self.vic_min_el = args['vic_min_el']
            self.vic_geo_angle = args['vic_geo_angle']
            self.vic_tracking_strat = args['vic_tracking_strat']
            self.vic_module = args['vic_module']
            self.vic_const_name = args['vic_const_name']
            self.vic_opt_args = args['vic_opt_args']

            self.inter_tle_file = args['inter_tle_file']
            self.inter_min_el = args['inter_min_el']
            self.inter_geo_angle = args['inter_geo_angle']
            self.inter_tracking_strat = args['inter_tracking_strat']
            self.inter_module = args['inter_module']
            self.inter_const_name = args['inter_const_name']
            self.inter_opt_args = args['inter_opt_args']

            try:
                self.vic_fixed_params = args['vic_fixed_params']
            except KeyError:
                self.vic_fixed_params = ""

            try:
                self.inter_fixed_params = args['inter_fixed_params']
            except KeyError:
                self.inter_fixed_params = ""

            if args['parallel']:
                self.parallel = args['parallel']
            else:
                self.parallel = 0
            if args['name']:
                self.name = args['name']
            else:
                self.name = ''

    def output_to_file(self, folder_name, linux=False):
        """
        Save a Config object to a batch or shell file.

        Args:
            folder_name (str or Path): Folder path for saving file.
            linux (bool): If True, Config object is saved as a shell file. Otherwise saved as batch.

        Returns: None

        """
        if linux:
            py_cmd = "python3"
            start_char = " "
            end_char = " \\\n"
            file_type = ".sh"
        else:
            py_cmd = "python"
            start_char = " "
            end_char = "^\n"
            file_type = ".bat"

        # Set save_data
        if self.save_data:
            save_data_str = "save_data"
        else:
            save_data_str = "no_save_data"

        text = f"{py_cmd} -m open_source_i_n{end_char}" \
            f'{start_char}--duration {int(self.duration)}{end_char}'  \
            f'{start_char}--sim_block_size {int(self.sim_block_size/86400)}{end_char}' \
            f'{start_char}--{save_data_str}{end_char}' \
            f'{start_char}--gs "{self.gs.lat}, {self.gs.lon}"{end_char}' \
            f'{start_char}--freq_dl {self.freq_dl / 1e9}{end_char}' \
            f'{start_char}--freq_ul {self.freq_ul / 1e9}{end_char}' \
            f'{start_char}--vic_tle_file "{self.vic_tle_file}"{end_char}' \
            f'{start_char}--vic_min_el {int(self.vic_min_el)}{end_char}' \
            f'{start_char}--vic_geo_angle {int(self.vic_geo_angle)}{end_char}' \
            f'{start_char}--vic_tracking_strat "{self.vic_tracking_strat.value}"{end_char}' \
            f'{start_char}--vic_module "{self.vic_module.__name__.split(".")[-1]}"{end_char}' \
            f'{start_char}--vic_const_name "{self.vic_module.AntennaModel.name}"{end_char}' \
            f'{start_char}--vic_fixed_params "{self.vic_fixed_params}"{end_char}' \
            f'{start_char}--inter_tle_file "{self.inter_tle_file}"{end_char}' \
            f'{start_char}--inter_min_el {int(self.inter_min_el)}{end_char}' \
            f'{start_char}--inter_geo_angle {int(self.inter_geo_angle)}{end_char}' \
            f'{start_char}--inter_tracking_strat "{self.inter_tracking_strat.value}"{end_char}' \
            f'{start_char}--inter_module "{self.inter_module.__name__.split(".")[-1]}"{end_char}' \
            f'{start_char}--inter_const_name "{self.inter_const_name}"{end_char}' \
            f'{start_char}--inter_fixed_params "{self.inter_fixed_params}"{end_char}' \
            f'{start_char}--parallel {self.parallel}{end_char}' \
            f'{start_char}--name "{self.name}"'

        with open(f"{folder_name}{self.name}{file_type}", 'w') as file:
            file.writelines(text)
