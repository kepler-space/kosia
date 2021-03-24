from time import monotonic
from pathlib import Path
from os import getcwd, scandir, mkdir
from re import search
from sys import stderr
import tkinter
from tkinter import filedialog

root_path_src = Path.cwd()
# FIXME: If running in an IDE, change this to Path.cwd().parent.parent. If running from a
#  terminal (and after having run pip3 install -e .[open_source_i_n]), then just use Path.cwd()


# File IO Functions
def relative_path(path):
    """
    Takes a string indicating a path relative to the program root and returns a Path object.

    Args:
        path (str): Path relative to the program root folder.

    Returns: Path object.

    """
    path = path.replace("\\", "/")
    return root_path_src / path


def open_file_dialog(init_folder=None, file_types=None, plural=False):
    """
    Opens a file dialog, allow the user to browse for a file.

    Args: init_folder (str): Optional. Path to which file dialog opens initially.
    file_types (list of tuples): List of tuples specifying accepted file info. For example:
        [("PNG Files", ".png")]. plural (bool): If true, file dialog accepts multiple file
        selection. Default is False.

    Returns (str): File path to selected file.
    """
    if not file_types:
        file_types = [('All files', '*.*')]

    if not init_folder:
        init_folder = getcwd()

    # Select file(s) from file dialog
    root = tkinter.Tk()
    root.withdraw()
    try:
        if plural:
            return filedialog.askopenfilenames(parent=root,
                                               initialdir=init_folder,
                                               filetypes=file_types)
        else:
            return filedialog.askopenfilenames(parent=root,
                                               initialdir=init_folder,
                                               filetypes=file_types)[0]
    except IndexError:
        print("Error: You must select a file in the file dialog.", file=stderr)
        return None


def open_folder_dialog(init_folder=None):
    """
    Opens a tkinter dialog window used to select a folder location.

    Args:
        init_folder (str or Path): Initial folder in which to open the dialog.

    Returns (str): File path selected in the dialog, or None if window is closed without
    a selection.

    """
    if not init_folder:
        init_folder = getcwd()

    root = tkinter.Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory(parent=root,
                                              initialdir=init_folder,
                                              title="Select folder")
    if folder_selected != "":
        return folder_selected
    else:
        print("No folder selected.")
        return ""


def check_folder(path):
    """
    Checks to see if a folder exists. If not, that folder is created.

    Args:
        path (str): Path of folder to be checked/created.

    Returns: None
    """
    # Create folder
    try:
        scandir(path)
    except FileNotFoundError:
        print("Folder %s not found. Creating new folder." % path)
        mkdir(path)


def read_config_file(file_name):
    """
    Creates a dictionary by parsing lines in a file with the following format:
    `<key> value`

    Args:
        file_name (str): Path of config file relative to the program root.

    Returns: Dictionary
    """
    # Reads the config.ini to parse the program settings.
    file_path = root_path_src / file_name
    with open(file_path, 'r') as f:
        dic = {
            search("\<(.*?)\>", i).group().strip("<>"): "".join(i.split()[1:])   # pylint: disable=anomalous-backslash-in-string
            for i in f.readlines() if i.startswith("<")
        }
    return dic


def get_setting(settings_dict, key, accepted_vals=None, is_boolean=False, optional=False):
    """
    Check whether a dictionary value meets a set of accepted values.

    Args:
        settings_dict (dict): Attached a dict of strings. Usually one loaded using read_config_file.
        key (str): The key to be fetched.
        accepted_vals (tuple): Tuple of strings containing accepted values for the setting.
        is_boolean (bool): Whether the returned value should be converted to a boolean.
        optional (bool): If False, a warning will be issued if a key is not found. Default is False.

    Returns: The value of the dictionary at the specified key, after error checking.

    """

    # For converting string input to a boolean val. Only used when if_boolean is set to true on a
    # function call.
    def __str_to_boolean(var):
        boolean_true_vals = ['True', 'T', '1']
        return bool(var in boolean_true_vals)

    # Load value from settings
    try:
        output = settings_dict[key]
    except KeyError():
        if optional:
            return None
        else:
            print(f"No {key} specified. Please specify an {key} in settings.ini")
            return None

    # If accepted values are specified, use them to validate input.
    if accepted_vals:
        if output not in accepted_vals:
            print(f"Error: {key} '{output}' not recognized. Accepted values are {accepted_vals}",
                  file=stderr)
            return None
        elif is_boolean:
            return __str_to_boolean(output)
        else:
            return output
    elif is_boolean:
        return __str_to_boolean(output)
    else:
        return output


# Performance functions
class Timer:
    """Minimal class used to time blocks of code."""
    print_mode = False  # For debug

    def __init__(self, msg):
        self.start_time = monotonic()
        self.duration = None
        self._print(msg, end=' ')

    def start(self, msg):
        """Reinitialize the object, starting its internal clock."""
        self.__init__(msg)

    def end(self):
        """End the objects internal clock and print the time elapsed."""
        self.duration = monotonic() - self.start_time
        self._print("Done. t = %.4f seconds" % self.duration)

    def _print(self, *args, **kwargs):
        """Debug print."""
        if self.print_mode:
            print(*args, **kwargs)


# Utility functions
def pp(arr):                    # pylint: disable=invalid-name
    """Simple pretty print."""
    for i in arr:
        print(i)


def pp_dict(dic):
    """Pretty print a dictionary."""
    for key in dic:
        print(key, dic[key])


def pp_ant_props(ant_model):
    """Print an antenna model's parameters."""
    print("Default es:", ant_model.default_es)
    print("Default sat:", ant_model.default_sat)
    pp_dict(ant_model.props.__dict__)


def print_config(config):
    """Print the full parameters of a simulation. Pass the config file as input."""
    print("\nINPUT PARAMETERS:")
    pp_dict(config.args)

    print("\nVICTIM PROPERTIES:")
    pp_ant_props(config.vic_module.AntennaModel)
    print("\nINTERFERER PROPERTIES:")
    pp_ant_props(config.inter_module.AntennaModel)
    print()
