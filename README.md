# Kepler Open Source Interference Analysis tool (KOSIA)

**Kepler Satellite Interference Simulator**

## Description
Computes perceived radiofrequency interference between two orbital systems.
For each system, the user must specify its desired frequency, antenna, and orbital characteristics, as well as simulation parameters (e.g. epoch, timestep resolution).


## Installation
Inside the root directory of the repository run:
```bash
pip3 install -e .[open_source_i_n]
```

To install developer dependencies for running tests, run:
```bash
pip3 install -e .[dev]
```

### Dependencies

* pip
* Numpy
* matplotlib
* scipy
* skyfield
* recordclass
* pandas
* openpyxl

#### Developer Dependencies

* pylint
* pytest
* yapf

## Usage
The program can be run in three different Run Modes:

1. **Command Line Interface**: Arguments are passed in-line from terminal/command prompt.
2. **Batch Select**: A batch/shell file containing simulation parameters is selected from file dialog.
3. **Batch List**: All batch/shell files within a specified folder are run in sequence.

The preferred Run Mode can be specified in the `settings.ini` file in the root folder.

It is recommended to run in either **Batch Select** or **Batch List**, since these utilize batch/shell files to run the program. These act as simple, easily accessible containers for saving and editing simulation input parameters. From here on, such files will be referred to as *config files*, as they are used to specify a simulation's configuration.

If you have chosen **Batch Select** or **Batch List**, you may run the program at any time from the command line by using the following command in the program root folder:

##### Linux
```bash
python3 -m open_source_i_n
```

##### Windows
```bash
python -m open_source_i_n
```


### Required inputs:
1. 1x Simulation config file (or alternatively, a set of command line arguments).
2. 2x TLE files. One for each constellation (interfering and victim).
3. 2x Antenna model files. One for each constellation (interfering and victim).


#### Config Files
Tells the program how to perform the simulation, including which TLE and antenna files to use, tracking algorithms to run, and how many processors to deploy. It is convenient to save a set of command line instructions in a Shell (.sh) or Batch (.bat) file, to be edited or re-ran later.

Config files (and therefore, command line instances) must contain all mandatory parameters, listed below.

##### Parameters

|Parameter | Required? | Description | Type | Units | Example Value|
|--- | --- | --- | --- | --- | ---|
|`duration` | Y | Duration of the simulation. | float | Days | 3.5|
|`sim_block_size` | Y | The number of days to simulate   at once. | float | n/a | 0.2|
|`save_data, no_save_data` | N | Boolean flag for whether to   save output data (.npy files and graphs) or not. Will save by default. | boolean | n/a | TRUE|
|`gs` | Y | The latitude/longitude pair   for the ground station, given as a comma-separated pair. | float,float | deg | 35,-79|
|`freq_dl` | Y | The downlink frequency, in   GHz. | float | GHz | 14.5|
|`freq_ul` | Y | The uplink frequency, in GHz. | float | GHz | 11.6|
|`vic_tle_file,   inter_tle_file` | Y | Path to the TLE file for the   constellation. | string | n/a | ~/TLE/kepler.txt|
|`vic_min_el, inter_min_el` | Y | Minimum elevation for the   constellation Earth station. | float | deg | 10|
|`vic_geo_angle,   inter_geo_angle` | Y | The GEO exclusion angle of the   constellation. | float | deg | 18.2|
|`vic_tracking_strat,   inter_tracking_strat` | Y | The strategy to use for   selecting interfering satellites in the I/N calculation. | string | n/a | One of: "Random", "Highest Elevation", "Longest Hold"|
|`vic_fixed_params` | N | The elevation/azimuth pair of a fixed pointing antenna at the ground station location, given as a comma-separated pair. | float,float | deg | 30,270|
|`vic_module, inter_module` | Y | Which antenna to use for the   simulation. | string | n/a | kepler_interference|
|`inter_const_name` | Y | The name of the interfering   constellation (for graph titles). | string | n/a | SpaceX4408|
|`parallel` | N | How many CPU threads to use   for the simulation. 1 means to   use a single core, while 0 lets the program choose. Default is 0. | integer | n/a | 0|
|`name` | N | Optional name for the simulation run. Is added to the   filenames of all generated outputs. | string | n/a | "my-sim-v3"|


Depending on your operating system, you can structure your shell/batch config files as shown:

##### Linux
```bash
python3 -m open_source_i_n \
 --duration 1 \
 --sim_block_size 0.2 \
 --save_data \
 --gs "35,-79" \
 --freq_dl 11.7 \
 --freq_ul 14.25 \
 --vic_tle_file "TLEs/TLES_Kepler.txt" \
 --vic_min_el 10 \
 --vic_geo_angle 20 \
 --vic_tracking_strat "Random" \
 --vic_module "kepler_antenna" \
 --inter_tle_file "TLEs/TLEs_Example_System.txt" \
 --inter_min_el 12 \
 --inter_geo_angle 14 \
 --inter_tracking_strat "Highest Elevation" \
 --inter_module "template_standard" \
 --inter_const_name "Company-X"
```

##### Windows 
```batch
python -m open_source_i_n^
 --duration 1^
 --sim_block_size 0.2^
 --no_save_data^
 --gs "35,-79"^
 --freq_dl 11.7^
 --freq_ul 14.25^
 --vic_tle_file "TLEs/TLES_Kepler.txt"^
 --vic_min_el 10^
 --vic_geo_angle 20^
 --vic_tracking_strat "Random"^
 --vic_module "kepler_antenna"^
 --inter_tle_file "TLEs/TLEs_Example_System.txt"^
 --inter_min_el 12^
 --inter_geo_angle 14^
 --inter_tracking_strat "Highest Elevation"^
 --inter_module "template_standard"^
 --inter_const_name "Company-X"
```

#### TLE Files

TLE files describe the orbital characteristics of a satellite system.
The TLE files should be a txt file, with  all constellation TLEs laid out as follows (with no blank lines):

```
0 Kepler Plane 1 sat 1
1 99859U          20001.00000000  .00015063  00000-0  14589-2 0 00009
2 99859 097.7532   0.0000 0006678 083.0376   0.0000 14.99917647000000
0 Kepler Plane 1 sat 2
1 99860U          20001.00000000  .00015063  00000-0  14589-2 0 00009
2 99860 097.7532   0.0000 0006678 083.0376  18.0000 14.99917647000000
```

**Note:** Hosted on Kepler's github (https://github.com/kepler-space/tle_generator) is a script that can read an ITU IFIC database and return a formatted text file of TLEs. The output text file can then be fed into the simulator.


#### Antenna Model Files
Antenna model files completely describe the antenna parameters for the earth and space components of a system.

Two template antenna model files are provided in src > open_source_i_n > antenna. One of these strictly uses standard antenna patterns pre-encoded in the local antenna library, and the other demonstrates how a custom antenna pattern might be constructed.


### Tools
The following tools have been included with the build.
- Plotter
- Antenna Pattern Tester

All tool scripts can be found in the ./tools folder.

#### Plotter
Reads the .simdata files generated by KOSIA and plots the simulation data as a CDF. If multiple files are selected, 
all datasets will be added to a single plot.

#### Antenna Pattern Tester
Contains several tools designed to plot (and validate) antenna patterns.

You must edit ant_patt_tester.py directly to select your antenna pattern in either of the following cases:

##### Validating standard antenna patterns
Has scripts to plot off-axis patterns in 2D or 3D (see implementation of plot_patt_list_2d() and plot_func_as_colormap()). 

##### Validating whole KOSIA antenna pattern .py files
Also has a script that reads an entire KOSIA antenna python file directly (see plot_antenna_file()).


## Tests
The tests are run by simply calling `pytest` in the root directory. Developer extras [must be installed](#installation).
```bash
pytest
```


## Known Limitations
- Only supports co-located earth stations. To enable non-co-location, four unique separation angles must be defined, rather than just one. Also, two ground station locations (and thus ground station objects) must be used.
- Antenna pattern S.1428 underestimates peak gain (i.e. off-axis angle = 0) since it doesn't actually take gain as an input, but calculates it based on an inputted diameter. That said, a feature has been implemented to "correct" an inputted diameter such that it matches with the peak gain inputted for a given antenna model. However, floating point inaccuracies prevent this from being exact.
- Atmospheric losses are not calculated, but can be entered as fixed values in simulator.py.
- The program currently does not handle two dimensional antenna patterns, it assumes radially symmetric, circular beams for all antenna patterns. Antenna pattern files cannot read in azimuthal angles, that could be used to determine where an earth station is in a projected beam pattern from space.


## Future Features
- Incorporation of ITU and national power limits to quickly assess compliance.
- Memory optimization. Propogation of the satellite networks over long periods of time requires arbitrarily large memory allocations. At present, the software does not account for system limitations in its allocation of memory.

