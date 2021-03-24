"""Auxiliary functions related to loading or handling Skyfield objects and their properties"""
# pylint: disable=import-error
from sys import stderr
from datetime import datetime
from numpy import pi, sqrt, power, deg2rad, loadtxt
from skyfield.api import load, EarthSatellite
from sgp4.api import Satrec, WGS84
from scipy import constants as consts
from scipy.constants import G

R_EARTH = 6371000
M_EARTH = 5.972e24
mu = consts.G * M_EARTH  # Universal gravitational parameter
ts = load.timescale()


def calc_period_from_alt(perigee, apogee, mode='rads/min'):
    """
    Calculates orbital period of a satellite.

    Modes:
    Output is returned according to one of the following modes.
    'rev/day'
    'rads/min'
    'deg/min'
    'seconds'
    'minutes'
    'hours'
    'days'

    Args:
        perigee (int/float): Perigee in kilometers above Earth's surface.
        apogee (int/float): Apogee in kilometers above Earth's surface.
        mode (str): Output mode (e.g. 'rads/min')

    Returns: Period in radians per minute

    """
    perigee *= 1000
    apogee *= 1000

    sem_mjr_ax = (perigee + apogee) / 2     # Semi major axis
    p_secs = sqrt(((4 * (pi**2)) / (G * M_EARTH)) * pow(R_EARTH + sem_mjr_ax, 3))  # period in
    # seconds

    prd = None
    if mode == 'rev/day':
        prd = 86400 / p_secs
    elif mode == 'rads/min':
        prd = deg2rad(360 / (p_secs / 60))
    elif mode == 'deg/min':
        prd = 360 / (p_secs / 60)
    elif mode == 'seconds':
        prd = p_secs
    elif mode == 'minutes':
        prd = p_secs / 60
    elif mode == 'hours':
        prd = p_secs / 3600
    elif mode == 'days':
        prd = p_secs / 86400
    else:
        print(f"Cannot calculate period, mode '{mode}' is not recognized.", file=stderr)

    return prd


# pylint: disable=too-many-arguments
def _return_sat_from_elements(satnum, epoch, ecc, arg_p, incl, mn_ano, sem_maj_ax, raan):
    """
    Return an EarthSatellite object from a set of orbital elements. Bstar drag coefficient,
    ndot ballistic coefficient, and nddot second derivative of mean motion ARE ALL ASSUMED TO BE
    ZERO.

    Args:
        satnum (int): Arbitrary integer, should not exceed 5 characters.
        epoch (int): Epoch in days since 1949, Dec 31.
        ecc (float): Eccentricity, float b/w 0 and 1.
        arg_p (float): Argument of perigee in degrees.
        incl (float): Inclination in degrees.
        mn_ano (float): Mean anomaly in degrees.
        sem_maj_ax (float): Semi major axis in meters.
        raan (float): Right ascension of ascending node in degrees.

    Returns:

    """
    def mn_mot_from_smmjax():
        period = 2 * pi * sqrt(power(sem_maj_ax, 3) / mu)
        rads_per_min = 2 * pi / (period / 60)
        return rads_per_min

    arg_p = deg2rad(arg_p)
    incl = deg2rad(incl)
    mn_ano = deg2rad(mn_ano)
    raan = deg2rad(raan)

    satrec = Satrec()
    satrec.sgp4init(
        WGS84,  # gravity model
        'i',  # 'a' = old AFSPC mode, 'i' = improved mode
        satnum,  # satnum: Satellite number
        epoch,  # epoch: days since 1949 December 31 00:00 UT
        0.0,  # bstar: drag coefficient (/earth radii)
        0.0,  # ndot: ballistic coefficient (revs/day)
        0.0,  # nddot: second derivative of mean motion (revs/day^3)
        ecc,  # ecco: eccentricity
        arg_p,  # argpo: argument of perigee (radians)
        incl,  # inclo: inclination (radians)
        mn_ano,  # mo: mean anomaly (radians)
        mn_mot_from_smmjax(),  # no_kozai: mean motion (radians/minute)
        raan,  # nodeo: right ascension of ascending node (radians)
    )

    sat = EarthSatellite.from_satrec(satrec, ts)

    return sat


def orbit_els_to_sat_list(path):
    """
    Takes a path leading to a text file containing comma-delimited orbital elements with six
    columns, in the following order: semi-major axis (m), eccentricity(-), inclination (deg),
    RAAN (deg), ArgPerig (deg), MeanAnom (deg), Epoch (days since 1949, Dec 31).

    Bstar drag coefficient, ndot ballistic coefficient, and nddot second derivative of mean
    motion ARE ALL ASSUMED TO BE ZERO.

    Args:
        path (str): Path to text file containing orbital elements.

    Returns: List of skyfield satellite objects.

    """
    table = loadtxt(path, delimiter=",")

    # Add 1 day to epoch date to correct an offset that is present.
    table[:, 6] += 1

    sats = []
    for sat in range(0, len(table)):
        sats.append(
            _return_sat_from_elements(sat + 1, table[sat, 6], table[sat, 1], table[sat, 4],
                                      table[sat, 2], table[sat, 5], table[sat, 0], table[sat, 3]))

    return sats


if __name__ == '__main__':
    ts = load.timescale()
    epoch_y = datetime.now().year
    epoch_m = datetime.now().month
    epoch_d = datetime.now().day
