"""Satellite/constellation-related functionality."""
from collections import namedtuple
from enum import Enum
import random
from numpy import (  # pylint: disable=no-name-in-module
    arange, arctan, array, cos, degrees, radians, repeat as np_repeat, sin, sqrt, tan, where)
from numpy import arctan2
# pylint: disable=import-error
from recordclass import recordclass
from skyfield.api import load, Topos
from skyfield.constants import ERAD

from open_source_i_n import skyfield_funcs
from open_source_i_n.utils import relative_path

from .geometry import IdxSphericalCoordinate

# Data type for storing data for both downlink and uplink.
UlDlPair = recordclass('UlDlPair', ['ul', 'dl'], defaults=(None, None))

# Geosynchronous equatorial orbit distance in kilometers.
GEO_ORBIT_KM = 35786

ExclusionZone = namedtuple('ExclusionZone', 'az_min_deg az_max_deg el_min_deg el_max_deg')

# Data type for storing things related to victim and interfering constellation.
VicInterPair = recordclass('VicInterPair', ['vic', 'inter'], defaults=(None, None))


def get_geo_belt_points(position):
    # pylint: disable=invalid-name
    """Return (azimuth, elevation) to a GEO sat given its longitude.
    Args:
      position (LatLon): Earth station position.
      sat_lon np.array(float): GEO sat longitude in degrees.

    Returns:
      list: An iterator of (azimuth, elevation) points as floats in degrees.
    """
    # Calculate longitude difference using longitude from [0, 360] instead of [-180, 180].
    lon_diff_r = radians((position.lon + 180) - arange(360))
    lat_r = radians(position.lat)

    # Azimuth.
    az = 180 + degrees(arctan2(tan(lon_diff_r), sin(lat_r)))
    az = where(position.lat < 0, az - 180, az)
    az = where(az < 0, az + 360.0, az)

    # Elevation.
    r_1 = 1 + GEO_ORBIT_KM / (ERAD / 1000)
    v_1 = r_1 * cos(lat_r) * cos(lon_diff_r) - 1
    v_2 = r_1 * sqrt(1 - (cos(lat_r)**2) * (cos(lon_diff_r)**2))
    el = degrees(arctan2(v_1 , v_2))

    return zip(az, el)


def is_in_exclusion_zones(exclusion_zones, az, el):
    # pylint: disable=invalid-name
    """Return true if the given azimuth and elevation is in any exclusion zones.

    Args:
        exclusion_zones (namedtuple(az_min_deg, az_max_deg, el_min_deg, el_max_deg)):
            return of get_exclusion_zones.
        az (float): azimuth angle to determine if  it's in exclusion zones.
        el (float): elevation angle to determine if it's in exclusion zones.

    Returns:
        bool: True if az, el is in exclusion zones.
    """

    return any([
        zone.az_min_deg <= az <= zone.az_max_deg and zone.el_min_deg <= el <= zone.el_max_deg
        for zone in exclusion_zones
    ])


def get_exclusion_zones(position, min_el, alpha_angle, geo_belt_points):
    # pylint: disable=invalid-name
    """Return a list of ExclusionZones where TX should be muted (both ground and satellite).
    Args:
      position (LatLon): Target position.
      min_el (float): Minimum elevation angle for establishing communication.
      alpha_angle (float): Half cone angle around GEO belt (defined by ITU doc).
      geo_belt_points: TODO

    Returns:
      list: A list of ExclusionZones
    """
    # Anywhere below minimum elevation.
    exclusion_zones = [ExclusionZone(0, 360, -90, min_el)]

    # Around the GEO belt.
    if geo_belt_points is None:
        geo_belt_points = get_geo_belt_points(position)

    for az, el in geo_belt_points:
        # Form a cone around the given az, el.
        for rotation in arange(5, 90, 10):
            delta_az = alpha_angle * cos(radians(rotation))
            delta_el = alpha_angle * sin(radians(rotation))
            az_min = (az + 360 - delta_az) % 360
            az_max = (az + delta_az) % 360
            el_min = el - delta_el
            el_max = min(90, el + delta_el)

            # Check if azimuth range wraps around.
            if az_min < az_max:
                exclusion_zones.append(ExclusionZone(az_min, az_max, el_min, el_max))
            else:
                # Azimuth ranges like (355, 5) get split into 2 zones like (355, 360)
                # and (0, 5).
                exclusion_zones.append(ExclusionZone(az_min, 360, el_min, el_max))
                exclusion_zones.append(ExclusionZone(0, az_max, el_min, el_max))

    return exclusion_zones


def load_satellites(tle_fname):
    """
    Loads a text file of TLE lines into a list of Skyfield EarthSatellite objects.

    Args:
        tle_fname (str/Path): Filepath to TLE text file.

    Returns (list): List of Skyfield EarthSatellite objects.

    """
    if tle_fname.parts[-1].startswith('ORBEL'):
        sat_list = skyfield_funcs.orbit_els_to_sat_list(tle_fname)
    else:
        # Try loading file as a list of TLEs first. If file are not TLEs, load.tle will return an
        # empty list.
        sat_list = load.tle_file(tle_fname.__str__())

    return sat_list


class Constellation:
    """Satellite constellation."""
    class TrackingStrategy(Enum):
        """Strategy for tracking satellites."""
        # pylint: disable=fixme
        # TODO(kepler): Use i18n instead of string values.
        RANDOM = 'Random'
        HIGHEST_ELEVATION = 'Highest Elevation'
        LONGEST_HOLD = 'Longest Hold'

    def __init__(  # pylint: disable=too-many-arguments
            self,
            module,
            tle_file,
            min_el,
            geo_angle,
            tracking_strat,
            frequency,
            opt_args,
            fixed_params=None,
            const_name=None):
        """Constructor."""
        self._antenna_model = module.AntennaModel(frequency, opt_args)
        self._min_el = min_el
        self._geo_angle = geo_angle
        self._tracking_strategy = tracking_strat
        self._tle_file = tle_file
        self._name = const_name or 'Unknown constellation'
        self._fixed_params = fixed_params

    antenna_model = property(lambda self: self._antenna_model)
    # Not pickle-able so regenerate as needed.
    sats = property(lambda self: load_satellites(relative_path(self._tle_file)))
    min_el = property(lambda self: self._min_el)
    geo_angle = property(lambda self: self._geo_angle)
    tracking_strat = property(lambda self: self._tracking_strategy)
    name = property(lambda self: self._name)
    fixed_params = property(lambda self: self._fixed_params)

    def __str__(self):
        return self.name

    def get_sat_coords(self, position, times, long_holds):
        """Returns a list of IdxSphericalCoordinate for satellites being tracked at each timestep.

        long_holds only used if LONGEST_HOLD TrackingStrategy selected for constellation.

        Args:
            position (LatLon): Groundstation position.
            times (array): Array of skyfield.timelib.Times to track satellites for.
            long_holds (list): Pre-calculated list of IdxSphericalCoordinates for longest hold.

        Returns:
            list: List of IdxSphericalCoordinate of the satellite tracked for each timestep.
        """
        if self.tracking_strat == Constellation.TrackingStrategy.LONGEST_HOLD:
            return long_holds  # Long holds have already been calculated! Just return the values.

        sat_coords = self.vis_sats(position, times)  # Calculate visible satellite positions.

        # If Random, return random pass.
        if self.tracking_strat == Constellation.TrackingStrategy.RANDOM:
            return [random.choice(passes) if passes else None for passes in sat_coords]

        # pylint: disable=fixme
        # Todo: Enable a fixed tracking mode, that models an antenna held at a fixed position
        #  relative to the ground station.
        # # If Fixed, return pass modelled by a fixed location.
        # if self.tracking_strat == Constellation.TrackingStrategy.FIXED:
        #     return array((0, *self.fixed_params), dtype=IdxSphericalCoordinate)

        # Else, return highest elevation pass
        return [max(passes, key=lambda y: y['el']) if passes else None for passes in sat_coords]

    def vis_sats(self, position, times):
        """Return list of visible satellites and coordinates for each timestep.

         Args:
            position (LatLon): Groundstation position.
            times (array): Array of skyfield.timelib.Times to track satellites for.

        Returns:
           ret_val (list): [[IdxSphericalCoordinate for each visible satellite] for each timestep]
        """
        excl_zones = get_exclusion_zones(position, self.min_el, self.geo_angle, None)
        sats = self.sats  # Cache satellites for this process.
        diff_vector = sats - np_repeat(Topos(*position), len(sats))  # Groundstn-sat vector.
        # Propagate the difference vectors.
        const_prop = array([*(sat.at(times) for sat in diff_vector)])
        # Calculate the elevation, azimuth and distance of the constellation sats. Optimized.
        _altaz = [None]
        _set_item = _altaz.__setitem__
        const_coords = array([
            *[
                _set_item(0, coords.altaz())
                or array([_altaz[0][0].degrees, _altaz[0][1].degrees, _altaz[0][2].m])  # pylint: disable=unsubscriptable-object
                for coords in const_prop
            ]
        ])
        # Aggregate visible satellites at each timestep.
        return [
            [
                array((sat_idx, *sat_coord), dtype=IdxSphericalCoordinate)
                for sat_idx, sat_coord in enumerate(coords) if sat_coord[0] > self.min_el
                and not is_in_exclusion_zones(excl_zones, sat_coord[1], sat_coord[0])
            ] for coords in const_coords.transpose((2, 0, 1))  # Transposed to time, sat, coord.
        ]

    def _get_fixed_params(self):
        """Get fixed earth station pointing parameters from user."""
        # pylint: disable=fixme
        # Fixme: Function is only a quick solution to implementing location of a Fixed transmitter.
        print(f"Tracking strategy FIXED was selected for Constellation '{self.name}'")
        elevation = float(input("\nEnter a fixed elevation, in degrees: "))
        azimuth = float(input("Enter a fixed azimuth, in degrees: "))
        distance = float(input("Enter a fixed link distance, in km: ")) * 1000

        return elevation, azimuth, distance



def longest_hold(vis_sats_by_timestep):
    """Return the satellite and coordinates tracked by "longest hold" tracking.

    "Longest hold" tracking greedily keeps tracking a satellite until it goes out of view.
    The next satellite tracked is the one that will remain visible for the longest time.
    Rows without any satellites in view are simply None.

    Args:
        vis_sats_by_timestep (list): [
                [array(satellite idx and coordinates) for visible satellite] for timestep
            ]

    Returns:
        list: List of IdxSphericalCoordinate of the satellite tracked for each timestep.
    """
    if not vis_sats_by_timestep:
        return []

    # IdxSphericalCoordinate ndarrays are unhashable, so we convert to lists.
    vis_sats_as_listy_sats = [[sat.tolist() for sat in timestep]
                              for timestep in vis_sats_by_timestep]
    long_holds = []
    last_seen = set(vis_sats_as_listy_sats[0])  # Most recently seen satellites.
    run_length = 0  # Most consecutive appearances.
    # Sentinel to ensure last value is written.
    for snapshot in vis_sats_as_listy_sats + [[]]:
        persistent = set(snapshot) & last_seen
        if persistent:  # Some satellites we saw last time are still here.
            last_seen = persistent  # Update seen-list.
            run_length += 1  # Update book-keeping.
            continue
        if last_seen:  # Longest-seen satellite no longer visible
            # Convert satellite back to numpy type and record duration of visibility.
            long_holds += [array(last_seen.pop(), dtype=IdxSphericalCoordinate)] * run_length
        else:  # We didn't see a satellite last snapshot.
            long_holds.append(None)  # Record a None.
        last_seen = set(snapshot)
        run_length = 1

    return long_holds
