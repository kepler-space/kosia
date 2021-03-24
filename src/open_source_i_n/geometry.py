"""Geometry/space/position related library."""
import numpy as np


class LatLon(tuple):
    """Latitude and longitude type."""
    def __new__(cls, latitude, longitude):
        return super().__new__(cls, (latitude, longitude))

    def __init__(self, latitude, longitude):  # pylint: disable=super-init-not-called
        """
        latitude: Positive values indicate north; negative values indicate south.
        longitude: Positive values indicate east; negative values indicate west.
        """
        self.lat = float(latitude)
        self.lon = float(longitude)

    def __getnewargs__(self):
        return tuple(self)

    @staticmethod
    def from_string(coord):
        """Return a LatLon object from string representation.

        Input is lat,lon . E.g., "35,-22".
        """
        return LatLon(*map(float, coord.split(',')))

    def __str__(self):
        card_lat = 'NS'[self.lat < 0]
        card_lon = 'EW'[self.lon < 0]
        return '%.1f %s, %.1f %s' % (abs(self.lat), card_lat, abs(self.lon), card_lon)


SphericalCoordinate = [('el', np.float64), ('az', np.float64), ('r', np.float64)]
IdxSphericalCoordinate = [('idx', np.uint32)] + SphericalCoordinate
