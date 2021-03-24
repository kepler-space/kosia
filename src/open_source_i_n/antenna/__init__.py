"""Base classes and functions for the antenna module."""
from abc import ABCMeta, abstractmethod
import importlib
from types import SimpleNamespace
import scipy.constants as consts
import numpy as np


# pylint: disable=too-few-public-methods
class AntennaProperties(SimpleNamespace):
    """Namespace for holding configurable antenna properties."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def load(module_name):
    """Load AntennaModel module by name."""
    return importlib.import_module('{}.{}'.format(__name__, module_name))


class BaseAntennaModel(metaclass=ABCMeta):  # pylint: disable=too-many-public-methods
    """The base class for implementing Antenna models."""

    name = "base"

    def __init__(self, frequency, opt_args=None):
        self._frequency = frequency

    @property
    def ul(self, _memo={}):
        """Namespace for uplink I/N calculation functions."""
        class UplinkINCalcs:
            es_psd = self.interfering_es_ul_psd
            sat_g_over_t = self.victim_sat_ul_g_over_t

        if self.__class__ not in _memo:
            _memo[self.__class__] = UplinkINCalcs()
        return _memo[self.__class__]

    @property
    def dl(self, _memo={}):
        """Namespace for downlink I/N calculation functions."""
        class DownlinkINCalcs:
            sat_psd = self.interfering_sat_dl_psd
            es_g_over_t = self.victim_es_dl_g_over_t

        if self.__class__ not in _memo:
            _memo[self.__class__] = DownlinkINCalcs()
        return _memo[self.__class__]

    frequency = property(lambda self: self._frequency)

    @abstractmethod
    def interfering_sat_dl_psd(self, elevation, sep_angle, nad_angle, dist):
        """Interfering satellite downlink PSD abstract method"""
        pass

    @abstractmethod
    def interfering_es_ul_psd(self, elevation, sep_angle, nad_angle, dist):
        """Interfering earth station uplink PSD abstract method"""
        pass

    @abstractmethod
    def victim_es_dl_g_over_t(self, elevation, sep_angle, nad_angle, dist):
        """Victim earth station downlink G/T abstract method"""
        pass

    @abstractmethod
    def victim_sat_ul_g_over_t(self, elevation, sep_angle, nad_angle, dist):
        """Victim satellite uplink G/T abstract method"""
        pass

    # Calculations
    @staticmethod
    def spr_loss(dist):
        """Returns spreading loss in dB"""
        return 10 * np.log10(1 / (4 * np.pi * (dist**2)))

    @staticmethod
    def freespace_loss(dist, freq_hz):
        """Returns freespace path loss in dB"""
        return -20 * np.log10(4 * np.pi * dist / (consts.c / freq_hz))

    @staticmethod
    def diameter(gain, freq_hz, efficiency):
        """Returns parabolic diameter in meters"""
        return ((consts.c / freq_hz) / consts.pi) * np.sqrt(np.power(10, gain / 10) / efficiency)

    # Standard PSD calculators
    @staticmethod
    def interfering_sat_dl_psd_basic(high_el_boundary,
                                     low_el_pfd_hz,
                                     high_el_pfd_hz,
                                     pointing=False):
        """Default interfering satellite downlink PSD function.

        Args:
            high_el_boundary (float): High elevation boundary.
            low_el_pfd_hz (float): Low elevation PFD.
            high_el_pfd_hz (float): High elevation PFD.
            pointing (bool): If True, separation angle will be fixed to zero.

        Returns:
            function: The satellite's PSD function with values.
        """
        def interfering_sat_dl_psd(elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            """Implementation of the default victim satellite downlink PSD,
                adjusted to be in dBW/Hz.

            Args:
                elevation (float): Elevation (degrees) of interfering satellite
                sep_angle (float): Separation angle (degrees) between interfering satellite and
                                  chosen victim satellite
                nad_angle (float): Assuming nadir pointing, angle from victim satellite to
                                  nadir pointing above the ground station (radians)
                dist (float): Distance to satellite from victim ground station (m)

            Returns:
                float: Satellite PSD at earth station in dBW/Hz
                       (adjusted over entire transmit bandwidth).
            """
            if pointing:
                sep_angle = 0

            # elevation in degrees
            if elevation > high_el_boundary:
                return low_el_pfd_hz
            return high_el_pfd_hz

        return interfering_sat_dl_psd

    @staticmethod
    def interfering_sat_dl_psd_s1528(gain_sat_dl,  # pylint: disable=invalid-name,too-many-arguments
                                     diam,
                                     power_db,
                                     bw_hz,
                                     hpbw,
                                     L_N=-15,
                                     pointing=False):
        """Standard pattern S.1528, as applied to calculate interfering satellite downlink PSD."""

        def interfering_sat_dl_psd(self, elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            if pointing:
                sep_angle = 0

            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_s1528_sat(
                gain_sat_dl, diam, sep_angle, self.frequency.dl, hpbw,
                L_N=L_N) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(dist, self.frequency.dl)

        return interfering_sat_dl_psd

    @staticmethod
    def interfering_es_ul_psd_app8(gain_es_ul, diam_es, power_db, bw_hz, pointing=False):
        """Standard Appendix 8 pattern, as applied to calculate interfering satellite uplink PSD."""

        def interfering_es_ul_psd(self, elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            if pointing:
                sep_angle = 0

            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_app8_es(gain_es_ul, diam_es, sep_angle,
                                        self.frequency.ul) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(dist, self.frequency.ul)

        return interfering_es_ul_psd

    @staticmethod
    def interfering_es_ul_psd_s580_6(gain_es_ul, power_db, bw_hz, pointing=False):
        """Standard pattern S.580-6, as applied to calculate interfering ES uplink PSD."""

        def interfering_es_ul_psd(self, elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            if pointing:
                sep_angle = 0

            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_s580_6_es(gain_es_ul, sep_angle) \
                + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(dist, self.frequency.ul)

        return interfering_es_ul_psd

    @staticmethod
    def interfering_es_ul_psd_s1428(gain_es_ul, diam_es, power_db, bw_hz, pointing=False):
        """Standard pattern S.1428, as applied to calculate interfering ES uplink PSD."""

        def interfering_es_ul_psd(self, elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            if pointing:
                sep_angle = 0

            # Because the 1428 pattern calculates its gain (as a function of diameter), we must
            # ensure the diameter entered will correspond to the peak gain provided by the
            # antenna model (gain_es_ul)
            lam = consts.c / self.frequency.ul
            d_lambda = diam_es / lam
            if d_lambda <= 100:
                corrected_es_1428_diam = lam * np.power(10, (gain_es_ul - 7.7) / 20)
            else:
                corrected_es_1428_diam = lam * np.power(10, (gain_es_ul - 8.4) / 20)

            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_s1428_es(corrected_es_1428_diam, sep_angle,
                                         self.frequency.ul) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(dist, self.frequency.ul)

        return interfering_es_ul_psd

    @staticmethod
    def interfering_es_ul_psd_s1528(gain_es_ul,    # pylint: disable=invalid-name,too-many-arguments
                                    diam_es,
                                    es_power_db,
                                    bw_hz,
                                    hpbw,
                                    L_N=-15,
                                    pointing=False):
        """Standard pattern S.1528, as applied to calculate interfering ES uplink PSD."""

        # print("Warning: Rec. 1528 is being used as an earth station pattern, but is only "
        #       "recommended for use as a satellite pattern.", file=stderr)

        def interfering_es_ul_psd(self, elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            if pointing:
                sep_angle = 0

            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_s1528_sat(
                gain_es_ul, diam_es, sep_angle, self.frequency.ul, hpbw,
                L_N=L_N) + es_power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(dist, self.frequency.ul)

        return interfering_es_ul_psd

    # Standard G/T calculators
    @staticmethod
    def victim_sat_ul_g_over_t_s1528(gain_sat_ul,  # pylint: disable=invalid-name,too-many-arguments
                                     diam,
                                     temp_sys_sat,
                                     hpbw,
                                     other_losses_db=0,
                                     L_N=-15,
                                     pointing=False,
                                     use_nad_ang=False):
        """Standard pattern S.1528, as applied to calculate victim satellite uplink G/T."""

        def victim_sat_ul_gain(self, elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            if pointing:
                sep_angle = 0
            if use_nad_ang:
                sep_angle = nad_angle

            # Return G/T
            return self.gain_s1528_sat(gain_sat_ul, diam, sep_angle, self.frequency.ul, hpbw,
                                       L_N) - 10 * np.log10(temp_sys_sat) - other_losses_db

        return victim_sat_ul_gain

    @staticmethod
    def victim_es_dl_g_over_t_app8(gain_es_dl,
                                   diam_es,
                                   temp_sys_es,
                                   other_losses_db=0,
                                   pointing=False):
        """Standard Appendix 8 pattern, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            if pointing:
                sep_angle = 0

            # Return G/T
            return self.gain_app8_es(
                gain_es_dl, diam_es, sep_angle,
                self.frequency.dl) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_s580_6(gain_es_dl,
                                     temp_sys_es,
                                     other_losses_db=0,
                                     pointing=False):
        """Standard pattern S.580-6, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            if pointing:
                sep_angle = 0

            # Return G/T
            return self.gain_s580_6_es(
                gain_es_dl, sep_angle) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_s1428(gain_es_dl,
                                    diam_es,
                                    temp_sys_es,
                                    other_losses_db=0,
                                    pointing=False):
        """Standard pattern S.1428, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            if pointing:
                sep_angle = 0

            # Because the 1428 pattern calculates its gain (as a function of diameter), we must
            # ensure the diameter entered will correspond to the peak gain provided by the
            # antenna model (gain_es_ul)
            lam = consts.c / self.frequency.dl
            d_lambda = diam_es / lam
            if d_lambda <= 100:
                corrected_es_1428_diam = lam * np.power(10, (gain_es_dl - 7.7) / 20)
            else:
                corrected_es_1428_diam = lam * np.power(10, (gain_es_dl - 8.4) / 20)

            # Return G/T
            return self.gain_s1428_es(
                corrected_es_1428_diam, sep_angle,
                self.frequency.dl) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_s1528(gain_es_dl,   # pylint: disable=invalid-name,too-many-arguments
                                    diam_es,
                                    temp_sys_es,
                                    hpbw,
                                    other_losses_db=0,
                                    L_N=-15,
                                    pointing=False):
        """Standard pattern S.1528, as applied to calculate victim ES downlink G/T."""

        # print("Warning: Rec. 1528 is being used as an earth station pattern, but is only "
        #       "recommended for use as a satellite pattern.", file=stderr)

        def victim_es_dl_gain(self, elevation, sep_angle, nad_angle, dist):  # pylint: disable=unused-argument
            if pointing:
                sep_angle = 0

            # Return G/T
            return self.gain_s1528_sat(gain_es_dl, diam_es, sep_angle, self.frequency.dl, hpbw,
                                       L_N) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    # Standard Off-axis gain calculators
    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_app8_es(gain_es_dl, diam_es, sep_angle, freq_hz):
        """
                    Off-axis gain - earth station (Appendix 8, Annex 3 pattern)

                    Args:
                        gain_es_dl (float): RX gain of earth station
                        diam_es: Diameter of earth station
                        sep_angle (float): Separation angle (degrees) between victim satellite and
                                           chosen interfering satellite.
                        freq_hz (float): Frequency (Hz)

                    Returns:
                        float: G off axis (dB) of ground station.
                    """

        D = diam_es
        lmbda = consts.c / freq_hz
        G_max = gain_es_dl

        # In cases where D/lmbda is not given, it may be estimated as 20*log(D/lmbda) =~ G_max - 7.7
        if not D or not lmbda:
            D_lmbda = np.exp((G_max - 7.7) / 20)
        else:
            D_lmbda = D / lmbda

        G_1 = 2 + 15 * np.log10(D_lmbda)
        phi = sep_angle  # degrees
        phi_m = 20 * np.power(D_lmbda, -1) * np.sqrt(G_max - G_1)  # in degrees
        phi_r = 15.85 * np.power(D_lmbda, -0.6)  # in degrees

        # Appendix 8, Annex 3, part a)
        if D_lmbda >= 100:
            if 0 <= phi < phi_m:
                G_phi = G_max - 2.5e-3 * ((D_lmbda * phi)**2)
            elif phi_m <= phi < phi_r:
                G_phi = G_1
            elif phi_m <= phi < 48:
                G_phi = 32 - 25 * np.log10(phi)
            elif 48 <= phi < 180:
                G_phi = -10
            else:
                raise ValueError(
                    "Inappropriate separation angle value submitted to antenna pattern. "
                    "Phi must be between 0 and 180. Phi: {}".format(phi))

        # Appendix 8, Annex 3, part b)
        else:
            if 0 <= phi < phi_m:
                G_phi = G_max - 2.5e-3 * ((D_lmbda * phi)**2)
            elif phi_m <= phi < (100 * np.power(D_lmbda, -1)):
                G_phi = G_1
            elif (100 * np.power(D_lmbda, -1)) <= phi < 48:
                G_phi = 52 - 10 * np.log10(D_lmbda) - 25 * np.log10(phi)
            elif 48 <= phi <= 180:
                G_phi = 10 - 10 * np.log10(D_lmbda)
            else:
                raise ValueError(
                    "Inappropriate separation angle value submitted to antenna pattern. "
                    "Phi must be between 0 and 180. Phi: {}".format(phi))

        # Return G
        return G_phi

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches,too-many-locals
    def gain_s580_6_es(gain_es, sep_angle):
        """
            Pattern from ITU Antenna patterns page, supposedly for 580-6.
            https://www.itu.int/en/ITU-R/software/Documents/ant-pattern/APL_DOC_BY_PATTERN_NAME/APEREC015V01.pdf

            Args:
                gain_es (float): RX gain of earth station
                sep_angle (float): Separation angle (degrees) between victim satellite and
                                   chosen interfering satellite.

            Returns:
                float: G off axis (dB) of ground station.

            Notes:
                Appendix 30B Earth station antenna pattern since WRC-03 applicable
                    for D/lambda > 50.
                Pattern is extended for D/lambda < 50 as in Appendix 8.
                Pattern is extended for angles greater than 20 degrees as in Recommendation
                    ITU-R S.465-5.
                Pattern is extended in the main-lobe range as in Appendix 7 to produce
                    continuous curves.

                Error: Phi_b is less than Phi_r
                Error: Gmax is less than G1. Square root of negative value.
        """
        eff = 0.7  # BR software sets antenna efficiency to 0.7 for technical examination.

        d_lambda = np.sqrt((np.power(10, gain_es / 10)) / (eff * (np.power(np.pi, 2))))
        phi = sep_angle

        if d_lambda < 50:
            g_1 = 2 + 15 * np.log10(d_lambda)
            phi_r = 100 * (1 / d_lambda)
        elif 50 <= d_lambda < 100:
            g_1 = -21 + 25 * np.log10(d_lambda)
            phi_r = 100 * (1 / d_lambda)
        else:
            g_1 = -1 + 15 * np.log10(d_lambda)
            phi_r = 15.85 * np.power(d_lambda, -0.6)

        phi_b = np.power(10, 42 / 25)
        phi_m = 20 * (1 / d_lambda) * np.sqrt(gain_es - g_1)

        # Intervals
        ivl0 = 0
        ivl1 = phi_m
        ivl2 = phi_r
        ivl3 = 19.95
        ivl4 = phi_b
        ivl5 = 180

        if d_lambda >= 50:
            if ivl0 <= phi < ivl1:
                G_phi = gain_es - 2.5 * 0.001 * np.power(d_lambda * phi, 2)
            elif ivl1 <= phi < ivl2:
                G_phi = g_1
            elif ivl2 <= phi <= ivl3:
                G_phi = 29 - 25 * np.log10(phi)
            elif ivl3 < phi < ivl4:
                G_phi = min(-3.5, 32 - 25 * np.log10(phi))
            elif ivl4 <= phi <= ivl5:
                G_phi = float(-10)
            else:
                raise ValueError(
                    "Inappropriate separation angle value submitted to antenna pattern."
                    " Phi must be between 0 and 180. Phi: {}".format(phi))
        else:
            # Interval 3 is skipped in this case.

            if ivl0 <= phi < ivl1:
                G_phi = gain_es - 2.5 * 0.001 * np.power(d_lambda * phi, 2)
            elif ivl1 <= phi < ivl2:
                G_phi = g_1
            elif ivl2 <= phi < ivl4:
                G_phi = 52 - 10 * np.log10(d_lambda) - 25 * np.log10(phi)
            elif ivl4 <= phi <= ivl5:
                G_phi = 10 - 10 * np.log10(d_lambda)
            else:
                raise ValueError(
                    "Inappropriate separation angle value submitted to antenna pattern."
                    " Phi must be between 0 and 180. Phi: {}".format(phi))

        # Return G
        return G_phi

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches,too-many-statements
    def gain_s1428_es(diam, sep_angle, freq_hz):
        """ Off-axis gain - earth station (ITU REC S.1428)
                    Args:
                        diam (float): Antenna diameter (m)
                        sep_angle (float): Separation angle (degrees) between victim satellite and
                                           chosen interfering satellite.
                        freq_hz (float): Frequency (Hz)
                    Returns:
                        G_phi (float): gain at the angle ψ from the main beam direction (dBi)
        """
        # Calculate transmit wavelength.
        wavelength = consts.c / freq_hz
        phi = abs(sep_angle)

        # Set variables based on D/λ
        d_lamba = diam / wavelength
        if d_lamba <= 100.0:
            g_max = 20.0 * np.log10(d_lamba) + 7.7
            g_1 = 29.0 - 25.0 * np.log10(95.0 * (1.0 / d_lamba))
            phi_m = 20.0 * (1.0 / d_lamba) * np.sqrt(g_max - g_1)
        else:
            g_max = 20.0 * np.log10(d_lamba) + 8.4
            g_1 = -1.0 + 15.0 * np.log10(d_lamba)
            phi_m = 20.0 * (1.0 / d_lamba) * np.sqrt(g_max - g_1)
            phi_r = 15.85 * d_lamba**-0.6

        # Calculate ES off-axis gain based on sep_angle.
        if 20.0 <= d_lamba <= 25.0:
            if 0 <= phi < phi_m:
                G_phi = g_max - (2.5 * 10**-3) * (d_lamba * phi)**2
            elif phi_m <= phi < (95.0 * (1 / d_lamba)):
                G_phi = g_1
            elif (95.0 * (1.0 / d_lamba)) <= phi < 33.1:
                G_phi = 29.0 - 25.0 * np.log10(phi)
            elif 33.1 < phi <= 80.0:
                G_phi = -9.0
            elif 80.0 < phi < 180.0:
                G_phi = -5.0
        elif 25.0 < d_lamba <= 100.0:
            if 0.0 <= phi < phi_m:
                G_phi = g_max - (2.5 * 10**-3) * (d_lamba * phi)**2
            elif phi_m <= phi < (95.0 * (1.0 / d_lamba)):
                G_phi = g_1
            elif (95.0 * (1.0 / d_lamba)) <= phi <= 33.1:
                G_phi = 29.0 - 25.0 * np.log10(phi)
            elif 33.1 < phi <= 80.0:
                G_phi = -9.0
            elif 80.0 < phi <= 120.0:
                G_phi = -4.0
            elif 120.0 < phi < 180.0:
                G_phi = -9.0
        elif d_lamba > 100.0:
            if 0 <= phi < phi_m:
                G_phi = g_max - (2.5 * 10**-3) * (d_lamba * phi)**2
            elif phi_m <= phi < phi_r:
                G_phi = g_1
            elif phi_r <= phi < 10.0:
                G_phi = 29.0 - 25.0 * np.log10(phi)
            elif 10.0 <= phi < 34.1:
                G_phi = 34.0 - 30.0 * np.log10(phi)
            elif 34.1 <= phi < 80.0:
                G_phi = -12.0
            elif 80.0 <= phi < 120.0:
                G_phi = -7.0
            elif 120.0 <= phi < 180.0:
                G_phi = -12.0

        return G_phi

    @staticmethod
    # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    def gain_s1528_sat(gain_sat, diam, sep_angle, freq_hz, hpbw, L_N=-15):
        """ Off-axis gain - space station (ITU REC S.1528, recommends 1.2)
            Args:
                gain_sat (float): Peak antenna gain
                diam (float): Antenna diameter (m)
                sep_angle (float): Separation angle (degrees) between victim satellite and
                                   chosen interfering satellite.
                freq_hz (float):
                hpbw (float): Half power (3 dB) beam width
                L_N (int): Near-in-side-lobe level (dB) relative to the peak gain. Can be
                    one of -15, -20, -25, -30.

            Returns:
                G_phi (float): gain at the angle ψ from the main beam direction (dBi)

            Notes:
                Pattern is configured for circular beams only.
            From ITU-R REC S.1528:
                phi_b: one-half the 3 dB beamwidth in the plane of interest (3 dB below Gm)(degrees)
                phi_b: sqrt(1200)/(D/λ) for minor axis (use actual values if known) (degrees)
                phi_b: (major axis / minor axis) sqrt(1200)/(D/λ) for major axis (use actual
                    values if known) (degrees)

                The numeric values of a, b, and α for LN = –15 dB, –20 dB, –25 dB, and –30 dB
                side-lobe levels are given in Table 1. The values of a and α for LN = –30 dB
                require further study. Administrations are invited to provide data to enable the
                values of a and α for LN = –30 dB to be refined.

                L_N(dB)
                (dB)    a                               b       α
                –15     2.58*np.sqrt(1 − 1.4 log(z))    6.32    1.5
                –20     2.58*np.sqrt(1 − 1.0 log(z))    6.32    1.5
                –25     2.58*np.sqrt(1 − 0.6 log(z))    6.32    1.5
                –30     2.58*np.sqrt(1 − 0.4 log(z))    6.32    1.5
        """
        D_lambda = diam / (consts.c / freq_hz)
        G_m = gain_sat  # maximum gain in the main lobe (dBi)
        z = 1  # (major axis/minor axis) for the radiated beam
        L_F = 0  # 0 dBi far-out side-lobe level (dBi)
        L_B = max(15 + L_N + 0.25 * G_m + 5 * np.log10(z), 0.0)  # back-lobe level (dBi)
        phi = sep_angle  # Off-axis angle (degrees)
        phi_b = hpbw / 2  # half the 3 dB beamwidth in the plane of interest (3 dB below G_m)(deg)
        a_coeff = {"-15": 1.4, "-20": 1.0, "-25": 0.6, "-30": 0.4}
        a = 2.58 * np.sqrt(1.0 - a_coeff[str(L_N)] * np.log10(z))
        b = 6.32
        alpha = 1.5
        X = G_m + L_N + 25 * np.log10(b * phi_b)
        Y = b * phi_b * (10**(0.04 * (G_m + L_N - L_F)))

        # Intervals
        ivl0 = 0
        ivl1 = a * phi_b
        ivl2 = 0.5 * b * phi_b
        ivl3 = b * phi_b
        ivl4 = Y
        ivl5 = 90

        # Reference pattern 1.2
        if ivl0 <= phi <= ivl1:
            G_phi = G_m - 3 * ((phi / phi_b)**alpha)
        elif ivl1 < phi <= ivl2:
            G_phi = G_m + L_N + 20 * np.log10(z)
        elif ivl2 < phi <= ivl3:
            G_phi = G_m + L_N
        elif ivl3 < phi <= ivl4:
            G_phi = X - 25.0 * np.log10(phi)
        elif ivl4 < phi <= ivl5:
            G_phi = L_F
        elif ivl5 < phi:
            G_phi = L_B
        else:
            raise ValueError("Inappropriate separation angle value submitted to antenna pattern."
                             " Phi must be between 0 and 180. Phi: {}".format(phi))

        return G_phi
