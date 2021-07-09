"""Base classes and functions for the antenna module."""
from abc import ABCMeta, abstractmethod
import importlib
from types import SimpleNamespace
import scipy.constants as consts
import numpy as np
from scipy.spatial.transform import Rotation as R



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
    # pylint: disable=unused-argument
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
    def interfering_sat_dl_psd(self, sim_geom):
        """Interfering satellite downlink PSD abstract method"""
        # pylint: disable=unnecessary-pass
        pass

    @abstractmethod
    def interfering_es_ul_psd(self, sim_geom):
        """Interfering earth station uplink PSD abstract method"""
        # pylint: disable=unnecessary-pass
        pass

    @abstractmethod
    def victim_es_dl_g_over_t(self, sim_geom, mode):
        """Victim earth station downlink G/T abstract method"""
        # pylint: disable=unnecessary-pass
        pass

    @abstractmethod
    def victim_sat_ul_g_over_t(self, sim_geom, mode):
        """Victim satellite uplink G/T abstract method"""
        # pylint: disable=unnecessary-pass
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
        def interfering_sat_dl_psd(sim_geom):  # pylint: disable=unused-argument
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

            # elevation in degrees
            if sim_geom['inter'].elev > high_el_boundary:
                return low_el_pfd_hz
            return high_el_pfd_hz

        return interfering_sat_dl_psd

    @staticmethod
    def interfering_sat_dl_psd_nd(gain_sat_dl, power_db, bw_hz, pointing=False):
        """Non Directional Antenna, as applied to calculate interfering satellite downlink PSD."""

        def interfering_sat_dl_psd(self, sim_geom):  # pylint: disable=unused-argument
            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_nd_sat(gain_sat_dl) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_sat_dl_psd

    @staticmethod
    def interfering_sat_dl_psd_abcdphi(gain_sat_dl, power_db, bw_hz, A, B, C, D, phi1, pointing=False):
        """ABCDphi Antenna, as applied to calculate interfering satellite downlink PSD."""

        def interfering_sat_dl_psd(self, sim_geom):  # pylint: disable=unused-argument
            off_axis_angle = sim_geom['inter'].nad_ang
            if pointing:
                off_axis_angle = 0
            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_abcdphi_sat(gain_sat_dl, off_axis_angle, A, B, C, D, phi1) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_sat_dl_psd

    @staticmethod
    def interfering_sat_dl_psd_465_5(gain_sat_dl, power_db, bw_hz, freq_hz, A, pointing=False):
        """465-5 Antenna, as applied to calculate interfering satellite downlink PSD."""

        def interfering_sat_dl_psd(self, sim_geom):  # pylint: disable=unused-argument
            off_axis_angle = sim_geom['inter'].nad_ang
            if pointing:
                off_axis_angle = 0
            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_465_5_sat(gain_sat_dl, off_axis_angle, freq_hz, A) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_sat_dl_psd


    @staticmethod
    def interfering_sat_dl_psd_672_4(gain_sat_dl, power_db, bw_hz, freq_hz, phi_0, pointing=False):
        """672-4 Antenna, as applied to calculate interfering satellite downlink PSD."""

        def interfering_sat_dl_psd(self, sim_geom):  # pylint: disable=unused-argument
            off_axis_angle = sim_geom['inter'].nad_ang
            if pointing:
                off_axis_angle = 0
            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_672_4_sat(gain_sat_dl, off_axis_angle, freq_hz, phi_0) + power_db - 10 * np.log10(
                bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_sat_dl_psd

    @staticmethod
    def interfering_sat_dl_psd_s1528(gain_sat_dl,  # pylint: disable=invalid-name,too-many-arguments
                                     power_db,
                                     bw_hz,
                                     hpbw,
                                     L_N=-15,
                                     pointing=False):
        """Standard pattern S.1528, as applied to calculate interfering satellite downlink PSD."""

        def interfering_sat_dl_psd(self, sim_geom):  # pylint: disable=unused-argument
            off_axis_angle = sim_geom['inter'].nad_ang
            if pointing:
                off_axis_angle = 0

            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_s1528_sat(
                gain_sat_dl, off_axis_angle, self.frequency.dl, hpbw,
                L_N=L_N) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['inter'].dist, self.frequency.dl)

        return interfering_sat_dl_psd

    @staticmethod
    def interfering_es_ul_psd_nd(gain_es_ul, power_db, bw_hz, pointing=False):
        """Non Directional Antenna, as applied to calculate interfering satellite uplink PSD."""

        def interfering_es_ul_psd(self, sim_geom):  # pylint: disable=unused-argument
            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_nd_es(gain_es_ul) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_es_ul_psd

    @staticmethod
    def interfering_es_ul_psd_abcdphi(gain_es_ul, power_db, bw_hz,A,B,C,D,phi1 ,pointing=False):
        """ABCDphi Antenna, as applied to calculate interfering satellite uplink PSD."""

        def interfering_es_ul_psd(self, sim_geom):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['vic'].fixed_ang_inter)>0.001:
                sep_angle = sim_geom['vic'].fixed_ang_inter
            # sep_angle = sim_geom['vic'].zen_ang
            if pointing:
                sep_angle = 0
            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_abcdphi_es(gain_es_ul,sep_angle,A,B,C,D,phi1) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_es_ul_psd

    @staticmethod
    def interfering_es_ul_psd_465_5(gain_es_ul, power_db, bw_hz,freq_hz, A, pointing=False):
        """ABCDphi Antenna, as applied to calculate interfering satellite uplink PSD."""

        def interfering_es_ul_psd(self, sim_geom):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['vic'].fixed_ang_inter) > 0.001:
                sep_angle = sim_geom['vic'].fixed_ang_inter
            if pointing:
                sep_angle = 0
            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_465_5_es(gain_es_ul,sep_angle,freq_hz,A) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_es_ul_psd


    @staticmethod
    def interfering_es_ul_psd_ap8(gain_es_ul, power_db, bw_hz, pointing=False):
        """Standard Appendix 8 pattern, as applied to calculate interfering satellite uplink PSD."""

        def interfering_es_ul_psd(self, sim_geom):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['vic'].fixed_ang_inter) > 0.001:
                sep_angle = sim_geom['vic'].fixed_ang_inter
            if pointing:
                sep_angle = 0

            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_ap8_es(gain_es_ul, sep_angle) + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_es_ul_psd

    @staticmethod
    def interfering_es_ul_psd_s580_6(gain_es_ul, power_db, bw_hz, pointing=False):
        """Standard pattern S.580-6, as applied to calculate interfering ES uplink PSD."""

        def interfering_es_ul_psd(self, sim_geom):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['vic'].fixed_ang_inter) > 0.001:
                sep_angle = sim_geom['vic'].fixed_ang_inter
            if pointing:
                sep_angle = 0

            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_s580_6_es(gain_es_ul, sep_angle) \
                + power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_es_ul_psd

    @staticmethod
    def interfering_es_ul_psd_s1428(gain_es_ul, diam_es, power_db, bw_hz, pointing=False):
        """Standard pattern S.1428, as applied to calculate interfering ES uplink PSD."""

        def interfering_es_ul_psd(self, sim_geom):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['vic'].fixed_ang_inter) > 0.001:
                sep_angle = sim_geom['vic'].fixed_ang_inter
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
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_es_ul_psd

    @staticmethod
    def interfering_es_ul_psd_s1528(gain_es_ul,    # pylint: disable=invalid-name,too-many-arguments
                                    es_power_db,
                                    bw_hz,
                                    hpbw,
                                    L_N=-15,
                                    pointing=False):
        """Standard pattern S.1528, as applied to calculate interfering ES uplink PSD."""

        # print("Warning: Rec. 1528 is being used as an earth station pattern, but is only "
        #       "recommended for use as a satellite pattern.", file=stderr)

        def interfering_es_ul_psd(self, sim_geom):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['vic'].fixed_ang_inter) > 0.001:
                sep_angle = sim_geom['vic'].fixed_ang_inter
            if pointing:
                sep_angle = 0

            # Return EIRP = G + P - BW - FSPL
            eirp_hz = self.gain_s1528_sat(
                gain_es_ul,  sep_angle, self.frequency.ul, hpbw,
                L_N=L_N) + es_power_db - 10 * np.log10(bw_hz)

            # Return the Received PSD in dBW/Hz/m^2
            return eirp_hz + self.freespace_loss(sim_geom['vic'].dist, self.frequency.ul)

        return interfering_es_ul_psd

    # Standard G/T calculators
    @staticmethod
    def victim_sat_ul_g_over_t_nd(gain_sat_ul,
                                 temp_sys_sat,
                                 other_losses_db=0,
                                 pointing=False):
        """Non directional satellite as applied to calculate victim Sat uplink G/T."""

        def victim_sat_ul_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            # Return G/T
            return self.gain_nd_sat(gain_sat_ul) - 10 * np.log10(temp_sys_sat) - other_losses_db

        return victim_sat_ul_gain

    @staticmethod
    def victim_sat_ul_g_over_t_abcdphi(gain_sat_ul,
                                 temp_sys_sat,
                                 A,
                                 B,
                                 C,
                                 D,
                                 phi1,
                                 other_losses_db=0,
                                 pointing=False):
        """ABCDphi satellite as applied to calculate victim Sat uplink G/T."""

        def victim_sat_ul_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            # Return G/T
            off_axis_angle = sim_geom['vic'].nad_ang
            if pointing:
                off_axis_angle = 0
            return self.gain_abcdphi_sat(gain_sat_ul, off_axis_angle, A, B, C, D, phi1) - 10 * np.log10(temp_sys_sat) - other_losses_db

        return victim_sat_ul_gain

    @staticmethod
    def victim_sat_ul_g_over_t_465_5(gain_sat_ul,
                                       temp_sys_sat,
                                       A,
                                       other_losses_db=0,
                                       pointing=False):
        """465-5 satellite as applied to calculate victim Sat uplink G/T."""

        def victim_sat_ul_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            # Return G/T
            off_axis_angle = sim_geom['vic'].nad_ang
            if pointing:
                off_axis_angle = 0
            return self.gain_465_5_sat(gain_sat_ul, off_axis_angle, A) - 10 * np.log10(
                temp_sys_sat) - other_losses_db

        return victim_sat_ul_gain

   

    @staticmethod
    def victim_sat_ul_g_over_t_672_4(gain_sat_ul,
                                       temp_sys_sat,
                                       freq_hz,
                                       phi_0,
                                       other_losses_db=0,
                                       pointing=False):
        """672-4 satellite as applied to calculate victim Sat uplink G/T."""

        def victim_sat_ul_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            # Return G/T
            off_axis_angle = sim_geom['vic'].nad_ang
            if pointing:
                off_axis_angle = 0
            return self.gain_672_4_sat(gain_sat_ul, off_axis_angle, freq_hz, phi_0) - 10 * np.log10(
                temp_sys_sat) - other_losses_db

        return victim_sat_ul_gain

    @staticmethod
    def victim_sat_ul_g_over_t_s1528(gain_sat_ul,  # pylint: disable=invalid-name,too-many-arguments
                                     temp_sys_sat,
                                     hpbw,
                                     other_losses_db=0,
                                     L_N=-15,
                                     pointing=False,
                                     use_nad_ang=False):
        """Standard pattern S.1528, as applied to calculate victim satellite uplink G/T."""

        def victim_sat_ul_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            off_axis_angle = sim_geom['vic'].nad_ang

            if pointing:
                off_axis_angle = 0
            if use_nad_ang:
                off_axis_angle = sim_geom['vic'].nad_ang

            # Return G/T
            return self.gain_s1528_sat(gain_sat_ul, off_axis_angle, self.frequency.ul, hpbw,
                                       L_N) - 10 * np.log10(temp_sys_sat) - other_losses_db

        return victim_sat_ul_gain

    

    @staticmethod
    def victim_es_dl_g_over_t_nd(gain_es_dl,
                                     temp_sys_es,
                                     other_losses_db=0,
                                     pointing=False):
        """Non directional earth station, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            # Return G/T
            return self.gain_nd_es(gain_es_dl) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_abcdphi(gain_es_dl,
                                     temp_sys_es,
                                      A,
                                      B,
                                      C,
                                      D,
                                      phi1,
                                     other_losses_db=0,
                                     pointing=False,):
        """abcdphi earth station, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0

            # Return G/T
            return self.gain_abcdphi_es(gain_es_dl,sep_angle,A,B,C,D,phi1) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_465_5(gain_es_dl,
                                      temp_sys_es,
                                      A,
                                      other_losses_db=0,
                                      pointing=False, ):
        """465-5 earth station, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0

            # Return G/T
            return self.gain_465_5_es(gain_es_dl,sep_angle,A) - 10 * np.log10(
                temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    
  
    
    @staticmethod
    def victim_es_dl_g_over_t_240V01(gain_es_dl,
                                 temp_sys_es,
                                 other_losses_db=0,
                                 pointing=False):
        """240V01 earth station, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0
            # Return G/T
            return self.gain_240V01_es(sep_angle) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_242V01(gain_es_dl,
                                     temp_sys_es,
                                     other_losses_db=0,
                                     pointing=False):
        """242V01 earth station, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0
            # Return G/T
            return self.gain_242V01_es(sep_angle) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_244V01(gain_es_dl,
                                     temp_sys_es,
                                     other_losses_db=0,
                                     pointing=False):
        """244V01 earth station, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0
            # Return G/T
            return self.gain_244V01_es(gain_es_dl,sep_angle) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_028V01(gain_es_dl,
                                     temp_sys_es,
                                     other_losses_db=0,
                                     pointing=False):
        """028V01 earth station, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0
            # Return G/T
            return self.gain_028V01_es(gain_es_dl, sep_angle) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_230V01(gain_es_dl,
                                     temp_sys_es,
                                     other_losses_db=0,
                                     pointing=False):
        """230V01 earth station, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0
            # Return G/T
            return self.gain_230V01_es(gain_es_dl, sep_angle) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_235V01(gain_es_dl,
                                     temp_sys_es,
                                     other_losses_db=0,
                                     pointing=False):
        """235V01 earth station, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0
            # Return G/T
            return self.gain_235V01_es(gain_es_dl, sep_angle) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_229V01(gain_es_dl,
                                     temp_sys_es,
                                     other_losses_db=0,
                                     pointing=False):
        """229V01 earth station, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0
            # Return G/T
            return self.gain_229V01_es(gain_es_dl, sep_angle) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_ap8(gain_es_dl,
                                   temp_sys_es,
                                   other_losses_db=0,
                                   pointing=False):
        """Standard Appendix 8 pattern, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0

            # Return G/T
            return self.gain_ap8_es(
                gain_es_dl, sep_angle) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    @staticmethod
    def victim_es_dl_g_over_t_s580_6(gain_es_dl,
                                     temp_sys_es,
                                     other_losses_db=0,
                                     pointing=False):
        """Standard pattern S.580-6, as applied to calculate victim ES downlink G/T."""

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
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

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
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
                                    temp_sys_es,
                                    hpbw,
                                    other_losses_db=0,
                                    L_N=-15,
                                    pointing=False):
        """Standard pattern S.1528, as applied to calculate victim ES downlink G/T."""

        # print("Warning: Rec. 1528 is being used as an earth station pattern, but is only "
        #       "recommended for use as a satellite pattern.", file=stderr)

        def victim_es_dl_gain(self, sim_geom, mode):  # pylint: disable=unused-argument
            sep_angle = sim_geom['vic'].sep_ang
            if abs(sim_geom['inter'].fixed_ang_vic) > 0.001:
                sep_angle = sim_geom['inter'].fixed_ang_vic
            if pointing:
                sep_angle = 0

            # Return G/T
            return self.gain_s1528_sat(gain_es_dl, sep_angle, self.frequency.dl, hpbw,
                                       L_N) - 10 * np.log10(temp_sys_es) - other_losses_db

        return victim_es_dl_gain

    # Standard Off-axis gain calculators
    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_nd_es(gain_es_dl):
        """
                    Non-directional earth station (ITU APEND_099V01)

                    Args:
                        gain_es_dl (float): RX gain of earth station

                    Returns:
                        gain_es_dl (float): RX gain of earth station
                    """
        # Return Gain
        return gain_es_dl

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_abcdphi_es(gain_es_dl,sep_angle,A,B,C,D,phi1):
        """
                    Non-standard generic earth station antenna pattern described
                    by 4 main coefficients: A, B, C, D and angle phi1. (ITU APENST807V01)

                    Args:
                        gain_es_dl (float): RX gain of earth station
                        sep_angle (float): Separation angle (degrees) between victim satellite and
                                           chosen interfering satellite.
                        A (float): Coeffecient A
                        B (float): Coeffecient B
                        C (float): Coeffecient C
                        D (float): Coeffecient D
                        phi1 (float): Angle phi1 (degrees)
                    Returns:
                        G_phi (float): Off axis gain of earth station
                    """
        # Return Gain

        if sep_angle>=0 and sep_angle<1:
            G_phi = gain_es_dl
        elif sep_angle>1 and sep_angle<=phi1:
            G_phi = A-B*np.log10(sep_angle)
        elif sep_angle>phi1 and sep_angle<=180:
            G_phi = max(min(A-B*np.log10(phi1),C-D*np.log10(sep_angle)),-10)
        else:
            raise ValueError(
                "Inappropriate separation angle value submitted to antenna pattern. "
                "Phi must be between 0 and 180. Phi: {}".format(sep_angle))

        if G_phi>gain_es_dl:
            G_phi = gain_es_dl
        if G_phi<-10:
            G_phi = -10
        return G_phi

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_465_5_es(gain_es_dl,sep_angle,A):
        """
                    Non-standard generic earth station antenna pattern similar to
                    that in Recommendation ITU-R S.465-5, where the side-lobe
                    radiation is represented by the expression CoefA - 25 log(phi). (ITU APENST806V01)

                    Args:
                        gain_es_dl (float): RX gain of earth station
                        sep_angle (float): Separation angle (degrees) between victim satellite and
                                           chosen interfering satellite.
                        A (float): Coeffecient A
                    Returns:
                        G_phi (float): Off axis gain of earth station
                    """
        # Return Gain
        G_max = gain_es_dl

        eta = 0.7 # Effeciency, taken from ITU doc
        D_lmbda = np.sqrt((10**(0.1*G_max))/(eta*np.power(np.pi,2)))


        if D_lmbda > 100:
            G1 = A
            phi_r = 1
        elif D_lmbda <= 100:
            G1 = A - 50 + 25*np.log10(D_lmbda)
            phi_r = 100/D_lmbda

        phi_m = (20 / D_lmbda) * np.sqrt(G_max - G1)
        phi_b = 10**((A+10)/25)

        if 0<= sep_angle and sep_angle< phi_m:
            G_phi = G_max -0.0025 * (D_lmbda*sep_angle)**2
        elif phi_m <= sep_angle and sep_angle < phi_r:
            G_phi = G1
        elif phi_r <= sep_angle and sep_angle <= 180:
            G_phi = max(A - 25*np.log10(sep_angle),-10)
        else:
            raise ValueError(
                "Inappropriate separation angle value submitted to antenna pattern. "
                "Phi must be between 0 and 180. Phi: {}".format(sep_angle))

        return G_phi

    

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_240V01_es(sep_angle):
        """
                    ITU APEMLA240V01 earth station for GSO

                    Args:
                        sep_angle (float): Separation angle (degrees)

                    Returns:
                        gain_es_dl (float): RX gain of earth station
                    """
        # Return Gain
        if 0<= sep_angle and sep_angle < 45:
            gain_es_dl = 5-0.002*sep_angle**2
        elif 45<= sep_angle and sep_angle <= 180:
            gain_es_dl = 1
        return gain_es_dl

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_242V01_es(sep_angle):
        """
                    ITU APEMLA242V01 earth station for GSO

                    Args:
                        sep_angle (float): Separation angle (degrees)

                    Returns:
                        gain_es_dl (float): RX gain of earth station
                    """
        # Return Gain
        D = 0.3
        c = 3e8
        f = 1.5e9
        lmbda = c /f
        D_lmbda = D/lmbda

        if 0 <= sep_angle and sep_angle < 11.4:
            gain_es_dl = 14 - 0.003 * sep_angle ** 2
        elif 11.4 <= sep_angle and sep_angle < 28.8:
            gain_es_dl = 10.1
        elif 28.8<= sep_angle and sep_angle < 73:
            gain_es_dl = 52 - 10*np.log10(D_lmbda) -25*np.log10(sep_angle)
        elif 73 <= sep_angle and sep_angle <= 180:
            gain_es_dl = 0
        return gain_es_dl

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_244V01_es(max_gain,sep_angle):
        """
                    ITU APEMLA244V01 earth station for GSO

                    Args:
                        max_gain (float): Maximum gain of ground station
                        sep_angle (float): Separation angle (degrees)

                    Returns:
                        gain_es_dl (float): RX gain of earth station
                    """
        # Return Gain
        D = 0.7
        c = 3e8
        f = 1.5e9
        lmbda = c / f
        D_lmbda = D / lmbda

        if 0 <= sep_angle and sep_angle < 6.7:
            gain_es_dl = max_gain - 0.12 * sep_angle ** 2
        elif 6.7 <= sep_angle and sep_angle < 14.4:
            gain_es_dl = 14.6
        elif 14.4 <= sep_angle and sep_angle < 53.3:
            gain_es_dl = 52 - 10 * np.log10(D_lmbda) - 25 * np.log10(sep_angle)
        elif 53.3 <= sep_angle and sep_angle <= 180:
            gain_es_dl = 0
        return gain_es_dl

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_028V01_es(max_gain, sep_angle):
        """
                    ITU APEREC028V01 earth station for GSO

                    Args:
                        max_gain (float): Maximum gain of ground station
                        sep_angle (float): Separation angle (degrees)

                    Returns:
                        gain_es_dl (float): RX gain of earth station
                    """
        # Return Gain
        A = 44
        B = 25
        phi1 = 40
        phib = 90
        G_min = -5

        if 0 <= sep_angle and sep_angle < phi1:
            gain_es_dl = max_gain
        elif phi1 <= sep_angle and sep_angle < phib:
            gain_es_dl = A-B*np.log10(sep_angle)
        elif phib<= sep_angle and sep_angle <= 180:
            gain_es_dl = G_min
        return gain_es_dl

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_230V01_es(max_gain, sep_angle):
        """
                    ITU APEUAE230V01 earth station for GSO

                    Args:
                        max_gain (float): Maximum gain of ground station
                        sep_angle (float): Separation angle (degrees)

                    Returns:
                        gain_es_dl (float): RX gain of earth station
                    """
        # Return Gain

        if 0 <= sep_angle and sep_angle < 60:
            gain_es_dl = max_gain
        elif 60 <= sep_angle and sep_angle < 90:
            gain_es_dl = -0.003*sep_angle**2 + 0.358*sep_angle - 4.68
        elif 90 <= sep_angle and sep_angle < 130:
            gain_es_dl = -0.1598*sep_angle + 17.622
        elif 130 <= sep_angle and sep_angle <= 180:
            gain_es_dl = -3.15
        return gain_es_dl

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_229V01_es(max_gain, sep_angle):
        """
                    ITU APEUAE229V01 earth station for GSO

                    Args:
                        max_gain (float): Maximum gain of ground station
                        sep_angle (float): Separation angle (degrees)

                    Returns:
                        gain_es_dl (float): RX gain of earth station
                    """
        # Return Gain
        phi0 = 28
        phim = 23
        phib = 50
        A = 38
        B = 25
        G_min = -5

        if 0 <= sep_angle and sep_angle < phim:
            gain_es_dl = max_gain - 12*(sep_angle/phi0)**2
        elif phim <= sep_angle and sep_angle < phib:
            gain_es_dl = A-B*np.log10(sep_angle)
        elif phib <= sep_angle and sep_angle <= 180:
            gain_es_dl = G_min
        return gain_es_dl

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_235V01_es(max_gain, sep_angle):
        """
                    ITU APEUAE235V01 earth station for GSO

                    Args:
                        max_gain (float): Maximum gain of ground station
                        sep_angle (float): Separation angle (degrees)

                    Returns:
                        gain_es_dl (float): RX gain of earth station
                    """
        # Return Gain
        phi1 = 6
        phim = 16
        phir = 30
        phib = 76
        A = 44.69
        B = 25.512
        G_min = -3.299
        G1 = 7
        phi2 = 13.334

        if 0 <= sep_angle and sep_angle < phi1:
            gain_es_dl = max_gain
        elif phi1 <= sep_angle and sep_angle < phim:
            gain_es_dl = max_gain*(1-((sep_angle-phi1)/phi2)**2)
        elif phim< sep_angle and sep_angle < phir:
            gain_es_dl = G1
        elif phir<= sep_angle and sep_angle< phib:
            gain_es_dl = A-B*np.log10(sep_angle)
        elif phib <= sep_angle and sep_angle <= 180:
            gain_es_dl = G_min
        return gain_es_dl

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_ap8_es(gain_es_dl, sep_angle):
        """
                    Off-axis gain - earth station (Appendix 8, Annex 3 pattern)

                    Args:
                        gain_es_dl (float): RX gain of earth station
                        sep_angle (float): Separation angle (degrees) between victim satellite and
                                           chosen interfering satellite.

                    Returns:
                        float: G off axis (dB) of ground station.
                    """

        # D = diam_es
        # lmbda = consts.c / freq_hz
        G_max = gain_es_dl

        # # In cases where D/lmbda is not given, it may be estimated as 20*log(D/lmbda) =~ G_max - 7.7
        # if not D or not lmbda:
        #     D_lmbda = np.exp((G_max - 7.7) / 20)
        # else:
        #     D_lmbda = D / lmbda

        D_lmbda = np.exp((G_max - 7.7) / 20)

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
    # pylint: disable=invalid-name,too-many-branches
    def gain_nd_sat(gain_sat):
        """
                    Non-directional satellite (ITU APEND_499V01)

                    Args:
                        gain_sat (float): RX gain of satellite

                    Returns:
                        gain_sat (float): RX gain of satellite
                    """
        # Return Gain
        return gain_sat

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_abcdphi_sat(gain_sat, sep_angle, A, B, C, D, phi1):
        """
                    Non-standard generic satellite antenna pattern described
                    by 4 main coefficients: A, B, C, D and angle phi1. (ITU APENST807V01)

                    Args:
                        gain_sat (float): RX gain of satellite
                        sep_angle (float): Separation angle (degrees) between victim satellite and
                                           chosen interfering satellite.
                        A (float): Coeffecient A
                        B (float): Coeffecient B
                        C (float): Coeffecient C
                        D (float): Coeffecient D
                        phi1 (float): Angle phi1 (degrees)
                    Returns:
                        G_phi (float): Off axis gain of earth station
                    """
        # Return Gain

        if sep_angle >= 0 and sep_angle < 1:
            G_phi = gain_sat
        elif sep_angle > 1 and sep_angle <= phi1:
            G_phi = A - B * np.log10(sep_angle)
        elif sep_angle > phi1 and sep_angle <= 180:
            G_phi = max(min(A - B * np.log10(phi1), C - D * np.log10(sep_angle)), -10)

        if G_phi > gain_sat:
            G_phi = gain_sat
        if G_phi < -10:
            G_phi = -10
        return G_phi

    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_465_5_sat(gain_sat, sep_angle, A):
        """
                    Non-standard generic satellite antenna pattern similar to
                    that in Recommendation ITU-R S.465-5, where the side-lobe
                    radiation is represented by the expression CoefA - 25 log(phi). (ITU APENST806V01)

                    Args:
                        gain_sat (float): RX gain of satellite
                        sep_angle (float): Separation angle (degrees) between victim satellite and
                                           chosen interfering satellite.
                        A (float): Coeffecient A
                    Returns:
                        G_phi (float): Off axis gain of satellite
                    """
        # Return Gain
        G_max = gain_sat
        eta = 0.7 # Effeciency, taken from ITU doc
        D_lmbda = np.sqrt((10**(0.1*G_max))/(eta*np.power(np.pi,2)))

        if D_lmbda > 100:
            G1 = A
            phi_r = 1
        elif D_lmbda <= 100:
            G1 = A - 50 + 25 * np.log10(D_lmbda)
            phi_r = 100 / D_lmbda

        phi_m = (20 / D_lmbda) * np.sqrt(G_max - G1)
        phi_b = 10 ** ((A + 10) / 25)

        if 0 <= sep_angle and sep_angle < phi_m:
            G_phi = G_max - 0.0025 * (D_lmbda * sep_angle) ** 2
        elif phi_m <= sep_angle and sep_angle < phi_r:
            G_phi = G1
        elif phi_r <= sep_angle and sep_angle <= 180:
            G_phi = max(A - 25 * np.log10(sep_angle), -10)

        return G_phi

  

     
    @staticmethod
    # pylint: disable=invalid-name,too-many-branches
    def gain_672_4_sat(gain_sat, sep_angle, freq_hz, phi_0):
        """
                    Non-standard generic satellite antenna pattern similar to
                    that in Recommendation ITU-R S.672-4 (ITU APSREC408V01)

                    Args:
                        gain_sat (float): RX gain of satellite
                        sep_angle (float): Separation angle (degrees) between victim satellite and
                                           chosen interfering satellite.
                        freq_hz (float): Frequency (Hz)
                        phi_0 (float): Constant phi0 from the definition
                    Returns:
                        G_phi (float): Off axis gain of satellite
                    """
        # Return Gain
        G_max = gain_sat
        lmbda = consts.c / freq_hz
        a = 2.58
        b = 6.32
        L_s = -20

        if 0 <= sep_angle/phi_0 and sep_angle/phi_0 <= a/2:
            G_phi = G_max - 12*(sep_angle/phi_0)**2
        elif a/2 < sep_angle/phi_0 and sep_angle/phi_0 <= b/2:
            G_phi = G_max +L_s
        elif sep_angle/phi_0>b/2:
            G_phi = G_max+L_s+20-25*np.log10(2*sep_angle/phi_0)

        return G_phi

    @staticmethod
    # pylint: disable=invalid-name,too-many-arguments,too-many-locals
    def gain_s1528_sat(gain_sat, sep_angle, freq_hz, hpbw, L_N=-15):
        """ Off-axis gain - space station (ITU REC S.1528, recommends 1.2)
            Args:
                gain_sat (float): Peak antenna gain
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
