"""Main simulation engine."""
import concurrent.futures
import datetime
from functools import wraps
from itertools import chain, repeat
import os
from collections import namedtuple

import numpy as np
from numpy import arange, array, log10, pi, repeat as np_repeat, dstack, empty
import scipy.constants as consts

# pylint: disable=import-error,unused-import
from skyfield.api import load, Topos
import itur
from .interval import partition_interval_chunks
from .satellite import Constellation, UlDlPair, VicInterPair, longest_hold
from .antenna import BaseAntennaModel as Bam

SimGeometry = namedtuple('SimGeometry', 'sep_ang elev nad_ang zen_ang fixed_ang_vic fixed_ang_inter dist lat lon name sat_grd_vec orb_vel_vec sat_nad_vec')


def _timed_calc_interval(time_per_step):  # pylint: disable=unused-argument
    """Time an InterferenceSim method that has an interval to operate on as the first argument.

    Args:
        time_per_step (float): Time required to complete a step of the calculation.

    Returns:
        function: Decorator function.
    """

    def _timed_calc_interval_decorator(func):
        """Time a function that has an interval to operate on as the first argument.

        Args:
            func (function): Function to wrap.

        Returns:
            function: Wrapped function.
        """

        @wraps(func)
        def wrapper(sim, interval, *args, **kwargs):
            sim_begin = datetime.datetime.now()
            print('\nSimulating: %i hours - %is step size - %i steps' %
                  (interval.length / 3600, interval.step, interval.steps))

            ret_val = func(sim, interval, *args, **kwargs)

            sim_end = datetime.datetime.now()
            sim_duration = sim_end - sim_begin
            print("%ss, %.5fs/step" % (sim_duration, sim_duration.total_seconds() / interval.steps))
            return ret_val

        return wrapper

    return _timed_calc_interval_decorator


class InterferenceSim:
    """Interference simulation engine."""

    def __init__(self, frequency, constels, position, parallelism=0):
        """Constructor.

        Args:
            frequency (UlDlPair(T, T)): T is float with frequency for up/downlink.
            constels (Constellation): Constellation to get tracked satellite coordinates for.
            position (LatLon): Ground station position.
            parallelism (int): Number of concurrent chunks to process.
        """
        self._victim, self._inter = constels
        self._frequency = frequency
        self._position = position
        self._parallelism = parallelism or os.cpu_count()
        self._epoch_start = self._victim.sats[0].epoch
        self.debug = True

    constellations = property(lambda self: VicInterPair(self._victim, self._inter))
    parallelism = property(lambda self: self._parallelism)

    # pylint: disable=invalid-name,too-many-locals,too-many-statements
    def _calculate_interference(self, t_curr, tracked, coords, vic_diff):
        """Return the C/I+N, I/N, C/I, and C/N for downlink and uplink for a single timestep.

         Args:
            t_curr (skyfield.timelib.Time): Time to calculate interference for.
            tracked (VicInterPair(T, T)): T is skyfield.sgp4lib.EarthSatellite being tracked.
            coords (VicInterPair(T, T)): T is IdxSphericalCoordinate for each tracked satellite.
            vic_diff (array): List of vectors between the victim satellites and the ground station.

        Returns:
            data (array or None): None if no satellites interfering or array with uplink and
                downlink I/N.
        """


        # One of the satellites not visible means no interference.
        if None in tracked:
            return empty((4, 2, 0))

        # Geometry Variables
        groundstation = Topos(*self._position)
        vic_to_grd = vic_diff[coords.vic['idx'].item()].at(t_curr)  # Vector from vic to ground.
        inter_to_grd = (tracked.inter - groundstation).at(t_curr)  # Vector from inter to ground.
        sep_angle = vic_to_grd.separation_from(inter_to_grd).degrees  # Angle between sats.
        vic_curr = tracked.vic.at(t_curr)  # Victim satellite position.
        inter_curr = tracked.inter.at(t_curr)  # Interferer satellite position.
        vic_nad_ang = vic_to_grd.separation_from(vic_curr).degrees  # Angle to nadir in radians.
        # Angle to nadir in radians.
        inter_nad_ang = inter_to_grd.separation_from(inter_curr).degrees
        zenith_point = groundstation.at(t_curr).from_altaz(alt_degrees=90., az_degrees=0.)
        vic_zen_ang = zenith_point.separation_from(vic_to_grd).degrees
        inter_zen_ang = zenith_point.separation_from(inter_to_grd).degrees

        # Find names of tracked satellites
        inter_name = tracked.inter.name
        vic_name = tracked.vic.name


        inter_subpoint = inter_curr.subpoint()
        vic_subpoint = vic_curr.subpoint()
        inter_lat, inter_lon = inter_subpoint.latitude.degrees, inter_subpoint.longitude.degrees
        vic_lat, vic_lon = vic_subpoint.latitude.degrees, vic_subpoint.longitude.degrees

        # Find orbital velocity vectors
        inter_curr = tracked.inter.at(t_curr)
        inter_orb_vel_vec = inter_curr.velocity.km_per_s
        vic_curr = tracked.vic.at(t_curr)
        vic_orb_vel_vec = vic_curr.velocity.km_per_s

        # Find nadir vectors
        inter_subpoint_gs = Topos(inter_lat,inter_lon)
        inter_nad_vec = (inter_subpoint_gs-tracked.inter).at(t_curr) # interfering nadir vector
        vic_subpoint_gs = Topos(vic_lat,vic_lon)
        vic_nad_vec = (vic_subpoint_gs- tracked.vic).at(t_curr) # victim nadir vector


        # If Victim ES is to be kept at a fixed azimuth/elevation, define that vector.
        if self.constellations.vic.fixed_params:
            fixed_el, fixed_az = self.constellations.vic.fixed_params
            fixed_point = groundstation.at(t_curr).from_altaz(alt_degrees=fixed_el,
                                                              az_degrees=fixed_az)
            fixed_vic_es_to_vic_sat_ang = fixed_point.separation_from(vic_to_grd).degrees
            fixed_vic_es_to_inter_sat_ang = fixed_point.separation_from(inter_to_grd).degrees
        else:
            fixed_vic_es_to_vic_sat_ang = 0
            fixed_vic_es_to_inter_sat_ang = 0

        # If Inter ES is to be kept at a fixed azimuth/elevation, define that vector.
        if self.constellations.inter.fixed_params:
            inter_fixed_el, inter_fixed_az = self.constellations.inter.fixed_params
            inter_fixed_point = groundstation.at(t_curr).from_altaz(alt_degrees=inter_fixed_el,
                                                              az_degrees=inter_fixed_az)
            fixed_inter_es_to_vic_sat_ang = inter_fixed_point.separation_from(vic_to_grd).degrees
            fixed_inter_es_to_inter_sat_ang = inter_fixed_point.separation_from(vic_to_grd).degrees
        else:
            fixed_inter_es_to_vic_sat_ang = 0
            fixed_inter_es_to_inter_sat_ang = 0
        
        sim_geom = {'vic': SimGeometry(sep_angle,
                                       coords.vic['el'].item(),
                                       vic_nad_ang,
                                       vic_zen_ang,
                                       fixed_vic_es_to_vic_sat_ang,
                                       fixed_inter_es_to_vic_sat_ang,
                                       coords.vic['r'].item(),
                                       vic_lat,
                                       vic_lon,
                                       vic_name,
                                       inter_to_grd,
                                       inter_orb_vel_vec,
                                       inter_nad_vec),
                    'inter': SimGeometry(sep_angle,          
                                         coords.inter['el'].item(),
                                         inter_nad_ang,   
                                         inter_zen_ang,
                                         fixed_vic_es_to_inter_sat_ang,
                                         fixed_inter_es_to_inter_sat_ang,
                                         coords.inter['r'].item(),
                                         inter_lat,
                                         inter_lon,
                                         inter_name,
                                         vic_to_grd,
                                         vic_orb_vel_vec,
                                         vic_nad_vec)
                    }

        # Other variables
        adj_factor = 0  # ITU-R BO.1696 Section 2.2.2, Z1
        rain_noise_temp = 0  # ITU-R BO.1696 Section 2.2.2, dT

        # Interfering PFD in dBW/Hz for DL and UL.
        psd = array((
            # Interfering ES PSD into victim sat
            self._inter.antenna_model.ul.es_psd(sim_geom),
            # Interfering satellite PSD to co-located ES's
            self._inter.antenna_model.dl.sat_psd(sim_geom),
            # Carrier ES PSD to into vic (carrier) sat
            self._victim.antenna_model.ul.es_psd(sim_geom),
            # Carrier satellite PSD to co-located ES's
            self._victim.antenna_model.dl.sat_psd(sim_geom)
        ))

        # G/T off axis for victim ground station and satellite, respectively.
        g_t = array((
            # Victim satellite G/T towards interfering ES signal.
            self._victim.antenna_model.ul.sat_g_over_t(sim_geom, 'inter'),
            # Victim ES G/T towards interfering satellite signal.
            self._victim.antenna_model.dl.es_g_over_t(sim_geom, 'inter'),
            # Victim satellite G/T towards carrier ES signal.
            self._victim.antenna_model.ul.sat_g_over_t(sim_geom, 'vic'),
            # Victim ES G/T towards carrier satellite signal.
            self._victim.antenna_model.dl.es_g_over_t(sim_geom, 'vic')
        ))

        # Rain fade
        inter_atmo_loss_dl = 0
        vic_atmo_loss_dl = 0
        inter_atmo_loss_ul = 0
        vic_atmo_loss_ul = 0

        # Downlink C/I+N (dB)
        g_t_vic_es = g_t[1]
        g_t_carrier_es = g_t[3]
        psd_int_dl = psd[1]
        psd_carrier_dl = psd[3]

        # Calculate the bandwidth overlap.
        vic_bw_dl = self._victim.antenna_model.props.es_bw_hz_dl
        vic_bw_ul = self._victim.antenna_model.props.sat_bw_hz_ul
        inter_bw_dl = self._inter.antenna_model.props.sat_bw_hz_dl
        inter_bw_ul = self._inter.antenna_model.props.es_bw_hz_ul
        inter_bw_overlap_dl = min(0, 10 * log10(vic_bw_dl / inter_bw_dl))
        inter_bw_overlap_ul = min(0, 10 * log10(vic_bw_ul / inter_bw_ul))
        # carrier_bw_overlaps are assumed to be 1, since carrier sat and ES should always be
        # using the same bandwidths.

        # Calculate the receive antenna capture area/aperture.
        lam_dl = consts.c / self._frequency.dl
        lam_ul = consts.c / self._frequency.ul
        a_eff_dl = 10 * log10(lam_dl * lam_dl / (4 * pi))
        a_eff_ul = 10 * log10(lam_ul * lam_ul / (4 * pi))

        # For more information on calculation of I/N or C/N+I, see ITU-R BO.1696.
        # The terms in the I/N below can be separated as follows:
        # Received PFD (I)
        # + Noise power (N)
        # + Bandwidth overlap
        # - Losses
        dl_I_N = psd_int_dl + 10 * log10(inter_bw_dl) + a_eff_dl \
                 + g_t_vic_es - 10 * log10(vic_bw_dl) - 10 * log10(consts.k) \
                 + inter_bw_overlap_dl \
                 - inter_atmo_loss_dl - adj_factor - rain_noise_temp
        dl_C_N = psd_carrier_dl + + a_eff_dl \
                 + g_t_carrier_es - 10 * log10(consts.k) \
                 - vic_atmo_loss_dl - adj_factor - rain_noise_temp
        dl_C_I = dl_C_N - dl_I_N
        dl_C_I_N = -10 * log10(
            10 ** (-0.1 * dl_C_N) + 10 ** (-0.1 * dl_C_I))  # eq 1(b) in ITU Rec. ITU-R BO.1696

        # Uplink C/I+N (dB)
        g_t_vic_sat = g_t[0]
        g_t_carrier_sat = g_t[2]
        psd_int_ul = psd[0]
        psd_carrier_ul = psd[2]

        ul_I_N = psd_int_ul + 10 * log10(inter_bw_ul) + a_eff_ul \
                 + g_t_vic_sat - 10 * log10(vic_bw_ul) - 10 * log10(consts.k) \
                 + inter_bw_overlap_ul \
                 - inter_atmo_loss_ul - adj_factor - rain_noise_temp
        ul_C_N = psd_carrier_ul + g_t_carrier_sat - 10 * log10(
            consts.k) + a_eff_ul - vic_atmo_loss_ul - adj_factor - rain_noise_temp
        ul_C_I = ul_C_N - ul_I_N
        ul_C_I_N = -10 * log10(
            10 ** (-0.1 * ul_C_N) + 10 ** (-0.1 * ul_C_I))  # eq 1b in ITU Rec. ITU-R BO.1696

        # FIXME Temporary: """ overwriting downlink I/N with the downlink PSD"""
        # dl_I_N = psd_int_dl+ 10 * np.log10(1e6)

        # If debug is set to true, output the calculation parameters at each timestep into a link
        # budget.
        if self.debug:
            inter_props = self._inter.antenna_model.props
            vic_props = self._victim.antenna_model.props

            # Uplink - Calculated parameters
            # Carrier
            car_gain_ul = vic_props.es_gain_ul
            car_eirp_ul = car_gain_ul + vic_props.es_power_db
            car_dist_ul = coords.vic['r'].item()
            car_elev_ul = coords.vic['el'].item()
            car_azi_ul = coords.vic['az'].item()
            car_fspl_ul = Bam.freespace_loss(car_dist_ul, self._frequency.ul)
            car_rcv_iso_pwr_ul = car_eirp_ul + car_fspl_ul
            car_carr_pwr_ul = car_rcv_iso_pwr_ul + vic_props.sat_gain_ul
            car_flux_dens_ul = car_eirp_ul + Bam.spr_loss(car_dist_ul)
            # Interferer
            int_gain_ul = inter_props.es_gain_ul
            int_eirp_ul = int_gain_ul + inter_props.es_power_db
            int_dist_ul = coords.vic['r'].item()
            # pylint: disable=fixme
            # Todo: Because earth stations are assumed to be co-located, the same 'coords' object
            #  is used to calculate distance for both carrier and interferer uplinks. Both links
            #  travel the same distance to the victim antenna. This will change with
            #  non-co-located stations.
            int_elev_ul = coords.inter['el'].item()
            int_azi_ul = coords.inter['az'].item()
            int_fspl_ul = Bam.freespace_loss(int_dist_ul, self._frequency.ul)
            int_rcv_iso_pwr_ul = int_eirp_ul + int_fspl_ul
            int_carr_pwr_ul = int_rcv_iso_pwr_ul + vic_props.sat_gain_ul
            int_flux_dens_ul = int_eirp_ul + Bam.spr_loss(int_dist_ul)

            # Downlink - Calculated parameters
            # Carrier
            car_gain_dl = vic_props.sat_gain_dl
            car_eirp_dl = car_gain_dl + vic_props.sat_power_db
            car_dist_dl = coords.vic['r'].item()
            # pylint: disable=fixme
            # Todo: Because earth stations are assumed to be co-located, the same 'coords' object
            #  is used to calculate distance for both carrier and interferer uplinks. Both links
            #  travel the same distance to the victim antenna. This will change with
            #  non-co-located stations.
            car_elev_dl = coords.vic['el'].item()
            car_azi_dl = coords.vic['az'].item()
            car_fspl_dl = Bam.freespace_loss(car_dist_dl, self._frequency.dl)
            car_rcv_iso_pwr_dl = car_eirp_dl + car_fspl_dl
            car_carr_pwr_dl = car_rcv_iso_pwr_dl + vic_props.es_gain_dl
            car_flux_dens_dl = car_eirp_dl + Bam.spr_loss(car_dist_dl)
            # Interferer
            int_gain_dl = inter_props.sat_gain_dl
            int_eirp_dl = int_gain_dl + inter_props.sat_power_db
            int_dist_dl = coords.inter['r'].item()
            int_elev_dl = coords.inter['el'].item()
            int_azi_dl = coords.inter['az'].item()
            int_fspl_dl = Bam.freespace_loss(int_dist_dl, self._frequency.dl)
            int_rcv_iso_pwr_dl = int_eirp_dl + int_fspl_dl
            int_carr_pwr_dl = int_rcv_iso_pwr_dl + vic_props.es_gain_dl
            int_flux_dens_dl = int_eirp_dl + Bam.spr_loss(int_dist_dl)

            row_ul = {
                # General
                'Time (UTC)': t_curr.utc_strftime('%b %d %H:%M:%S'),
                'Sep. Angle': sep_angle,
                'Nad. Angle': vic_nad_ang,
                'Rcvd. Freq.': self._frequency.ul,
                'Rcvr. Gain': vic_props.sat_gain_ul,
                'Tequiv': vic_props.sat_temp_sys,
                'g/T (peak)': vic_props.sat_gain_ul - 10 * log10(vic_props.sat_temp_sys),

                # Carrier uplink
                'Car Xmtr gain': car_gain_ul,
                'Car EIRP': car_eirp_ul,
                'Car Range (km)': car_dist_ul * 0.001,
                'Car Elev (deg)': car_elev_ul,
                'Car Azi (deg)': car_azi_ul,
                'Car Rcvd. Iso.': car_rcv_iso_pwr_ul,
                'Car Carrier Power': car_carr_pwr_ul,
                'Car Flux Density': car_flux_dens_ul,
                'Car Free Space Loss': Bam.freespace_loss(car_dist_ul, self._frequency.ul),
                # Todo: Same atmo loss is assumed for carrier and interferer cases, due to
                #  co-located ES's
                'Car Atmos Loss': vic_atmo_loss_ul,
                'Car g/T (actual, off-axis, with losses)': g_t_carrier_sat,
                'Car Tx Bandwidth': vic_props.es_bw_hz_ul * 1e-6,
                'Car Rx Bandwidth': vic_props.sat_bw_hz_ul * 1e-6,
                'Car Rcvd. PSD': psd_carrier_ul,
                'C/N': ul_C_N,

                # Interferer uplink
                'Int Xmtr gain': int_gain_ul,
                'Int EIRP': int_eirp_ul,
                'Int Range (km)': int_dist_ul * 0.001,
                'Int Elev (deg)': int_elev_ul,
                'Int Azi (deg)': int_azi_ul,
                'Int Rcvd. Iso.': int_rcv_iso_pwr_ul,
                'Int Carrier Power': int_carr_pwr_ul,
                'Int Flux Density': int_flux_dens_ul,
                'Int Free Space Loss': Bam.freespace_loss(int_dist_ul, self._frequency.ul),
                'Int Atmos Loss': vic_atmo_loss_ul,
                'Int g/T (actual, off-axis, with losses)': g_t_vic_sat,
                'Int Tx Bandwidth': inter_props.es_bw_hz_ul * 1e-6,
                'Int Rx Bandwidth': inter_props.sat_bw_hz_ul * 1e-6,
                'Int BW Overlap': inter_bw_overlap_ul,
                'Int Rcvd. PSD': psd_int_ul,
                'I/N': ul_I_N,
                'C/N+I': ul_C_I_N,
                'C/I': ul_C_I
            }

            row_dl = {
                # General
                'Time (UTC)': t_curr.utc_strftime('%b %d %H:%M:%S'),
                'Sep. Angle': sep_angle,
                'Nad. Angle': vic_nad_ang,
                'Rcvd. Freq.': self._frequency.dl,
                'Rcvr. Gain': vic_props.es_gain_dl,
                'Tequiv': vic_props.es_temp_sys,
                'g/T (peak)': vic_props.es_gain_dl - 10 * log10(vic_props.es_temp_sys),

                # Carrier downlink
                'Car Xmtr gain': car_gain_dl,
                'Car EIRP': car_eirp_dl,
                'Car Range (km)': car_dist_dl * 0.001,
                'Car Elev (deg)': car_elev_dl,
                'Car Azi (deg)': car_azi_dl,
                'Car Rcvd. Iso.': car_rcv_iso_pwr_dl,
                'Car Carrier Power': car_carr_pwr_dl,
                'Car Flux Density': car_flux_dens_dl,
                'Car Free Space Loss': Bam.freespace_loss(car_dist_dl, self._frequency.dl),
                # Todo: Same atmo loss is assumed for carrier and interferer cases, due to
                #  co-located ES's
                'Car Atmos Loss': vic_atmo_loss_dl,
                'Car g/T (actual, off-axis, with losses)': g_t_carrier_es,
                'Car Tx Bandwidth': vic_props.sat_bw_hz_dl * 1e-6,
                'Car Rx Bandwidth': vic_props.es_bw_hz_dl * 1e-6,
                'Car Rcvd. PSD': psd_carrier_dl,
                'C/N': dl_C_N,

                # Interferer downlink
                'Int Xmtr gain': int_gain_dl,
                'Int EIRP': int_eirp_dl,
                'Int Range (km)': int_dist_dl * 0.001,
                'Int Elev (deg)': int_elev_dl,
                'Int Azi (deg)': int_azi_dl,
                'Int Rcvd. Iso.': int_rcv_iso_pwr_dl,
                'Int Carrier Power': int_carr_pwr_dl,
                'Int Flux Density': int_flux_dens_dl,
                'Int Free Space Loss': Bam.freespace_loss(int_dist_dl, self._frequency.dl),
                'Int Atmos Loss': inter_atmo_loss_dl,
                'Int g/T (actual, off-axis, with losses)': g_t_vic_es,
                'Int Tx Bandwidth': inter_props.sat_bw_hz_dl * 1e-6,
                'Int Rx Bandwidth': inter_props.es_bw_hz_dl * 1e-6,
                'Int BW Overlap': inter_bw_overlap_dl,
                'Int Rcvd. PSD': psd_int_dl,
                'I/N': dl_I_N,
                'C/N+I': dl_C_I_N,
                'C/I': dl_C_I
            }

            data = array(
                ((ul_I_N, dl_I_N), (ul_C_I_N, dl_C_I_N), (ul_C_I, dl_C_I), (ul_C_N, dl_C_N)))
            budget = ([list(row_ul.values()), list(row_dl.values())])

            return [data, budget]

        return array(((ul_I_N, dl_I_N), (ul_C_I_N, dl_C_I_N), (ul_C_I, dl_C_I), (ul_C_N, dl_C_N)))

    def _interference_i_n(self, interval, longest_holds_list):
        """Return interference for a set of timesteps.

         Args:
            interval (Interval): Start and end of time to calculate in seconds from epoch start.
            longest_holds_list (list): List of IdxSphericalCoordinate of the satellite tracked for
                    each timestep or None.

        Returns:
                i_n_results (VicInterPair(T, T)): T is [dB of interference or None for timestep].
        """

        # If interval is empty, simply return a shell.
        if interval.steps == 0:
            return UlDlPair([], [])

        # Otherwise, initialize geometry.
        timescale = load.timescale(builtin=True)
        times = timescale.utc(self._epoch_start.J, second=arange(*interval, interval.step))
        groundstation = Topos(*self._position)
        sat_coords = VicInterPair()  # Index and coordinates of tracked satellite at each timestep.
        sat_gs_diffs = VicInterPair()  # Vector between satellites and ground station.
        tracked = VicInterPair()  # [None/tracked sat object for each timestep].
        for idx in tracked.__fields__:  # Just need VicInterPair indices.
            sats = self.constellations[idx].sats  # Temporary alias.
            sat_gs_diffs[idx] = sats - np_repeat(groundstation, len(sats))
            sat_coords[idx] = self.constellations[idx].get_sat_coords(self._position, times,
                                                                          longest_holds_list[idx])
            tracked[idx] = [None if row is None else sats[row['idx']] for row in sat_coords[idx]]

        # Calculate C/I+N, I/N, C/N, and C/I by mapping to the _calculate_interference method.
        calc_list = list(
            map(self._calculate_interference, times, [VicInterPair(*tup) for tup in zip(*tracked)],
                [VicInterPair(*tup) for tup in zip(*sat_coords)], repeat(sat_gs_diffs.vic)))

        """
        Create a 3D array. Output type x UlDlPair x Timestep.
        Output type: i_n, c_i_n, c_i, c_n
        UlDlPair: Ul i_n, DL i_n
        Timestep: seconds
        """

        # If debug is true, link budget information will be bundled with calc data, and must be
        # separated out.
        if self.debug:
            calc_list_filtered = list(filter(None, calc_list))

            # If calc_list is not empty, separate out the link budget info from the raw
            # interference data. Otherwise, return an empty shell.
            if len(calc_list_filtered) > 0:
                calc_array = array(calc_list_filtered, dtype=object)

                # Extraction
                block = calc_array[:, 0]
                lnk_bdgt = calc_array[:, 1]

                bdgt_ul = []
                bdgt_dl = []
                for i in lnk_bdgt:
                    bdgt_ul.append(i[0])
                    bdgt_dl.append(i[1])

                link_budget_ul = array(bdgt_ul)
                link_budget_dl = array(bdgt_dl)

                budget_pair = UlDlPair(link_budget_ul, link_budget_dl)
                out = [dstack(block), budget_pair]

                return out
            return [empty((4, 2, 0)), UlDlPair([], [])]
        return dstack(tuple(calc_list))

    @_timed_calc_interval(time_per_step=0.33)
    def run(self, interval):
        """Return interference for a set of timesteps.

         Args:
            interval (Interval): Interval to simulate, in seconds since epoch start.

        Returns:
                i_n_results (VicInterPair(T, T)): T is [dB of interference or None for timestep].
        """
        timescale = load.timescale(builtin=True)

        # Divide interval further into partitions, one for each processor to be used.
        if self.parallelism < 0:  # Negative parallelism is interpreted as serial processing.
            intervals = list(partition_interval_chunks(*interval, 1, interval.step))
        else:
            intervals = list(partition_interval_chunks(*interval, self._parallelism, interval.step))

        times = [
            timescale.utc(self._epoch_start.J, second=arange(*subint, interval.step))
            for subint in intervals
        ]

        # Pre-compute the longest holds where necessary. This needs to be done as globally as
        # possible to whole passes. If sim_block_size requires this be broken up, there may be
        # accuracy issues near the seams.
        longest_holds_list = VicInterPair()
        for idx, constel in enumerate(self.constellations):
            if constel.tracking_strat == Constellation.TrackingStrategy.LONGEST_HOLD:
                # Compute visibility and reduce to a single list.
                with concurrent.futures.ProcessPoolExecutor() as executor:
                    vis = executor.map(constel.vis_sats, repeat(self._position), times)
                vis_sats_by_timestep = list(chain(*vis))
                # Compute longest holds and reshape result into intervals for parallel operation.
                holds_iterator = iter(longest_hold(vis_sats_by_timestep))
                longest_holds_list[idx] = [[next(holds_iterator) for _ in range(subint.steps)]
                                           for subint in intervals]
            else:
                longest_holds_list[idx] = repeat(None, interval.steps)

        # Perform interference calculation.
        if self.parallelism < 0:  # Serial processing
            print("Running with serial processing...")
            results = map(self._interference_i_n, intervals,
                          [VicInterPair(*tup) for tup in zip(*longest_holds_list)])
        else:  # Parallel processing
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = executor.map(self._interference_i_n, intervals,
                                       [VicInterPair(*tup) for tup in zip(*longest_holds_list)])

        # If debug is enabled, link budget information will be included in the results array.
        # This must be separated before being returned. If debug is not enabled, only actual
        # interference data will be returned. Data from each processor partition is then dstacked
        # into a single 3D array before being returned.
        if self.debug:
            out = []
            for res in results:
                data = res[0]
                budget = res[1]
                out.append([data, budget])
            return out
        return dstack(list(results))
