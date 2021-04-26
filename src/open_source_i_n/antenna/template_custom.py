"""Template antenna module using several custom antenna patterns."""
from numpy import power, cos, deg2rad, log10
from . import AntennaProperties, BaseAntennaModel   # pylint: disable=import-error


# pylint: disable=unused-argument
class AntennaModel(BaseAntennaModel):
    """Template antenna model."""

    name = 'template_custom'
    default_sat = 'main_beam'
    default_es = 'user_terminal_small'

    __SAT_PROPERTIES = {
        'main_beam': {
            # Downlink
            'sat_diam_dl': 0.5,
            'sat_gain_dl': 33.8786,
            'sat_hpbw_dl': 3.5874,
            'sat_eff_dl': 0.65,
            'sat_power_db': 2,
            'sat_eirp': 35.8786,
            'sat_losses_dl': 0,
            'sat_bw_hz_dl': 150e6,

            # Uplink
            'sat_diam_ul': 0.5,
            'sat_gain_ul': 35.5912,
            'sat_hpbw_ul': 2.9454,
            'sat_eff_ul': 0.65,
            'sat_temp_sys': 650.01,
            'sat_bw_hz_ul': 150e6,
            'sat_losses_ul': 0,
        },
        'ttc_beam': {
            # Downlink
            'sat_diam_dl': 0.0677,
            'sat_gain_dl': 2,
            'sat_hpbw_dl': 140.8331,
            'sat_eff_dl': 0.65,
            'sat_power_db': 3,
            'sat_eirp': 5,
            'sat_losses_dl': 0,
            'sat_bw_hz_dl': 1e6,

            # Uplink
            'sat_diam_ul': 0.0677,
            'sat_gain_ul': 1.4876,
            'sat_hpbw_ul': 149.3905,
            'sat_eff_ul': 0.65,
            'sat_temp_sys': 676,
            'sat_bw_hz_ul': 1e6,
            'sat_losses_ul': 0
        }
    }

    __ES_PROPERTIES = {
        'gateway': {
            'es_eff': 0.65,

            # Downlink
            'es_diam_dl': 3.4,
            'es_gain_dl': 52.4,
            'es_bw_hz_dl': 125e6,
            'es_temp_sys': 115.09,
            'es_losses_dl': 0,

            # Uplink
            'es_diam_ul': 3.4,
            'es_gain_ul': 54.10,
            'es_eirp': 64.10,
            'es_power_db': 10.0,
            'es_bw_hz_ul': 125e6,
            'es_losses_ul': 0
        },
        'user_terminal_small': {
            'es_eff': 0.6525,

            # Downlink
            'es_diam_dl': 0.65,
            'es_gain_dl': 36.3,
            'es_bw_hz_dl': 40e6,
            'es_temp_sys': 140,
            'es_losses_dl': 0,

            # Uplink
            'es_diam_ul': 0.65,
            'es_gain_ul': 37.7,
            'es_eirp': 46.73,
            'es_power_db': 9.031,
            'es_bw_hz_ul': 20e6,
            'es_losses_ul': 0
        }
    }

    props = AntennaProperties(**__SAT_PROPERTIES[default_sat], **__ES_PROPERTIES[default_es])

    def interfering_sat_dl_psd(self, sim_geom):
        """Constant EIRP throughout pass. This implies the interfering downlink antenna is pointed
        at the victim ground station."""
        eirp_hz = self.props.sat_eirp - 10 * log10(self.props.sat_bw_hz_dl)

        # Return the Received PSD in dBW/Hz/m^2
        return eirp_hz + self.spr_loss(sim_geom['inter'].dist)

    def interfering_es_ul_psd(self, sim_geom):
        """This antenna pattern will only use the peak gain. Sep angle and nadir angle are
        ignored. This allows us to model the antenna (in this case, the satellite downlink
        antenna) so as to be pointed directly into the victim earth station at all times."""
        if sim_geom['inter'].elev <= 40.0:
            # These are set so they can be written properly later to the link budget output.
            self.props.es_gain_ul = 32.0
            self.props.es_eirp = -39.7 + (10 * log10(self.props.es_bw_hz_ul))
            self.props.es_power_db = self.props.es_eirp - self.props.es_gain_ul

            # Otherwise, at <= 40 deg elevation, EIRP density is set to -35 dBW/Hz
            eirp_ul_hz = -35.0
        else:
            # These are set so they can be written properly later to the link budget output.
            self.props.es_gain_ul = 34.6
            self.props.es_eirp = -44.37 + (10 * log10(self.props.es_bw_hz_ul))
            self.props.es_power_db = self.props.es_eirp - self.props.es_gain_ul

            # Otherwise, at > 40 deg elevation, EIRP density is set to -42 dBW/Hz
            eirp_ul_hz = -42

        # Calculate Received PSD in dBW/Hz/m^2 at victim ground station
        return eirp_ul_hz + self.spr_loss(sim_geom['vic'].dist)

    # A standard antenna pattern (Appendix 8) is used for the victim earth station receive pattern.
    victim_es_dl_g_over_t = BaseAntennaModel.victim_es_dl_g_over_t_app8(
        props.es_gain_dl, props.es_diam_dl, props.es_temp_sys)

    def victim_sat_ul_g_over_t(self, sim_geom):
        """Implement a cosine^1.35 scan loss on the satellite receive antenna."""
        scan_loss = power(cos(deg2rad(sim_geom['vic'].nad_ang)), 1.35)
        scan_loss_bottom = 10 * log10(
            0.0042316
        )  # There's a drop to infinity at nad_angle=90, so the min is set to cos(89)^1.35
        scan_loss = max(scan_loss_bottom, 10 * log10(scan_loss))

        # Return G/T
        return self.props.sat_gain_ul - 10 * log10(self.props.sat_temp_sys) + scan_loss
