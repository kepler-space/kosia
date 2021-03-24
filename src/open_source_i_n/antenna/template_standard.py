"""Template antenna module using standard antenna patterns only."""
from . import AntennaProperties, BaseAntennaModel   # pylint: disable=import-error


# pylint: disable=too-few-public-methods
class AntennaModel(BaseAntennaModel):
    """Template antenna model."""

    name = 'template_standard'
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

    interfering_sat_dl_psd = BaseAntennaModel.interfering_sat_dl_psd_s1528(props.sat_gain_dl,
                                                                           props.sat_diam_dl,
                                                                           props.sat_power_db,
                                                                           props.sat_bw_hz_dl,
                                                                           props.sat_hpbw_dl,
                                                                           L_N=-15,
                                                                           pointing=False)

    interfering_es_ul_psd = BaseAntennaModel.interfering_es_ul_psd_s1428(
        props.es_gain_ul, props.es_diam_ul, props.es_power_db, props.es_bw_hz_ul, True)

    victim_es_dl_g_over_t = BaseAntennaModel.victim_es_dl_g_over_t_s580_6(
        props.es_gain_dl, props.es_temp_sys, props.es_losses_dl)

    victim_sat_ul_g_over_t = BaseAntennaModel.victim_sat_ul_g_over_t_s1528(
        props.sat_gain_ul, props.sat_diam_ul, props.sat_temp_sys, props.sat_hpbw_ul)
