import math

from src.channels.wireless_channel import (
    broadcast_rate_bps,
    channel_gain,
    path_loss_db,
    uplink_rate_bps,
)
from src.utils.units import (
    dbm_to_watt,
    ghz_to_hz,
    noise_dbm_per_hz_to_watt_per_hz,
    tflops_to_flops,
)


def test_unit_conversions_match_reference_values():
    assert math.isclose(dbm_to_watt(30), 1.0, rel_tol=1e-12)
    assert math.isclose(ghz_to_hz(2), 2e9, rel_tol=1e-12)
    assert math.isclose(tflops_to_flops(10), 10e12, rel_tol=1e-12)
    assert math.isclose(
        noise_dbm_per_hz_to_watt_per_hz(-140), 1e-17, rel_tol=1e-12
    )


def test_path_loss_and_channel_gain_decrease_with_distance():
    assert math.isclose(path_loss_db(10), 65.0, rel_tol=1e-12)
    assert channel_gain(5, fading_power=1.0) > channel_gain(
        10, fading_power=1.0
    )


def test_uplink_rate_increases_with_rho_and_power():
    noise = noise_dbm_per_hz_to_watt_per_hz(-140)
    gain = channel_gain(8, fading_power=1.0)
    low_rho = uplink_rate_bps(
        dbm_to_watt(20), gain, ghz_to_hz(2), noise, rho=0.2
    )
    high_rho = uplink_rate_bps(
        dbm_to_watt(20), gain, ghz_to_hz(2), noise, rho=0.6
    )
    high_power = uplink_rate_bps(
        dbm_to_watt(23), gain, ghz_to_hz(2), noise, rho=0.2
    )
    assert high_rho > low_rho
    assert high_power > low_rho


def test_broadcast_rate_uses_worst_channel():
    noise = noise_dbm_per_hz_to_watt_per_hz(-140)
    good = channel_gain(5, fading_power=1.0)
    bad = channel_gain(10, fading_power=1.0)
    rate = broadcast_rate_bps(
        dbm_to_watt(30), [good, bad], ghz_to_hz(2), noise
    )
    bad_only = broadcast_rate_bps(
        dbm_to_watt(30), [bad], ghz_to_hz(2), noise
    )
    assert math.isclose(rate, bad_only, rel_tol=1e-12)
