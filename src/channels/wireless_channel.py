import math


def path_loss_db(distance_m: float) -> float:
    if distance_m <= 0:
        raise ValueError("distance_m must be positive")
    return 30.0 + 35.0 * math.log10(distance_m)


def channel_gain(distance_m: float, fading_power: float = 1.0) -> float:
    if fading_power < 0:
        raise ValueError("fading_power must be nonnegative")
    return fading_power * 10 ** (-path_loss_db(distance_m) / 10.0)


def uplink_rate_bps(
    power_watt: float,
    gain: float,
    bandwidth_hz: float,
    noise_psd_w_per_hz: float,
    rho: float,
) -> float:
    if not 0 < rho <= 1:
        raise ValueError("rho must be in (0, 1]")
    effective_bandwidth = rho * bandwidth_hz
    snr = power_watt * gain / (effective_bandwidth * noise_psd_w_per_hz)
    return effective_bandwidth * math.log2(1.0 + snr)


def broadcast_rate_bps(
    power_watt: float,
    gains: list[float],
    bandwidth_hz: float,
    noise_psd_w_per_hz: float,
) -> float:
    if not gains:
        raise ValueError("gains must be nonempty")
    return min(
        bandwidth_hz
        * math.log2(
            1.0
            + power_watt * gain / (bandwidth_hz * noise_psd_w_per_hz)
        )
        for gain in gains
    )
