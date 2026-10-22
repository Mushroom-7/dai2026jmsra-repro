def dbm_to_watt(dbm: float) -> float:
    return 10 ** ((dbm - 30.0) / 10.0)


def noise_dbm_per_hz_to_watt_per_hz(dbm_per_hz: float) -> float:
    return dbm_to_watt(dbm_per_hz)


def ghz_to_hz(ghz: float) -> float:
    return ghz * 1e9


def tflops_to_flops(tflops: float) -> float:
    return tflops * 1e12
