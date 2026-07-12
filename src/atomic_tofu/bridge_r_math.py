from __future__ import annotations

import math


def centered_uniform_tail(values: list[float], tau: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if tau <= 0:
        raise ValueError("tau must be positive")
    center = sum(values) / len(values)
    scaled = [(value - center) / tau for value in values]
    maximum = max(scaled)
    return tau * (maximum + math.log(sum(math.exp(value - maximum) for value in scaled) / len(values)))


def bridge_r_objective(erase: float, global_kl: float, local_values: list[float], lambda_g: float, lambda_l: float, lambda_t: float, tau: float) -> float:
    local_mean = sum(local_values) / len(local_values)
    return erase + lambda_g * global_kl + lambda_l * local_mean + lambda_t * centered_uniform_tail(local_values, tau)

