from dataclasses import dataclass

@dataclass(frozen=True)
class ThresholdConfig:
    T_hi: float
    T_lo: float
    q_hi: float
    q_lo: float
    p_hi: float
    heavy_action: str
    light_action: str


def choose_threshold_action(T, q, p, previous_action, cfg: ThresholdConfig):
    """No-training conventional baseline. Main version does not call the shield."""
    if T >= cfg.T_hi or q >= cfg.q_hi:
        return cfg.light_action
    if T <= cfg.T_lo and q <= cfg.q_lo and p >= cfg.p_hi:
        return cfg.heavy_action
    return previous_action if previous_action is not None else cfg.light_action
