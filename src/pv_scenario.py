"""Illustrative *self-consumption potential*, not measured electricity or CO2.

Assumes a horizontal solar panel, fixed simple conversion efficiency,
no batteries, no shading, no inverter or load effects, and no other loads.
Input solar irradiance is an ERA5 model estimate, not a meter reading.
This standalone calculation cannot justify a carbon-savings claim.
"""
from __future__ import annotations

import numpy as np


def hypothetical_self_use_kwh(ghi_w_m2: np.ndarray, *, area_m2: float = 4.0,
                              efficiency: float = 0.20, load_w: float = 300.0) -> float:
    """Potential direct PV self-use during successive one-hour task intervals.

    An illustrative 4 m² horizontal panel at 20% nominal efficiency supplying
    a 300 W flexible computer. Every array entry covers precisely one hour.
    """
    ghi = np.asarray(ghi_w_m2, dtype=float)
    if ghi.ndim != 1 or not np.isfinite(ghi).all() or (ghi < 0).any():
        raise ValueError('Expected a finite nonnegative 1D hourly GHI array')
    if not (area_m2 > 0 and 0 < efficiency <= 1 and load_w > 0):
        raise ValueError('PV scenario parameters must be positive and valid')
    assumed_pv_w = ghi * area_m2 * efficiency
    return float(np.minimum(assumed_pv_w, load_w).sum() / 1000.0)
