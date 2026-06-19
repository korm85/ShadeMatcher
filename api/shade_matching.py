"""
CIEDE2000 shade matching against the VITA Classical dictionary.
"""

import math
import numpy as np
from vita_shades import VITA_SHADES, GROUP_NAMES


def ciede2000(lab1: tuple, lab2: tuple) -> float:
    """
    Compute CIEDE2000 colour difference between two L*a*b* triples.
    Pure-Python implementation (no external dependency path).
    """
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2

    # Step 1: a' adjustments
    C1 = math.sqrt(a1 ** 2 + b1 ** 2)
    C2 = math.sqrt(a2 ** 2 + b2 ** 2)
    C_bar7 = ((C1 + C2) / 2) ** 7
    G = 0.5 * (1 - math.sqrt(C_bar7 / (C_bar7 + 25 ** 7)))
    a1p = a1 * (1 + G)
    a2p = a2 * (1 + G)

    # Step 2: C', h'
    C1p = math.sqrt(a1p ** 2 + b1 ** 2)
    C2p = math.sqrt(a2p ** 2 + b2 ** 2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360

    # Step 3: Delta L', Delta C', Delta H'
    dLp = L2 - L1
    dCp = C2p - C1p

    if C1p * C2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360

    dHp = 2 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2))

    # Step 4: CIEDE2000
    Lbar = (L1 + L2) / 2
    Cbarp = (C1p + C2p) / 2

    if C1p * C2p == 0:
        hbarp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbarp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbarp = (h1p + h2p + 360) / 2
    else:
        hbarp = (h1p + h2p - 360) / 2

    T = (1
         - 0.17 * math.cos(math.radians(hbarp - 30))
         + 0.24 * math.cos(math.radians(2 * hbarp))
         + 0.32 * math.cos(math.radians(3 * hbarp + 6))
         - 0.20 * math.cos(math.radians(4 * hbarp - 63)))

    SL = 1 + 0.015 * (Lbar - 50) ** 2 / math.sqrt(20 + (Lbar - 50) ** 2)
    SC = 1 + 0.045 * Cbarp
    SH = 1 + 0.015 * Cbarp * T

    Cbarp7 = Cbarp ** 7
    RC = 2 * math.sqrt(Cbarp7 / (Cbarp7 + 25 ** 7))
    d_theta = 30 * math.exp(-((hbarp - 275) / 25) ** 2)
    RT = -math.sin(math.radians(2 * d_theta)) * RC

    return math.sqrt(
        (dLp / SL) ** 2 +
        (dCp / SC) ** 2 +
        (dHp / SH) ** 2 +
        RT * (dCp / SC) * (dHp / SH)
    )


def match_vita(L: float, a: float, b: float) -> list[dict]:
    """
    Compute CIEDE2000 distances to all 16 VITA shades.
    Returns list of dicts sorted by distance (closest first).
    """
    tooth = (L, a, b)
    results = []
    for shade, ref in VITA_SHADES.items():
        ref_lab = (ref["L"], ref["a"], ref["b"])
        dE = ciede2000(tooth, ref_lab)
        results.append({
            "shade": shade,
            "group": ref["group"],
            "group_name": GROUP_NAMES[ref["group"]],
            "dE00": dE,
            "ref_L": ref["L"],
            "ref_a": ref["a"],
            "ref_b": ref["b"],
        })
    results.sort(key=lambda x: x["dE00"])
    return results


def confidence_score(dE00: float) -> tuple[str, str]:
    """
    Return (label, description) confidence tier based on dE00 threshold.
    Clinical rule-of-thumb: dE00 < 1 imperceptible, < 3 acceptable.
    """
    if dE00 < 1.0:
        return "Excellent", "Imperceptible difference"
    elif dE00 < 2.0:
        return "Very Good", "Perceptible only to trained observers"
    elif dE00 < 3.5:
        return "Good", "Clinically acceptable"
    elif dE00 < 6.0:
        return "Fair", "Noticeable difference — verify lighting"
    else:
        return "Poor", "Large colour gap — check calibration"
