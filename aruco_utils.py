"""
ArUco-based calibration card detection and perspective warping.
"""

import cv2
import numpy as np


# Expected ArUco IDs at the four corners: TL, TR, BR, BL
CORNER_IDS = [0, 1, 2, 3]

# Physical card dimensions (pixels in the output warped image)
WARP_W, WARP_H = 600, 400


def detect_aruco(bgr: np.ndarray):
    """
    Detect ArUco markers in the image.
    Returns (corners, ids) or (None, None) if detection fails.
    """
    # Try DICT_4X4_50 first, fall back to 6X6_250
    for dict_id in (cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_6X6_250):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(bgr)
        if ids is not None and len(ids) >= 4:
            return corners, ids.flatten()
    return None, None


def warp_card(bgr: np.ndarray, corners, ids: np.ndarray):
    """
    Perspective-warp the calibration card region to a canonical rectangle.
    corner_ids order: TL=0, TR=1, BR=2, BL=3.
    Returns the warped BGR image or None on failure.
    """
    if corners is None or ids is None:
        return None

    id_to_corner = {}
    for marker_corners, marker_id in zip(corners, ids):
        if marker_id in CORNER_IDS:
            # Use the centroid of each ArUco marker as its card-corner proxy
            id_to_corner[marker_id] = marker_corners[0].mean(axis=0)

    if len(id_to_corner) < 4:
        return None

    src_pts = np.float32([
        id_to_corner[0],  # TL
        id_to_corner[1],  # TR
        id_to_corner[2],  # BR
        id_to_corner[3],  # BL
    ])
    dst_pts = np.float32([
        [0,      0],
        [WARP_W, 0],
        [WARP_W, WARP_H],
        [0,      WARP_H],
    ])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    warped = cv2.warpPerspective(bgr, H, (WARP_W, WARP_H))
    return warped


# ---------------------------------------------------------------------------
# Swatch sampling on the warped card
# ---------------------------------------------------------------------------

# Relative positions (cx, cy) of each reference swatch centre as fraction of
# card width/height.  These must match the physical card layout.
# Layout: 4 rows × 4 cols grid, first swatch at (col=0,row=0) = top-left.
# Adjust margins to taste (0.08 margin, 0.23 step).
_COL_XS = [0.08 + 0.23 * c for c in range(4)]
_ROW_YS = [0.15 + 0.23 * r for r in range(4)]

# Swatch order (left→right, top→bottom) matching the physical card
SWATCH_LABELS = [
    "A1", "A2", "A3", "A3.5",
    "B1", "B2", "B3", "B4",
    "C1", "C2", "C3", "C4",
    "D2", "D3", "D4", "A4",   # A4 fills the 16th slot on many cards
]

SWATCH_RADIUS = 15  # half-side of the sampling square in pixels


def sample_swatches(warped_bgr: np.ndarray):
    """
    Return N×3 uint8 array of median BGR values from each swatch centre.
    """
    h, w = warped_bgr.shape[:2]
    medians = []
    for ry in _ROW_YS:
        for rx in _COL_XS:
            cx, cy = int(rx * w), int(ry * h)
            r = SWATCH_RADIUS
            patch = warped_bgr[
                max(0, cy - r): cy + r,
                max(0, cx - r): cx + r
            ]
            medians.append(np.median(patch.reshape(-1, 3), axis=0))
    return np.array(medians, dtype=np.float32)
