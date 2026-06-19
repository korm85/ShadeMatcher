"""
Card corner detection (ArUco primary, black-square-contour fallback)
and swatch sampling for the Gavan.ai calibration card.

Card layout:
  - 4 small black squares at the outer corners (TL, TR, BR, BL)
  - Two circular fiducials in the centre (ignored here)
  - 4×4 colour swatch grid occupying the LEFT ~47% of the card
  - Barcode/stripe region on the RIGHT half (ignored)
"""

import cv2
import numpy as np


# ArUco corner ID convention (if the card encodes them)
CORNER_IDS = [0, 1, 2, 3]

# Output canonical warp dimensions
WARP_W, WARP_H = 600, 400


# ---------------------------------------------------------------------------
# ArUco detection
# ---------------------------------------------------------------------------

def detect_aruco(bgr: np.ndarray):
    """Try multiple ArUco dictionaries; return (corners, ids) or (None, None)."""
    for dict_id in (cv2.aruco.DICT_4X4_50, cv2.aruco.DICT_6X6_250,
                    cv2.aruco.DICT_5X5_50, cv2.aruco.DICT_4X4_100):
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        corners, ids, _ = detector.detectMarkers(bgr)
        if ids is not None and len(ids) >= 4:
            return corners, ids.flatten()
    return None, None


# ---------------------------------------------------------------------------
# Fallback: find 4 black square fiducials by contour analysis
# ---------------------------------------------------------------------------

def _sort_corners(pts: np.ndarray) -> np.ndarray:
    """
    Sort 4 points into [TL, TR, BR, BL] order.
    """
    pts = pts[np.argsort(pts[:, 1])]   # sort by y
    top = pts[:2][np.argsort(pts[:2, 0])]    # TL, TR
    bot = pts[2:][np.argsort(pts[2:, 0])]    # BL, BR
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)


def detect_black_square_corners(bgr: np.ndarray):
    """
    Detect the 4 small black corner squares on the Gavan.ai card using
    adaptive thresholding + contour filtering.
    Returns N×2 float32 array of corner centroids sorted [TL,TR,BR,BL],
    or None if fewer than 4 candidates found.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to cope with lighting variation
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, blockSize=21, C=10
    )
    # Morphological close to fill small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    h, w = bgr.shape[:2]
    img_area = h * w
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter by area: between 0.05% and 1.5% of image
        if area < img_area * 0.0005 or area > img_area * 0.015:
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        aspect = cw / max(ch, 1)
        if not (0.5 < aspect < 2.0):
            continue
        # Solidity: real squares are solid
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        solidity = area / max(hull_area, 1)
        if solidity < 0.7:
            continue
        cx, cy = x + cw / 2, y + ch / 2
        candidates.append((cx, cy, area))

    if len(candidates) < 4:
        return None

    # Keep the 4 largest candidates (the corner squares are typically the biggest
    # solid black blobs on the card)
    candidates.sort(key=lambda c: c[2], reverse=True)
    pts = np.array([[c[0], c[1]] for c in candidates[:4]], dtype=np.float32)
    return _sort_corners(pts)


# ---------------------------------------------------------------------------
# Perspective warp
# ---------------------------------------------------------------------------

def warp_card(bgr: np.ndarray, aruco_corners, aruco_ids):
    """
    Perspective-warp the calibration card to a canonical rectangle.
    Tries ArUco corners first; falls back to black-square contour detection.
    Returns warped BGR image, or None on failure.
    """
    src_pts = None

    # --- ArUco path ---
    if aruco_corners is not None and aruco_ids is not None:
        id_to_corner = {}
        for marker_corners, marker_id in zip(aruco_corners, aruco_ids):
            if marker_id in CORNER_IDS:
                id_to_corner[marker_id] = marker_corners[0].mean(axis=0)
        if len(id_to_corner) == 4:
            src_pts = np.float32([
                id_to_corner[0], id_to_corner[1],
                id_to_corner[2], id_to_corner[3],
            ])

    # --- Contour fallback ---
    if src_pts is None:
        src_pts = detect_black_square_corners(bgr)
        if src_pts is None:
            return None

    dst_pts = np.float32([
        [0,      0],
        [WARP_W, 0],
        [WARP_W, WARP_H],
        [0,      WARP_H],
    ])
    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        return None
    return cv2.warpPerspective(bgr, H, (WARP_W, WARP_H))


# ---------------------------------------------------------------------------
# Swatch sampling — Gavan.ai card layout
#
# The 4×4 colour grid occupies the left ~47% of the card width.
# Layout (fraction of warped card dimensions):
#   Columns (x): 0.10, 0.21, 0.33, 0.44
#   Rows    (y): 0.22, 0.40, 0.60, 0.78
# ---------------------------------------------------------------------------

_COL_XS = [0.10, 0.21, 0.33, 0.44]
_ROW_YS = [0.22, 0.40, 0.60, 0.78]

# 16 swatches in row-major order (top-left → bottom-right)
SWATCH_LABELS = [
    "S01", "S02", "S03", "S04",   # row 1 — neutral grey scale
    "S05", "S06", "S07", "S08",   # row 2 — chromatic primaries
    "S09", "S10", "S11", "S12",   # row 3 — secondary/mixed tones
    "S13", "S14", "S15", "S16",   # row 4 — light / skin tones
]

SWATCH_RADIUS = 12   # half-side of sampling square (pixels in warped space)


def sample_swatches(warped_bgr: np.ndarray) -> np.ndarray:
    """Return 16×3 float32 array of median BGR values from each swatch centre."""
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
            if patch.size == 0:
                medians.append(np.array([128.0, 128.0, 128.0]))
            else:
                medians.append(np.median(patch.reshape(-1, 3), axis=0))
    return np.array(medians, dtype=np.float32)
