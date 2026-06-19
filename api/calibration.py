"""
Root-Polynomial Color Correction (RPCC) and CIELAB conversion utilities.
"""

import numpy as np
import cv2
from scipy.linalg import lstsq


def bgr_to_lab(bgr_img: np.ndarray) -> np.ndarray:
    """Convert uint8 BGR image to float32 CIELAB."""
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB).astype(np.float32)


def lab_float_to_display(lab_img: np.ndarray) -> np.ndarray:
    """Convert OpenCV-scale LAB (L:0-255, a/b:0-255) to standard L*a*b* floats."""
    L = lab_img[..., 0] * (100.0 / 255.0)
    a = lab_img[..., 1] - 128.0
    b = lab_img[..., 2] - 128.0
    return np.stack([L, a, b], axis=-1)


def expand_rpcc(rgb: np.ndarray) -> np.ndarray:
    """
    Expand N×3 RGB (0-1 float) into N×6 root-polynomial basis:
    [R, G, B, sqrt(R*G), sqrt(R*B), sqrt(G*B)]
    """
    R, G, B = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    RG = np.sqrt(np.clip(R * G, 0, None))
    RB = np.sqrt(np.clip(R * B, 0, None))
    GB = np.sqrt(np.clip(G * B, 0, None))
    return np.column_stack([R, G, B, RG, RB, GB])


def fit_ccm(source_rgb: np.ndarray, target_lab: np.ndarray):
    """
    Fit a 6×3 Root-Polynomial Color Correction Matrix via OLS.
    source_rgb : N×3 float, range [0,1]
    target_lab : N×3 float, standard L*a*b* scale
    Returns 6×3 matrix M such that expand_rpcc(rgb) @ M ≈ lab
    """
    X = expand_rpcc(source_rgb)
    M, _, _, _ = lstsq(X, target_lab)
    return M


def apply_ccm(bgr_img: np.ndarray, ccm: np.ndarray) -> np.ndarray:
    """
    Apply RPCC matrix to a full BGR image.
    Returns an image-shaped array of L*a*b* floats.
    """
    h, w = bgr_img.shape[:2]
    rgb = bgr_img[:, :, ::-1].astype(np.float32) / 255.0
    flat = rgb.reshape(-1, 3)
    expanded = expand_rpcc(flat)
    lab_flat = expanded @ ccm
    return lab_flat.reshape(h, w, 3)


# ---------------------------------------------------------------------------
# Specular glare masking
# ---------------------------------------------------------------------------

def glare_mask(lab_img: np.ndarray, l_threshold: float = 95.0) -> np.ndarray:
    """
    Return boolean mask (True = valid pixel) excluding saturated-glare pixels.
    lab_img is standard float L*a*b* with L in [0,100].
    """
    return lab_img[..., 0] < l_threshold


# ---------------------------------------------------------------------------
# Multi-Scale Self-Quotient (MSQ) shading normalisation
# ---------------------------------------------------------------------------

def msq_normalise(lab_img: np.ndarray,
                  sigmas: tuple = (3, 9, 27),
                  epsilon: float = 1e-3) -> np.ndarray:
    """
    Flatten 3D Lambertian shading on the L* channel via multi-scale
    self-quotient image (bilateral-filter illumination estimate).
    a* and b* channels are passed through unchanged.
    """
    L = lab_img[..., 0].astype(np.float32)
    illum_stack = []
    for sigma in sigmas:
        d = max(1, int(sigma * 3) | 1)   # odd diameter
        illum = cv2.bilateralFilter(L, d, sigma * 2, sigma * 2)
        illum_stack.append(illum)
    illumination = np.mean(illum_stack, axis=0)
    L_norm = L / (illumination + epsilon) * np.mean(illumination)
    L_norm = np.clip(L_norm, 0, 100)
    result = lab_img.copy()
    result[..., 0] = L_norm
    return result
