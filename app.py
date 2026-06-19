"""
Dental Shade Matcher — Streamlit PoC
"""

import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image

from aruco_utils import detect_aruco, warp_card, sample_swatches, SWATCH_LABELS
from calibration import fit_ccm, apply_ccm, glare_mask, msq_normalise
from shade_matching import match_vita, confidence_score
from vita_shades import VITA_SHADES

# ---------------------------------------------------------------------------
# Ground-truth L*a*b* for the calibration card swatches (same order as
# SWATCH_LABELS, i.e. the known values printed on / certified for the card).
# These must match the physical card used in production.
# ---------------------------------------------------------------------------
CARD_GROUND_TRUTH_LAB = np.array([
    [74.0,  0.6, 16.0],   # A1
    [71.5,  1.5, 18.5],   # A2
    [68.5,  2.8, 20.5],   # A3
    [65.0,  3.8, 22.0],   # A3.5
    [76.5, -0.5, 14.0],   # B1
    [73.5,  0.5, 17.5],   # B2
    [70.0,  1.8, 20.0],   # B3
    [65.5,  2.5, 22.5],   # B4
    [73.5, -1.0, 10.0],   # C1
    [69.0, -0.5, 12.5],   # C2
    [65.0,  0.5, 14.0],   # C3
    [60.0,  1.0, 15.5],   # C4
    [72.0,  0.8, 13.0],   # D2
    [67.0,  1.5, 15.5],   # D3
    [62.0,  2.0, 17.0],   # D4
    [61.0,  4.5, 23.5],   # A4 (slot 16)
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Helper: decode uploaded file → BGR numpy array
# ---------------------------------------------------------------------------
def load_bgr(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return bgr


def bgr_to_rgb_display(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def lab_to_bgr_preview(lab_float: np.ndarray) -> np.ndarray:
    """Convert float L*a*b* array → uint8 BGR for display."""
    L = np.clip(lab_float[..., 0] * (255.0 / 100.0), 0, 255)
    a = np.clip(lab_float[..., 1] + 128.0, 0, 255)
    b = np.clip(lab_float[..., 2] + 128.0, 0, 255)
    lab8 = np.stack([L, a, b], axis=-1).astype(np.uint8)
    return cv2.cvtColor(lab8, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Tooth region selection (simple centre-crop heuristic when no mask given)
# ---------------------------------------------------------------------------
def extract_tooth_region(bgr: np.ndarray, margin: float = 0.25) -> np.ndarray:
    """
    Return the central crop of the image as a proxy for the tooth region.
    In production this would be replaced by a segmentation mask.
    """
    h, w = bgr.shape[:2]
    y0 = int(h * margin)
    y1 = int(h * (1 - margin))
    x0 = int(w * margin)
    x1 = int(w * (1 - margin))
    return bgr[y0:y1, x0:x1]


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dental Shade Matcher",
    page_icon="🦷",
    layout="wide",
)

st.title("🦷 Dental Shade Matcher — PoC")
st.markdown(
    "Upload a clinical photograph containing the calibration card. "
    "The app will apply Root-Polynomial Colour Correction, suppress glare and "
    "3D shading artefacts, then report the closest VITA Classical shade."
)

# ---------------------------------------------------------------------------
# Sidebar: controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    glare_threshold = st.slider("Glare L* threshold", 80, 100, 95,
                                help="Pixels with L* above this are masked out.")
    show_debug = st.checkbox("Show debug panels", value=True)
    st.markdown("---")
    st.markdown("**Calibration card:** ArUco IDs 0-3 at corners (DICT_4X4_50)")

# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
col_upload, col_ref = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Upload clinical photograph (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
    )

with col_ref:
    st.markdown("#### VITA shade reference")
    group_colors = {"A": "#c8956c", "B": "#d4b483", "C": "#b0a898", "D": "#b89b8a"}
    for shade, val in VITA_SHADES.items():
        grp = val["group"]
        st.markdown(
            f'<span style="background:{group_colors[grp]};padding:2px 6px;'
            f'border-radius:4px;color:#222;font-size:0.8em">{shade}</span>',
            unsafe_allow_html=True,
        )

if uploaded is None:
    st.info("No image uploaded yet. Please upload a clinical photograph to begin.")
    st.stop()

# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------
bgr = load_bgr(uploaded)
if bgr is None:
    st.error("Could not decode the image. Please upload a valid JPEG or PNG.")
    st.stop()

with st.spinner("Detecting calibration card…"):
    corners, ids = detect_aruco(bgr)
    warped = warp_card(bgr, corners, ids) if corners is not None else None

calibration_available = warped is not None
ccm = None

if calibration_available:
    with st.spinner("Fitting colour correction matrix (RPCC)…"):
        raw_swatches = sample_swatches(warped)          # N×3 uint8 BGR
        rgb_norm = raw_swatches[:, ::-1] / 255.0        # N×3 float, RGB order
        ccm = fit_ccm(rgb_norm, CARD_GROUND_TRUTH_LAB)
    st.success("Calibration card detected — RPCC matrix fitted.")
else:
    st.warning(
        "⚠️ No calibration card detected (ArUco markers not found). "
        "Falling back to sRGB→Lab conversion without card calibration. "
        "Results may be less accurate."
    )

with st.spinner("Processing tooth region…"):
    tooth_bgr = extract_tooth_region(bgr)

    if ccm is not None:
        tooth_lab = apply_ccm(tooth_bgr, ccm)
    else:
        # Fallback: naive sRGB→Lab
        tooth_lab_u8 = cv2.cvtColor(tooth_bgr, cv2.COLOR_BGR2LAB)
        L = tooth_lab_u8[..., 0].astype(np.float32) * (100.0 / 255.0)
        a = tooth_lab_u8[..., 1].astype(np.float32) - 128.0
        b = tooth_lab_u8[..., 2].astype(np.float32) - 128.0
        tooth_lab = np.stack([L, a, b], axis=-1)

    # Glare mask
    valid = glare_mask(tooth_lab, l_threshold=glare_threshold)
    n_valid = valid.sum()
    n_total = valid.size

    # MSQ shading normalisation
    tooth_lab_norm = msq_normalise(tooth_lab)

    # Average over valid pixels
    if n_valid == 0:
        st.error("All pixels were masked as glare. Lower the L* threshold.")
        st.stop()

    avg_L = float(tooth_lab_norm[..., 0][valid].mean())
    avg_a = float(tooth_lab_norm[..., 1][valid].mean())
    avg_b = float(tooth_lab_norm[..., 2][valid].mean())

# ---------------------------------------------------------------------------
# Shade matching
# ---------------------------------------------------------------------------
rankings = match_vita(avg_L, avg_a, avg_b)
best = rankings[0]
conf_label, conf_desc = confidence_score(best["dE00"])

# ---------------------------------------------------------------------------
# Results layout
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Results")

res_col1, res_col2, res_col3, res_col4 = st.columns(4)
with res_col1:
    st.metric("Best VITA Shade", best["shade"])
with res_col2:
    st.metric("ΔE₀₀", f"{best['dE00']:.2f}")
with res_col3:
    st.metric("Confidence", conf_label)
with res_col4:
    st.metric("Shade Group", f"{best['group']} — {best['group_name']}")

st.caption(f"Confidence note: {conf_desc}")

st.markdown("#### Tooth L\\*a\\*b\\* values")
lab_col1, lab_col2, lab_col3 = st.columns(3)
with lab_col1:
    st.metric("L* (Lightness)", f"{avg_L:.1f}")
with lab_col2:
    st.metric("a* (Red–Green)", f"{avg_a:.2f}")
with lab_col3:
    st.metric("b* (Blue–Yellow)", f"{avg_b:.2f}")

coverage_pct = 100 * n_valid / n_total
st.caption(f"Valid (non-glare) pixels used: {n_valid:,} / {n_total:,} ({coverage_pct:.1f}%)")

# ---------------------------------------------------------------------------
# Top-5 shade ranking table
# ---------------------------------------------------------------------------
st.markdown("#### Top 5 closest VITA shades")
table_rows = []
for r in rankings[:5]:
    table_rows.append({
        "Shade": r["shade"],
        "Group": r["group_name"],
        "ΔE₀₀": f"{r['dE00']:.2f}",
        "Ref L*": r["ref_L"],
        "Ref a*": r["ref_a"],
        "Ref b*": r["ref_b"],
    })
st.table(table_rows)

# ---------------------------------------------------------------------------
# Image panels
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Image panels")

img_cols = st.columns(3 if calibration_available else 2)

with img_cols[0]:
    st.markdown("**Original upload**")
    st.image(bgr_to_rgb_display(bgr), use_container_width=True)

if calibration_available and warped is not None:
    with img_cols[1]:
        st.markdown("**Warped calibration card**")
        st.image(bgr_to_rgb_display(warped), use_container_width=True)
    tooth_panel_col = img_cols[2]
else:
    tooth_panel_col = img_cols[1]

with tooth_panel_col:
    st.markdown("**Tooth region (shading-normalised)**")
    preview_bgr = lab_to_bgr_preview(tooth_lab_norm)
    st.image(bgr_to_rgb_display(preview_bgr), use_container_width=True)

# ---------------------------------------------------------------------------
# Debug: glare overlay
# ---------------------------------------------------------------------------
if show_debug:
    st.markdown("---")
    st.subheader("Debug: glare mask overlay")
    glare_overlay = tooth_bgr.copy()
    glare_overlay[~valid] = [0, 0, 255]   # red = glare pixels
    st.image(bgr_to_rgb_display(glare_overlay), use_container_width=True,
             caption="Red pixels excluded from shade calculation (glare/saturation)")
