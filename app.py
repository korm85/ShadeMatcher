"""
Dental Shade Matcher — Streamlit PoC
"""

import cv2
import numpy as np
import streamlit as st

from aruco_utils import detect_aruco, warp_card, sample_swatches, SWATCH_LABELS
from calibration import fit_ccm, apply_ccm, glare_mask, msq_normalise
from shade_matching import match_vita, confidence_score
from vita_shades import VITA_SHADES

# ---------------------------------------------------------------------------
# Ground-truth L*a*b* for the 16 Gavan.ai calibration card swatches.
# Row-major order (TL→TR then next row), matching SWATCH_LABELS in aruco_utils.
# Values are PoC estimates — replace with the card's certificate in production.
# ---------------------------------------------------------------------------
CARD_GROUND_TRUTH_LAB = np.array([
    # Row 1 — neutral grey scale (dark → light)
    [ 16.0,  0.0,   0.0],   # S01  near-black
    [ 35.0,  0.0,   0.0],   # S02  dark grey
    [ 52.0,  0.0,   0.0],   # S03  medium grey
    [ 68.0,  0.0,   0.0],   # S04  light grey
    # Row 2 — chromatic primaries / saturated
    [ 38.0, 25.0,  16.0],   # S05  reddish-brown / maroon
    [ 45.0,  2.0, -44.0],   # S06  bright blue
    [ 58.0,-28.0,  29.0],   # S07  green
    [ 27.0,  8.0, -28.0],   # S08  dark navy
    # Row 3 — secondary / mixed tones
    [ 52.0, 24.0,  36.0],   # S09  orange-brown
    [ 65.0, 22.0,  48.0],   # S10  orange / amber
    [ 55.0, -8.0,  30.0],   # S11  olive / yellow-green
    [ 62.0, -1.0, -15.0],   # S12  blue-grey
    # Row 4 — light / skin tones
    [ 72.0,  8.0,  16.0],   # S13  tan / skin tone
    [ 78.0,  5.0,  10.0],   # S14  light tan
    [ 85.0,  1.0,   5.0],   # S15  cream / off-white
    [ 93.0,  0.0,   2.0],   # S16  near-white
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
# Tooth region selection
# ---------------------------------------------------------------------------
def extract_tooth_region(bgr: np.ndarray, card_detected: bool = False) -> np.ndarray:
    """
    Heuristic tooth-region crop.

    When a calibration card is present in the lower half of a clinical photo
    (Gavan.ai layout), the teeth are typically in the upper ~50% of the frame,
    centred horizontally.  Without a card we fall back to a symmetric centre crop.

    In production this would be replaced by a dedicated segmentation model.
    """
    h, w = bgr.shape[:2]
    if card_detected:
        # Upper 52% of image, middle 65% horizontally
        y0, y1 = 0,            int(h * 0.52)
        x0, x1 = int(w * 0.17), int(w * 0.83)
    else:
        y0, y1 = int(h * 0.25), int(h * 0.75)
        x0, x1 = int(w * 0.25), int(w * 0.75)
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
    st.markdown("**Calibration card:** Gavan.ai — 4×4 swatch grid, black square corner fiducials")

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
    tooth_bgr = extract_tooth_region(bgr, card_detected=calibration_available)

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
