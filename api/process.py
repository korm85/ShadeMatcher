"""
Vercel serverless function — dental shade matching pipeline.
All dependencies are co-located in api/ so Vercel bundles them reliably.
"""

import base64
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify

# Local siblings (all copied into api/)
from aruco_utils import detect_aruco, warp_card, sample_swatches, WARP_W, WARP_H
from calibration import fit_ccm, apply_ccm, glare_mask, msq_normalise
from shade_matching import match_vita, confidence_score

app = Flask(__name__)

# HTML is a sibling file — guaranteed to exist in the same directory
_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# Gavan.ai calibration card — 16-swatch PoC ground-truth L*a*b*
# ---------------------------------------------------------------------------
CARD_GT_LAB = np.array([
    [ 16.0,  0.0,   0.0],
    [ 35.0,  0.0,   0.0],
    [ 52.0,  0.0,   0.0],
    [ 68.0,  0.0,   0.0],
    [ 38.0, 25.0,  16.0],
    [ 45.0,  2.0, -44.0],
    [ 58.0,-28.0,  29.0],
    [ 27.0,  8.0, -28.0],
    [ 52.0, 24.0,  36.0],
    [ 65.0, 22.0,  48.0],
    [ 55.0, -8.0,  30.0],
    [ 62.0, -1.0, -15.0],
    [ 72.0,  8.0,  16.0],
    [ 78.0,  5.0,  10.0],
    [ 85.0,  1.0,   5.0],
    [ 93.0,  0.0,   2.0],
], dtype=np.float32)


def _bgr_to_b64(bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    L = np.clip(lab[..., 0] * (255.0 / 100.0), 0, 255)
    a = np.clip(lab[..., 1] + 128.0, 0, 255)
    b = np.clip(lab[..., 2] + 128.0, 0, 255)
    return cv2.cvtColor(np.stack([L, a, b], -1).astype(np.uint8), cv2.COLOR_LAB2BGR)


def extract_tooth_region(bgr, card_detected, region=None):
    h, w = bgr.shape[:2]
    if region:
        # Manual selection: region is fractions [0-1] of original image
        x0 = max(0, int(region['x0'] * w))
        y0 = max(0, int(region['y0'] * h))
        x1 = min(w, int(region['x1'] * w))
        y1 = min(h, int(region['y1'] * h))
        return bgr[y0:y1, x0:x1], True
    if card_detected:
        y0, y1 = 0,             int(h * 0.52)
        x0, x1 = int(w * 0.17), int(w * 0.83)
    else:
        y0, y1 = int(h * 0.25), int(h * 0.75)
        x0, x1 = int(w * 0.25), int(w * 0.75)
    return bgr[y0:y1, x0:x1], False


@app.route("/api/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    data = request.files["image"].read()
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Downscale for serverless timeout safety
    h, w = bgr.shape[:2]
    if max(h, w) > 1600:
        scale = 1600 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

    glare_threshold = float(request.form.get("glare_threshold", 95))

    # Optional manual tooth region (fractions sent from the canvas selector)
    manual_region = None
    try:
        if all(k in request.form for k in ('tooth_x0','tooth_y0','tooth_x1','tooth_y1')):
            manual_region = {k: float(request.form[k])
                             for k in ('tooth_x0','tooth_y0','tooth_x1','tooth_y1')}
            # Rekey to x0/y0/x1/y1
            manual_region = {k.replace('tooth_',''):v for k,v in manual_region.items()}
    except ValueError:
        pass

    # Optional manual card region (fractions sent from the canvas selector)
    manual_card_region = None
    try:
        if all(k in request.form for k in ('card_x0','card_y0','card_x1','card_y1')):
            manual_card_region = {k: float(request.form[k])
                                  for k in ('card_x0','card_y0','card_x1','card_y1')}
            manual_card_region = {k.replace('card_',''):v for k,v in manual_card_region.items()}
    except ValueError:
        pass

    # Card detection & CCM fitting
    ccm, warped_b64 = None, None
    if manual_card_region is not None:
        # Use the user-drawn region directly — no ArUco/contour detection needed
        ch, cw = bgr.shape[:2]
        r = manual_card_region
        cx0 = max(0, int(r['x0'] * cw));  cy0 = max(0, int(r['y0'] * ch))
        cx1 = min(cw, int(r['x1'] * cw)); cy1 = min(ch, int(r['y1'] * ch))
        card_crop = bgr[cy0:cy1, cx0:cx1]
        warped = cv2.resize(card_crop, (WARP_W, WARP_H))
        card_detected = True
    else:
        corners, ids = detect_aruco(bgr)
        warped = warp_card(bgr, corners, ids)
        card_detected = warped is not None

    if card_detected:
        raw = sample_swatches(warped)
        ccm = fit_ccm(raw[:, ::-1] / 255.0, CARD_GT_LAB)
        warped_b64 = _bgr_to_b64(warped)

    # Tooth region
    tooth_bgr, used_manual = extract_tooth_region(bgr, card_detected, manual_region)

    if ccm is not None:
        tooth_lab = apply_ccm(tooth_bgr, ccm)
    else:
        u8 = cv2.cvtColor(tooth_bgr, cv2.COLOR_BGR2LAB)
        tooth_lab = np.stack([
            u8[..., 0].astype(np.float32) * (100.0 / 255.0),
            u8[..., 1].astype(np.float32) - 128.0,
            u8[..., 2].astype(np.float32) - 128.0,
        ], -1)

    valid = glare_mask(tooth_lab, glare_threshold)
    n_valid, n_total = int(valid.sum()), int(valid.size)

    if n_valid == 0:
        return jsonify({"error": "All pixels masked as glare — lower threshold"}), 422

    tooth_norm = msq_normalise(tooth_lab)
    avg_L = float(tooth_norm[..., 0][valid].mean())
    avg_a = float(tooth_norm[..., 1][valid].mean())
    avg_b = float(tooth_norm[..., 2][valid].mean())

    rankings = match_vita(avg_L, avg_a, avg_b)
    best = rankings[0]
    conf_label, conf_desc = confidence_score(best["dE00"])

    tooth_preview = _bgr_to_b64(_lab_to_bgr(tooth_norm))
    glare_overlay = tooth_bgr.copy()
    glare_overlay[~valid] = [0, 0, 255]

    disp = cv2.resize(bgr, (640, int(bgr.shape[0] * 640 / bgr.shape[1])))

    region_label = None
    if used_manual and manual_region:
        r = manual_region
        region_label = (f"x {r['x0']:.2f}–{r['x1']:.2f} · "
                        f"y {r['y0']:.2f}–{r['y1']:.2f}")

    return jsonify({
        "card_detected": card_detected,
        "manual_region": region_label,
        "tooth_lab": {"L": round(avg_L, 2), "a": round(avg_a, 2), "b": round(avg_b, 2)},
        "best_shade": best["shade"],
        "best_group": best["group_name"],
        "dE00": round(best["dE00"], 3),
        "confidence": conf_label,
        "confidence_desc": conf_desc,
        "valid_pixels_pct": round(100 * n_valid / n_total, 1),
        "top5": [
            {"shade": r["shade"], "group": r["group_name"], "dE00": round(r["dE00"], 3)}
            for r in rankings[:5]
        ],
        "images": {
            "original": _bgr_to_b64(disp),
            "warped_card": warped_b64,
            "tooth_norm": tooth_preview,
            "glare_overlay": _bgr_to_b64(glare_overlay),
        },
    })


@app.route("/", methods=["GET"])
def index():
    return _HTML, 200, {"Content-Type": "text/html; charset=utf-8"}
