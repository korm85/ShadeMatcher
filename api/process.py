"""
Vercel serverless function — dental shade matching pipeline.
Accepts a multipart POST with an image, returns JSON results + base64 previews.
"""

import os, sys, base64, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np
from flask import Flask, request, jsonify

from aruco_utils import detect_aruco, warp_card, sample_swatches
from calibration import fit_ccm, apply_ccm, glare_mask, msq_normalise
from shade_matching import match_vita, confidence_score

app = Flask(__name__)

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


def extract_tooth_region(bgr, card_detected):
    h, w = bgr.shape[:2]
    if card_detected:
        y0, y1 = 0,            int(h * 0.52)
        x0, x1 = int(w * 0.17), int(w * 0.83)
    else:
        y0, y1 = int(h * 0.25), int(h * 0.75)
        x0, x1 = int(w * 0.25), int(w * 0.75)
    return bgr[y0:y1, x0:x1]


@app.route("/api/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    data = request.files["image"].read()
    arr = np.frombuffer(data, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        return jsonify({"error": "Could not decode image"}), 400

    # Downscale large images for speed (serverless timeout)
    max_dim = 1600
    h, w = bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)))

    glare_threshold = float(request.form.get("glare_threshold", 95))

    # --- Card detection & CCM fitting ---
    corners, ids = detect_aruco(bgr)
    warped = warp_card(bgr, corners, ids)
    card_detected = warped is not None
    ccm = None
    warped_b64 = None

    if card_detected:
        raw = sample_swatches(warped)
        rgb_norm = raw[:, ::-1] / 255.0
        ccm = fit_ccm(rgb_norm, CARD_GT_LAB)
        warped_b64 = _bgr_to_b64(warped)

    # --- Tooth region ---
    tooth_bgr = extract_tooth_region(bgr, card_detected)

    if ccm is not None:
        tooth_lab = apply_ccm(tooth_bgr, ccm)
    else:
        u8 = cv2.cvtColor(tooth_bgr, cv2.COLOR_BGR2LAB)
        L = u8[..., 0].astype(np.float32) * (100.0 / 255.0)
        a = u8[..., 1].astype(np.float32) - 128.0
        b = u8[..., 2].astype(np.float32) - 128.0
        tooth_lab = np.stack([L, a, b], -1)

    valid = glare_mask(tooth_lab, glare_threshold)
    n_valid = int(valid.sum())
    n_total = int(valid.size)

    if n_valid == 0:
        return jsonify({"error": "All pixels masked as glare — lower threshold"}), 422

    tooth_norm = msq_normalise(tooth_lab)
    avg_L = float(tooth_norm[..., 0][valid].mean())
    avg_a = float(tooth_norm[..., 1][valid].mean())
    avg_b = float(tooth_norm[..., 2][valid].mean())

    # --- Shade matching ---
    rankings = match_vita(avg_L, avg_a, avg_b)
    best = rankings[0]
    conf_label, conf_desc = confidence_score(best["dE00"])

    # --- Preview images ---
    tooth_preview = _bgr_to_b64(_lab_to_bgr(tooth_norm))
    glare_overlay = tooth_bgr.copy()
    glare_overlay[~valid] = [0, 0, 255]
    glare_b64 = _bgr_to_b64(glare_overlay)

    # Resize original for display
    disp_bgr = cv2.resize(bgr, (640, int(bgr.shape[0] * 640 / bgr.shape[1])))
    original_b64 = _bgr_to_b64(disp_bgr)

    return jsonify({
        "card_detected": card_detected,
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
            "original": original_b64,
            "warped_card": warped_b64,
            "tooth_norm": tooth_preview,
            "glare_overlay": glare_b64,
        },
    })


@app.route("/", methods=["GET"])
def index():
    html_path = os.path.join(os.path.dirname(__file__), "..", "public", "index.html")
    with open(html_path) as f:
        return f.read(), 200, {"Content-Type": "text/html"}
