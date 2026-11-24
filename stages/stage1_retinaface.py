"""
STAGE 1 — RetinaFace: detect and crop faces from a crowded input image.

Input:  a single image (path or BGR array)
Output: list of dicts, each with:
        face_id, crop (BGR array), detector_score, accepted (bool),
        box [x1,y1,x2,y2], landmarks (relative to crop)

Faces with detector_score >= KEEP_CONF go forward to SHQS.
Low-confidence detections are saved separately for inspection.
"""

import cv2
from retinaface import RetinaFace

DET_THRESHOLD = 0.2   # RetinaFace's own cutoff (low to see every candidate)
KEEP_CONF     = 0.8   # score >= this -> accepted for SHQS
MARGIN        = 0.10  # 10 % extra around the box on each side


def _lm_list(lm):
    """Landmarks as 5 points, sorted by x so naming can't flip them."""
    eyes  = sorted([lm["left_eye"], lm["right_eye"]], key=lambda p: p[0])
    mouth = sorted([lm["mouth_left"], lm["mouth_right"]], key=lambda p: p[0])
    pts = [eyes[0], eyes[1], lm["nose"], mouth[0], mouth[1]]
    return [[float(x), float(y)] for x, y in pts]


def detect_and_crop(img):
    """
    Detect all faces in *img* (BGR numpy array) and return a list of crop dicts.

    Each dict:
        face_id        : int (1-based)
        crop           : BGR numpy array (original pixels, no resize)
        detector_score : float
        accepted       : bool   (score >= KEEP_CONF)
        box            : [x1, y1, x2, y2] in the original image
        landmarks      : [[x,y], ...] relative to the crop
    """
    if img is None or img.size == 0:
        raise ValueError("Invalid input image")

    H, W = img.shape[:2]
    resp = RetinaFace.detect_faces(img, threshold=DET_THRESHOLD)

    results = []
    if not isinstance(resp, dict) or not resp:
        return results

    for i, (_, f) in enumerate(resp.items(), start=1):
        x1, y1, x2, y2 = [int(v) for v in f["facial_area"]]
        bw, bh = x2 - x1, y2 - y1
        x1m = max(int(x1 - MARGIN * bw), 0)
        x2m = min(int(x2 + MARGIN * bw), W)
        y1m = max(int(y1 - MARGIN * bh), 0)
        y2m = min(int(y2 + MARGIN * bh), H)

        crop = img[y1m:y2m, x1m:x2m].copy()
        if crop.size == 0:
            continue

        score = float(f["score"])
        lms = _lm_list(f["landmarks"])

        results.append({
            "face_id": i,
            "crop": crop,
            "detector_score": score,
            "accepted": score >= KEEP_CONF,
            "box": [x1m, y1m, x2m, y2m],
            "landmarks": [[x - x1m, y - y1m] for x, y in lms],
        })

    return results
