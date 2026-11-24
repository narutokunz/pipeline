"""
STAGE 2 — SHQS: Score each cropped face and classify as NO / SOFT / STRONG.

This is the SHQS scoring engine (OpenCV + NumPy only, no model downloads).

SHQS = 100 * S * R * F * N * C * L   (fuzzy memberships multiplied, no weights)
  S = sharpness (blur + low-res, noise-compensated, relative to face contrast)
  R = resolution (native pixel size of the crop)
  F = fidelity  (Haar cascade face check)
  N = noise     (Immerkaer estimate)
  C = contrast  (percentile range)
  L = illumination (distance of mean brightness from mid-grey)

Thresholds:
  NO      SHQS >= SOFT_THR      (good quality, no enhancement)
  SOFT    STRONG_THR <= SHQS < SOFT_THR   (medium, CLAHE)
  STRONG  SHQS < STRONG_THR    (bad, CodeFormer)
"""

import math
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")

# ---- thresholds ----
SOFT_THR   = 75.0
STRONG_THR = 35.0

# ---- membership ramps: (bad, good) ----
RAMPS = {
    "sharp_rel": (0.05, 0.22),
    "sharp_abs": (0.5, 3.0),
    "res":       (10.0, 48.0),
    "noise":     (20.0, 5.0),
    "contrast":  (0.10, 0.35),
    "illum":     (0.48, 0.30),
}
NO_FACE_F = 0.05

VERSION      = "v5 (S*R*F*N*C*L, native-size factor)"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FACE_INTERIOR = 0.45
BAND_SIGMAS   = (1.5, 3.5)
_BAND_GAIN    = []
_CASCADES     = None


def ramp(x, key):
    bad, good = RAMPS[key]
    return float(np.clip((x - bad) / (good - bad), 0.0, 1.0))


# ---- measurements ----
def _central(g, f):
    n = g.shape[0]
    m = int(n * (1 - f) / 2)
    return g[m:n - m, m:n - m]


def _band(g):
    return cv2.GaussianBlur(g, (0, 0), BAND_SIGMAS[0]) - cv2.GaussianBlur(g, (0, 0), BAND_SIGMAS[1])


def _immerkaer(g):
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    h, w = g.shape
    return math.sqrt(math.pi / 2) / (6 * (w - 2) * (h - 2)) * float(np.abs(cv2.filter2D(g, -1, k)).sum())


def measure(bgr):
    """Raw measurements of one face crop."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    interp = cv2.INTER_AREA if min(g.shape) >= 128 else cv2.INTER_CUBIC
    g = cv2.resize(g, (128, 128), interpolation=interp)

    noise = _immerkaer(g)
    if not _BAND_GAIN:
        wn = np.random.RandomState(0).normal(0, 1, (256, 256)).astype(np.float32)
        _BAND_GAIN.append(float(_central(_band(wn), 0.6).std()))
    e = float(_central(_band(g), FACE_INTERIOR).std())
    sharp = math.sqrt(max(e * e - (_BAND_GAIN[0] * noise) ** 2, 0.0))
    sharp_rel = sharp / (float(_central(g, FACE_INTERIOR).std()) + 1e-6)

    sm = cv2.GaussianBlur(g, (0, 0), 1.5)
    p2, p98 = np.percentile(sm, [2, 98])
    clipped = float(((sm <= 5) | (sm >= 250)).mean())
    contrast = ((p98 - p2) / 255.0) * (1 - clipped)
    illum = abs(float(g.mean()) / 255.0 - 0.5)
    return {"sharp": sharp, "sharp_rel": sharp_rel, "noise": noise,
            "contrast": contrast, "illum": illum}


def face_found(bgr):
    """Fidelity check with OpenCV's bundled Haar cascades (frontal + profile)."""
    global _CASCADES
    if _CASCADES is None:
        try:
            names = ["haarcascade_frontalface_default.xml", "haarcascade_profileface.xml"]
            _CASCADES = [cv2.CascadeClassifier(cv2.data.haarcascades + n) for n in names]
            if any(c.empty() for c in _CASCADES):
                raise RuntimeError
        except Exception:
            print("WARNING: Haar cascade files not found — face check skipped (F = 1).")
            _CASCADES = []
    if not _CASCADES:
        return None
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    s = 160.0 / max(g.shape)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    p = int(0.3 * max(g.shape))
    g = cv2.equalizeHist(cv2.copyMakeBorder(g, p, p, p, p, cv2.BORDER_REFLECT))
    front, prof = _CASCADES
    return (len(front.detectMultiScale(g, 1.1, 3, minSize=(40, 40))) > 0
            or len(prof.detectMultiScale(g, 1.1, 3, minSize=(40, 40))) > 0
            or len(prof.detectMultiScale(cv2.flip(g, 1), 1.1, 3, minSize=(40, 40))) > 0)


# ---- score ----
def shqs_score(bgr):
    """Return (score_0_to_100, factor_dict, face_found_bool, raw_measurements)."""
    m = measure(bgr)
    found = face_found(bgr)
    mu = {
        "S": min(ramp(m["sharp_rel"], "sharp_rel"), ramp(m["sharp"], "sharp_abs")),
        "R": ramp(float(min(bgr.shape[:2])), "res"),
        "F": 1.0 if found is None else (1.0 if found else NO_FACE_F),
        "N": ramp(m["noise"], "noise"),
        "C": ramp(m["contrast"], "contrast"),
        "L": ramp(m["illum"], "illum"),
    }
    q = 100.0 * float(np.prod(list(mu.values())))
    return q, mu, found, m


def level_of(q):
    """Classify a SHQS score into NO / SOFT / STRONG."""
    if q >= SOFT_THR:
        return "NO"
    return "STRONG" if q < STRONG_THR else "SOFT"
