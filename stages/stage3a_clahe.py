"""
STAGE 3a — CLAHE enhancement for SOFT-quality faces ONLY.

Takes ONLY the face crops classified as SOFT by SHQS. Never touches NO or STRONG images.
Applies adaptive CLAHE on the LAB-L channel (preserves colour).

Input:  BGR numpy array of a single face crop
Output: BGR numpy array (enhanced)
"""

import cv2
import numpy as np

CLAHE_CLIP       = 2.0
CLAHE_MIN_TILE_PX = 16
CLAHE_MAX_TILES  = 8


def clahe_enhance(bgr, clip=CLAHE_CLIP, min_tile_px=CLAHE_MIN_TILE_PX,
                  max_tiles=CLAHE_MAX_TILES):
    """CLAHE on LAB-L only (colour kept); tile grid adapts to image size."""
    h, w = bgr.shape[:2]
    tiles = int(np.clip(min(h, w) // min_tile_px, 2, max_tiles))
    l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
