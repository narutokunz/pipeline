"""
STAGE 3b — CodeFormer enhancement for STRONG-quality faces ONLY.

Takes ONLY the face crops classified as STRONG by SHQS.
Calls CodeFormer's inference script as a subprocess (it has its own venv / deps).

IMPORTANT: set CODEFORMER_ROOT to the CodeFormer installation folder that
contains inference_codeformer.py and the weights/ directory.

Input:  list of (face_id, BGR crop) tuples
Output: dict  { face_id: BGR_enhanced_array | None }
"""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# ---- CONFIGURATION ----
# Point this to the root of the CodeFormer repo that has inference_codeformer.py
CODEFORMER_ROOT = Path(os.environ.get(
    "CODEFORMER_ROOT",
    r"C:\Users\SSD\Downloads\CodeFormer-master (2)\CodeFormer-master\CodeFormer-master"
))


def _resolve_codeformer_python():
    """Find a valid working Python executable."""
    env_py = os.environ.get("CODEFORMER_PYTHON")
    if env_py and Path(env_py).exists():
        return Path(env_py)

    # Check if CodeFormer's own venv works by testing it
    cf_venv_py = CODEFORMER_ROOT / "venv" / "Scripts" / "python.exe"
    if cf_venv_py.exists():
        try:
            res = subprocess.run([str(cf_venv_py), "-c", "import sys"], capture_output=True, timeout=5)
            if res.returncode == 0:
                return cf_venv_py
        except Exception:
            pass

    # Default to the current active Python running the pipeline
    return Path(sys.executable)


CODEFORMER_PYTHON = _resolve_codeformer_python()

W           = 0.8     # fidelity weight (0.7-0.9 for degraded faces)
PAD         = 0.35    # context padding around the crop for the detector
CROP_MARGIN = 0.0     # extra shrink of the final crop
OUT_SIZE    = None    # None = keep at 512-scale resolution
MIN_SIM     = 0.65    # low-freq similarity check


def _prep(bgr, size=512):
    """Pad + make square + upscale to size x size.
    Returns (prepared_image, box_of_original_inside_it)."""
    h, w = bgr.shape[:2]
    p = int(PAD * max(h, w))
    x = cv2.copyMakeBorder(bgr, p, p, p, p, cv2.BORDER_REPLICATE)
    H, Wd = x.shape[:2]
    s = max(H, Wd)
    top, left = (s - H) // 2, (s - Wd) // 2
    x = cv2.copyMakeBorder(x, top, s - H - top, left, s - Wd - left, cv2.BORDER_REPLICATE)
    x = cv2.resize(x, (size, size), interpolation=cv2.INTER_LANCZOS4)
    k = size / s
    box = ((left + p) * k, (top + p) * k, (left + p + w) * k, (top + p + h) * k)
    return x, box


def _crop_to_original(img, box):
    """Cut padding away, keeping only the original image area."""
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * CROP_MARGIN, (y1 - y0) * CROP_MARGIN
    x0, y0 = int(round(x0 + mx)), int(round(y0 + my))
    x1, y1 = int(round(x1 - mx)), int(round(y1 - my))
    out = img[max(y0, 0):y1, max(x0, 0):x1]
    if OUT_SIZE is not None:
        if isinstance(OUT_SIZE, int):
            sc = OUT_SIZE / max(out.shape[:2])
            interp = cv2.INTER_AREA if sc < 1 else cv2.INTER_LANCZOS4
            out = cv2.resize(out, None, fx=sc, fy=sc, interpolation=interp)
        else:
            out = cv2.resize(out, OUT_SIZE, interpolation=cv2.INTER_AREA)
    return out


def _similarity(orig, restored_crop):
    """Low-frequency correlation between original and restored crop."""
    a = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY), (32, 32),
                   interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    b = cv2.resize(cv2.cvtColor(restored_crop, cv2.COLOR_BGR2GRAY), (32, 32),
                   interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))


def enhance_strong_faces(faces):
    """
    Enhance a batch of STRONG-quality face crops using CodeFormer.

    Args:
        faces: list of (face_id: str/int, crop: BGR ndarray) tuples
               — ONLY faces that SHQS classified as STRONG.

    Returns:
        dict { face_id: enhanced_BGR_array | None }
        None means CodeFormer could not process (face not detected / rejected).
    """
    if not faces:
        return {}

    # Create temp workspace
    tmp = Path(__file__).resolve().parent.parent / "_codeformer_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    in_dir  = tmp / "in"
    out_dir = tmp / "out"
    in_dir.mkdir(parents=True, exist_ok=True)

    # Prepare each crop: pad + upscale, save to temp input folder
    todo = {}
    for face_id, crop in faces:
        stem = str(face_id)
        padded, box = _prep(crop)
        cv2.imwrite(str(in_dir / f"{stem}.png"), padded)
        todo[stem] = (crop, box)

    print(f"  [CodeFormer] Enhancing {len(todo)} STRONG faces (w={W}) ...")

    # Run CodeFormer inference as a subprocess
    cmd = [
        str(CODEFORMER_PYTHON),
        "inference_codeformer.py",
        "-w", str(W),
        "--input_path",  str(in_dir.resolve()),
        "--output_path", str(out_dir.resolve()),
        "--detection_model", "retinaface_resnet50",
        "--bg_upsampler", "None",
        "--only_center_face",
        "--face_upsample",
    ]

    try:
        subprocess.run(cmd, cwd=str(CODEFORMER_ROOT), check=True,
                       capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"  [CodeFormer] ERROR: {e.stderr[:500] if e.stderr else e}")
        # Return None for all faces
        shutil.rmtree(tmp, ignore_errors=True)
        return {fid: None for fid, _ in faces}
    except FileNotFoundError:
        print(f"  [CodeFormer] ERROR: Python not found at {CODEFORMER_PYTHON}")
        print(f"  Set CODEFORMER_PYTHON env var or check CODEFORMER_ROOT={CODEFORMER_ROOT}")
        shutil.rmtree(tmp, ignore_errors=True)
        return {fid: None for fid, _ in faces}

    # Collect results: crop padding away, check similarity
    results = {}
    for stem, (orig_crop, box) in todo.items():
        res_path = out_dir / "final_results" / f"{stem}.png"
        if not res_path.exists():
            print(f"    {stem}: no output (face not detected by CodeFormer)")
            results[stem] = None
            continue

        full = cv2.imread(str(res_path))
        k = full.shape[0] / 512.0
        box_k = tuple(v * k for v in box)
        restored = _crop_to_original(full, box_k)
        sim = _similarity(orig_crop, restored)

        if sim >= MIN_SIM:
            results[stem] = restored
            print(f"    {stem}: OK (similarity={sim:.3f})")
        else:
            print(f"    {stem}: REJECTED (similarity={sim:.3f} < {MIN_SIM})")
            results[stem] = None  # fallback to original in merge

    shutil.rmtree(tmp, ignore_errors=True)
    return results
