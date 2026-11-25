"""
server.py — FastAPI backend for the Face Recognition Pipeline web application.

Serves the React frontend, provides REST + SSE endpoints for uploading images,
running the pipeline as a background task, streaming progress, and downloading
results as JSON / CSV / annotated images.

Usage:
    cd pipeline/web
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import csv
import io
import json
import os
import shutil
import sys
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
#  Resolve paths & import pipeline stages
# ---------------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parent.parent          # …/pipeline/
sys.path.insert(0, str(PIPELINE_ROOT))

from stages.stage1_retinaface import detect_and_crop            # noqa: E402
from stages.stage2_shqs import (                                # noqa: E402
    shqs_score, level_of, SOFT_THR, STRONG_THR,
)
from stages.stage3a_clahe import clahe_enhance                  # noqa: E402
from stages.stage3b_codeformer import enhance_strong_faces      # noqa: E402
from stages.stage4_arcface import (                             # noqa: E402
    generate_embedding, cosine_similarity,
)

# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------
MATCH_THRESHOLD = 0.5
UPLOAD_DIR      = PIPELINE_ROOT / "web" / "_uploads"
OUTPUT_DIR      = PIPELINE_ROOT / "web" / "_output"
FRONTEND_DIR    = PIPELINE_ROOT / "web" / "frontend" / "dist"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# ---------------------------------------------------------------------------
#  In-memory task store  (task_id → task dict)
# ---------------------------------------------------------------------------
tasks: Dict[str, dict] = {}

# ---------------------------------------------------------------------------
#  FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Face Recognition Pipeline",
    version="1.0.0",
    description="Web backend for quality-aware selective-enhancement face recognition.",
)

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _validate_image_ext(filename: str) -> None:
    """Raise 400 if the file extension is not a supported image format."""
    if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format. Allowed: {', '.join(sorted(IMAGE_EXTENSIONS))}",
        )


def _save_upload(upload: UploadFile, sub_dir: str) -> Path:
    """Read an UploadFile and persist it under UPLOAD_DIR/<sub_dir>/."""
    _validate_image_ext(upload.filename)
    dest_dir = UPLOAD_DIR / sub_dir
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Prefix with a short uuid to avoid collisions
    safe_name = f"{uuid.uuid4().hex[:8]}_{upload.filename}"
    dest = dest_dir / safe_name
    with open(dest, "wb") as f:
        f.write(upload.file.read())

    # Quick sanity check — can OpenCV decode it?
    if cv2.imread(str(dest)) is None:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return dest


def _resolve_image_path(filename: str) -> Path:
    """
    Look up *filename* across the upload and output trees.
    Returns the first existing match or raises 404.
    """
    # 1. Direct match under uploads
    for p in UPLOAD_DIR.rglob(filename):
        if p.is_file():
            return p
    # 2. Direct match under task outputs
    for p in OUTPUT_DIR.rglob(filename):
        if p.is_file():
            return p
    raise HTTPException(status_code=404, detail=f"Image not found: {filename}")


def _draw_annotated(crowd_img: np.ndarray, faces: list) -> np.ndarray:
    """
    Draw bounding boxes on the crowd image.
      • green  (0,200,0)   → MATCH   (similarity >= 0.5)
      • orange (0,165,255) → NO MATCH
      • grey   (160,160,160) → embedding failed / low confidence
    """
    annotated = crowd_img.copy()
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        sim = f.get("similarity_score")
        match = f.get("match_status", "UNKNOWN")

        if match == "MATCH":
            color = (0, 200, 0)       # green
        elif match == "NO MATCH":
            color = (0, 165, 255)     # orange
        else:
            color = (160, 160, 160)   # grey

        thickness = max(2, int(min(crowd_img.shape[:2]) / 300))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label
        label = f"#{f['face_id']}"
        if sim is not None:
            label += f" {sim:.2f}"
        font_scale = max(0.4, min(crowd_img.shape[:2]) / 1200)
        cv2.putText(
            annotated, label, (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness,
        )
    return annotated


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE RUNNER  (executes in a background thread)
# ═══════════════════════════════════════════════════════════════════════════

def _run_pipeline_task(task_id: str) -> None:
    """
    Execute the full face-recognition pipeline for *task_id* and update the
    shared ``tasks`` dict with progress events and final results.
    """
    task = tasks[task_id]
    events: list = task["events"]           # SSE event buffer
    t_total = time.time()

    def emit(stage: str, status: str, message: str, **extra):
        evt = {
            "stage": stage,
            "status": status,
            "message": message,
            "elapsed": round(time.time() - t_total, 2),
            **extra,
        }
        events.append(evt)

    try:
        ref_path  = Path(task["reference_path"])
        crowd_path = Path(task["crowd_path"])

        ref_img   = cv2.imread(str(ref_path))
        crowd_img = cv2.imread(str(crowd_path))
        if ref_img is None:
            raise ValueError(f"Cannot load reference image: {ref_path.name}")
        if crowd_img is None:
            raise ValueError(f"Cannot load crowd image: {crowd_path.name}")

        # Prepare task output directory
        task_out = OUTPUT_DIR / task_id
        for sub in ("raw", "enhanced", "low_conf"):
            (task_out / sub).mkdir(parents=True, exist_ok=True)

        stage_times: Dict[str, float] = {}

        # ── STAGE 1: Face Detection ──────────────────────────────────────
        emit("detecting_faces", "active", "Detecting faces with RetinaFace...")
        t0 = time.time()

        faces_raw = detect_and_crop(crowd_img)
        accepted  = [f for f in faces_raw if f["accepted"]]
        rejected  = [f for f in faces_raw if not f["accepted"]]

        stage_times["detection"] = round(time.time() - t0, 2)
        emit(
            "detecting_faces", "complete",
            f"Detected {len(faces_raw)} faces ({len(accepted)} accepted, "
            f"{len(rejected)} low-confidence)",
        )

        if not accepted:
            emit("complete", "complete", "No accepted faces found in the image.",
                 results_ready=True)
            task["status"] = "complete"
            task["results"] = {
                "total_detected": len(faces_raw),
                "accepted": 0,
                "low_confidence": len(rejected),
                "faces": [],
                "stage_times": stage_times,
                "total_time": round(time.time() - t_total, 2),
            }
            return

        # Save low-conf crops
        for f in rejected:
            cv2.imwrite(str(task_out / "low_conf" / f"face_{f['face_id']}.png"), f["crop"])

        # Save raw accepted crops
        for f in accepted:
            cv2.imwrite(str(task_out / "raw" / f"face_{f['face_id']}.png"), f["crop"])

        # ── STAGE 2: Quality Assessment ──────────────────────────────────
        emit("evaluating_quality", "active", "Evaluating face quality with SHQS...")
        t0 = time.time()

        classified = {"NO": [], "SOFT": [], "STRONG": []}
        face_data: Dict[int, dict] = {}

        for f in accepted:
            fid = f["face_id"]
            crop = f["crop"]
            score, factors, found, measurements = shqs_score(crop)
            level = level_of(score)
            classified[level].append(fid)

            face_data[fid] = {
                "face_id":          fid,
                "crop":             crop,                       # kept in-memory only
                "detector_confidence": round(float(f["detector_score"]), 4),
                "bbox":             f["box"],                   # [x1,y1,x2,y2]
                "landmarks":        f["landmarks"],
                "shqs_score":       round(float(score), 2),
                "quality_factors":  {k: round(float(v), 3) for k, v in factors.items()},
                "quality_level":    level,
                "face_found":       found,
                "measurements":     {k: round(float(v), 4) for k, v in measurements.items()},
            }

        cnt = Counter(fd["quality_level"] for fd in face_data.values())
        stage_times["quality"] = round(time.time() - t0, 2)
        emit(
            "evaluating_quality", "complete",
            f"Quality: NO={cnt.get('NO',0)}  SOFT={cnt.get('SOFT',0)}  "
            f"STRONG={cnt.get('STRONG',0)}",
        )

        # ── STAGE 3: Enhancement ─────────────────────────────────────────
        emit("enhancing_faces", "active", "Enhancing degraded faces...")
        t0 = time.time()

        final_images: Dict[int, np.ndarray] = {}

        # NO — pass through
        for fid in classified["NO"]:
            final_images[fid] = face_data[fid]["crop"]
            face_data[fid]["enhancement_method"] = "none"

        # SOFT — CLAHE
        for fid in classified["SOFT"]:
            enhanced = clahe_enhance(face_data[fid]["crop"])
            final_images[fid] = enhanced
            face_data[fid]["enhancement_method"] = "CLAHE"
            cv2.imwrite(
                str(task_out / "enhanced" / f"face_{fid}_clahe.png"), enhanced,
            )

        # STRONG — CodeFormer
        if classified["STRONG"]:
            strong_input = [(fid, face_data[fid]["crop"]) for fid in classified["STRONG"]]
            cf_results = enhance_strong_faces(strong_input)
            for fid in classified["STRONG"]:
                result = cf_results.get(str(fid))
                if result is not None:
                    final_images[fid] = result
                    face_data[fid]["enhancement_method"] = "CodeFormer"
                    cv2.imwrite(
                        str(task_out / "enhanced" / f"face_{fid}_codeformer.png"), result,
                    )
                else:
                    final_images[fid] = face_data[fid]["crop"]
                    face_data[fid]["enhancement_method"] = "none (CodeFormer failed)"
        else:
            pass  # no STRONG faces

        # Save final images (one per face)
        for fid, img in final_images.items():
            cv2.imwrite(str(task_out / "enhanced" / f"face_{fid}_final.png"), img)

        stage_times["enhancement"] = round(time.time() - t0, 2)
        emit("enhancing_faces", "complete",
             f"Enhanced {len(classified['SOFT'])} SOFT + {len(classified['STRONG'])} STRONG faces")

        # ── STAGE 4: Embedding & Matching ────────────────────────────────
        emit("matching_faces", "active", "Generating embeddings & matching against reference...")
        t0 = time.time()

        # Generate reference embedding
        try:
            ref_emb = generate_embedding(ref_img)
        except ValueError as e:
            emit("matching_faces", "complete",
                 f"Reference embedding failed: {e}")
            ref_emb = None

        if ref_emb is not None:
            for fid in sorted(final_images.keys()):
                try:
                    query_emb = generate_embedding(final_images[fid])
                    sim = cosine_similarity(ref_emb, query_emb)
                    matched = sim >= MATCH_THRESHOLD
                    face_data[fid]["similarity_score"] = round(sim, 4)
                    face_data[fid]["match_status"] = "MATCH" if matched else "NO MATCH"
                except ValueError as e:
                    face_data[fid]["similarity_score"] = None
                    face_data[fid]["match_status"] = f"ERROR: {e}"
        else:
            for fid in face_data:
                face_data[fid]["similarity_score"] = None
                face_data[fid]["match_status"] = "SKIPPED (reference failed)"

        stage_times["matching"] = round(time.time() - t0, 2)
        emit("matching_faces", "complete", "Embedding & similarity complete")

        # ── STAGE 5: Ranking ─────────────────────────────────────────────
        emit("ranking_results", "active", "Ranking faces by similarity & quality...")
        t0 = time.time()

        ranking = []
        for fid in sorted(face_data.keys()):
            fd = face_data[fid]
            sim = fd.get("similarity_score")
            if sim is None:
                continue
            ranking.append({
                "face_id":    fid,
                "similarity": sim,
                "shqs":       fd["shqs_score"],
                "level":      fd["quality_level"],
                "match":      fd.get("match_status", "UNKNOWN"),
            })
        ranking.sort(key=lambda e: (e["similarity"], e["shqs"]), reverse=True)

        stage_times["ranking"] = round(time.time() - t0, 2)
        emit("ranking_results", "complete",
             f"Ranked {len(ranking)} faces")

        # ── Annotated image ──────────────────────────────────────────────
        emit("generating_output", "active", "Drawing annotated image...")
        t0 = time.time()

        face_results = []
        for fid in sorted(face_data.keys()):
            fd = face_data[fid]
            # Determine crop paths (relative to task output)
            original_crop = f"raw/face_{fid}.png"
            enh_method = fd.get("enhancement_method", "none")
            if enh_method == "CLAHE":
                enhanced_crop = f"enhanced/face_{fid}_clahe.png"
            elif enh_method == "CodeFormer":
                enhanced_crop = f"enhanced/face_{fid}_codeformer.png"
            else:
                enhanced_crop = None

            face_results.append({
                "face_id":              fid,
                "bbox":                 fd["bbox"],
                "shqs_score":           fd["shqs_score"],
                "quality_factors":      fd["quality_factors"],
                "quality_level":        fd["quality_level"],
                "enhancement_method":   enh_method,
                "similarity_score":     fd.get("similarity_score"),
                "match_status":         fd.get("match_status", "UNKNOWN"),
                "detector_confidence":  fd["detector_confidence"],
                "original_crop":        original_crop,
                "enhanced_crop":        enhanced_crop,
                "final_crop":           f"enhanced/face_{fid}_final.png",
                "measurements":         fd.get("measurements", {}),
            })

        annotated = _draw_annotated(crowd_img, face_results)
        annotated_path = task_out / "annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])

        stage_times["output"] = round(time.time() - t0, 2)
        total_time = round(time.time() - t_total, 2)

        # ── Assemble final results ───────────────────────────────────────
        results = {
            "task_id":          task_id,
            "total_detected":   len(faces_raw),
            "accepted":         len(accepted),
            "low_confidence":   len(rejected),
            "quality_distribution": {
                "NO":     cnt.get("NO", 0),
                "SOFT":   cnt.get("SOFT", 0),
                "STRONG": cnt.get("STRONG", 0),
            },
            "thresholds": {
                "SHQS_SOFT":    SOFT_THR,
                "SHQS_STRONG":  STRONG_THR,
                "MATCH":        MATCH_THRESHOLD,
            },
            "faces":          face_results,
            "ranking":        ranking[:10],      # top-10
            "stage_times":    stage_times,
            "total_time":     total_time,
            "annotated_image": "annotated.jpg",
        }

        task["results"] = results
        task["status"]  = "complete"
        emit("generating_output", "complete", "Output generated")
        emit("complete", "complete", f"Pipeline finished in {total_time}s",
             results_ready=True)

    except Exception as exc:
        task["status"] = "error"
        task["error"]  = str(exc)
        emit("error", "error", f"Pipeline failed: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

# ── Upload endpoints ─────────────────────────────────────────────────────

@app.post("/api/upload/reference")
async def upload_reference(file: UploadFile = File(...)):
    """Accept a reference face image, save it, and return a preview URL."""
    dest = _save_upload(file, "reference")
    return JSONResponse({
        "filename":    dest.name,
        "preview_url": f"/api/images/{dest.name}",
        "message":     "Reference image uploaded successfully.",
    })


@app.post("/api/upload/crowd")
async def upload_crowd(file: UploadFile = File(...)):
    """Accept a crowd / target image, save it, and return a preview URL."""
    dest = _save_upload(file, "crowd")
    return JSONResponse({
        "filename":    dest.name,
        "preview_url": f"/api/images/{dest.name}",
        "message":     "Crowd image uploaded successfully.",
    })


# ── Image serving ────────────────────────────────────────────────────────

@app.get("/api/images/{filename:path}")
async def serve_image(filename: str):
    """
    Serve an image from the upload directory or any task output directory.
    Accepts plain filenames and paths like ``<task_id>/raw/face_1.png``.
    """
    path = _resolve_image_path(filename)
    return FileResponse(path, media_type="image/png")


# ── Analysis trigger ─────────────────────────────────────────────────────

@app.post("/api/analyze")
async def start_analysis(
    reference_filename: Optional[str] = None,
    crowd_filename: Optional[str] = None,
):
    """
    Kick off the pipeline as a background task.

    Body (JSON):
        reference_filename  – name of a previously uploaded reference image
        crowd_filename      – name of a previously uploaded crowd image

    Returns:
        task_id to poll progress via SSE.
    """
    if not reference_filename or not crowd_filename:
        raise HTTPException(status_code=400,
                            detail="Both reference_filename and crowd_filename are required.")

    # Locate the actual files
    try:
        ref_path   = _resolve_image_path(reference_filename)
        crowd_path = _resolve_image_path(crowd_filename)
    except HTTPException:
        raise HTTPException(
            status_code=404,
            detail="One or both uploaded images could not be found. Please re-upload.",
        )

    task_id = uuid.uuid4().hex
    tasks[task_id] = {
        "task_id":        task_id,
        "status":         "running",
        "reference_path": str(ref_path),
        "crowd_path":     str(crowd_path),
        "events":         [],       # SSE event buffer
        "results":        None,
        "error":          None,
        "created_at":     time.time(),
    }

    thread = threading.Thread(
        target=_run_pipeline_task, args=(task_id,), daemon=True,
    )
    thread.start()

    return JSONResponse({
        "task_id": task_id,
        "message": "Pipeline analysis started.",
        "status_url": f"/api/status/{task_id}",
        "results_url": f"/api/results/{task_id}",
    })


# ── SSE progress stream ─────────────────────────────────────────────────

@app.get("/api/status/{task_id}")
async def stream_status(task_id: str):
    """
    Server-Sent Events endpoint that streams pipeline progress.

    Each event is a JSON object with ``stage``, ``status``, ``message``,
    ``elapsed``, and optionally ``results_ready``.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    def _event_generator():
        sent = 0
        while True:
            task = tasks.get(task_id)
            if task is None:
                break

            events = task["events"]
            # Send any new events that appeared since last check
            while sent < len(events):
                evt = events[sent]
                yield f"data: {json.dumps(evt)}\n\n"
                sent += 1
                # If pipeline is done or errored, close the stream
                if evt.get("stage") in ("complete", "error"):
                    return

            # Also break if the task status changed externally
            if task["status"] in ("complete", "error") and sent >= len(events):
                return

            time.sleep(0.3)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Results ──────────────────────────────────────────────────────────────

@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    """Return completed analysis results as JSON."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    task = tasks[task_id]
    if task["status"] == "running":
        return JSONResponse({"status": "running", "message": "Analysis still in progress."},
                            status_code=202)
    if task["status"] == "error":
        raise HTTPException(status_code=500, detail=task.get("error", "Unknown error"))

    return JSONResponse(task["results"])


@app.get("/api/results/{task_id}/annotated")
async def get_annotated_image(task_id: str):
    """Return the annotated crowd image with bounding boxes drawn."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    task = tasks[task_id]
    if task["status"] != "complete":
        raise HTTPException(status_code=400, detail="Results not ready yet.")

    annotated_path = OUTPUT_DIR / task_id / "annotated.jpg"
    if not annotated_path.exists():
        raise HTTPException(status_code=404, detail="Annotated image not found.")

    return FileResponse(annotated_path, media_type="image/jpeg",
                        filename="annotated.jpg")


# ── Downloads ────────────────────────────────────────────────────────────

@app.get("/api/results/{task_id}/download/json")
async def download_json(task_id: str):
    """Download the full results as a JSON report."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    task = tasks[task_id]
    if task["status"] != "complete" or task["results"] is None:
        raise HTTPException(status_code=400, detail="Results not ready yet.")

    content = json.dumps(task["results"], indent=2, default=str)
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="report_{task_id[:8]}.json"'},
    )


@app.get("/api/results/{task_id}/download/csv")
async def download_csv(task_id: str):
    """Download per-face results as CSV."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Unknown task_id")

    task = tasks[task_id]
    if task["status"] != "complete" or task["results"] is None:
        raise HTTPException(status_code=400, detail="Results not ready yet.")

    faces = task["results"].get("faces", [])

    fieldnames = [
        "face_id", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "detector_confidence", "shqs_score", "quality_level",
        "S", "R", "F", "N", "C", "L",
        "enhancement_method", "similarity_score", "match_status",
        "original_crop", "enhanced_crop",
    ]

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for f in faces:
        qf = f.get("quality_factors", {})
        row = {
            "face_id":              f["face_id"],
            "bbox_x1":             f["bbox"][0],
            "bbox_y1":             f["bbox"][1],
            "bbox_x2":             f["bbox"][2],
            "bbox_y2":             f["bbox"][3],
            "detector_confidence": f["detector_confidence"],
            "shqs_score":          f["shqs_score"],
            "quality_level":       f["quality_level"],
            "S":                   qf.get("S", ""),
            "R":                   qf.get("R", ""),
            "F":                   qf.get("F", ""),
            "N":                   qf.get("N", ""),
            "C":                   qf.get("C", ""),
            "L":                   qf.get("L", ""),
            "enhancement_method":  f["enhancement_method"],
            "similarity_score":    f.get("similarity_score", ""),
            "match_status":        f.get("match_status", ""),
            "original_crop":       f.get("original_crop", ""),
            "enhanced_crop":       f.get("enhanced_crop", ""),
        }
        writer.writerow(row)

    csv_bytes = buf.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="faces_{task_id[:8]}.csv"'},
    )


# ── Session clear ────────────────────────────────────────────────────────

@app.post("/api/clear")
async def clear_session():
    """
    Remove all uploaded images, task output directories, and in-memory state.
    """
    removed_uploads = 0
    removed_tasks   = 0

    # Clean uploads
    if UPLOAD_DIR.exists():
        for child in UPLOAD_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            removed_uploads += 1

    # Clean task outputs
    if OUTPUT_DIR.exists():
        for child in OUTPUT_DIR.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed_tasks += 1

    tasks.clear()

    return JSONResponse({
        "message":          "Session cleared.",
        "removed_uploads":  removed_uploads,
        "removed_tasks":    removed_tasks,
    })


# ═══════════════════════════════════════════════════════════════════════════
#  Static frontend (mount LAST so API routes take priority)
# ═══════════════════════════════════════════════════════════════════════════

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


# ═══════════════════════════════════════════════════════════════════════════
#  Run with:  python server.py
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    print(f"Pipeline root : {PIPELINE_ROOT}")
    print(f"Upload dir    : {UPLOAD_DIR}")
    print(f"Output dir    : {OUTPUT_DIR}")
    print(f"Frontend      : {FRONTEND_DIR}  ({'found' if FRONTEND_DIR.exists() else 'NOT FOUND'})")
    print()
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
