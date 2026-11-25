"""
run_pipeline.py — Quality-Aware Selective Enhancement for Low-Quality Face Recognition

UNIFIED PIPELINE:
  Input Image (crowded scene)
       │
  ┌────▼─────────────────────┐
  │ STAGE 1: RetinaFace      │  Detect & crop individual faces
  └────┬─────────────────────┘
       │
  ┌────▼─────────────────────┐
  │ STAGE 2: SHQS             │  Score each crop → NO / SOFT / STRONG
  └────┬─────────────────────┘
       │
       ├── NO     → pass through (original crop)
       ├── SOFT   → STAGE 3a: CLAHE  (only SOFT images)
       └── STRONG → STAGE 3b: CodeFormer (only STRONG images)
       │
  ┌────▼─────────────────────┐
  │ MERGE (no duplicates)     │  Exactly one image per face
  └────┬─────────────────────┘
       │
  ┌────▼─────────────────────┐
  │ STAGE 4: ArcFace          │  Embedding + cosine similarity → rank
  └────┬─────────────────────┘
       │
       ▼
   report.json

Usage:
  python run_pipeline.py <input_image> [--reference <ref_image_or_gallery_dir>]
  python run_pipeline.py --build-gallery   (build reference gallery first)

Output goes to:  pipeline_output/<image_stem>/
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# Add the pipeline root to sys.path so stages can be imported
PIPELINE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_ROOT))

from stages.stage1_retinaface import detect_and_crop
from stages.stage2_shqs import shqs_score, level_of, SOFT_THR, STRONG_THR
from stages.stage3a_clahe import clahe_enhance
from stages.stage3b_codeformer import enhance_strong_faces
from stages.stage4_arcface import (
    generate_embedding, cosine_similarity,
    rank_faces, build_gallery, load_gallery,
)

OUTPUT_ROOT   = PIPELINE_ROOT / "pipeline_output"
GALLERY_DIR   = PIPELINE_ROOT / "Reference_Gallery"
GALLERY_FILE  = PIPELINE_ROOT / "models" / "reference_embeddings.npz"
MATCH_THRESHOLD = 0.5

# =====================================================================================
#  DEFAULT TEST PATHS (Change these anytime, or leave None to use command-line arguments)
# =====================================================================================
DEFAULT_INPUT_FOLDER = r"C:\Users\SSD\Desktop\TESTING\DBForMatching"
DEFAULT_REFERENCE    = r"C:\Users\SSD\Desktop\TESTING\ref.jpeg"
# =====================================================================================


def _build_gallery():
    """Build the reference gallery from Reference_Gallery/ folder."""
    if not GALLERY_DIR.exists() or not any(GALLERY_DIR.iterdir()):
        print(f"\n✗ Gallery folder is empty or missing: {GALLERY_DIR}")
        print(f"  Create sub-folders per person with their face images:")
        print(f"    {GALLERY_DIR}/person_A/img1.jpg")
        print(f"    {GALLERY_DIR}/person_B/img1.jpg")
        sys.exit(1)

    print("\n===== BUILDING REFERENCE GALLERY =====")
    build_gallery(GALLERY_DIR, GALLERY_FILE)
    print("Done.\n")


def run_pipeline(image_path, reference_path=None, skip_codeformer=False):
    """
    Run the full face recognition pipeline on a single input image.

    Args:
        image_path:       path to the crowded input image
        reference_path:   path to a single reference image (for 1:1 comparison)
                          OR None to use the pre-built gallery
        skip_codeformer:  if True, STRONG faces fall back to original (no CodeFormer)

    Returns:
        dict: the full pipeline report
    """
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load input image: {image_path}")

    # Set up output directory
    output = OUTPUT_ROOT / image_path.stem
    for folder in ["raw", "no", "soft", "strong", "enhanced", "low_conf"]:
        (output / folder).mkdir(parents=True, exist_ok=True)

    # ================================================================
    #  STAGE 1: RetinaFace — detect and crop
    # ================================================================
    print("\n" + "=" * 60)
    print("  STAGE 1: RetinaFace — Face Detection & Cropping")
    print("=" * 60)

    faces = detect_and_crop(image)
    print(f"  Total faces detected: {len(faces)}")
    accepted = [f for f in faces if f["accepted"]]
    rejected = [f for f in faces if not f["accepted"]]
    print(f"  Accepted (score >= 0.8): {len(accepted)}")
    print(f"  Low confidence (saved separately): {len(rejected)}")

    # Save low-conf crops
    for f in rejected:
        name = f"face_{f['face_id']}.png"
        cv2.imwrite(str(output / "low_conf" / name), f["crop"])

    # Save all accepted raw crops
    for f in accepted:
        name = f"face_{f['face_id']}.png"
        cv2.imwrite(str(output / "raw" / name), f["crop"])

    # ================================================================
    #  STAGE 2: SHQS — quality scoring & classification
    # ================================================================
    print("\n" + "=" * 60)
    print("  STAGE 2: SHQS — Quality Assessment")
    print(f"  Thresholds: NO >= {SOFT_THR} | SOFT >= {STRONG_THR} | STRONG < {STRONG_THR}")
    print("=" * 60)

    classified = {"NO": [], "SOFT": [], "STRONG": []}
    face_data = {}  # face_id -> all metadata

    for f in accepted:
        fid = f["face_id"]
        crop = f["crop"]
        score, factors, found, measurements = shqs_score(crop)
        level = level_of(score)
        weakest = min(factors, key=factors.get)

        classified[level].append(fid)
        face_data[fid] = {
            "face_id": fid,
            "crop": crop,
            "detector_score": f["detector_score"],
            "box": f["box"],
            "landmarks": f["landmarks"],
            "shqs_score": round(float(score), 2),
            "quality_factors": {k: round(float(v), 3) for k, v in factors.items()},
            "level": level,
            "weakest": weakest,
            "face_found": found,
        }

        # Save to level-specific folder
        name = f"face_{fid}.png"
        cv2.imwrite(str(output / level.lower() / name), crop)

        print(f"  face_{fid}: SHQS={score:.1f}  level={level}  weakest={weakest}")

    cnt = Counter(fd["level"] for fd in face_data.values())
    print(f"\n  Summary: NO={cnt.get('NO', 0)}  SOFT={cnt.get('SOFT', 0)}  "
          f"STRONG={cnt.get('STRONG', 0)}")

    # ================================================================
    #  STAGE 3: Selective Enhancement
    #    3a. CLAHE  — ONLY on SOFT images
    #    3b. CodeFormer — ONLY on STRONG images
    #    NO images pass through untouched
    # ================================================================
    print("\n" + "=" * 60)
    print("  STAGE 3: Selective Enhancement")
    print("=" * 60)

    # This dict will hold: face_id -> final_image (BGR array)
    final_images = {}

    # --- NO: pass through ---
    for fid in classified["NO"]:
        final_images[fid] = face_data[fid]["crop"]
        face_data[fid]["enhancement"] = "none"
        face_data[fid]["status"] = "READY"
    if classified["NO"]:
        print(f"  NO ({len(classified['NO'])} faces): passed through as-is")

    # --- SOFT: CLAHE only ---
    for fid in classified["SOFT"]:
        enhanced = clahe_enhance(face_data[fid]["crop"])
        final_images[fid] = enhanced
        face_data[fid]["enhancement"] = "CLAHE"
        face_data[fid]["status"] = "READY"
        cv2.imwrite(str(output / "enhanced" / f"face_{fid}_clahe.png"), enhanced)
    if classified["SOFT"]:
        print(f"  SOFT ({len(classified['SOFT'])} faces): enhanced with CLAHE")

    # --- STRONG: CodeFormer only ---
    if classified["STRONG"]:
        if skip_codeformer:
            print(f"  STRONG ({len(classified['STRONG'])} faces): "
                  "CodeFormer SKIPPED (--skip-codeformer flag)")
            for fid in classified["STRONG"]:
                final_images[fid] = face_data[fid]["crop"]  # fallback to original
                face_data[fid]["enhancement"] = "none (CodeFormer skipped)"
                face_data[fid]["status"] = "READY (fallback)"
        else:
            strong_input = [(fid, face_data[fid]["crop"]) for fid in classified["STRONG"]]
            cf_results = enhance_strong_faces(strong_input)

            for fid in classified["STRONG"]:
                key = str(fid)
                enhanced = cf_results.get(key)
                if enhanced is not None:
                    final_images[fid] = enhanced
                    face_data[fid]["enhancement"] = "CodeFormer"
                    face_data[fid]["status"] = "READY"
                    cv2.imwrite(str(output / "enhanced" / f"face_{fid}_codeformer.png"),
                                enhanced)
                else:
                    # Fallback to original if CodeFormer failed
                    final_images[fid] = face_data[fid]["crop"]
                    face_data[fid]["enhancement"] = "none (CodeFormer failed)"
                    face_data[fid]["status"] = "READY (fallback)"
                    print(f"    face_{fid}: CodeFormer failed, using original")

            ok = sum(1 for fid in classified["STRONG"]
                     if face_data[fid]["enhancement"] == "CodeFormer")
            print(f"  STRONG ({len(classified['STRONG'])} faces): "
                  f"{ok} enhanced with CodeFormer")

    # ================================================================
    #  MERGE CHECK: exactly one final image per accepted face
    # ================================================================
    assert len(final_images) == len(accepted), \
        f"Merge error: {len(final_images)} final images vs {len(accepted)} accepted faces"
    print(f"\n  Merged: {len(final_images)} final images (no duplicates ✓)")

    # Save all final images to enhanced/ for ArcFace
    for fid, img in final_images.items():
        cv2.imwrite(str(output / "enhanced" / f"face_{fid}_final.png"), img)

    # ================================================================
    #  STAGE 4: ArcFace — embedding & matching
    # ================================================================
    print("\n" + "=" * 60)
    print("  STAGE 4: ArcFace — Recognition & Ranking")
    print("=" * 60)

    # Mode A: single reference image comparison
    if reference_path:
        ref_path = Path(reference_path)
        ref_img = cv2.imread(str(ref_path))
        if ref_img is None:
            print(f"  ✗ Could not load reference image: {ref_path}")
        else:
            print(f"  Reference: {ref_path.name}")
            try:
                ref_emb = generate_embedding(ref_img)
            except ValueError as e:
                print(f"  ✗ Reference embedding failed: {e}")
                ref_emb = None

            if ref_emb is not None:
                for fid in sorted(final_images.keys()):
                    try:
                        query_emb = generate_embedding(final_images[fid])
                        sim = cosine_similarity(ref_emb, query_emb)
                        matched = sim >= MATCH_THRESHOLD
                        face_data[fid]["recognition"] = {
                            "reference": ref_path.name,
                            "similarity": round(sim, 4),
                            "status": "MATCH" if matched else "NO MATCH",
                            "threshold": MATCH_THRESHOLD,
                        }
                        status = "MATCH ✓" if matched else "NO MATCH"
                        print(f"  face_{fid}: similarity={sim:.4f}  → {status}")
                    except ValueError as e:
                        face_data[fid]["recognition"] = {"error": str(e)}
                        print(f"  face_{fid}: embedding failed ({e})")

    # Mode B: gallery-based matching
    elif GALLERY_FILE.exists():
        print(f"  Gallery: {GALLERY_FILE.name}")
        gallery_embs, gallery_labels = load_gallery(GALLERY_FILE)
        print(f"  Gallery size: {len(gallery_labels)} images, "
              f"{len(set(gallery_labels))} identities")

        for fid in sorted(final_images.keys()):
            try:
                result = rank_faces(final_images[fid], gallery_embs, gallery_labels,
                                    top_k=5, threshold=MATCH_THRESHOLD)
                face_data[fid]["recognition"] = result
                print(f"  face_{fid}: {result['status']} → "
                      f"{result['predicted_identity']} "
                      f"(sim={result['similarity']:.4f})")
            except ValueError as e:
                face_data[fid]["recognition"] = {"error": str(e)}
                print(f"  face_{fid}: embedding failed ({e})")
    else:
        print("  ⚠ No reference image or gallery found.")
        print(f"    Provide --reference <image> or run --build-gallery first.")
        print(f"    Gallery folder: {GALLERY_DIR}")

    # ================================================================
    #  STAGE 5: Ranking — Top 5 matches
    # ================================================================
    print("\n" + "=" * 60)
    print("  STAGE 5: Ranking — Top 5 Matches")
    print("=" * 60)

    # Build ranking entries from face_data
    ranking_entries = []
    for fid in sorted(face_data.keys()):
        fd = face_data[fid]
        rec = fd.get("recognition", {})
        sim = rec.get("similarity")
        if sim is None:
            continue  # skip faces where embedding failed

        # Determine enhancement status
        enhancement = fd.get("enhancement", "none")
        if enhancement in ("none", "none (CodeFormer skipped)", "none (CodeFormer failed)"):
            enhanced_label = "Not Enhanced"
        elif enhancement == "CLAHE":
            enhanced_label = "Enhanced (CLAHE)"
        elif enhancement == "CodeFormer":
            enhanced_label = "Enhanced (CodeFormer)"
        else:
            enhanced_label = f"Enhanced ({enhancement})"

        status = rec.get("status", "NO MATCH")
        matched_label = "MATCHED ✓" if status == "MATCH" else "NOT MATCHED"

        ranking_entries.append({
            "face_id": fid,
            "name": f"face_{fid}.png",
            "similarity": sim,
            "shqs_score": fd.get("shqs_score", 0),
            "level": fd.get("level", "?"),
            "matched": matched_label,
            "enhanced": enhanced_label,
        })

    # Sort by similarity descending (primary), SHQS descending (secondary)
    ranking_entries.sort(key=lambda e: (e["similarity"], e["shqs_score"]), reverse=True)

    top5 = ranking_entries[:5]

    # Print ranking table
    print(f"\n  {'Rank':<6}{'Face':<16}{'Similarity':<13}{'SHQS':<8}"
          f"{'Status':<16}{'Enhancement'}")
    print("  " + "-" * 75)
    ranking_lines = []
    for rank, entry in enumerate(top5, start=1):
        line = (f"  {f'#{rank}':<6}{entry['name']:<16}{entry['similarity']:<13.4f}"
                f"{entry['shqs_score']:<8.1f}{entry['matched']:<16}{entry['enhanced']}")
        print(line)
        ranking_lines.append(line)

    if not top5:
        print("  No faces with valid embeddings to rank.")

    # Save ranking to text file
    ranking_path = output / "ranking.txt"
    with open(ranking_path, "w", encoding="utf-8") as f:
        f.write(f"STAGE 5: Ranking — Top 5 Matches\n")
        f.write(f"Input Image: {image_path.name}\n")
        if reference_path:
            f.write(f"Reference: {Path(reference_path).name}\n")
        f.write(f"Match Threshold: {MATCH_THRESHOLD}\n")
        f.write(f"\n{'Rank':<6}{'Face':<16}{'Similarity':<13}{'SHQS':<8}"
                f"{'Status':<16}{'Enhancement'}\n")
        f.write("-" * 75 + "\n")
        for rank, entry in enumerate(top5, start=1):
            f.write(f"{f'#{rank}':<6}{entry['name']:<16}{entry['similarity']:<13.4f}"
                    f"{entry['shqs_score']:<8.1f}{entry['matched']:<16}"
                    f"{entry['enhanced']}\n")
        f.write(f"\nTotal faces ranked: {len(ranking_entries)}\n")
        f.write(f"Faces shown: top {len(top5)}\n")

    print(f"\n  Ranking saved to: {ranking_path}")

    # Add ranking to report
    report_ranking = []
    for rank, entry in enumerate(top5, start=1):
        report_ranking.append({
            "rank": rank,
            "face": entry["name"],
            "similarity": entry["similarity"],
            "shqs_score": entry["shqs_score"],
            "quality_level": entry["level"],
            "status": entry["matched"],
            "enhancement": entry["enhanced"],
        })

    # ================================================================
    #  REPORT
    # ================================================================
    report = {
        "input_image": str(image_path.resolve()),
        "total_faces_detected": len(faces),
        "accepted_faces": len(accepted),
        "low_confidence_faces": len(rejected),
        "quality_distribution": {
            "NO": cnt.get("NO", 0),
            "SOFT": cnt.get("SOFT", 0),
            "STRONG": cnt.get("STRONG", 0),
        },
        "thresholds": {
            "SHQS_SOFT": SOFT_THR,
            "SHQS_STRONG": STRONG_THR,
            "MATCH": MATCH_THRESHOLD,
        },
        "faces": [],
    }

    for fid in sorted(face_data.keys()):
        fd = face_data[fid].copy()
        fd.pop("crop", None)  # don't serialize numpy arrays
        report["faces"].append(fd)

    report_path = output / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, default=str)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Input:   {image_path.name}")
    print(f"  Faces:   {len(accepted)} accepted, {len(rejected)} low-conf")
    print(f"  Quality: NO={cnt.get('NO',0)}  SOFT={cnt.get('SOFT',0)}  "
          f"STRONG={cnt.get('STRONG',0)}")
    print(f"  Output:  {output}")
    print(f"  Report:  {report_path}")

    return report


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def run_on_input(input_target, reference_path=None, skip_codeformer=False):
    """
    Runs the pipeline on either a single image file OR all images in a folder.
    """
    target = Path(input_target)
    if not target.exists():
        raise FileNotFoundError(f"Input path not found: {target}")

    if target.is_dir():
        image_files = sorted([p for p in target.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if not image_files:
            print(f"No images found in folder: {target}")
            return []
        print(f"\n========================================================")
        print(f"  FOUND {len(image_files)} IMAGES IN FOLDER: {target.name}")
        print(f"========================================================")
        all_reports = []
        for img_p in image_files:
            print(f"\n>>> Running for: {img_p.name} <<<")
            rep = run_pipeline(img_p, reference_path=reference_path, skip_codeformer=skip_codeformer)
            all_reports.append(rep)
        return all_reports
    else:
        return [run_pipeline(target, reference_path=reference_path, skip_codeformer=skip_codeformer)]


# ================================================================
#  CLI
# ================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Quality-Aware Selective Enhancement for Face Recognition"
    )
    parser.add_argument("input_image", nargs="?",
                        help="Path to the input image OR directory of images")
    parser.add_argument("--reference", "-r", type=str, default=None,
                        help="Reference face image for 1:1 comparison")
    parser.add_argument("--build-gallery", action="store_true",
                        help="Build reference gallery from Reference_Gallery/ folder")
    parser.add_argument("--skip-codeformer", action="store_true",
                        help="Skip CodeFormer stage (STRONG faces use original)")
    parser.add_argument("--threshold", "-t", type=float, default=0.5,
                        help="Match threshold for cosine similarity (default: 0.5)")

    args = parser.parse_args()

    global MATCH_THRESHOLD
    MATCH_THRESHOLD = args.threshold

    if args.build_gallery:
        _build_gallery()
        return

    if not args.input_image:
        if DEFAULT_INPUT_FOLDER and Path(DEFAULT_INPUT_FOLDER).exists():
            print(f"Using default input path: {DEFAULT_INPUT_FOLDER}")
            ref = args.reference or (DEFAULT_REFERENCE if DEFAULT_REFERENCE and Path(DEFAULT_REFERENCE).exists() else None)
            if ref:
                print(f"Using reference: {ref}")
            run_on_input(DEFAULT_INPUT_FOLDER, ref, args.skip_codeformer)
            return

        parser.print_help()
        print("\nExamples:")
        print("  python run_pipeline.py crowd_photo.jpg --reference known_face.jpg")
        print("  python run_pipeline.py C:\\Users\\SSD\\Desktop\\TESTING\\DBForMatching --reference known_face.jpg")
        print("  python run_pipeline.py --build-gallery")
        sys.exit(1)

    run_on_input(args.input_image, args.reference, args.skip_codeformer)


if __name__ == "__main__":
    main()
