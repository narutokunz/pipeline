# """
# STAGE 4 — ArcFace: generate embeddings and match faces against a reference gallery.

# Uses InsightFace's buffalo_l model for recognition (512-dim embeddings, L2-normalized).
# Since input images are already cropped faces, detection is skipped — the recognition
# model is used directly after resizing to 112×112.

# Reference gallery:
#   - Build once with  build_gallery()
#   - Stored as  reference_embeddings.npz  in the models/ folder
#   - Structure: Reference_Gallery/<person_name>/img1.jpg, img2.jpg, ...

# Matching:
#   - rank_faces()  returns top-K matches with cosine similarities
# """

# import cv2
# import numpy as np
# from pathlib import Path

# _recognizer = None


# def _load_model():
#     """Lazy-load the ArcFace recognition model (singleton)."""
#     global _recognizer
#     if _recognizer is None:
#         from insightface.app import FaceAnalysis
#         app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
#         app.prepare(ctx_id=-1, det_size=(640, 640))
#         _recognizer = app.models["recognition"]
#     return _recognizer


# def generate_embedding(image):
#     """
#     Generate a 512-dim L2-normalized ArcFace embedding from a face crop (BGR array).

#     Args:
#         image: BGR numpy array (already cropped face)

#     Returns:
#         np.ndarray of shape (512,), dtype float32, L2-normalized

#     Raises:
#         ValueError if image is invalid or embedding is degenerate
#     """
#     if image is None or image.size == 0:
#         raise ValueError("Invalid input image")

#     model = _load_model()
#     face = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
#     embedding = model.get_feat(face).flatten().astype(np.float32)

#     norm = np.linalg.norm(embedding)
#     if not np.isfinite(norm) or norm == 0:
#         raise ValueError("Invalid embedding (zero or non-finite norm)")

#     return embedding / norm


# def cosine_similarity(emb1, emb2):
#     """Cosine similarity between two L2-normalized embeddings."""
#     return float(np.clip(np.dot(emb1, emb2), -1, 1))


# # ---- Gallery ----

# def build_gallery(gallery_dir, output_path):
#     """
#     Build a reference gallery from a folder of person sub-folders.

#     Structure:
#         gallery_dir/
#             person_A/
#                 img1.jpg
#                 img2.jpg
#             person_B/
#                 img1.jpg

#     Saves reference_embeddings.npz with 'embeddings' and 'labels' arrays.
#     """
#     gallery_dir = Path(gallery_dir)
#     output_path = Path(output_path)

#     embeddings = []
#     labels = []

#     for person_folder in sorted(gallery_dir.iterdir()):
#         if not person_folder.is_dir():
#             continue
#         for image_path in sorted(person_folder.iterdir()):
#             if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
#                 continue
#             image = cv2.imread(str(image_path))
#             if image is None:
#                 print(f"  Cannot read: {image_path}")
#                 continue
#             try:
#                 emb = generate_embedding(image)
#                 embeddings.append(emb)
#                 labels.append(person_folder.name)
#                 print(f"  Gallery: {person_folder.name}/{image_path.name}  ✓")
#             except ValueError as e:
#                 print(f"  Gallery: {person_folder.name}/{image_path.name}  SKIP ({e})")

#     if embeddings:
#         np.savez_compressed(output_path,
#                             embeddings=np.array(embeddings),
#                             labels=np.array(labels))
#         print(f"\nGallery saved: {len(embeddings)} images, "
#               f"{len(set(labels))} identities -> {output_path}")
#     else:
#         print("No valid embeddings generated for the gallery.")


# def load_gallery(gallery_path):
#     """Load a saved gallery .npz file."""
#     gallery_path = Path(gallery_path)
#     if not gallery_path.exists():
#         raise FileNotFoundError(
#             f"Gallery not found: {gallery_path}\n"
#             "Run:  python run_pipeline.py --build-gallery  first."
#         )
#     with np.load(gallery_path, allow_pickle=False) as data:
#         return data["embeddings"], data["labels"]


# def rank_faces(image, gallery_embeddings, gallery_labels,
#                top_k=5, threshold=0.5):
#     """
#     Match a face crop against the reference gallery.

#     Args:
#         image:              BGR numpy array (enhanced or original face crop)
#         gallery_embeddings: (N, 512) array of reference embeddings
#         gallery_labels:     (N,) array of identity labels
#         top_k:              how many top matches to return
#         threshold:          minimum similarity for a positive match

#     Returns:
#         dict with: predicted_identity, similarity, status, threshold, top_matches
#     """
#     query_emb = generate_embedding(image)

#     # Cosine similarity (embeddings are already L2-normalized)
#     similarities = np.clip(gallery_embeddings @ query_emb, -1, 1)

#     # Best score per unique identity
#     identity_scores = {}
#     for label, score in zip(gallery_labels, similarities):
#         label = str(label)
#         score = float(score)
#         if label not in identity_scores:
#             identity_scores[label] = score
#         else:
#             identity_scores[label] = max(identity_scores[label], score)

#     ranked = sorted(identity_scores.items(), key=lambda x: x[1], reverse=True)
#     if not ranked:
#         raise ValueError("Reference gallery is empty")

#     top_matches = []
#     for rank, (identity, score) in enumerate(ranked[:top_k], start=1):
#         top_matches.append({
#             "rank": rank,
#             "identity": identity,
#             "similarity": round(score, 4),
#         })

#     best_identity, best_score = ranked[0]
#     matched = best_score >= threshold

#     return {
#         "predicted_identity": best_identity if matched else "Unknown",
#         "similarity": round(best_score, 4),
#         "status": "MATCH" if matched else "NO MATCH",
#         "threshold": threshold,
#         "top_matches": top_matches,
#     }
"""
STAGE 4 — ArcFace: generate embeddings and match faces against a reference gallery.

Uses InsightFace's buffalo_l model for recognition (512-dim embeddings, L2-normalized).
Since input images are already cropped faces, detection is skipped — the recognition
model is used directly after resizing to 112×112.

Reference gallery:
  - Build once with  build_gallery()
  - Stored as  reference_embeddings.npz  in the models/ folder
  - Structure: Reference_Gallery/<person_name>/img1.jpg, img2.jpg, ...

Matching:
  - rank_faces()  returns top-K matches with cosine similarities
"""

import cv2
import numpy as np
from pathlib import Path

_recognizer = None


def _load_model():
    """Lazy-load the ArcFace recognition model (singleton)."""
    global _recognizer
    if _recognizer is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _recognizer = app.models["recognition"]
    return _recognizer


def generate_embedding(image):
    """
    Generate a 512-dim L2-normalized ArcFace embedding from a face crop (BGR array).

    Args:
        image: BGR numpy array (already cropped face)

    Returns:
        np.ndarray of shape (512,), dtype float32, L2-normalized

    Raises:
        ValueError if image is invalid or embedding is degenerate
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid input image")

    model = _load_model()
    face = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC)
    embedding = model.get_feat(face).flatten().astype(np.float32)

    norm = np.linalg.norm(embedding)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Invalid embedding (zero or non-finite norm)")

    return embedding / norm


def cosine_similarity(emb1, emb2):
    """Cosine similarity between two L2-normalized embeddings."""
    return float(np.clip(np.dot(emb1, emb2), -1, 1))


# ---- Gallery ----

def build_gallery(gallery_dir, output_path):
    """
    Build a reference gallery from a folder of person sub-folders.

    Structure:
        gallery_dir/
            person_A/
                img1.jpg
                img2.jpg
            person_B/
                img1.jpg

    Saves reference_embeddings.npz with 'embeddings' and 'labels' arrays.
    """
    gallery_dir = Path(gallery_dir)
    output_path = Path(output_path)

    embeddings = []
    labels = []

    for person_folder in sorted(gallery_dir.iterdir()):
        if not person_folder.is_dir():
            continue
        for image_path in sorted(person_folder.iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  Cannot read: {image_path}")
                continue
            try:
                emb = generate_embedding(image)
                embeddings.append(emb)
                labels.append(person_folder.name)
                print(f"  Gallery: {person_folder.name}/{image_path.name}  ✓")
            except ValueError as e:
                print(f"  Gallery: {person_folder.name}/{image_path.name}  SKIP ({e})")

    if embeddings:
        np.savez_compressed(output_path,
                            embeddings=np.array(embeddings),
                            labels=np.array(labels))
        print(f"\nGallery saved: {len(embeddings)} images, "
              f"{len(set(labels))} identities -> {output_path}")
    else:
        print("No valid embeddings generated for the gallery.")


def load_gallery(gallery_path):
    """Load a saved gallery .npz file."""
    gallery_path = Path(gallery_path)
    if not gallery_path.exists():
        raise FileNotFoundError(
            f"Gallery not found: {gallery_path}\n"
            "Run:  python run_pipeline.py --build-gallery  first."
        )
    with np.load(gallery_path, allow_pickle=False) as data:
        return data["embeddings"], data["labels"]


def rank_faces(image, gallery_embeddings, gallery_labels,
               top_k=5, threshold=0.5):
    """
    Match a face crop against the reference gallery.

    Args:
        image:              BGR numpy array (enhanced or original face crop)
        gallery_embeddings: (N, 512) array of reference embeddings
        gallery_labels:     (N,) array of identity labels
        top_k:              how many top matches to return
        threshold:          minimum similarity for a positive match

    Returns:
        dict with: predicted_identity, similarity, status, threshold, top_matches
    """
    query_emb = generate_embedding(image)

    # Cosine similarity (embeddings are already L2-normalized)
    similarities = np.clip(gallery_embeddings @ query_emb, -1, 1)

    # Best score per unique identity
    identity_scores = {}
    for label, score in zip(gallery_labels, similarities):
        label = str(label)
        score = float(score)
        if label not in identity_scores:
            identity_scores[label] = score
        else:
            identity_scores[label] = max(identity_scores[label], score)

    ranked = sorted(identity_scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        raise ValueError("Reference gallery is empty")

    top_matches = []
    for rank, (identity, score) in enumerate(ranked[:top_k], start=1):
        top_matches.append({
            "rank": rank,
            "identity": identity,
            "similarity": round(score, 4),
        })

    best_identity, best_score = ranked[0]
    matched = best_score >= threshold

    return {
        "predicted_identity": best_identity if matched else "Unknown",
        "similarity": round(best_score, 4),
        "status": "MATCH" if matched else "NO MATCH",
        "threshold": threshold,
        "top_matches": top_matches,
    }