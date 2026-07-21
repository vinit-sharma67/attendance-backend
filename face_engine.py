"""InsightFace wrapper — cloud/low-memory friendly.

Defaults are tuned for a small free server (Render 512 MB):
  - FACE_PACK=buffalo_s  (light detection + recognition models, bundled in ./models)
  - only detection + recognition modules are loaded (no landmarks/genderage)
  - DET_SIZE=640, MAX_SIDE=1600 keep RAM spikes small

For a bigger server you can set env: FACE_PACK=buffalo_l, DET_SIZE=1280, MAX_SIDE=2560.
The model is loaded ONCE at startup; embeddings are L2-normalised so cosine
similarity is a dot product.
"""
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

PACK = os.getenv("FACE_PACK", "buffalo_s")
DET_SIZE = int(os.getenv("DET_SIZE", "640"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "1600"))
# "." means: look for ./models/<PACK>/*.onnx (bundled in the repo).
# If not found there, insightface falls back to downloading into this root.
ROOT = os.getenv("INSIGHTFACE_ROOT", ".")

_app: FaceAnalysis | None = None


def load_model():
    """Call once on server startup."""
    global _app
    if _app is None:
        _app = FaceAnalysis(name=PACK, root=ROOT,
                            allowed_modules=["detection", "recognition"],
                            providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=-1, det_size=(DET_SIZE, DET_SIZE))
    return _app


def _decode(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        scale = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def extract_faces(image_bytes: bytes) -> list[np.ndarray]:
    """Return a list of normalised embeddings, one per detected face."""
    app = load_model()
    img = _decode(image_bytes)
    faces = app.get(img)
    out = []
    for f in faces:
        emb = f.normed_embedding  # already L2-normalised
        if emb is not None:
            out.append(np.asarray(emb, dtype=np.float32))
    return out


def extract_single_face(image_bytes: bytes) -> np.ndarray:
    """For enrollment: expect one clear face, take the biggest."""
    app = load_model()
    img = _decode(image_bytes)
    faces = app.get(img)
    if not faces:
        raise ValueError("No face found in photo")
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    return np.asarray(faces[0].normed_embedding, dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both normalised


def match_faces(photo_embeddings: list[np.ndarray],
                known: dict[int, list[np.ndarray]],
                threshold: float) -> dict[int, float]:
    """Each detected face is compared with every student's embeddings; best
    score per student wins. Faces below threshold (outsiders) are ignored.
    Returns {student_id: best_similarity} for matched students."""
    matched: dict[int, float] = {}
    for emb in photo_embeddings:
        best_id, best_score = None, threshold
        for sid, emb_list in known.items():
            for known_emb in emb_list:
                s = cosine(emb, known_emb)
                if s > best_score:
                    best_id, best_score = sid, s
        if best_id is not None:
            if best_id not in matched or best_score > matched[best_id]:
                matched[best_id] = best_score
    return matched
