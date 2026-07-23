"""InsightFace wrapper.

The model is loaded ONCE at startup (slow, ~2-5s) and then every request
is fast. Embeddings are 512-dim, L2-normalised, so cosine similarity is
just a dot product.
"""
import numpy as np
import cv2
from insightface.app import FaceAnalysis

_app: FaceAnalysis | None = None


def load_model():
    """Call once on server startup."""
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        # det_size 960 = better detection of small faces in a group photo
        _app.prepare(ctx_id=-1, det_size=(1280, 1280))
    return _app


def _decode(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    # Cap very large images to keep detection fast
    h, w = img.shape[:2]
    max_side = 2560
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
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


def extract_single_face_with_thumb(image_bytes: bytes, thumb_size: int = 144):
    """For enrollment: biggest face -> (embedding, small JPEG thumbnail bytes)."""
    app = load_model()
    img = _decode(image_bytes)
    faces = app.get(img)
    if not faces:
        raise ValueError("No face found in photo")
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
               reverse=True)
    f = faces[0]
    emb = np.asarray(f.normed_embedding, dtype=np.float32)
    x1, y1, x2, y2 = [int(v) for v in f.bbox]
    m = int(0.30 * max(x2 - x1, y2 - y1))          # thodi margin, achha dikhta hai
    x1 = max(0, x1 - m); y1 = max(0, y1 - m)
    x2 = min(img.shape[1], x2 + m); y2 = min(img.shape[0], y2 + m)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        crop = img
    crop = cv2.resize(crop, (thumb_size, thumb_size))
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return emb, (buf.tobytes() if ok else b"")


def extract_single_face(image_bytes: bytes) -> np.ndarray:
    """For enrollment: expect exactly one clear face, take the biggest."""
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
    """Greedy matching: each detected face is compared with every student's
    embeddings; best score per student wins. Faces below threshold
    (outsiders) are ignored.

    Returns {student_id: best_similarity} for matched students.
    """
    matched: dict[int, float] = {}
    for emb in photo_embeddings:
        best_id, best_score = None, threshold
        for sid, emb_list in known.items():
            for known_emb in emb_list:
                s = cosine(emb, known_emb)
                if s > best_score:
                    best_id, best_score = sid, s
        if best_id is not None:
            # keep the highest score if same student appears in 2 photos
            if best_id not in matched or best_score > matched[best_id]:
                matched[best_id] = best_score
    return matched
