"""Face Attendance API — FastAPI backend.

Flow:
  1. POST /api/students          -> enroll student (name + roll no + 1-3 photos)
  2. POST /api/attendance/scan   -> upload 1-2 class photos, get present/absent preview
  3. POST /api/attendance/confirm-> save the (possibly edited) result to DB
Photos are processed in memory and never stored — only embeddings are kept.
"""
import os
from datetime import date

import numpy as np
from dotenv import load_dotenv
from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form,
                     HTTPException, Request, UploadFile)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy.orm import Session

import face_engine
import sheets_sync
from auth import (create_token, get_current_user, hash_password, verify_password)
from database import (AttendanceRecord, FaceEmbedding, Student, User, get_db,
                      init_db, SessionLocal)

load_dotenv()
THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.50"))

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Face Attendance API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory cache of embeddings: {student_id: [np.array, ...]}
EMB_CACHE: dict[int, list[np.ndarray]] = {}


def refresh_cache(db: Session):
    EMB_CACHE.clear()
    for e in db.query(FaceEmbedding).all():
        EMB_CACHE.setdefault(e.student_id, []).append(
            np.asarray(e.embedding, dtype=np.float32))


@app.on_event("startup")
def startup():
    init_db()
    face_engine.load_model()          # load once — requests stay fast
    db = SessionLocal()
    try:
        # create first admin user from .env if missing
        uname = os.getenv("ADMIN_USERNAME", "monitor")
        if not db.query(User).filter(User.username == uname).first():
            db.add(User(username=uname,
                        password_hash=hash_password(os.getenv("ADMIN_PASSWORD", "admin123"))))
            db.commit()
        refresh_cache(db)
    finally:
        db.close()


# ---------- Auth ----------

class LoginIn(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
@limiter.limit("10/minute")
def login(request: Request, body: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == body.username).first()
    if not user or not verify_password(body.password, user.password_hash):
        raise HTTPException(401, "Wrong username or password")
    return {"access_token": create_token(user.username), "token_type": "bearer"}


# ---------- Students (enrollment) ----------

@app.get("/api/students")
def list_students(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    out = []
    for s in db.query(Student).order_by(Student.roll_no).all():
        out.append({"id": s.id, "roll_no": s.roll_no, "name": s.name,
                    "photos": len(s.embeddings)})
    return out


@app.post("/api/students")
async def add_student(name: str = Form(...), roll_no: str = Form(...),
                      photos: list[UploadFile] = File(...),
                      db: Session = Depends(get_db),
                      _: User = Depends(get_current_user)):
    if db.query(Student).filter(Student.roll_no == roll_no).first():
        raise HTTPException(400, f"Roll no {roll_no} already exists")
    if not 1 <= len(photos) <= 3:
        raise HTTPException(400, "Upload 1 to 3 photos")

    embeddings = []
    for p in photos:
        data = await p.read()
        if len(data) > 8 * 1024 * 1024:
            raise HTTPException(400, "Photo too large (max 8 MB)")
        try:
            embeddings.append(face_engine.extract_single_face(data))
        except ValueError:
            raise HTTPException(400, f"No clear face found in '{p.filename}'. Use a front-facing photo.")

    student = Student(name=name.strip(), roll_no=roll_no.strip())
    db.add(student)
    db.flush()
    for emb in embeddings:
        db.add(FaceEmbedding(student_id=student.id, embedding=emb.tolist()))
    db.commit()
    refresh_cache(db)
    return {"id": student.id, "name": student.name, "roll_no": student.roll_no,
            "photos": len(embeddings)}


@app.delete("/api/students/{student_id}")
def delete_student(student_id: int, db: Session = Depends(get_db),
                   _: User = Depends(get_current_user)):
    s = db.get(Student, student_id)
    if not s:
        raise HTTPException(404, "Student not found")
    db.delete(s)
    db.commit()
    refresh_cache(db)
    return {"ok": True}


# ---------- Attendance ----------

@app.post("/api/attendance/scan")
@limiter.limit("20/minute")
async def scan(request: Request, photos: list[UploadFile] = File(...),
               db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """Detect faces in class photo(s) and return a PREVIEW (nothing saved yet)."""
    if not EMB_CACHE:
        raise HTTPException(400, "No students enrolled yet")
    if not 1 <= len(photos) <= 4:
        raise HTTPException(400, "Upload 1 to 4 photos")

    all_embs: list[np.ndarray] = []
    for p in photos:
        data = await p.read()
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(400, "Photo too large (max 10 MB)")
        all_embs.extend(face_engine.extract_faces(data))

    matched = face_engine.match_faces(all_embs, EMB_CACHE, THRESHOLD)

    students = db.query(Student).order_by(Student.roll_no).all()
    present, absent = [], []
    for s in students:
        if s.id in matched:
            present.append({"id": s.id, "roll_no": s.roll_no, "name": s.name,
                            "confidence": round(matched[s.id] * 100)})
        else:
            absent.append({"id": s.id, "roll_no": s.roll_no, "name": s.name})

    return {"faces_detected": len(all_embs),
            "unknown_faces_ignored": len(all_embs) - len(matched),
            "present": present, "absent": absent}


class ConfirmIn(BaseModel):
    present_ids: list[int]
    confidences: dict[int, int] = {}   # optional {student_id: percent}


@app.post("/api/attendance/confirm")
def confirm(body: ConfirmIn, background: BackgroundTasks,
            db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """Save today's attendance. Re-running replaces today's records.
    Also syncs to Google Sheets in the background (if configured)."""
    today = date.today()
    db.query(AttendanceRecord).filter(AttendanceRecord.day == today).delete()
    saved = 0
    sheet_rows = []
    for s in db.query(Student).order_by(Student.roll_no).all():
        status = "present" if s.id in body.present_ids else "absent"
        db.add(AttendanceRecord(student_id=s.id, day=today, status=status,
                                confidence=body.confidences.get(s.id, 0)))
        sheet_rows.append({"roll_no": s.roll_no, "name": s.name,
                           "status": status,
                           "confidence": body.confidences.get(s.id, 0)})
        saved += 1
    db.commit()
    # Sheets sync runs AFTER the response is sent — app stays fast
    background.add_task(sheets_sync.sync_attendance, today, sheet_rows)
    return {"ok": True, "date": str(today), "records": saved}


@app.get("/api/attendance/history")
def history(day: str | None = None, db: Session = Depends(get_db),
            _: User = Depends(get_current_user)):
    q = db.query(AttendanceRecord, Student).join(Student)
    target = date.fromisoformat(day) if day else date.today()
    q = q.filter(AttendanceRecord.day == target)
    rows = [{"roll_no": s.roll_no, "name": s.name, "status": r.status,
             "confidence": r.confidence} for r, s in q.all()]
    rows.sort(key=lambda x: x["roll_no"])
    return {"date": str(target), "records": rows,
            "present": sum(1 for r in rows if r["status"] == "present"),
            "absent": sum(1 for r in rows if r["status"] == "absent")}
