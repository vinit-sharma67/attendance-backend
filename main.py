"""Face Attendance API — FastAPI backend (v2: subject/lecture-wise).

Flow:
  1. POST /api/students          -> enroll student (name + roll no + 1-3 photos)
  2. GET/POST /api/subjects      -> manage lecture subjects (Maths, Physics, ...)
  3. POST /api/attendance/scan   -> upload class photo(s), get present/absent preview
  4. POST /api/attendance/confirm-> save the result for a given subject
Photos are processed in memory and never stored — only embeddings are kept.
"""
import base64
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
from database import (AttendanceRecord, FaceEmbedding, Student, Subject, User,
                      get_db, init_db, SessionLocal)

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


def _students_payload(db: Session) -> list[dict]:
    """Full enrolled list for the Sheet's 'Enrolled Students' tab."""
    out = []
    for s in db.query(Student).order_by(Student.roll_no).all():
        out.append({"roll_no": s.roll_no, "name": s.name,
                    "photos": len(s.embeddings),
                    "enrolled_on": s.created_at.strftime("%d %b %Y") if s.created_at else ""})
    return out


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
                    "photos": len(s.embeddings), "photo": s.photo or ""})
    return out


@app.post("/api/students")
async def add_student(background: BackgroundTasks,
                      name: str = Form(...), roll_no: str = Form(...),
                      photos: list[UploadFile] = File(...),
                      db: Session = Depends(get_db),
                      _: User = Depends(get_current_user)):
    if db.query(Student).filter(Student.roll_no == roll_no).first():
        raise HTTPException(400, f"Roll no {roll_no} already exists")
    if not 1 <= len(photos) <= 3:
        raise HTTPException(400, "Upload 1 to 3 photos")

    embeddings = []
    thumb_b64 = None
    for i, p in enumerate(photos):
        data = await p.read()
        if len(data) > 8 * 1024 * 1024:
            raise HTTPException(400, "Photo too large (max 8 MB)")
        try:
            if i == 0:
                emb, thumb = face_engine.extract_single_face_with_thumb(data)
                embeddings.append(emb)
                if thumb:
                    thumb_b64 = base64.b64encode(thumb).decode()
            else:
                embeddings.append(face_engine.extract_single_face(data))
        except ValueError:
            raise HTTPException(400, f"No clear face found in '{p.filename}'. Use a front-facing photo.")

    student = Student(name=name.strip(), roll_no=roll_no.strip(), photo=thumb_b64)
    db.add(student)
    db.flush()
    for emb in embeddings:
        db.add(FaceEmbedding(student_id=student.id, embedding=emb.tolist()))
    db.commit()
    refresh_cache(db)
    background.add_task(sheets_sync.sync_students, _students_payload(db))
    return {"id": student.id, "name": student.name, "roll_no": student.roll_no,
            "photos": len(embeddings)}


@app.delete("/api/students/{student_id}")
def delete_student(student_id: int, background: BackgroundTasks,
                   db: Session = Depends(get_db),
                   _: User = Depends(get_current_user)):
    s = db.get(Student, student_id)
    if not s:
        raise HTTPException(404, "Student not found")
    db.delete(s)
    db.commit()
    refresh_cache(db)
    background.add_task(sheets_sync.sync_students, _students_payload(db))
    return {"ok": True}


# ---------- Subjects ----------

class SubjectIn(BaseModel):
    name: str


@app.get("/api/subjects")
def list_subjects(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return [{"id": s.id, "name": s.name}
            for s in db.query(Subject).order_by(Subject.name).all()]


@app.post("/api/subjects")
def add_subject(body: SubjectIn, db: Session = Depends(get_db),
                _: User = Depends(get_current_user)):
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "Subject name required")
    if len(name) > 50:
        raise HTTPException(400, "Subject name too long (max 50)")
    if db.query(Subject).filter(Subject.name == name).first():
        raise HTTPException(400, f"Subject '{name}' already exists")
    s = Subject(name=name)
    db.add(s)
    db.commit()
    return {"id": s.id, "name": s.name}


@app.delete("/api/subjects/{subject_id}")
def delete_subject(subject_id: int, db: Session = Depends(get_db),
                   _: User = Depends(get_current_user)):
    s = db.get(Subject, subject_id)
    if not s:
        raise HTTPException(404, "Subject not found")
    db.delete(s)
    db.commit()
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
    subject: str = "General"
    replace: bool = False              # False = MERGE (default), True = overwrite day


@app.post("/api/attendance/confirm")
def confirm(body: ConfirmIn, background: BackgroundTasks,
            db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """Save today's attendance for one subject.

    MERGE mode (default): new presents are added/updated, students already
    marked present earlier today STAY present, everyone still unseen is absent.
    So multiple saves in one lecture accumulate — nothing is lost.

    REPLACE mode (replace=true): today's records for this subject are wiped
    and rewritten exactly as sent (use for corrections).
    Also syncs the FINAL merged state to Google Sheets in the background.
    """
    subject = (body.subject or "General").strip() or "General"
    today = date.today()

    existing = {r.student_id: r for r in
                db.query(AttendanceRecord)
                  .filter(AttendanceRecord.day == today,
                          AttendanceRecord.subject == subject).all()}

    if body.replace:
        for r in existing.values():
            db.delete(r)
        db.flush()
        existing = {}

    for s in db.query(Student).all():
        conf = body.confidences.get(s.id, 0)
        if s.id in body.present_ids:
            rec = existing.get(s.id)
            if rec:                          # was absent (or present) → present
                rec.status = "present"
                if conf > (rec.confidence or 0):
                    rec.confidence = conf
            else:
                db.add(AttendanceRecord(student_id=s.id, day=today,
                                        subject=subject, status="present",
                                        confidence=conf))
        else:
            if s.id not in existing:         # never seen today → absent
                db.add(AttendanceRecord(student_id=s.id, day=today,
                                        subject=subject, status="absent",
                                        confidence=0))
            # else: keep existing status (earlier present stays present)
    db.commit()

    # Build FINAL state (after merge) for the Sheet
    final = (db.query(AttendanceRecord, Student).join(Student)
             .filter(AttendanceRecord.day == today,
                     AttendanceRecord.subject == subject).all())
    sheet_rows = sorted(
        [{"roll_no": s.roll_no, "name": s.name, "status": r.status,
          "confidence": r.confidence} for r, s in final],
        key=lambda x: x["roll_no"])
    present_count = sum(1 for x in sheet_rows if x["status"] == "present")

    # Sheets sync runs AFTER the response is sent — app stays fast
    background.add_task(sheets_sync.sync_attendance, today, subject, sheet_rows)
    return {"ok": True, "date": str(today), "subject": subject,
            "mode": "replace" if body.replace else "merge",
            "records": len(sheet_rows), "present_total": present_count}


@app.post("/api/attendance/resync")
def resync_sheet(background: BackgroundTasks, subject: str | None = None,
                 db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    """Rebuild Google Sheet tab(s) from the database — full history.
    Use when a tab was deleted or looks wrong. Optional ?subject=ML for one tab."""
    q = db.query(AttendanceRecord, Student).join(Student)
    if subject:
        q = q.filter(AttendanceRecord.subject == subject)

    by_subject: dict[str, dict[str, dict]] = {}
    for r, s in q.all():
        stus = by_subject.setdefault(r.subject, {})
        stu = stus.setdefault(s.roll_no, {"name": s.name, "marks": {}})
        stu["marks"][r.day.isoformat()] = "P" if r.status == "present" else "A"

    if not by_subject:
        raise HTTPException(404, "No attendance records found to sync")

    for subj, stus in by_subject.items():
        dates = sorted({d for stu in stus.values() for d in stu["marks"]})
        rows = [{"roll_no": rn, "name": v["name"],
                 "marks": [v["marks"].get(d, "") for d in dates]}
                for rn, v in sorted(stus.items())]
        background.add_task(sheets_sync.sync_rebuild, subj, dates, rows)

    return {"ok": True, "subjects": sorted(by_subject.keys())}


@app.get("/api/attendance/history")
def history(day: str | None = None, subject: str | None = None,
            db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    q = db.query(AttendanceRecord, Student).join(Student)
    target = date.fromisoformat(day) if day else date.today()
    q = q.filter(AttendanceRecord.day == target)
    if subject:
        q = q.filter(AttendanceRecord.subject == subject)
    rows = [{"roll_no": s.roll_no, "name": s.name, "status": r.status,
             "confidence": r.confidence, "subject": r.subject}
            for r, s in q.all()]
    rows.sort(key=lambda x: (x["subject"], x["roll_no"]))
    return {"date": str(target), "subject": subject, "records": rows,
            "present": sum(1 for r in rows if r["status"] == "present"),
            "absent": sum(1 for r in rows if r["status"] == "absent")}
