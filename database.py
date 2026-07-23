"""Database engine, session and ORM models.

Works with BOTH:
  - SQLite  (zero setup — local dev)
  - PostgreSQL (Neon — production)

Face embeddings are stored as JSON lists. Matching happens in memory
(see main.py EMB_CACHE), so no vector extension is needed.

v2: subject/lecture-wise attendance. Existing databases are migrated
automatically on startup (old records get subject='General').
"""
import os
from datetime import date, datetime

from dotenv import load_dotenv
from sqlalchemy import (JSON, Column, Date, DateTime, ForeignKey, Integer,
                        String, Text, UniqueConstraint, create_engine, text)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///attendance.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class User(Base):
    """App login users (monitor / teacher). Not students."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    role = Column(String(20), default="monitor")
    created_at = Column(DateTime, default=datetime.utcnow)


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True)
    roll_no = Column(String(20), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    photo = Column(Text)          # small base64 JPEG face thumbnail (optional)
    created_at = Column(DateTime, default=datetime.utcnow)
    embeddings = relationship("FaceEmbedding", back_populates="student",
                              cascade="all, delete-orphan")


class FaceEmbedding(Base):
    """One student can have multiple embeddings (2-3 photos = better accuracy).
    Stored as a JSON list of 512 floats."""
    __tablename__ = "face_embeddings"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    student = relationship("Student", back_populates="embeddings")


class Subject(Base):
    """Lectures/subjects for the class (Maths, Physics, ...)."""
    __tablename__ = "subjects"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    day = Column(Date, default=date.today, nullable=False)
    subject = Column(String(50), nullable=False, default="General")
    status = Column(String(10), nullable=False)  # present / absent
    confidence = Column(Integer, default=0)      # match % for present students
    marked_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("student_id", "day", "subject",
                                       name="one_record_per_day_subject"),)


def _try(sql: str):
    """Run a migration statement; ignore errors (already applied / not needed)."""
    try:
        with engine.connect() as conn:
            conn.execute(text(sql))
            conn.commit()
    except Exception:
        pass


def init_db():
    Base.metadata.create_all(engine)
    # Migrations for databases created before subjects existed:
    _try("ALTER TABLE attendance_records ADD COLUMN subject VARCHAR(50) NOT NULL DEFAULT 'General'")
    _try("ALTER TABLE attendance_records DROP CONSTRAINT one_record_per_day")
    _try("CREATE UNIQUE INDEX IF NOT EXISTS one_record_per_day_subject "
         "ON attendance_records (student_id, day, subject)")
    _try("ALTER TABLE students ADD COLUMN photo TEXT")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
