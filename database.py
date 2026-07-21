"""Database engine, session and ORM models.

Works with BOTH:
  - SQLite  (zero setup — default, perfect for daily local use)
  - PostgreSQL (for cloud deployment later)

Face embeddings are stored as JSON lists. Matching happens in memory
(see main.py EMB_CACHE), so no vector extension is needed.
"""
import os
from datetime import date, datetime

from dotenv import load_dotenv
from sqlalchemy import (JSON, Column, Date, DateTime, ForeignKey, Integer,
                        String, UniqueConstraint, create_engine)
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


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    id = Column(Integer, primary_key=True)
    student_id = Column(Integer, ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    day = Column(Date, default=date.today, nullable=False)
    status = Column(String(10), nullable=False)  # present / absent
    confidence = Column(Integer, default=0)      # match % for present students
    marked_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("student_id", "day", name="one_record_per_day"),)


def init_db():
    Base.metadata.create_all(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
