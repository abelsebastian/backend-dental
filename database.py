"""
SQLite Database for Smart DentalOps
Handles persistent storage for users, appointments, and analytics
"""

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# Use DATABASE_URL env var (Railway PostgreSQL) or fall back to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dentalops.db")

# Fix for Railway PostgreSQL URL format (postgres:// -> postgresql://)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Create engine
if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class UserDB(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="patient")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)


class AppointmentDB(Base):
    __tablename__ = "appointments"

    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, nullable=False)
    patient_id = Column(String)
    age = Column(Integer)
    procedure_type = Column(String)
    risk_score = Column(Float, default=0.0)
    slot_type = Column(String)
    slot_time = Column(String)
    status = Column(String, default="Scheduled")
    dentist = Column(String)
    chair = Column(String)
    duration = Column(String)
    notes = Column(Text, default="")
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class SentimentLogDB(Base):
    __tablename__ = "sentiment_logs"

    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, default="Unknown")
    message = Column(Text)
    sentiment = Column(String)
    polarity = Column(Float)
    subjectivity = Column(Float)
    intent = Column(String)
    confidence = Column(String)
    risk_before = Column(Float)
    risk_after = Column(Float)
    created_at = Column(DateTime, default=datetime.now)


# ── Create all tables ─────────────────────────────────────────────────────────
def init_db():
    Base.metadata.create_all(bind=engine)


# ── Dependency for FastAPI ────────────────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
