"""
SQLite Database for Smart DentalOps
Full schema: users, appointments, waitlist, notifications, doctors, slots
"""

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Float, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./dentalops.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if "sqlite" in DATABASE_URL:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="patient")
    phone = Column(String, default="")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)


class DoctorDB(Base):
    __tablename__ = "doctors"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    specialty = Column(String, default="General Dentistry")
    email = Column(String, default="")
    phone = Column(String, default="")
    available_days = Column(String, default="Mon,Tue,Wed,Thu,Fri")
    start_time = Column(String, default="09:00")
    end_time = Column(String, default="17:00")
    slot_duration_min = Column(Integer, default=30)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.now)


class AppointmentDB(Base):
    __tablename__ = "appointments"
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, nullable=False)
    patient_email = Column(String, default="")
    patient_phone = Column(String, default="")
    patient_id = Column(String)
    age = Column(Integer, default=30)
    procedure_type = Column(String, default="Cleaning")
    risk_score = Column(Float, default=0.0)
    risk_category = Column(String, default="Low")   # Low / Medium / High
    slot_type = Column(String, default="Standard Slot")
    slot_time = Column(String, default="10:00 AM")
    appointment_date = Column(String, default="")
    status = Column(String, default="Scheduled")    # Scheduled/Confirmed/In Progress/Completed/Cancelled/No-Show
    confirmation_status = Column(String, default="Pending")  # Pending/Confirmed/Declined
    dentist = Column(String, default="Dr. Smith")
    chair = Column(String, default="Chair 1")
    duration = Column(String, default="30 min")
    notes = Column(Text, default="")
    reminder_sent = Column(Boolean, default=False)
    reminder_24h_sent = Column(Boolean, default=False)
    reminder_2h_sent = Column(Boolean, default=False)
    deposit_required = Column(Boolean, default=False)
    deposit_paid = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)


class WaitlistDB(Base):
    __tablename__ = "waitlist"
    id = Column(Integer, primary_key=True, index=True)
    patient_name = Column(String, nullable=False)
    patient_email = Column(String, default="")
    patient_phone = Column(String, default="")
    procedure_type = Column(String, default="Cleaning")
    preferred_dentist = Column(String, default="Any")
    preferred_time = Column(String, default="Any")   # morning/afternoon/evening/any
    preferred_date_from = Column(String, default="")
    preferred_date_to = Column(String, default="")
    risk_score = Column(Float, default=0.0)
    status = Column(String, default="Waiting")       # Waiting/Notified/Booked/Expired
    notified_at = Column(DateTime, nullable=True)
    booked_appointment_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.now)


class NotificationLogDB(Base):
    __tablename__ = "notification_logs"
    id = Column(Integer, primary_key=True, index=True)
    appointment_id = Column(Integer, nullable=True)
    patient_name = Column(String, default="")
    patient_contact = Column(String, default="")
    channel = Column(String, default="SMS")          # SMS/Email/WhatsApp
    notification_type = Column(String, default="")   # Confirmation/Reminder24h/Reminder2h/DoubleConfirm/WaitlistAlert
    message = Column(Text, default="")
    status = Column(String, default="Sent")          # Sent/Failed/Delivered/Read
    sent_at = Column(DateTime, default=datetime.now)


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


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
