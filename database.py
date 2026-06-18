import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

# Create data directory before engine initializes
os.makedirs("data", exist_ok=True)

engine = create_engine("sqlite:///data/support.db", echo=False)
SessionLocal = sessionmaker(bind=engine)

class Ticket(Base):
    __tablename__ = "tickets"

    id              = Column(Integer, primary_key=True, index=True)
    ticket_id       = Column(String, unique=True, index=True)  # TKT-001
    customer_id     = Column(String, index=True)
    customer_name   = Column(String, nullable=True)
    customer_email  = Column(String, nullable=True)
    issue_summary   = Column(Text)
    full_conversation = Column(Text)  # full chat history
    category        = Column(String)  # billing, technical, general, complaint
    priority        = Column(String, default="medium")  # low, medium, high, urgent
    status          = Column(String, default="open")  # open, in_progress, resolved, closed
    resolution      = Column(Text, nullable=True)
    agent_notes     = Column(Text, nullable=True)
    escalated       = Column(Integer, default=0)  # 0=no, 1=yes
    confidence_score = Column(Float, nullable=True)  # agent confidence when resolving
    created_at      = Column(DateTime, default=datetime.utcnow)
    updated_at      = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at     = Column(DateTime, nullable=True)

class CustomerMemory(Base):
    __tablename__ = "customer_memory"

    id              = Column(Integer, primary_key=True, index=True)
    customer_id     = Column(String, unique=True, index=True)
    customer_name   = Column(String, nullable=True)
    customer_email  = Column(String, nullable=True)
    total_tickets   = Column(Integer, default=0)
    resolved_tickets = Column(Integer, default=0)
    last_contact    = Column(DateTime, nullable=True)
    preferences     = Column(Text, nullable=True)  # JSON string of preferences
    issues_history  = Column(Text, nullable=True)  # summary of past issues
    sentiment       = Column(String, default="neutral")  # positive, neutral, negative, frustrated
    notes           = Column(Text, nullable=True)  # agent notes about this customer

class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id              = Column(Integer, primary_key=True, index=True)
    ticket_id       = Column(String, index=True)
    customer_id     = Column(String, index=True)
    role            = Column(String)  # customer, agent, system
    message         = Column(Text)
    timestamp       = Column(DateTime, default=datetime.utcnow)
    node            = Column(String, nullable=True)  # which LangGraph node generated this

def init_db():
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
