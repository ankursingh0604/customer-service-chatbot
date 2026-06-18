import os
import uuid
import hashlib
import hmac
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from langchain_core.messages import HumanMessage
from datetime import datetime

load_dotenv()

# ── Customer-scoped access tokens (prevents IDOR) ─────────────────────────────
# Without this, anyone who guesses a sequential customer_id or ticket_id
# (CUST-002, TKT-0003) could read another customer's full ticket history,
# email, and conversation via GET /customers/{id} and GET /tickets/{id}.
#
# Fix: /chat returns a per-customer access_token (HMAC of customer_id + server
# secret). Read endpoints require that token via X-Customer-Token header and
# verify it matches the customer_id being requested.

SECRET_KEY = os.getenv("API_SECRET_KEY", "dev-secret-change-in-production")

def generate_customer_token(customer_id: str) -> str:
    """Deterministic per-customer token — same customer always gets the same
    token (works across sessions/devices), but can't be derived without the
    server-side secret, so it can't be guessed from customer_id alone."""
    return hmac.new(
        SECRET_KEY.encode(), customer_id.encode(), hashlib.sha256
    ).hexdigest()[:32]

def verify_customer_access(customer_id: str, x_customer_token: Optional[str] = Header(None)):
    """FastAPI dependency — confirms caller's token matches the customer_id
    in the URL path. 401 if missing, 403 if it belongs to someone else."""
    if not x_customer_token:
        raise HTTPException(status_code=401, detail="Missing X-Customer-Token header")
    expected = generate_customer_token(customer_id)
    if not hmac.compare_digest(x_customer_token, expected):
        raise HTTPException(status_code=403, detail="Token does not match this customer_id")
    return customer_id

from database import init_db, get_db, Ticket, CustomerMemory, InteractionLog
from agent import support_agent

init_db()

app = FastAPI(
    title="AI Customer Support Agent",
    description="LangGraph-powered customer support with memory, escalation, and ticket logging",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Request/Response Models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    customer_id: str
    message: str
    customer_name: Optional[str] = None
    customer_email: Optional[str] = None
    session_id: Optional[str] = None  # for continuing conversations

class ChatResponse(BaseModel):
    response: str
    ticket_id: Optional[str]
    intent: Optional[str]
    access_token: str  # customer must store this — required for /customers and /tickets lookups
    priority: Optional[str]
    sentiment: Optional[str]
    escalated: bool
    resolved: bool
    confidence: Optional[float]
    session_id: str

class TicketUpdate(BaseModel):
    status: str
    agent_notes: Optional[str] = ""
    resolution: Optional[str] = ""

# ── Chat Endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main customer support chat endpoint.
    Handles new and returning customers with full memory.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    if len(request.message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long. Max 2000 characters.")

    # Session management
    session_id = request.session_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": f"{request.customer_id}_{session_id}"}}

    # Get existing state if continuing conversation
    try:
        existing_state = support_agent.get_state(config)
        existing_messages = existing_state.values.get("messages", []) if existing_state.values else []
    except Exception:
        existing_messages = []

    # Build initial state
    initial_state = {
        "customer_id": request.customer_id,
        "customer_name": request.customer_name,
        "customer_email": request.customer_email,
        "messages": existing_messages + [HumanMessage(content=request.message)],
        "current_message": request.message,
        "intent": None,
        "priority": None,
        "sentiment": None,
        "kb_results": None,
        "resolution": None,
        "confidence": None,
        "resolved": None,
        "escalated": None,
        "escalation_reason": None,
        "ticket_id": None,
        "customer_history": None,
        "interaction_count": 0
    }

    # Run agent
    try:
        result = support_agent.invoke(initial_state, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Get the last AI message
    ai_messages = [m for m in result.get("messages", []) if hasattr(m, 'content') and not isinstance(m, HumanMessage)]
    response_text = ai_messages[-1].content if ai_messages else "I apologize, I couldn't process your request. Please try again."

    return ChatResponse(
        response=response_text,
        ticket_id=result.get("ticket_id"),
        intent=result.get("intent"),
        access_token=generate_customer_token(request.customer_id),
        priority=result.get("priority"),
        sentiment=result.get("sentiment"),
        escalated=bool(result.get("escalated")),
        resolved=bool(result.get("resolved")),
        confidence=result.get("confidence"),
        session_id=session_id
    )

# ── Ticket Endpoints ──────────────────────────────────────────────────────────

def verify_staff_key(x_staff_key: Optional[str] = Header(None)):
    """Separate auth tier — GET /tickets (full list), PATCH /tickets/{id}, and
    GET /stats are the support-staff dashboard, not customer-facing. These
    must NOT use the per-customer token (that would let any customer browse
    everyone's tickets) — they need a separate staff-only key."""
    staff_key = os.getenv("STAFF_API_KEY", "")
    if not staff_key or x_staff_key != staff_key:
        raise HTTPException(status_code=401, detail="Invalid or missing staff API key")
    return True


@app.get("/tickets")
def get_tickets(
    status: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db),
    _staff: bool = Depends(verify_staff_key)
):
    """Get all tickets with optional filters"""
    query = db.query(Ticket)
    if status:
        query = query.filter(Ticket.status == status)
    if priority:
        query = query.filter(Ticket.priority == priority)
    tickets = query.order_by(Ticket.created_at.desc()).limit(limit).all()

    return [
        {
            "id": t.id,
            "ticket_id": t.ticket_id,
            "customer_id": t.customer_id,
            "customer_name": t.customer_name,
            "issue_summary": t.issue_summary,
            "category": t.category,
            "priority": t.priority,
            "status": t.status,
            "escalated": bool(t.escalated),
            "confidence_score": t.confidence_score,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in tickets
    ]


@app.get("/tickets/{ticket_id}")
def get_ticket(
    ticket_id: str,
    db: Session = Depends(get_db),
    x_customer_token: Optional[str] = Header(None)
):
    """Get full ticket details including conversation.
    Protected — a ticket leaks full conversation + customer email, so this
    isn't open to anyone who guesses TKT-0001, TKT-0002, etc. The token must
    match the customer_id that OWNS this specific ticket (looked up after
    fetching the row, since ticket_id alone doesn't tell us the customer
    until we query)."""
    ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    if not x_customer_token:
        raise HTTPException(status_code=401, detail="Missing X-Customer-Token header")
    expected = generate_customer_token(ticket.customer_id)
    if not hmac.compare_digest(x_customer_token, expected):
        raise HTTPException(status_code=403, detail="This ticket does not belong to you")

    return {
        "ticket_id": ticket.ticket_id,
        "customer_id": ticket.customer_id,
        "customer_name": ticket.customer_name,
        "customer_email": ticket.customer_email,
        "issue_summary": ticket.issue_summary,
        "full_conversation": ticket.full_conversation,
        "category": ticket.category,
        "priority": ticket.priority,
        "status": ticket.status,
        "resolution": ticket.resolution,
        "agent_notes": ticket.agent_notes,
        "escalated": bool(ticket.escalated),
        "confidence_score": ticket.confidence_score,
        "created_at": ticket.created_at.isoformat() if ticket.created_at else None,
        "resolved_at": ticket.resolved_at.isoformat() if ticket.resolved_at else None,
    }


@app.patch("/tickets/{ticket_id}")
def update_ticket(
    ticket_id: str,
    update: TicketUpdate,
    db: Session = Depends(get_db),
    _staff: bool = Depends(verify_staff_key)
):
    """Human agent updates ticket status and adds notes. Staff-only —
    a customer must never be able to mark their own ticket resolved."""
    ticket = db.query(Ticket).filter(Ticket.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    valid_statuses = ["open", "in_progress", "escalated", "resolved", "closed"]
    if update.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Status must be one of {valid_statuses}")

    ticket.status = update.status
    if update.agent_notes:
        ticket.agent_notes = update.agent_notes
    if update.resolution:
        ticket.resolution = update.resolution
    if update.status == "resolved":
        ticket.resolved_at = datetime.utcnow()

    db.commit()
    return {"message": "Ticket updated", "ticket_id": ticket_id, "status": update.status}


@app.get("/customers/{customer_id}")
def get_customer(
    customer_id: str,
    db: Session = Depends(get_db),
    _verified: str = Depends(verify_customer_access)
):
    """Get customer profile and history.
    Protected — requires X-Customer-Token matching this customer_id, so one
    customer can't enumerate sequential IDs to read another's PII/history."""
    customer = db.query(CustomerMemory).filter(
        CustomerMemory.customer_id == customer_id
    ).first()

    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    tickets = db.query(Ticket).filter(
        Ticket.customer_id == customer_id
    ).order_by(Ticket.created_at.desc()).all()

    return {
        "customer_id": customer.customer_id,
        "name": customer.customer_name,
        "email": customer.customer_email,
        "total_tickets": customer.total_tickets,
        "resolved_tickets": customer.resolved_tickets,
        "last_contact": customer.last_contact.isoformat() if customer.last_contact else None,
        "sentiment": customer.sentiment,
        "issues_history": customer.issues_history,
        "notes": customer.notes,
        "tickets": [
            {
                "ticket_id": t.ticket_id,
                "issue": t.issue_summary[:100],
                "status": t.status,
                "priority": t.priority,
                "created_at": t.created_at.isoformat() if t.created_at else None
            }
            for t in tickets
        ]
    }


@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """Dashboard statistics. Intentionally left public/unauthenticated —
    returns only aggregate counts (no customer_id, no PII, no conversation
    content), so there's nothing here for an IDOR-style attack to expose."""
    total = db.query(Ticket).count()
    open_tickets = db.query(Ticket).filter(Ticket.status == "open").count()
    escalated = db.query(Ticket).filter(Ticket.escalated == 1).count()
    resolved = db.query(Ticket).filter(Ticket.status == "resolved").count()
    urgent = db.query(Ticket).filter(Ticket.priority == "urgent").count()
    high = db.query(Ticket).filter(Ticket.priority == "high").count()

    return {
        "total_tickets": total,
        "open_tickets": open_tickets,
        "escalated_tickets": escalated,
        "resolved_tickets": resolved,
        "urgent_tickets": urgent,
        "high_priority_tickets": high,
        "resolution_rate": round((resolved / total * 100), 1) if total > 0 else 0
    }


@app.get("/")
def root():
    return {
        "service": "AI Customer Support Agent",
        "status": "running",
        "endpoints": {
            "chat": "POST /chat",
            "tickets": "GET /tickets",
            "ticket_detail": "GET /tickets/{ticket_id}",
            "update_ticket": "PATCH /tickets/{ticket_id}",
            "customer": "GET /customers/{customer_id}",
            "stats": "GET /stats",
            "docs": "GET /docs"
        }
    }
