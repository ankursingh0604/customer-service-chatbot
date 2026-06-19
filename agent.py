import os
import uuid
import json
from typing import TypedDict, Annotated, Optional, List
import operator
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from knowledge_base import search_knowledge_base
from database import SessionLocal, Ticket, CustomerMemory, InteractionLog, init_db

load_dotenv()
init_db()

# ── State ─────────────────────────────────────────────────────────────────────

class SupportState(TypedDict):
    # Customer info
    customer_id: str
    customer_name: Optional[str]
    customer_email: Optional[str]

    # Conversation
    messages: Annotated[list[AnyMessage], operator.add]
    current_message: str

    # Analysis
    intent: Optional[str]          # billing, technical, account, general, complaint
    priority: Optional[str]        # low, medium, high, urgent
    sentiment: Optional[str]       # positive, neutral, negative, frustrated

    # Resolution
    kb_results: Optional[list]
    resolution: Optional[str]
    confidence: Optional[float]    # 0.0 to 1.0
    resolved: Optional[bool]

    # Escalation
    escalated: Optional[bool]
    escalation_reason: Optional[str]

    # Ticket
    ticket_id: Optional[str]

    # Customer history
    customer_history: Optional[str]
    interaction_count: Optional[int]

# ── LLM ──────────────────────────────────────────────────────────────────────

def get_llm(temperature=0.1):
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile",
        temperature=temperature
    )

# ── Helper: Get/Create Customer Memory ────────────────────────────────────────

def get_customer_memory(customer_id: str) -> dict:
    db = SessionLocal()
    try:
        customer = db.query(CustomerMemory).filter(
            CustomerMemory.customer_id == customer_id
        ).first()

        if customer:
            return {
                "exists": True,
                "name": customer.customer_name,
                "email": customer.customer_email,
                "total_tickets": customer.total_tickets,
                "resolved_tickets": customer.resolved_tickets,
                "last_contact": str(customer.last_contact) if customer.last_contact else None,
                "issues_history": customer.issues_history,
                "sentiment": customer.sentiment,
                "notes": customer.notes
            }
        return {"exists": False}
    finally:
        db.close()

def update_customer_memory(customer_id: str, state: SupportState, resolved: bool):
    db = SessionLocal()
    try:
        customer = db.query(CustomerMemory).filter(
            CustomerMemory.customer_id == customer_id
        ).first()

        if not customer:
            customer = CustomerMemory(customer_id=customer_id)
            db.add(customer)

        customer.customer_name = state.get("customer_name") or customer.customer_name
        customer.customer_email = state.get("customer_email") or customer.customer_email
        customer.total_tickets = (customer.total_tickets or 0) + 1
        if resolved:
            customer.resolved_tickets = (customer.resolved_tickets or 0) + 1
        customer.last_contact = datetime.utcnow()
        customer.sentiment = state.get("sentiment", "neutral")

        # Update issues history
        current_history = customer.issues_history or ""
        new_entry = f"[{datetime.utcnow().strftime('%Y-%m-%d')}] {state.get('intent', 'general')}: {state.get('current_message', '')[:100]}"
        customer.issues_history = (current_history + "\n" + new_entry).strip()[-2000:]  # keep last 2000 chars

        db.commit()
    finally:
        db.close()

# ── Node 1: Load Customer Context ─────────────────────────────────────────────

def load_customer_context(state: SupportState) -> dict:
    """Load customer history and context from memory"""
    print(f"📋 Loading context for customer: {state['customer_id']}")

    memory = get_customer_memory(state["customer_id"])

    if memory["exists"]:
        history_summary = (
            f"Returning customer. Total tickets: {memory['total_tickets']}. "
            f"Resolved: {memory['resolved_tickets']}. "
            f"Last contact: {memory.get('last_contact', 'Unknown')}. "
            f"Past issues: {memory.get('issues_history', 'None')}. "
            f"Customer sentiment: {memory.get('sentiment', 'neutral')}."
        )
        print(f"✅ Returning customer found")
    else:
        history_summary = "New customer — no previous interactions."
        print(f"✅ New customer")

    return {
        "customer_history": history_summary,
        "interaction_count": memory.get("total_tickets", 0)
    }

# ── Node 2: Analyze Intent & Sentiment ───────────────────────────────────────

def analyze_intent(state: SupportState) -> dict:
    """Classify the customer's intent, priority and sentiment"""
    print(f"🔍 Analyzing intent...")

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert customer support analyst. "
            "Analyze the customer message and return ONLY a JSON object. "
            "No other text, no markdown, just the JSON."
        )),
        ("human", (
            "Customer history: {history}\n\n"
            "Customer message: {message}\n\n"
            "Return this exact JSON structure:\n"
            '{{"intent": "billing|technical|account|general|complaint", '
            '"priority": "low|medium|high|urgent", '
            '"sentiment": "positive|neutral|negative|frustrated", '
            '"summary": "one line summary of the issue"}}'
        ))
    ])

    llm = get_llm(temperature=0)
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "history": state.get("customer_history", "No history"),
        "message": state["current_message"]
    })

    try:
        # Clean and parse JSON
        result = result.strip()
        if "```" in result:
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        data = json.loads(result)
    except Exception:
        data = {
            "intent": "general",
            "priority": "medium",
            "sentiment": "neutral",
            "summary": state["current_message"][:100]
        }

    # Auto-upgrade priority for frustrated customers
    if data.get("sentiment") == "frustrated" and data.get("priority") == "medium":
        data["priority"] = "high"

    # Auto-upgrade priority for returning customers with many unresolved issues
    if state.get("interaction_count", 0) > 3:
        if data.get("priority") == "low":
            data["priority"] = "medium"

    print(f"✅ Intent: {data.get('intent')} | Priority: {data.get('priority')} | Sentiment: {data.get('sentiment')}")

    return {
        "intent": data.get("intent", "general"),
        "priority": data.get("priority", "medium"),
        "sentiment": data.get("sentiment", "neutral")
    }

# ── Node 3: Search Knowledge Base ────────────────────────────────────────────

def search_kb(state: SupportState) -> dict:
    """Search knowledge base for relevant information"""
    print(f"📚 Searching knowledge base...")

    results = search_knowledge_base(state["current_message"], k=3)
    print(f"✅ Found {len(results)} relevant articles")

    return {"kb_results": results}

# ── Node 4: Generate Response ─────────────────────────────────────────────────

def generate_response(state: SupportState) -> dict:
    """Generate a response based on KB results and conversation history"""
    print(f"💬 Generating response...")

    kb_text = "\n\n".join([
        f"[{r['topic'].upper()}]: {r['content']}"
        for r in (state.get("kb_results") or [])
    ])

    # Build conversation history
    conv_history = ""
    for msg in state.get("messages", [])[-6:]:  # last 6 messages for context
        if isinstance(msg, HumanMessage):
            conv_history += f"Customer: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            conv_history += f"Agent: {msg.content}\n"

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful, empathetic customer support agent. "
            "Use the knowledge base to answer accurately. "
            "Be concise but complete. "
            "If you can fully resolve the issue, end with a confidence statement. "
            "If you cannot fully resolve, say so honestly. "
            "Never make up information not in the knowledge base. "
            "Customer sentiment: {sentiment}. "
            "If frustrated, acknowledge their frustration first."
        )),
        ("human", (
            "Customer history: {history}\n\n"
            "Conversation so far:\n{conv_history}\n\n"
            "Knowledge base:\n{kb_text}\n\n"
            "Current message: {message}\n\n"
            "Provide a helpful response. End with: "
            "CONFIDENCE: [HIGH/MEDIUM/LOW] - [brief reason]"
        ))
    ])

    llm = get_llm(temperature=0.3)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({
        "sentiment": state.get("sentiment", "neutral"),
        "history": state.get("customer_history", "New customer"),
        "conv_history": conv_history,
        "kb_text": kb_text or "No specific knowledge base articles found.",
        "message": state["current_message"]
    })

    # Extract confidence
    confidence = 0.5
    resolved = False

    if "CONFIDENCE: HIGH" in response:
        confidence = 0.9
        resolved = True
    elif "CONFIDENCE: MEDIUM" in response:
        confidence = 0.6
        resolved = True
    elif "CONFIDENCE: LOW" in response:
        confidence = 0.3
        resolved = False

    # Clean response — remove confidence line for display
    clean_response = response
    for line in response.split("\n"):
        if "CONFIDENCE:" in line:
            clean_response = response.replace(line, "").strip()
            break

    print(f"✅ Response generated | Confidence: {confidence}")

    return {
        "resolution": clean_response,
        "confidence": confidence,
        "resolved": resolved,
        "messages": [AIMessage(content=clean_response)]
    }

# ── Node 5: Create Ticket ─────────────────────────────────────────────────────

def create_ticket(state: SupportState) -> dict:
    """Create a support ticket in the database.

    Runs FIRST in both the escalate and no-escalate paths (see graph wiring
    in build_support_agent). Determines its own escalation decision by
    calling should_escalate() directly, rather than relying on an
    "escalated" key already being in state — at this point in the graph,
    if we're on the escalation path, escalate() hasn't run yet, so
    state.get("escalated") would still be None/falsy if we trusted it
    blindly. Computing it explicitly here keeps the ticket's escalated
    flag correct regardless of node execution order.
    """
    print(f"🎫 Creating ticket...")

    will_escalate = should_escalate(state) == "escalate"

    db = SessionLocal()
    try:
        # Generate ticket ID
        ticket_num = db.query(Ticket).count() + 1
        ticket_id = f"TKT-{ticket_num:04d}"

        # Build full conversation
        conv = []
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                conv.append(f"Customer: {msg.content}")
            elif isinstance(msg, AIMessage):
                conv.append(f"Agent: {msg.content}")
        full_conv = "\n".join(conv)

        ticket = Ticket(
            ticket_id=ticket_id,
            customer_id=state["customer_id"],
            customer_name=state.get("customer_name"),
            customer_email=state.get("customer_email"),
            issue_summary=state["current_message"][:500],
            full_conversation=full_conv,
            category=state.get("intent", "general"),
            priority=state.get("priority", "medium"),
            status="escalated" if will_escalate else ("resolved" if state.get("resolved") else "open"),
            resolution=state.get("resolution", ""),
            confidence_score=state.get("confidence", 0.5),
            escalated=1 if will_escalate else 0,
            resolved_at=datetime.utcnow() if (state.get("resolved") and not will_escalate) else None
        )
        db.add(ticket)

        # Log interaction
        log = InteractionLog(
            ticket_id=ticket_id,
            customer_id=state["customer_id"],
            role="system",
            message=f"Ticket created. Intent: {state.get('intent')}. Priority: {state.get('priority')}. Resolved: {state.get('resolved')}. Will escalate: {will_escalate}",
            node="create_ticket"
        )
        db.add(log)
        db.commit()

        # Update customer memory
        update_customer_memory(state["customer_id"], state, state.get("resolved", False))

        print(f"✅ Ticket created: {ticket_id} (escalating: {will_escalate})")
        return {"ticket_id": ticket_id}

    finally:
        db.close()

# ── Node 6: Escalate ──────────────────────────────────────────────────────────

def escalate(state: SupportState) -> dict:
    """Escalate to human agent.

    Runs AFTER create_ticket in the graph now (see build_support_agent),
    so state["ticket_id"] is always a real value here, not None — that was
    the root cause of "Your ticket ID is None" in escalation messages.
    create_ticket already set status="escalated" and escalated=1 on the
    row, so this function's job is just to log the specific reason and
    produce the customer-facing message — not to re-fetch/re-mutate the
    ticket.
    """
    print(f"🚨 Escalating to human agent...")

    db = SessionLocal()
    try:
        reasons = []
        if state.get("confidence", 1.0) < 0.4:
            reasons.append("low agent confidence")
        if state.get("sentiment") == "frustrated":
            reasons.append("frustrated customer")
        if state.get("priority") in ["high", "urgent"]:
            reasons.append(f"{state.get('priority')} priority issue")
        if state.get("interaction_count", 0) > 5:
            reasons.append("frequent contact customer")

        escalation_reason = ", ".join(reasons) if reasons else "complex issue requiring human review"

        log = InteractionLog(
            ticket_id=state["ticket_id"],
            customer_id=state["customer_id"],
            role="system",
            message=f"ESCALATED: {escalation_reason}",
            node="escalate"
        )
        db.add(log)
        db.commit()

        escalation_message = (
            f"I've escalated your case to our specialist team. "
            f"Your ticket ID is {state['ticket_id']}. "
            f"A human agent will contact you within 2-4 hours. "
            f"We apologize for the inconvenience."
        )

        print(f"✅ Escalated: {escalation_reason} | Ticket: {state['ticket_id']}")

        return {
            "escalated": True,
            "escalation_reason": escalation_reason,
            "messages": [AIMessage(content=escalation_message)]
        }
    finally:
        db.close()

# ── Routing Functions ─────────────────────────────────────────────────────────

def should_escalate(state: SupportState) -> str:
    """Decide whether to escalate or resolve.

    NOTE on graph ordering — this fixes a real bug found in production
    (ticket ID showed as "None" in escalation messages): the graph used to
    route generate_response -> escalate -> create_ticket -> END. But
    escalate() reads state["ticket_id"] to update/reference the ticket, and
    create_ticket() reads state["escalated"] to set the ticket's escalated
    flag — each node depended on something only the OTHER node produces,
    with no edge ordering that could satisfy both at once.

    Fix: create_ticket now ALWAYS runs first (it determines its own
    escalation decision by calling this same should_escalate() function
    internally — see create_ticket below — so the `escalated` field is
    correct on creation). escalate() then runs conditionally AFTER, by
    which point state["ticket_id"] is a real value, not None.
    """
    if state.get("confidence", 1.0) < 0.4:
        return "escalate"
    if state.get("sentiment") == "frustrated" and not state.get("resolved"):
        return "escalate"
    if state.get("priority") == "urgent":
        return "escalate"
    if state.get("intent") == "complaint" and not state.get("resolved"):
        return "escalate"
    return "no_escalate"

# ── Build Graph ───────────────────────────────────────────────────────────────

def build_support_agent():
    graph = StateGraph(SupportState)

    # Add all nodes
    graph.add_node("load_context", load_customer_context)
    graph.add_node("analyze_intent", analyze_intent)
    graph.add_node("search_kb", search_kb)
    graph.add_node("generate_response", generate_response)
    graph.add_node("create_ticket", create_ticket)
    graph.add_node("escalate", escalate)

    # Entry point
    graph.set_entry_point("load_context")

    # Edges
    graph.add_edge("load_context", "analyze_intent")
    graph.add_edge("analyze_intent", "search_kb")
    graph.add_edge("search_kb", "generate_response")

    # create_ticket ALWAYS runs right after generate_response — it computes
    # its own escalation decision internally (see create_ticket docstring)
    # so the ticket row is correct regardless of which path comes next.
    graph.add_edge("generate_response", "create_ticket")

    # AFTER the ticket exists (with a real ticket_id), conditionally route
    # to escalate — which now has a real ID to put in the customer message,
    # instead of "None".
    graph.add_conditional_edges(
        "create_ticket",
        should_escalate,
        {
            "escalate": "escalate",
            "no_escalate": END
        }
    )

    graph.add_edge("escalate", END)

    # SQLite checkpointer for persistence.
    # IMPORTANT: SqliteSaver.from_conn_string() returns a context manager
    # (_GeneratorContextManager), not a usable saver instance — calling
    # graph.compile(checkpointer=that) raises:
    #   TypeError: Invalid checkpointer provided. Expected an instance of
    #   `BaseCheckpointSaver`, `True`, `False`, or `None`.
    # This is a known breaking change in recent langgraph-checkpoint-sqlite
    # versions (see langchain-ai/langgraph#2042 and #1262). The `with` form
    # only works for short-lived scripts where the graph runs entirely
    # inside the `with` block — it doesn't work here because this module
    # builds the agent once at import time and FastAPI keeps using it
    # across many requests.
    # Fix: open the raw sqlite3 connection ourselves (check_same_thread=False
    # because FastAPI may call this from different threads) and pass the
    # connection directly to SqliteSaver(), bypassing the context manager.
    import sqlite3
    os.makedirs("data", exist_ok=True)

    conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
    memory = SqliteSaver(conn)
    return graph.compile(checkpointer=memory)

support_agent = build_support_agent()
