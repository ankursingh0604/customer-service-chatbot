import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
STAFF_API_KEY = os.getenv("STAFF_API_KEY", "")

try:
    API_URL = st.secrets["API_URL"]
    STAFF_API_KEY = st.secrets["STAFF_API_KEY"]
except Exception:
    pass

if "customer_token" not in st.session_state:
    st.session_state["customer_token"] = None
if "input_counter" not in st.session_state:
    st.session_state["input_counter"] = 0

st.set_page_config(
    page_title="AI Support Agent",
    page_icon="🎧",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
    * { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Syne', sans-serif !important; }
    .block-container { padding: 2rem 2rem; max-width: 1400px; }

    .hero {
        background: linear-gradient(135deg, #0a0a14 0%, #14100a 50%, #0a0a14 100%);
        border: 1px solid #2a2a1e;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .hero-title { font-family: 'Syne', sans-serif !important; font-size: 2rem; font-weight: 800; color: #f0f0f8; margin: 0 0 0.5rem 0; }
    .hero-sub { font-size: 0.9rem; color: #8888aa; margin: 0; }

    .chat-bubble-user {
        background: #1a1a2e;
        border: 1px solid #2a2a4e;
        border-radius: 12px 12px 2px 12px;
        padding: 10px 14px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        color: #c0c0f0;
        font-size: 14px;
    }
    .chat-bubble-agent {
        background: #0f1a0f;
        border: 1px solid #1a3a1a;
        border-radius: 12px 12px 12px 2px;
        padding: 10px 14px;
        margin: 8px 0;
        max-width: 80%;
        color: #c0d8c0;
        font-size: 14px;
    }

    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 500;
        margin-right: 4px;
    }
    .badge-low { background: #1a2a1a; color: #44aa44; }
    .badge-medium { background: #2a2a1a; color: #aaaa44; }
    .badge-high { background: #2a1a1a; color: #aa4444; }
    .badge-urgent { background: #3a0a0a; color: #ff4444; }
    .badge-open { background: #1a1a2a; color: #4444aa; }
    .badge-resolved { background: #1a2a1a; color: #44aa44; }
    .badge-escalated { background: #2a1a0a; color: #aa6644; }
    .badge-billing { background: #1a2a2a; color: #44aaaa; }
    .badge-technical { background: #2a1a2a; color: #aa44aa; }
    .badge-complaint { background: #2a1a1a; color: #aa4444; }
    .badge-general { background: #1a1a2a; color: #4488aa; }

    .metric-card {
        background: #0a0f0a;
        border: 1px solid #1a2a1a;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-number { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #44ee88; }
    .metric-label { font-size: 11px; color: #446644; text-transform: uppercase; letter-spacing: .1em; }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #1a5a2a, #1a3a5a);
        color: white; border: none; border-radius: 8px;
        font-family: 'Syne', sans-serif; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-title">🎧 AI Customer Support Agent</div>
    <p class="hero-sub">LangGraph agent with memory · Intent classification · Auto-escalation · Ticket logging</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["💬 Customer Chat", "🎫 Ticket Dashboard", "👤 Customer Lookup"])


def send_message_to_api(customer_id, message, customer_name, customer_email):
    """Send message to API and update session state."""
    st.session_state["chat_messages"].append({"role": "user", "content": message})
    with st.spinner("Agent thinking..."):
        try:
            response = requests.post(
                f"{API_URL}/chat",
                json={
                    "customer_id": customer_id,
                    "message": message,
                    "customer_name": customer_name or None,
                    "customer_email": customer_email or None,
                    "session_id": st.session_state.get("session_id")
                },
                timeout=120
            )
            if response.status_code == 200:
                result = response.json()
                st.session_state["session_id"] = result["session_id"]
                st.session_state["customer_token"] = result["access_token"]
                st.session_state["chat_messages"].append({
                    "role": "agent",
                    "content": result["response"]
                })
                st.session_state["last_result"] = result
                st.session_state["input_counter"] += 1
                st.rerun()
            else:
                st.error(f"Error: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Run: `uvicorn api:app --reload`")
        except Exception as e:
            st.error(f"Error: {str(e)}")


# ── Tab 1 — Customer Chat ─────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### Start a Support Conversation")

        c1, c2, c3 = st.columns(3)
        with c1:
            customer_id = st.text_input("Customer ID*", placeholder="CUST-001", key="cust_id")
        with c2:
            customer_name = st.text_input("Name (optional)", placeholder="John Doe", key="cust_name")
        with c3:
            customer_email = st.text_input("Email (optional)", placeholder="john@email.com", key="cust_email")

        if "chat_messages" not in st.session_state:
            st.session_state["chat_messages"] = []
        if "session_id" not in st.session_state:
            st.session_state["session_id"] = None

        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["chat_messages"]:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-bubble-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-bubble-agent">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        message = st.text_area(
            "Your message",
            placeholder="Describe your issue...",
            height=100,
            key=f"chat_input_{st.session_state['input_counter']}",
            label_visibility="collapsed"
        )

        col_send, col_clear = st.columns([3, 1])
        with col_send:
            send_btn = st.button("📨 Send Message", use_container_width=True)
        with col_clear:
            if st.button("🔄 New Chat", use_container_width=True):
                st.session_state["chat_messages"] = []
                st.session_state["session_id"] = None
                st.rerun()

        if send_btn:
            if not customer_id.strip():
                st.error("Please enter a Customer ID")
            elif not message.strip():
                st.error("Please enter a message")
            else:
                send_message_to_api(customer_id, message, customer_name, customer_email)

    with col2:
        st.markdown("#### Last Response Details")
        if "last_result" in st.session_state:
            r = st.session_state["last_result"]

            if r.get("ticket_id"):
                st.success(f"🎫 {r['ticket_id']}")

            intent = r.get("intent", "general")
            priority = r.get("priority", "medium")
            sentiment = r.get("sentiment", "neutral")

            st.markdown(
                f'<span class="badge badge-{intent}">{intent.upper()}</span>'
                f'<span class="badge badge-{priority}">{priority.upper()}</span>',
                unsafe_allow_html=True
            )

            if r.get("escalated"):
                st.error("🚨 Escalated to human agent")
            elif r.get("resolved"):
                st.success("✅ Resolved")
            else:
                st.warning("⏳ Open")

            conf = r.get("confidence", 0)
            conf_color = "🟢" if conf > 0.7 else "🟡" if conf > 0.4 else "🔴"
            st.metric("Agent Confidence", f"{conf_color} {conf*100:.0f}%")

            sentiment_emoji = "😊" if sentiment == "positive" else "😐" if sentiment == "neutral" else "😤" if sentiment == "frustrated" else "😞"
            st.metric("Customer Sentiment", f"{sentiment_emoji} {sentiment.title()}")

        else:
            st.info("Send a message to see analysis details here")

        st.markdown("#### 💡 Try these examples")
        examples = [
            "I want to cancel my subscription",
            "My account is locked, I can't login",
            "I was charged twice this month!",
            "How do I export my data?",
            "This is absolutely unacceptable! I've been waiting for 3 days!"
        ]
        for ex in examples:
            if st.button(ex[:40] + "...", key=f"ex_{ex[:10]}", use_container_width=True):
                if not customer_id.strip():
                    st.error("Please enter a Customer ID first")
                else:
                    send_message_to_api(customer_id, ex, customer_name, customer_email)


# ── Tab 2 — Ticket Dashboard ──────────────────────────────────────────────────
with tab2:
    st.markdown("#### Ticket Dashboard")

    try:
        stats_resp = requests.get(f"{API_URL}/stats", timeout=10)
        if stats_resp.status_code == 200:
            stats = stats_resp.json()
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.markdown(f'<div class="metric-card"><div class="metric-number">{stats["total_tickets"]}</div><div class="metric-label">Total</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="metric-card"><div class="metric-number" style="color:#4488ee">{stats["open_tickets"]}</div><div class="metric-label">Open</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="metric-card"><div class="metric-number" style="color:#ee4444">{stats["escalated_tickets"]}</div><div class="metric-label">Escalated</div></div>', unsafe_allow_html=True)
            c4.markdown(f'<div class="metric-card"><div class="metric-number">{stats["resolved_tickets"]}</div><div class="metric-label">Resolved</div></div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="metric-card"><div class="metric-number" style="color:#ee4444">{stats["urgent_tickets"]}</div><div class="metric-label">Urgent</div></div>', unsafe_allow_html=True)
            c6.markdown(f'<div class="metric-card"><div class="metric-number">{stats["resolution_rate"]}%</div><div class="metric-label">Resolved %</div></div>', unsafe_allow_html=True)
    except Exception:
        st.warning("Start the API to see stats")

    st.markdown("---")

    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        filter_status = st.selectbox("Status", ["all", "open", "escalated", "in_progress", "resolved", "closed"])
    with col_f2:
        filter_priority = st.selectbox("Priority", ["all", "urgent", "high", "medium", "low"])
    with col_f3:
        if st.button("🔄 Refresh", use_container_width=False):
            st.rerun()

    try:
        params = {}
        if filter_status != "all":
            params["status"] = filter_status
        if filter_priority != "all":
            params["priority"] = filter_priority

        tickets_resp = requests.get(
            f"{API_URL}/tickets",
            params=params,
            headers={"X-Staff-Key": STAFF_API_KEY},
            timeout=10
        )
        if tickets_resp.status_code == 401:
            st.error("Staff key missing/invalid — set STAFF_API_KEY in .env or secrets")
        elif tickets_resp.status_code == 200:
            tickets = tickets_resp.json()
            if not tickets:
                st.info("No tickets found. Start a chat to create tickets.")
            else:
                for t in tickets:
                    priority = t.get("priority", "medium")
                    status = t.get("status", "open")
                    category = t.get("category", "general")
                    escalated = t.get("escalated", False)

                    with st.expander(
                        f"{'🚨 ' if escalated else ''}{t['ticket_id']} — {t['issue_summary'][:60]}... | {priority.upper()} | {status.upper()}"
                    ):
                        col_l, col_r = st.columns([2, 1])
                        with col_l:
                            st.markdown(
                                f'<span class="badge badge-{category}">{category}</span>'
                                f'<span class="badge badge-{priority}">{priority}</span>'
                                f'<span class="badge badge-{status}">{status}</span>',
                                unsafe_allow_html=True
                            )
                            st.markdown(f"**Customer:** {t.get('customer_id')} | {t.get('customer_name', 'Unknown')}")
                            st.markdown(f"**Issue:** {t['issue_summary']}")
                            st.markdown(f"**Confidence:** {(t.get('confidence_score', 0) or 0)*100:.0f}%")
                            if t.get("created_at"):
                                st.markdown(f"*Created: {t['created_at'][:19]}*")

                        with col_r:
                            new_status = st.selectbox(
                                "Update status",
                                ["open", "in_progress", "escalated", "resolved", "closed"],
                                index=["open", "in_progress", "escalated", "resolved", "closed"].index(status) if status in ["open", "in_progress", "escalated", "resolved", "closed"] else 0,
                                key=f"status_{t['ticket_id']}"
                            )
                            notes = st.text_input("Agent notes", key=f"notes_{t['ticket_id']}")
                            if st.button("Update", key=f"update_{t['ticket_id']}"):
                                update_resp = requests.patch(
                                    f"{API_URL}/tickets/{t['ticket_id']}",
                                    json={"status": new_status, "agent_notes": notes},
                                    headers={"X-Staff-Key": STAFF_API_KEY},
                                    timeout=10
                                )
                                if update_resp.status_code == 200:
                                    st.success("Updated!")
                                    st.rerun()
                                elif update_resp.status_code == 401:
                                    st.error("Staff key missing/invalid — set STAFF_API_KEY")
    except requests.exceptions.ConnectionError:
        st.warning("Start the API server to see tickets")


# ── Tab 3 — Customer Lookup ───────────────────────────────────────────────────
with tab3:
    st.markdown("#### Customer Profile Lookup")
    st.caption(
        "You can only look up the customer_id you most recently chatted as in "
        "Tab 1 — the API verifies the access token before returning any history."
    )

    lookup_id = st.text_input("Enter Customer ID", placeholder="CUST-001")

    if st.button("🔍 Lookup Customer", use_container_width=False):
        if not lookup_id.strip():
            st.error("Enter a customer ID")
        elif not st.session_state.get("customer_token"):
            st.error("Chat with this customer_id in Tab 1 first to get an access token")
        else:
            try:
                resp = requests.get(
                    f"{API_URL}/customers/{lookup_id}",
                    headers={"X-Customer-Token": st.session_state["customer_token"]},
                    timeout=10
                )
                if resp.status_code == 403:
                    st.error("This token doesn't belong to that customer_id — can't view their data")
                elif resp.status_code == 200:
                    customer = resp.json()

                    col_l, col_r = st.columns([1, 1])
                    with col_l:
                        st.markdown("**Profile**")
                        st.markdown(f"**Name:** {customer.get('name', 'Unknown')}")
                        st.markdown(f"**Email:** {customer.get('email', 'Unknown')}")
                        st.markdown(f"**Total Tickets:** {customer.get('total_tickets', 0)}")
                        st.markdown(f"**Resolved:** {customer.get('resolved_tickets', 0)}")
                        st.markdown(f"**Last Contact:** {customer.get('last_contact', 'Never')[:19] if customer.get('last_contact') else 'Never'}")
                        st.markdown(f"**Sentiment:** {customer.get('sentiment', 'neutral').title()}")

                    with col_r:
                        st.markdown("**Issue History**")
                        st.text(customer.get("issues_history", "No history"))

                    st.markdown("**Past Tickets**")
                    for t in customer.get("tickets", []):
                        st.markdown(f"- `{t['ticket_id']}` — {t['issue'][:60]} | **{t['status'].upper()}** | {t['priority']}")

                elif resp.status_code == 404:
                    st.warning("Customer not found")
                else:
                    st.error(f"Error: {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API")