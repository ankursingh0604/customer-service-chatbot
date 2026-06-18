from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ── Knowledge base content ────────────────────────────────────────────────────
# In production this would be loaded from docs/PDFs/database
# Here we define it as structured documents

KNOWLEDGE_BASE = [
    # Billing
    Document(page_content="How to cancel subscription: Go to Settings > Billing > Cancel Subscription. Cancellation takes effect at end of billing period. No refunds for partial months.", metadata={"category": "billing", "topic": "cancellation"}),
    Document(page_content="Refund policy: Refunds are available within 7 days of purchase. Contact support with your order ID. Refunds processed within 3-5 business days.", metadata={"category": "billing", "topic": "refund"}),
    Document(page_content="Payment methods accepted: Credit cards (Visa, Mastercard, Amex), PayPal, and bank transfers. Crypto not accepted. All payments in USD.", metadata={"category": "billing", "topic": "payment"}),
    Document(page_content="Billing cycle: Monthly subscriptions renew on the same date each month. Annual plans renew yearly. You receive email reminder 7 days before renewal.", metadata={"category": "billing", "topic": "billing_cycle"}),
    Document(page_content="How to update payment method: Go to Settings > Billing > Payment Methods > Add New Card. Old card remains until new one is verified.", metadata={"category": "billing", "topic": "payment_update"}),
    Document(page_content="Invoice and receipts: All invoices sent to registered email. Access past invoices at Settings > Billing > Invoice History. Download as PDF.", metadata={"category": "billing", "topic": "invoice"}),

    # Technical
    Document(page_content="Password reset: Click 'Forgot Password' on login page. Enter registered email. Reset link valid for 24 hours. Check spam folder if not received.", metadata={"category": "technical", "topic": "password"}),
    Document(page_content="Account locked: Account locked after 5 failed login attempts. Wait 30 minutes or contact support to unlock immediately.", metadata={"category": "technical", "topic": "account_locked"}),
    Document(page_content="App not loading: Clear browser cache and cookies. Try incognito mode. Disable browser extensions. Try different browser. Check status page at status.ourapp.com.", metadata={"category": "technical", "topic": "app_loading"}),
    Document(page_content="Data export: Export all your data at Settings > Privacy > Export Data. CSV format. Processing takes up to 24 hours. Download link sent to email.", metadata={"category": "technical", "topic": "data_export"}),
    Document(page_content="API rate limits: Free plan: 100 requests/hour. Pro plan: 1000 requests/hour. Enterprise: unlimited. Rate limit resets every hour.", metadata={"category": "technical", "topic": "api_limits"}),
    Document(page_content="Two factor authentication: Enable 2FA at Settings > Security > Two Factor Auth. Supports authenticator apps (Google, Authy) and SMS.", metadata={"category": "technical", "topic": "2fa"}),
    Document(page_content="Integration setup: Connect third party apps at Settings > Integrations. Supports Slack, Zapier, Google Workspace. API keys at Settings > Developer.", metadata={"category": "technical", "topic": "integration"}),

    # Account
    Document(page_content="How to change email address: Settings > Account > Email. Verification sent to new email. Old email receives notification. Change takes effect after verification.", metadata={"category": "account", "topic": "email_change"}),
    Document(page_content="Delete account: Settings > Account > Delete Account. All data deleted within 30 days. This action is irreversible. Subscription cancelled immediately.", metadata={"category": "account", "topic": "delete_account"}),
    Document(page_content="Team management: Add team members at Settings > Team. Roles available: Admin, Editor, Viewer. Each member needs own account.", metadata={"category": "account", "topic": "team"}),
    Document(page_content="Plan upgrade: Upgrade at Settings > Billing > Change Plan. Upgrade takes effect immediately. Prorated credit applied for remaining days.", metadata={"category": "account", "topic": "upgrade"}),
    Document(page_content="Plan downgrade: Downgrade takes effect at next billing cycle. Features lost immediately on some plans. Review what you lose before downgrading.", metadata={"category": "account", "topic": "downgrade"}),

    # General
    Document(page_content="Support hours: Live chat available Monday-Friday 9am-6pm EST. Email support 24/7 with response within 4 hours. Emergency line for critical issues.", metadata={"category": "general", "topic": "support_hours"}),
    Document(page_content="SLA and uptime: We guarantee 99.9% uptime. Scheduled maintenance announced 48 hours in advance. Status updates at status.ourapp.com.", metadata={"category": "general", "topic": "sla"}),
    Document(page_content="Privacy and data: We are GDPR compliant. Data stored in EU servers. No data sold to third parties. Full privacy policy at ourapp.com/privacy.", metadata={"category": "general", "topic": "privacy"}),
]

# ── Build vector store ────────────────────────────────────────────────────────

_vectorstore = None

def get_knowledge_base():
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        _vectorstore = FAISS.from_documents(KNOWLEDGE_BASE, embeddings)
    return _vectorstore

def search_knowledge_base(query: str, k: int = 3) -> list:
    """Search knowledge base and return relevant documents"""
    kb = get_knowledge_base()
    results = kb.similarity_search_with_score(query, k=k)
    return [
        {
            "content": doc.page_content,
            "category": doc.metadata.get("category", "general"),
            "topic": doc.metadata.get("topic", ""),
            "relevance_score": float(score)
        }
        for doc, score in results
    ]
