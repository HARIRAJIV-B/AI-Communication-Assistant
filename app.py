import os
import re
import io
import time
import sqlite3
import datetime as dt
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from dotenv import load_dotenv

# --- one-time NLTK download ---
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# Optional OpenAI import (won't break if not installed yet)
USE_OPENAI = False
try:
    from openai import OpenAI
    if OPENAI_API_KEY:
        client = OpenAI(api_key=OPENAI_API_KEY)
        USE_OPENAI = True
except Exception:
    USE_OPENAI = False

# ------------------- Simple DB helpers -------------------
DB_PATH = "emails.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """CREATE TABLE IF NOT EXISTS emails (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender TEXT,
            subject TEXT,
            body TEXT,
            received_at TEXT,
            sentiment TEXT,
            priority TEXT,
            extracted_phone TEXT,
            extracted_alt_email TEXT,
            product_mentions TEXT,
            response TEXT,
            status TEXT
        )"""
    )
    conn.commit()
    conn.close()

def insert_email_rows(rows: List[Dict[str, Any]]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for r in rows:
        cur.execute(
            """INSERT INTO emails
            (sender, subject, body, received_at, sentiment, priority, extracted_phone,
             extracted_alt_email, product_mentions, response, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                r["sender"], r["subject"], r["body"], r["received_at"], r["sentiment"],
                r["priority"], r["extracted_phone"], r["extracted_alt_email"],
                r["product_mentions"], r.get("response", ""), r.get("status", "pending")
            )
        )
    conn.commit()
    conn.close()

def fetch_emails(order_by_priority: bool = True) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM emails", conn)
    conn.close()
    if df.empty:
        return df
    # urgent first
    if order_by_priority:
        df["priority_rank"] = df["priority"].apply(lambda x: 0 if str(x).lower()=="urgent" else 1)
        df = df.sort_values(["priority_rank", "id"])
        df = df.drop(columns=["priority_rank"])
    return df

def update_email_response(email_id: int, response: str, status: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE emails SET response=?, status=? WHERE id=?", (response, status, email_id))
    conn.commit()
    conn.close()

# ------------------- NLP utilities -------------------
FILTER_KEYWORDS = ["support", "help", "query", "request"]
URGENT_KEYWORDS = [
    "urgent", "immediately", "asap", "critical", "escalate", "cannot access",
    "blocked", "down", "outage", "security", "breach", "fail", "failed", "error"
]

PHONE_REGEX = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")
EMAIL_REGEX = re.compile(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)")

sia = SentimentIntensityAnalyzer()

def simple_sentiment(text: str) -> str:
    s = sia.polarity_scores(text or "")
    if s["compound"] >= 0.25:
        return "positive"
    elif s["compound"] <= -0.25:
        return "negative"
    else:
        return "neutral"

def is_filtered_subject(subject: str) -> bool:
    if not subject:
        return False
    s = subject.lower()
    return any(k in s for k in FILTER_KEYWORDS)

def detect_priority(text: str) -> str:
    t = (text or "").lower()
    return "urgent" if any(k in t for k in URGENT_KEYWORDS) else "not urgent"

def extract_info(body: str) -> Tuple[str, str, str]:
    phones = PHONE_REGEX.findall(body or "")
    emails = EMAIL_REGEX.findall(body or "")
    # naive product mentions: capitalized words over 2 letters (you can customize)
    products = re.findall(r"\b([A-Z][A-Za-z0-9]{2,})\b", body or "")
    products = ", ".join(sorted(set(products))[:5])
    phone = phones[0] if phones else ""
    alt_email = emails[1] if len(emails) > 1 else (emails[0] if emails else "")
    return phone, alt_email, products

# ------------------- RAG over FAQs -------------------
def load_kb_text(path: str = "kb/faqs.md") -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Split by FAQ blocks
    chunks = re.split(r"\n##\s+", text)
    cleaned = []
    for ch in chunks:
        ch = ch.strip()
        if ch and len(ch) > 40:
            cleaned.append(ch)
    return cleaned

KB_CHUNKS = load_kb_text()
VECTORIZER = TfidfVectorizer(stop_words="english")
KB_MATRIX = None
if KB_CHUNKS:
    KB_MATRIX = VECTORIZER.fit_transform(KB_CHUNKS)

def retrieve_kb(query: str, topk: int = 2) -> List[str]:
    if not KB_CHUNKS or KB_MATRIX is None:
        return []
    q_vec = VECTORIZER.transform([query])
    sims = linear_kernel(q_vec, KB_MATRIX).flatten()
    idxs = sims.argsort()[::-1][:topk]
    return [KB_CHUNKS[i] for i in idxs if sims[i] > 0.05]

# ------------------- Draft response generation -------------------
SYSTEM_PROMPT = """You are a professional, friendly customer support assistant.
- Be empathetic if the user is frustrated.
- Provide clear, concise, actionable steps.
- Reference any relevant product or order details found.
- Include any helpful information from the provided knowledge base context.
- Close with an offer to help further.
"""

def generate_draft_reply(email_row: Dict[str, Any]) -> str:
    context_chunks = retrieve_kb(email_row.get("body",""))
    context = "\n\n".join(context_chunks)
    sentiment = email_row.get("sentiment", "neutral")
    product = email_row.get("product_mentions", "")
    body = email_row.get("body","")
    subject = email_row.get("subject","")
    sender = email_row.get("sender","")

    if USE_OPENAI:
        try:
            prompt = f"""Subject: {subject}
From: {sender}

Email body:
\"\"\"
{body}
\"\"\"

Detected sentiment: {sentiment}
Product mentions: {product}

Knowledge base context:
\"\"\"
{context}
\"\"\"

Write a professional reply. Acknowledge any frustration. Include specific next steps.
"""
            # New-style Responses API (compatible with openai>=1.0.0)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":SYSTEM_PROMPT},
                    {"role":"user","content":prompt},
                ],
                temperature=0.4,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            # Fallback to template if API fails
            pass

    # Template fallback
    greeting = "Hi there,"
    if "@" in sender:
        name = sender.split("@")[0].replace("."," ").title()
        greeting = f"Hi {name},"
    empathy = "I'm sorry for the trouble you're facing" if sentiment=="negative" else "Thanks for reaching out"
    steps = "Could you share any error messages and the steps you took before the issue occurred?"

    kb_hint = f"\n\nHelpful context from our docs:\n{context}\n" if context else ""
    product_ref = f" regarding **{product}**" if product else ""
    closing = "Best regards,\nSupport Team"

    return f"""{greeting}

{empathy}.{product_ref} We’re here to help.
{steps} This will help us resolve it quickly.

Once we have this, we can investigate immediately and provide a fix or workaround.{kb_hint}

If this is urgent, please reply with 'urgent' and your phone number so we can call you back.

{closing}
"""

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="AI Communication Assistant", layout="wide")
st.title("📧 AI-Powered Communication Assistant")

init_db()

with st.sidebar:
    st.header("Settings")
    st.write("**OpenAI status:** " + ("✅ enabled" if USE_OPENAI else "❌ disabled (using template)"))
    default_path = "data/Sample_Support_Emails_Dataset.csv"
    st.caption("Upload a CSV with columns: Sender, Subject, Body, Date (or similar)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if st.button("Ingest CSV"):
        if uploaded:
            df = pd.read_csv(uploaded)
        elif os.path.exists(default_path):
            df = pd.read_csv(default_path)
        else:
            st.error("No CSV found. Upload one or place it in data/.")
            df = None

        if df is not None:
            # standardize columns
            cols_map = {c.lower(): c for c in df.columns}
            sender_col = cols_map.get("sender") or cols_map.get("from") or list(df.columns)[0]
            subject_col = cols_map.get("subject") or list(df.columns)[1]
            body_col = cols_map.get("body") or cols_map.get("message") or list(df.columns)[2]
            date_col = cols_map.get("date") or cols_map.get("received") or list(df.columns)[-1]

            rows = []
            for _, row in df.iterrows():
                subject = str(row.get(subject_col, ""))
                if not is_filtered_subject(subject):
                    continue
                body = str(row.get(body_col, ""))
                sender = str(row.get(sender_col, ""))
                received_at = str(row.get(date_col, ""))

                sentiment = simple_sentiment(subject + " " + body)
                priority = detect_priority(subject + " " + body)
                phone, alt_email, products = extract_info(body)

                rows.append({
                    "sender": sender,
                    "subject": subject,
                    "body": body,
                    "received_at": received_at,
                    "sentiment": sentiment,
                    "priority": priority,
                    "extracted_phone": phone,
                    "extracted_alt_email": alt_email,
                    "product_mentions": products,
                    "response": "",
                    "status": "pending"
                })
            if rows:
                insert_email_rows(rows)
                st.success(f"Ingested {len(rows)} filtered support emails.")
            else:
                st.warning("No emails matched the filter keywords in the subject.")

# Data view
df = fetch_emails(order_by_priority=True)

# --- Analytics
st.subheader("📊 Analytics (last 24h shown if timestamps available)")
col1, col2, col3, col4 = st.columns(4)
total = len(df)
resolved = int((df["status"] == "resolved").sum()) if not df.empty else 0
pending = total - resolved
neg = int((df["sentiment"] == "negative").sum()) if not df.empty else 0
urgent = int((df["priority"] == "urgent").sum()) if not df.empty else 0
col1.metric("Total Emails", total)
col2.metric("Resolved", resolved)
col3.metric("Pending", pending)
col4.metric("Urgent", urgent)

# --- Table of emails
st.subheader("📬 Filtered Support Emails")
if df.empty:
    st.info("No emails yet. Upload/ingest a CSV from the sidebar.")
else:
    st.dataframe(df[["id","sender","subject","received_at","sentiment","priority","status"]], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🧠 Process & Respond")
    for _, r in df.iterrows():
        with st.expander(f"#{r['id']} • {r['subject']}  —  {r['sender']}"):
            st.write("**Received:**", r["received_at"])
            st.write("**Sentiment:**", r["sentiment"], " • **Priority:**", r["priority"])
            st.write("**Body:**")
            st.write(r["body"])
            st.write("---")
            st.write("**Extracted Info**")
            st.write(f"Phone: {r['extracted_phone'] or '—'}")
            st.write(f"Alt. Email: {r['extracted_alt_email'] or '—'}")
            st.write(f"Products: {r['product_mentions'] or '—'}")

            if not r["response"]:
                if st.button(f"Generate Draft Reply #{r['id']}", key=f"gen_{r['id']}"):
                    draft = generate_draft_reply(r.to_dict())
                    update_email_response(int(r["id"]), draft, r["status"])
                    st.experimental_rerun()

            current_response = st.text_area("Draft Response", value=r["response"] or "", height=220, key=f"resp_{r['id']}")
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Save Draft", key=f"save_{r['id']}"):
                    update_email_response(int(r["id"]), current_response, r["status"])
                    st.success("Draft saved.")
            with c2:
                if st.button("Mark Resolved (Send)", key=f"send_{r['id']}"):
                    update_email_response(int(r["id"]), current_response, "resolved")
                    # simulate sending by appending to CSV
                    os.makedirs("output", exist_ok=True)
                    out_path = "output/sent_emails.csv"
                    row = {
                        "id": int(r["id"]),
                        "to": r["sender"],
                        "subject": f"Re: {r['subject']}",
                        "body": current_response,
                        "sent_at": dt.datetime.utcnow().isoformat() + "Z"
                    }
                    if os.path.exists(out_path):
                        odf = pd.read_csv(out_path)
                        odf = pd.concat([odf, pd.DataFrame([row])], ignore_index=True)
                        odf.to_csv(out_path, index=False)
                    else:
                        pd.DataFrame([row]).to_csv(out_path, index=False)
                    st.success("Response marked as sent (saved to output/sent_emails.csv).")
                    st.experimental_rerun()
            with c3:
                if st.button("Reset Status", key=f"reset_{r['id']}"):
                    update_email_response(int(r["id"]), current_response, "pending")
                    st.info("Status reset to pending.")
                    st.experimental_rerun()

st.caption("Tip: Urgent emails appear first. Use the sidebar to ingest a CSV again if you update data.")
