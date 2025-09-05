# AI-Powered Communication Assistant (Streamlit + Python)

A fast, end-to-end solution that ingests support emails (from CSV), filters and prioritizes them,
extracts key information, performs sentiment analysis, retrieves context from a small FAQ
knowledge base (RAG), generates draft replies with an LLM (OpenAI optional), and presents
everything on a clean dashboard.

> Use this as your Coding Assessment Challenge submission. Just run locally and push this repo.

## ✨ Features
- Email ingestion from CSV (simulates inbox)
- Filtering by subject keywords (Support, Help, Query, Request)
- Priority classification (urgent / not urgent) via keyword rules
- Sentiment analysis (VADER: positive / neutral / negative)
- Information extraction (phone, alternate email, product mentions)
- RAG over a mini FAQ knowledge base (TF‑IDF retrieval)
- Draft AI responses (OpenAI GPT if key present; otherwise smart template)
- Dashboard with analytics and editable/savable responses
- Export of “sent” replies to `output/sent_emails.csv` (SMTP optional hook)

## 🧱 Tech Stack
- Python 3.9+
- Streamlit
- SQLite (via `sqlite3` – auto-created if needed)
- scikit-learn (TF‑IDF), nltk VADER for sentiment
- Optional: OpenAI for LLM responses

## ✅ Quick Start
1. **Install software**  
   - Python 3.9 or newer (recommend 3.10 / 3.11)  
   - Git  
   - VS Code (optional but recommended)

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place your dataset**
   - Put your CSV (e.g., `Sample_Support_Emails_Dataset.csv`) in `data/`
   - Or upload it from within the app

5. **(Optional) OpenAI API key**
   - Copy `.env.example` to `.env` and place your key
   - Or set environment variable in your shell:
     ```bash
     setx OPENAI_API_KEY "YOUR_KEY"         # Windows PowerShell
     export OPENAI_API_KEY="YOUR_KEY"       # macOS/Linux
     ```

6. **Run the app**
   ```bash
   streamlit run app.py
   ```

7. **Open in browser**
   - Streamlit will print a local URL (e.g., `http://localhost:8501`)

## 🗂 Repo Structure
```
AI-Communication-Assistant/
├─ app.py
├─ requirements.txt
├─ README.md
├─ kb/
│  └─ faqs.md
├─ data/
│  └─ (place your CSV here)
├─ output/
│  └─ sent_emails.csv (auto-created after first send)
└─ .env.example
```

## 🧠 RAG Notes
We do a simple TF‑IDF retrieval over small FAQ entries. The top‑k most relevant chunks are passed
into the prompt (or the template if OpenAI is not configured). This keeps things lightweight and
robust for a hackathon setting.

## 🚀 What to Submit
- Push this entire folder to a **GitHub repository**
- Add a short **demo video link** (screen recording) to the README
- Include brief **architecture notes** (this README is a good start)

---

**Author**: BYPUREDDY HARI RAJIV
**License**: MIT
