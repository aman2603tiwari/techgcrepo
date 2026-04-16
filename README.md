# Multi-Agent Conference Planning System

Production-grade **multi-agent AI system** that autonomously generates end-to-end conference plans (sponsors, speakers, venues, pricing, and GTM strategy) from minimal inputs.

Built for **High Prep PS1 — Pinch × IIT Roorkee (TGC 2026)**.

---

## 🚀 Overview

This system takes **3 inputs**:
- Event Category  
- Geography  
- Audience Size  

…and outputs a **complete conference plan** using a coordinated multi-agent pipeline.

### Problems Solved
- Fragmented event data
- Manual sponsor/speaker discovery
- Unstructured pricing & planning

---

## 🧠 Architecture

Sequential **multi-agent pipeline** orchestrated using LangGraph:

```
User Input (Streamlit UI)
        ↓
LangGraph StateGraph
        ↓
Sponsor → Speaker → Venue → Pricing → GTM
        ↓
RAG (Qdrant) + Web Search (Tavily)
        ↓
Structured JSON → UI + Chatbot
```

- Shared **state object (TypedDict)** ensures safe data flow  
- Each agent writes only to its own output key  

---

## 🤖 Agents

| Agent     | Output |
|----------|--------|
| Sponsor  | Sponsor recommendations + funding estimates |
| Speaker  | Speaker/artist suggestions |
| Venue    | Venue shortlist with cost & capacity |
| Pricing  | Ticket pricing + revenue model |
| GTM      | Marketing & go-to-market strategy |

**All agents:**
- Use **RAG + live web search**
- Generate **strict JSON outputs**
- Run with **temperature = 0 (deterministic)**

---

## 🧰 Tech Stack

- **LangGraph (StateGraph)** – orchestration  
- **Groq LLaMA-3.3-70B** – LLM  
- **Qdrant** – vector DB (RAG)  
- **Tavily API** – web search  
- **Sentence Transformers (MiniLM)** – embeddings  
- **Streamlit** – UI  
- **Python 3.14**

---

## 📊 Data Layer

- **1,000+ events dataset (2025–2026)**
- Domains:
  - Conferences
  - Music Festivals
  - Sporting Events
- Regions:
  - India
  - USA
  - Europe
  - Singapore

### Dataset Fields
- Event metadata (name, location, year)
- Pricing tiers
- Sponsors & speakers
- Attendance & revenue

---

## 🔎 RAG System

- Local **Qdrant vector store**
- Query:
  ```
  "{category} conference {geography}"
  ```
- Retrieves **top-3 similar past events**

### Features
- Persistent storage  
- Smart reload via **MD5 hash**  
- API compatibility fallback  

---

## 🌐 Data Collection Pipeline

Multi-platform scraping system:

- Eventbrite  
- Cvent  
- Luma  
- 10times  
- ConcertArchives (custom JS engine)

### Features
- Deduplication + validation  
- Anti-bot protection  
- Dynamic content scraping (Playwright + JS)  

---

## 💻 User Interface

Built with **Streamlit**:

- Two-column layout:
  - Left → Inputs + outputs  
  - Right → AI chatbot  

### Features
- Real-time agent streaming  
- Tabbed structured outputs  
- Context-aware chatbot  
- Auto-generated suggestion chips  

---

## 📁 Project Structure

```
main.py          # CLI entry
app.py           # Streamlit UI
graph.py         # LangGraph pipeline
state.py         # Shared state schema

agents/
  sponsor.py
  speaker.py
  venue.py
  pricing.py
  gtm.py

tools/
  search.py      # RAG + Tavily + JSON parsing
  llm.py         # LLM wrapper

data/
  events.csv     # dataset

scrapers/        # data pipeline

qdrant_db/       # vector store
```

---

## ⚙️ Setup

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
```

Add:
```
GROQ_API_KEY=
TAVILY_API_KEY=
```

### 3. Run

**CLI**
```bash
python main.py
```

**UI**
```bash
streamlit run app.py
```

---

## ⚠️ Limitations

- Sequential agents → latency (~30–60s)  
- Static RAG dataset (no live ingestion yet)  
- Exhibitor & Ops agents not implemented  
- LinkedIn scraping requires authentication  

---

## 🔮 Future Improvements

- Parallel agent execution  
- Outreach email generation agent  
- Pricing simulation dashboard  
- Persistent memory across sessions  
- Visual agent execution graph  

---

## 🏆 Highlights

- End-to-end automated conference planning  
- Production-grade multi-agent architecture  
- Robust JSON parsing fallback system  
- Large-scale structured dataset + scraping pipeline  

---
