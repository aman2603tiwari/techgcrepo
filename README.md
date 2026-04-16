# Conference Planner — Multi-Agent AI System

## Setup

```bash
cd conference_agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# fill in GROQ_API_KEY and TAVILY_API_KEY
```

## Run CLI
```bash
python main.py
```

## Run Streamlit UI
```bash
streamlit run app.py
```

---

## data/events.csv — Column Reference (for Person 2)

| Column | Type | Example | Notes |
|--------|------|---------|-------|
| event_name | string | `PyCon India 2024` | Full official name |
| category | string | `AI/ML` | Must match one of: AI/ML, Web3 / Blockchain, ClimateTech, Fintech, DevOps / Cloud, Cybersecurity, Edtech, Healthtech |
| city | string | `Bangalore` | City only (no country) |
| country | string | `India` | |
| year | integer | `2024` | |
| venue_name | string | `NIMHANS Convention Centre` | Full venue name |
| attendance | integer | `1200` | Actual/reported attendance |
| duration_days | integer | `2` | |
| ticket_price_early_bird | integer | `800` | INR, 0 if free |
| ticket_price_standard | integer | `1200` | INR |
| ticket_price_vip | integer | `3500` | INR |
| sponsors | string | `Google\|AWS\|Microsoft` | Pipe-separated company names |
| speakers | string | `Guido van Rossum\|Armin Ronacher` | Pipe-separated full names |
| communities_promoted | string | `Python India Discord\|LinkedIn` | Pipe-separated |
| total_revenue_inr | integer | `3200000` | Estimated total (tickets + sponsorships) |

**Target**: 50-100 rows covering events from 2022-2025 across India + Singapore + global.

**Sources to scrape**: 10times.com, Luma.com, Hasgeek.com, Devfolio.co, NASSCOM.in, individual event websites.
