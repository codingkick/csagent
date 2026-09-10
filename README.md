# Hiver AI Support Agent — FastAPI Boilerplate

FastAPI microservice boilerplate for handling customer support messages, classification, RAG reply drafting, and escalation routing.

## 📁 Directory Structure

```
hiver/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application & API endpoints
│   ├── schemas.py       # Pydantic request/response data models
│   └── agent.py         # Support agent pipeline logic (Intent + RAG + Escalation)
├── requirements.txt     # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run FastAPI Server

```bash
uvicorn app.main:app --reload --port 8000
```

The API documentation will be available at:
* **Interactive OpenAPI Docs (Swagger UI):** `http://localhost:8000/docs`
* **Alternative Redoc:** `http://localhost:8000/redoc`

---

## 📡 API Endpoints

### 1. Health Check
`GET /health`

**Response:**
```json
{
  "status": "ok",
  "service": "support-agent-api"
}
```

---

### 2. Process Customer Message
`POST /api/v1/message`

**Request Payload:**
```json
{
  "user_id": "user_456",
  "tweet_id": "tweet_7890",
  "message": "Where is my order #12345? It's delayed by 3 days.",
  "brand": "AppleSupport"
}
```

**Response Payload:**
```json
{
  "intent": "order_status",
  "intent_confidence": 0.94,
  "drafted_reply": "Hello! Thank you for reaching out to @AppleSupport. Regarding your query ('Where is my order #12345? It's delayed by 3 days.'), we're looking into this right away.",
  "auto_handled": true,
  "escalation_reason": null,
  "retrieved_context_ids": [
    "tweet_hist_001",
    "tweet_hist_002"
  ]
}
```

---

### 3. Stream Drafted Reply (Server-Sent Events)
`POST /api/v1/message/stream`

Streams drafted response tokens directly to web/client interfaces.
