# 🚧 GenAI Platform App — Work in Progress

> ⚠️ This project is actively under development. Features and architecture may change.

An **Agentic Generative AI Platform** for developing, testing, and deploying AI agents with RAG (Retrieval-Augmented Generation) capabilities. The platform consists of a **FastAPI backend** and a **Streamlit chat interface frontend**.

---

## 🏗️ Project Structure

```
GenAI_Platform_App/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── api/               # REST API endpoints (agent, documents, query)
│   │   ├── core/              # Core logic (workflows, prompts)
│   │   └── services/          # LLM, RAG, vector store services
│   └── tests/                 # Unit tests
├── frontend/
│   └── streamlit_app/         # Streamlit chat interface
├── docker-compose.yaml        # Docker orchestration
├── requirements.txt           # Python dependencies
└── setup.sql                  # Database setup
```

---

## 🛠️ Tech Stack

**Backend:**
- FastAPI + Uvicorn
- LangChain / LangGraph
- Vector stores: FAISS, Pinecone, Qdrant, Milvus
- Redis, PostgreSQL
- (future):
    - langflow, Prometheus, grafana

**Frontend:**
- Streamlit (chat UI)

**Dependencies:** See `requirements.txt`

---

## 🚀 Quick Start

### Prerequisite
```bash
pip install -r ../requirements.txt
```

### Backend
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend/streamlit_app
streamlit run app.py
```

### Docker (will be iplemented in the future)
```bash
docker-compose up --build
```

---

## 📸 Streamlit App Screenshots

![Sample Screen](Sample_Screen.png)

---

## 📋 Todo

- improve observability
    - Langfuse → tracing
    - Prometheus → metrics
    - Grafana → dashboard
- Agent Memory Management 
    - current chat session context memory
    - Episodic memory(daily and per session log)
    - Semmantic (user info)
- agent tools
    - memory retrieval (RAG, cached memory search)
    - context compaction
    - other usefull tools
- research about:
    - LlamaIndex – structured data + retrieval
    - DSPy – declarative prompting + optimization

---

## 📄 License

_TBD_