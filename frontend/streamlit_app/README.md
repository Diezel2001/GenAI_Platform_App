# RAG Chatbot Frontend

A Streamlit-based chatbot frontend for the Agentic Generative AI Platform.

## Features

- Interactive Chat Interface - Ask questions about your documents
- Configurable Server Connection - Set host and port via sidebar
- GET/POST Request Support - Full HTTP request capabilities
- Request History - Track all API calls
- Health Monitoring - Check server status
- Semantic Search - Query documents using RAG pipeline

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit requests
```

## Usage

### Start the Frontend

```bash
cd frontend/streamlit_app
streamlit run app.py
```

The app will be available at: http://localhost:8501

### Configuration

1. Server Settings (Sidebar):
   - Host: Server hostname (default: localhost)
   - Port: Server port (default: 8000)

2. Quick Actions:
   - Health Check (GET /)
   - Query Health (GET /query/health)

3. Chat:
   - Type your question in the chat input
   - The app will query the backend RAG API
   - Results are displayed with relevance scores

### Advanced Usage

Use the Custom API Request section to make arbitrary GET/POST requests:

GET /query/health
POST /query/rag
Body: {"query": "your question", "k": 5}

## API Endpoints

The frontend connects to these backend endpoints:

- GET / - Health check
- GET /query/health - Query service health
- POST /query/ - Semantic document search
- POST /query/rag - RAG query with LLM response

## Running with Backend

### Option 1: Local Development

Terminal 1 - Start Backend:
```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Terminal 2 - Start Frontend:
```bash
cd frontend/streamlit_app
streamlit run app.py
```

### Option 2: Docker Compose

```bash
docker-compose up
```

## Project Structure

```
frontend/streamlit_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Troubleshooting

### Connection Refused

If you see "Connection failed - server may be down":
1. Ensure the backend server is running
2. Check the host/port settings in the sidebar
3. Verify firewall settings

### Timeout Errors

If requests timeout:
1. Check server load
2. Increase timeout in app.py if needed
3. Verify network connectivity

## License

Part of the Agentic Generative AI Platform.
