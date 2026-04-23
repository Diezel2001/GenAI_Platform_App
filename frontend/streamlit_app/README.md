# RAG Chatbot Frontend

A Streamlit-based chatbot frontend for the Agentic Generative AI Platform.

## Features

- **Interactive Chat Interface** - Chat with AI Agent about anything
- **Configurable Server Connection** - Set host and port via sidebar
- **GET/POST Request Support** - Full HTTP request capabilities with custom endpoint support
- **Request History** - Track all API calls with timestamps and status
- **Health Monitoring** - Real-time server status checking with visual indicators
- **Automatic Retry Logic** - Exponential backoff for failed requests
- **Custom API Requests** - Advanced section for arbitrary GET/POST requests
- **Chat Controls** - Clear chat history and manage conversation state

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

1. **Server Settings (Sidebar)**:
   - Host: Server hostname (default: localhost)
   - Port: Server port (default: 8000)
   - Real-time server URL display

2. **Quick Actions**:
   - Health Check (GET /)
   - Query Health (GET /query/health)
   - Check Server Status button

3. **Chat Interface**:
   - Type your message in the chat input
   - The app will query the backend Agent API
   - Responses displayed with metadata details (expandable)

4. **Chat Controls**:
   - Clear Chat History button to reset conversation

5. **Advanced: Custom API Request**:
   - Make arbitrary GET/POST requests to any endpoint
   - JSON body editor for POST requests
   - Automatic retry with exponential backoff

### Advanced Usage

Use the Custom API Request section to make arbitrary GET/POST requests:

**GET Example:**
```
Endpoint: /query/health
```

**POST Example:**
```
Endpoint: /agent/
Body: {"message": "your question", "k": 5}
```

## API Endpoints

The frontend connects to these backend endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/query/health` | Query service health |
| POST | `/agent/` | Agent chat endpoint (main) |

### Agent Endpoint Details

**Request Body:**
```json
{
  "message": "string (1-5000 chars)",
  "k": "integer (1-100, default: 5)"
}
```

**Response:**
```json
{
  "message": "original message",
  "results": "agent response text"
}
```

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
4. Try clicking "Check Server Status" to verify connectivity

### Timeout Errors

If requests timeout:
1. Check server load
2. The app automatically retries with exponential backoff
3. Try simplifying your question
4. Consider increasing timeout in app.py if needed

### Invalid JSON Errors

If you see JSON parsing errors in Custom API Request:
1. Verify your JSON body is properly formatted
2. Use double quotes for strings
3. Check for trailing commas

## License

Part of the Agentic Generative AI Platform.