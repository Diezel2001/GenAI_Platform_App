"""
Streamlit Chatbot Frontend for RAG Platform
============================================
A chatbot interface with GET/POST request capabilities to communicate with the backend server.
"""

import streamlit as st
import requests
import json
from datetime import datetime
from typing import Optional, Dict, Any

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------
# Custom CSS
# ----------------------------
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .server-status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .server-status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
    .server-status-unknown {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Session State Initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "server_host" not in st.session_state:
    st.session_state.server_host = "localhost"

if "server_port" not in st.session_state:
    st.session_state.server_port = 8000

if "server_status" not in st.session_state:
    st.session_state.server_status = "unknown"

if "request_history" not in st.session_state:
    st.session_state.request_history = []


# ----------------------------
# Helper Functions
# ----------------------------
def get_base_url() -> str:
    """Construct the base URL from host and port."""
    return f"http://{st.session_state.server_host}:{st.session_state.server_port}"


def make_get_request(endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """Make a GET request to the server."""
    url = f"{get_base_url()}{endpoint}"
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return {
            "success": True,
            "status_code": response.status_code,
            "data": response.json(),
            "url": url
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Connection failed - server may be down",
            "url": url
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out",
            "url": url
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "url": url
        }


def make_post_request(endpoint: str, data: Dict[str, Any], params: Optional[Dict] = None) -> Dict[str, Any]:
    """Make a POST request to the server."""
    url = f"{get_base_url()}{endpoint}"
    try:
        response = requests.post(
            url,
            json=data,
            params=params,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        response.raise_for_status()
        return {
            "success": True,
            "status_code": response.status_code,
            "data": response.json(),
            "url": url
        }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "error": "Connection failed - server may be down",
            "url": url
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Request timed out",
            "url": url
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "url": url
        }


def check_server_health() -> str:
    """Check if the server is healthy."""
    result = make_get_request("/")
    if result.get("success"):
        st.session_state.server_status = "healthy"
        return "healthy"
    else:
        st.session_state.server_status = "unhealthy"
        return "unhealthy"


def query_documents(query: str, k: int = 5, use_rag: bool = False) -> Dict[str, Any]:
    """Query the document search endpoint."""
    if use_rag:
        endpoint = "/query/rag"
        data = {
            "query": query,
            "k": k,
            "include_llm_response": True
        }
    else:
        endpoint = "/query/"
        data = {
            "query": query,
            "k": k
        }
    
    return make_post_request(endpoint, data)


def add_to_request_history(method: str, endpoint: str, result: Dict[str, Any]):
    """Add request to history."""
    st.session_state.request_history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": method,
        "endpoint": endpoint,
        "success": result.get("success", False),
        "status_code": result.get("status_code", "N/A")
    })
    # Keep only last 20 requests
    if len(st.session_state.request_history) > 20:
        st.session_state.request_history.pop(0)


# ----------------------------
# Sidebar - Server Configuration
# ----------------------------
with st.sidebar:
    st.title("⚙️ Server Configuration")
    
    st.divider()
    
    # Server Settings
    st.subheader("Connection Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        new_host = st.text_input(
            "Host",
            value=st.session_state.server_host,
            help="Server hostname or IP address"
        )
    with col2:
        new_port = st.number_input(
            "Port",
            value=st.session_state.server_port,
            min_value=1,
            max_value=65535,
            help="Server port number"
        )
    
    # Update session state if changed
    if new_host != st.session_state.server_host:
        st.session_state.server_host = new_host
    if new_port != st.session_state.server_port:
        st.session_state.server_port = new_port
    
    st.caption(f"📡 Server URL: {get_base_url()}")
    
    # Health Check Button
    if st.button("🔍 Check Server Status", use_container_width=True):
        with st.spinner("Checking server..."):
            status = check_server_health()
            if status == "healthy":
                st.success("✅ Server is healthy!")
            else:
                st.error("❌ Server is unreachable")
    
    # Display current status
    status_class = f"server-status-{st.session_state.server_status}"
    status_emoji = {"healthy": "🟢", "unhealthy": "🔴", "unknown": "⚪"}
    st.markdown(
        f'Status: <span class="{status_class}">'
        f'{status_emoji.get(st.session_state.server_status, "⚪")} '
        f'{st.session_state.server_status.upper()}</span>',
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Quick Actions
    st.subheader("Quick Actions")
    
    if st.button("🏥 Health Check (GET /)", use_container_width=True):
        result = make_get_request("/")
        add_to_request_history("GET", "/", result)
        if result.get("success"):
            st.json(result["data"])
        else:
            st.error(result.get("error", "Unknown error"))
    
    if st.button("📊 Query Health (GET /query/health)", use_container_width=True):
        result = make_get_request("/query/health")
        add_to_request_history("GET", "/query/health", result)
        if result.get("success"):
            st.json(result["data"])
        else:
            st.error(result.get("error", "Unknown error"))
    
    st.divider()
    
    # Request History
    st.subheader("📜 Request History")
    if st.session_state.request_history:
        for req in reversed(st.session_state.request_history[-5:]):
            status_icon = "✅" if req["success"] else "❌"
            st.caption(f"{status_icon} [{req['timestamp']}] {req['method']} {req['endpoint']}")
    else:
        st.caption("No requests yet")
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.request_history = []
        st.rerun()
    
    st.divider()
    
    # Chat Controls
    st.subheader("💬 Chat Controls")
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------
# Main Chat Area
# ----------------------------
st.title("🤖 RAG Chatbot")
st.caption("Ask questions about your documents using semantic search")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show metadata if available
        if "metadata" in message:
            with st.expander("📋 Response Details"):
                st.json(message["metadata"])


# ----------------------------
# Chat Input
# ----------------------------
user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process the query
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            # Query the backend
            result = query_documents(user_input, k=5, use_rag=True)
            add_to_request_history("POST", "/query/rag", result)
            
            if result.get("success"):
                data = result["data"]
                
                # Build response
                response_parts = []
                
                # Add search results
                if data.get("results"):
                    response_parts.append(f"**Found {data['total_results']} relevant results:**\n")
                    
                    for i, res in enumerate(data["results"], 1):
                        score = res.get("score", 0)
                        text = res.get("text", "No text available")
                        # Truncate long texts
                        if len(text) > 300:
                            text = text[:300] + "..."
                        response_parts.append(f"**Result {i}** (Score: {score:.4f}):\n{text}\n")
                
                # Add LLM response if available
                if data.get("llm_response"):
                    response_parts.append(f"\n**AI Response:**\n{data['llm_response']}")
                
                if not response_parts:
                    response_parts.append("No results found for your query.")
                
                assistant_reply = "\n".join(response_parts)
                
                st.markdown(assistant_reply)
                
                # Store message with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_reply,
                    "metadata": {
                        "query": data.get("query"),
                        "total_results": data.get("total_results"),
                        "vector_store": data.get("vector_store"),
                        "server_url": result.get("url")
                    }
                })
            else:
                error_msg = f"❌ **Error:** {result.get('error', 'Unknown error')}\n\n"
                error_msg += f"Server URL: {result.get('url', 'N/A')}\n\n"
                error_msg += "Please check your server configuration in the sidebar."
                
                st.error(error_msg)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })


# ----------------------------
# Custom Request Section (Expandable)
# ----------------------------
with st.expander("🔧 Advanced: Custom API Request"):
    st.subheader("Make Custom Requests")
    
    request_type = st.selectbox("Request Type", ["GET", "POST"])
    
    endpoint = st.text_input(
        "Endpoint",
        value="/",
        help="API endpoint path (e.g., /query/health)"
    )
    
    if request_type == "POST":
        st.caption("Request Body (JSON):")
        request_body = st.text_area(
            "Body",
            value='{"query": "example", "k": 5}',
            height=150,
            label_visibility="collapsed"
        )
    
    if st.button("📤 Send Request", use_container_width=True):
        with st.spinner("Sending request..."):
            if request_type == "GET":
                result = make_get_request(endpoint)
                add_to_request_history("GET", endpoint, result)
            else:
                try:
                    body = json.loads(request_body)
                    result = make_post_request(endpoint, body)
                    add_to_request_history("POST", endpoint, result)
                except json.JSONDecodeError:
                    st.error("Invalid JSON in request body")
                    result = None
            
            if result:
                if result.get("success"):
                    st.success(f"✅ Success (Status: {result.get('status_code')})")
                    st.json(result["data"])
                else:
                    st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")


# ----------------------------
# Footer
# ----------------------------
st.divider()
st.caption(
    "🔗 Connected to: " + get_base_url() + " | "
    "Built with Streamlit for the Agentic Generative AI Platform"
)
