"""
Streamlit Chatbot Frontend for Agentic Generative AI Platform
"""

import streamlit as st
import requests
import json
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI AGENT CHATBOT",
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

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


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


def make_get_request_with_retry(endpoint: str, params: Optional[Dict] = None, max_retries: int = 3, backoff_factor: float = 1.0) -> Dict[str, Any]:
    """Make a GET request with automatic retry logic."""
    last_result = None
    
    for attempt in range(max_retries):
        if attempt > 0:
            # Show retry status
            st.info(f"🔄 Retry attempt {attempt + 1}/{max_retries}...")
            time.sleep(backoff_factor * attempt)  # Exponential backoff
        
        last_result = make_get_request(endpoint, params)
        
        if last_result.get("success"):
            if attempt > 0:
                st.success("✅ Request succeeded after retry!")
            return last_result
        
        # Don't retry certain errors
        error = last_result.get("error", "")
        if "Invalid JSON" in error or "404" in error or "400" in error:
            # These errors won't be fixed by retrying
            break
    
    return last_result


def make_post_request_with_retry(endpoint: str, data: Dict[str, Any], params: Optional[Dict] = None, max_retries: int = 3, backoff_factor: float = 1.0) -> Dict[str, Any]:
    """Make a POST request with automatic retry logic."""
    last_result = None
    
    for attempt in range(max_retries):
        if attempt > 0:
            # Show retry status
            st.info(f"🔄 Retry attempt {attempt + 1}/{max_retries}...")
            time.sleep(backoff_factor * attempt)  # Exponential backoff
        
        last_result = make_post_request(endpoint, data, params)
        
        if last_result.get("success"):
            if attempt > 0:
                st.success("✅ Request succeeded after retry!")
            return last_result
        
        # Don't retry certain errors
        error = last_result.get("error", "")
        if "Invalid JSON" in error or "404" in error or "400" in error:
            # These errors won't be fixed by retrying
            break
    
    return last_result


def check_server_health() -> str:
    """Check if the server is healthy."""
    result = make_get_request("/")
    if result.get("success"):
        st.session_state.server_status = "healthy"
        return "healthy"
    else:
        st.session_state.server_status = "unhealthy"
        return "unhealthy"


def ask_agent(message: str, k: int = 5) -> Dict[str, Any]:
    """Ask the agent endpoint."""
    endpoint = "/agent/"
    data = {
        "message": message,
        "k": k,
        "session_id": st.session_state.session_id
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
st.title("🤖 AI AGENT CHATBOT")
st.caption("Chat with your agent")

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
user_input = st.chat_input("Enter message for Agent")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Process the query with retry logic
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Ask the agent with retry logic
            result = make_post_request_with_retry(
                "/agent/",
                {"message": user_input, "k": 5, "session_id": st.session_state.session_id}
            )
            add_to_request_history("POST", "/agent/", result)
            
            if result.get("success"):
                data = result["data"]
                
                # Format the response as: analysis, route, result
                analysis = data.get("analysis", "N/A")
                route = data.get("route", "N/A")
                results = data.get("results", "No response from agent.")
                
                assistant_reply = f"Analysis: {analysis}\n\nRoute: {route}\n\nResult: {results}"
                
                st.markdown(assistant_reply)
                
                # Store message with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_reply,
                    "metadata": {
                        "query": data.get("message"),
                        "server_url": result.get("url")
                    }
                })
            else:
                # Provide more helpful error messages based on error type
                error = result.get("error", "Unknown error")
                error_msg = f"❌ **Error:** {error}\n\n"
                error_msg += f"Server URL: {result.get('url', 'N/A')}\n\n"
                
                # Add specific guidance based on error type
                if "Connection failed" in error:
                    error_msg += "💡 **Suggestions:**\n"
                    error_msg += "1. Make sure the backend server is running\n"
                    error_msg += "2. Check your host and port settings in the sidebar\n"
                    error_msg += "3. Try clicking 'Check Server Status' to verify connectivity"
                elif "timed out" in error:
                    error_msg += "💡 **Suggestions:**\n"
                    error_msg += "1. The server might be under heavy load\n"
                    error_msg += "2. Try again in a moment\n"
                    error_msg += "3. Consider reducing the complexity of your question"
                else:
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
                result = make_get_request_with_retry(endpoint)
                add_to_request_history("GET", endpoint, result)
            else:
                try:
                    body = json.loads(request_body)
                    result = make_post_request_with_retry(endpoint, body)
                    add_to_request_history("POST", endpoint, result)
                except json.JSONDecodeError as e:
                    st.error(f"❌ Invalid JSON in request body: {e}")
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
