import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="AI AGENT_00",
    layout="wide"
)

# ----------------------------
# Session state initialization
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pdf_files" not in st.session_state:
    st.session_state.pdf_files = []

# ----------------------------
# SIDEBAR (LEFT)
# ----------------------------
with st.sidebar:
    st.title("Document List")

    st.subheader("Loaded PDFs")
    if st.session_state.pdf_files:
        for pdf in st.session_state.pdf_files:
            st.write("•", pdf["name"])
    else:
        st.caption("No PDFs uploaded")

    st.divider()

    uploaded_files = st.file_uploader(
        "Drag PDF files here!",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in [f["name"] for f in st.session_state.pdf_files]:
                st.session_state.pdf_files.append({
                    "name": file.name,
                    "file": file
                })

# ----------------------------
# MAIN CHAT AREA
# ----------------------------
st.title("Chat with AGENT_00")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask something about your PDFs...")

if user_input:
    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # ----------------------------
    # LLM RESPONSE (PLACEHOLDER)
    # LLM workflow
    # ----------------------------
    assistant_reply = (
        "This is a placeholder response.\n\n"
        "You asked:\n\n"
        f"> {user_input}"
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply
    })

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)
