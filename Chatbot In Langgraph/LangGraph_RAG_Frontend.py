
# LangGraph_RAG_Frontend.py

import queue
import uuid

import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from LangGraph_RAG_Backend import (
    chatbot,
    retrieve_all_threads,
    retrieve_chat_history,
    retrieve_chat_title,
    save_chat_title,
    submit_async_task,
    ingest_pdf,
    thread_has_document,
    thread_document_metadata,
    thread_config,
)


# ==========================================================
# 1. Page configuration
# ==========================================================

st.set_page_config(
    page_title="LangGraph RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LangGraph RAG + MCP Chatbot")

st.caption(
    "Powered by LangGraph, Ollama, FAISS, SQLite, and MCP tools."
)


# ==========================================================
# 2. Chat utilities
# ==========================================================

def generate_thread_id():
    return str(uuid.uuid4())


def generate_chat_title(text):
    title = " ".join(text.strip().split())

    if len(title) > 38:
        return title[:35] + "..."

    return title or "New Chat"


def add_thread(thread_id):
    thread_id = str(thread_id)

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []
    st.session_state["title_initialized"] = False

    # Reset uploader state for the new conversation.
    st.session_state["pdf_upload_key"] += 1

    add_thread(thread_id)


def format_message(message):
    """Convert a LangChain message into a Streamlit chat message."""

    if isinstance(message, HumanMessage):
        role = "user"

    elif isinstance(message, AIMessage):
        # Don't display internal tool-call-only messages.
        if not message.content:
            return None

        role = "assistant"

    else:
        # Tool results are handled by the graph and not displayed
        # as independent assistant messages.
        return None

    content = message.content

    if isinstance(content, list):
        parts = []

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))

        content = "\n".join(parts)

    if not isinstance(content, str):
        content = str(content)

    return {
        "role": role,
        "content": content,
    }


def load_conversation(thread_id):
    """Load a conversation from the LangGraph SQLite checkpoint."""

    try:
        messages = retrieve_chat_history(str(thread_id))

        formatted_messages = []

        for message in messages:
            formatted = format_message(message)

            if formatted is not None:
                formatted_messages.append(formatted)

        st.session_state["message_history"] = formatted_messages
        st.session_state["thread_id"] = str(thread_id)
        st.session_state["title_initialized"] = True

        # Clear any previously selected file from the uploader.
        st.session_state["pdf_upload_key"] += 1

    except Exception as exc:
        st.error(f"Unable to load conversation: {exc}")


# ==========================================================
# 3. Session initialization
# ==========================================================

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "title_initialized" not in st.session_state:
    st.session_state["title_initialized"] = False

if "pdf_upload_key" not in st.session_state:
    st.session_state["pdf_upload_key"] = 0

if "chat_threads" not in st.session_state:
    try:
        st.session_state["chat_threads"] = [
            str(thread_id)
            for thread_id in retrieve_all_threads()
        ]

    except Exception as exc:
        st.error(f"Could not retrieve chat history: {exc}")
        st.session_state["chat_threads"] = []

add_thread(st.session_state["thread_id"])


# ==========================================================
# 4. Sidebar
# ==========================================================

st.sidebar.title("💬 RAG + MCP Chatbot")

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True,
):
    reset_chat()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("My Conversations")

for thread_id in reversed(st.session_state["chat_threads"]):
    title = retrieve_chat_title(thread_id)

    if thread_id == st.session_state["thread_id"]:
        label = f"🟢 {title}"
    else:
        label = f"💬 {title}"

    if st.sidebar.button(
        label,
        key=f"conversation_{thread_id}",
        use_container_width=True,
    ):
        load_conversation(thread_id)
        st.rerun()


# ==========================================================
# 5. PDF uploader and RAG indexing
# ==========================================================

current_thread_id = str(st.session_state["thread_id"])

st.sidebar.divider()
st.sidebar.subheader("📄 PDF Knowledge Base")

if thread_has_document(current_thread_id):
    metadata = thread_document_metadata(current_thread_id)

    st.sidebar.success(
        f"Indexed: {metadata.get('filename', 'PDF')}"
    )

    st.sidebar.caption(
        f"Pages: {metadata.get('documents', 0)} | "
        f"Chunks: {metadata.get('chunks', 0)}"
    )

else:
    st.sidebar.info(
        "No PDF has been indexed for this conversation."
    )

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    key=f"pdf_uploader_{st.session_state['pdf_upload_key']}",
)

if uploaded_file is not None:
    st.sidebar.caption(f"Selected: {uploaded_file.name}")

    if st.sidebar.button(
        "📚 Index PDF",
        use_container_width=True,
        key=f"index_pdf_{current_thread_id}",
    ):
        try:
            with st.spinner(
                "Reading PDF and creating the FAISS index..."
            ):
                result = ingest_pdf(
                    file_bytes=uploaded_file.getvalue(),
                    thread_id=current_thread_id,
                    filename=uploaded_file.name,
                )

            st.sidebar.success(
                f"Indexed {result['filename']} successfully!"
            )

            st.sidebar.write(
                f"Pages: {result['documents']}"
            )

            st.sidebar.write(
                f"Text chunks: {result['chunks']}"
            )

        except Exception as exc:
            st.sidebar.error(
                f"PDF indexing failed: {exc}"
            )


# ==========================================================
# 6. Render existing conversation
# ==========================================================

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================================
# 7. Chat input
# ==========================================================

user_input = st.chat_input("Ask me anything...")


if user_input:
    thread_id = str(st.session_state["thread_id"])

    # Display and store the user's message in the UI.
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate a title for the first message.
    if not st.session_state["title_initialized"]:
        save_chat_title(
            thread_id,
            generate_chat_title(user_input),
        )

        st.session_state["title_initialized"] = True

    # The same thread ID is used for graph state and PDF retrieval.
    config = thread_config(thread_id)

    event_queue = queue.Queue()
    full_response = []
    status_holder = {"box": None}

    # ------------------------------------------------------
    # 8. Run LangGraph asynchronously
    # ------------------------------------------------------

    async def run_stream():
        try:
            async for message_chunk, metadata in chatbot.astream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=config,
                stream_mode="messages",
            ):
                event_queue.put((
                    "message",
                    message_chunk,
                    metadata,
                ))

        except Exception as exc:
            event_queue.put(("error", exc, None))

        finally:
            event_queue.put(("done", None, None))

    # ------------------------------------------------------
    # 9. Stream the response
    # ------------------------------------------------------

    with st.chat_message("assistant"):

        def response_stream():
            # Schedule execution on the backend event loop.
            submit_async_task(run_stream())

            while True:
                event_type, chunk, metadata = event_queue.get()

                if event_type == "done":
                    break

                if event_type == "error":
                    error_text = (
                        f"\n\n⚠️ Chat execution failed: {chunk}"
                    )

                    full_response.append(error_text)
                    yield error_text
                    continue

                message_chunk = chunk

                # Display tool execution status when a tool result
                # is emitted by the graph.
                if isinstance(message_chunk, ToolMessage):
                    tool_name = (
                        message_chunk.name or "Tool"
                    )

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Executing {tool_name}...",
                            state="running",
                            expanded=True,
                        )

                    else:
                        status_holder["box"].update(
                            label=f"🔧 Executing {tool_name}...",
                            state="running",
                            expanded=True,
                        )

                    continue

                # Display assistant text, not tool-call arguments.
                if isinstance(message_chunk, AIMessage):
                    content = message_chunk.content

                    if isinstance(content, str):
                        if content:
                            full_response.append(content)
                            yield content

                    elif isinstance(content, list):
                        for block in content:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "text"
                            ):
                                text = block.get("text", "")

                                if text:
                                    full_response.append(text)
                                    yield text

        displayed_response = st.write_stream(
            response_stream()
        )

        # Finalize the tool status.
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool execution finished",
                state="complete",
                expanded=False,
            )

    # ------------------------------------------------------
    # 10. Save the visible assistant response in the UI
    # ------------------------------------------------------

    if not isinstance(displayed_response, str):
        displayed_response = "".join(full_response)

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": displayed_response,
    })
