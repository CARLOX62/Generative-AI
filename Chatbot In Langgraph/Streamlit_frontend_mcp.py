# Streamlit_frontend_mcp.py

import queue
import uuid

import streamlit as st

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from LangGraph_MCP_Backend import (
    chatbot,
    retrieve_all_threads,
    retrieve_chat_history,
    retrieve_chat_title,
    save_chat_title,
    submit_async_task,
)


# ==========================================================
# 1. Page configuration
# ==========================================================

st.set_page_config(
    page_title="LangGraph MCP Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LangGraph MCP Chatbot")

st.caption(
    "Powered by LangGraph, Ollama, SQLite, and MCP tools."
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

    add_thread(thread_id)


def format_message(message):
    if isinstance(message, HumanMessage):
        role = "user"

    elif isinstance(message, AIMessage):
        # Tool-call messages may not contain visible text.
        if not message.content:
            return None

        role = "assistant"

    else:
        # Tool results are not rendered as separate chat bubbles.
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
    try:
        messages = retrieve_chat_history(str(thread_id))

        st.session_state["message_history"] = [
            formatted
            for message in messages
            if (
                formatted := format_message(message)
            ) is not None
        ]

        st.session_state["thread_id"] = str(thread_id)
        st.session_state["title_initialized"] = True

    except Exception as exc:
        st.error(f"Unable to load conversation: {exc}")


# ==========================================================
# 3. Session initialization
# ==========================================================

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    try:
        st.session_state["chat_threads"] = [
            str(thread_id)
            for thread_id in retrieve_all_threads()
        ]
    except Exception as exc:
        st.error(f"Could not retrieve chat history: {exc}")
        st.session_state["chat_threads"] = []

if "title_initialized" not in st.session_state:
    st.session_state["title_initialized"] = False

add_thread(st.session_state["thread_id"])


# ==========================================================
# 4. Sidebar
# ==========================================================

st.sidebar.title("💬 MCP Chatbot")

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
# 5. Render existing conversation
# ==========================================================

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================================
# 6. Chat input
# ==========================================================

user_input = st.chat_input("Ask me anything...")


if user_input:
    thread_id = str(st.session_state["thread_id"])

    # Display user message immediately.
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Save a title based on the first message.
    if not st.session_state["title_initialized"]:
        save_chat_title(
            thread_id,
            generate_chat_title(user_input),
        )

        st.session_state["title_initialized"] = True

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "metadata": {
            "thread_id": thread_id,
        },
        "run_name": "chat_turn",
    }

    event_queue = queue.Queue()
    status_holder = {"box": None}
    full_response = []

    # ------------------------------------------------------
    # Async graph execution
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
    # Stream assistant tokens in Streamlit
    # ------------------------------------------------------

    with st.chat_message("assistant"):

        def response_stream():
            # Run the graph on the backend's event loop.
            submit_async_task(run_stream())

            while True:
                event_type, chunk, metadata = event_queue.get()

                if event_type == "done":
                    break

                if event_type == "error":
                    error_text = f"\n\n⚠️ Chat execution failed: {chunk}"

                    full_response.append(error_text)
                    yield error_text
                    continue

                message_chunk = chunk

                # Show tool execution status.
                if isinstance(message_chunk, ToolMessage):
                    tool_name = (
                        message_chunk.name or "MCP tool"
                    )

                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Running {tool_name}...",
                            expanded=True,
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Running {tool_name}...",
                            state="running",
                            expanded=True,
                        )

                    continue

                # Stream assistant text only.
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

        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool execution finished",
                state="complete",
                expanded=False,
            )

    # ------------------------------------------------------
    # Save assistant message in the Streamlit session.
    # LangGraph separately saves the graph state in SQLite.
    # ------------------------------------------------------

    if not isinstance(displayed_response, str):
        displayed_response = "".join(full_response)

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": displayed_response,
    })