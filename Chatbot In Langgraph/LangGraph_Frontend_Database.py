# LangGraph_Frontend_Database.py

import streamlit as st
import uuid

from langchain_core.messages import HumanMessage, AIMessage, AIMessageChunk

from LangGraph_Tool_Backend import (
    chatbot,
    retrieve_all_threads,
    save_chat_title,
    retrieve_chat_title,
)


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="LangGraph AI Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 LangGraph AI Chatbot")


# --------------------------------------------------
# Session state initialization
# --------------------------------------------------

def generate_thread_id():
    return str(uuid.uuid4())


def generate_chat_title(message):
    """Create a simple title from the user's first message."""
    title = " ".join(message.strip().split())
    return title[:40] + ("..." if len(title) > 40 else "")


def add_thread(thread_id):
    """Add a thread to the current session's thread list."""
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def get_thread_title(thread_id):
    """Retrieve a saved title, with a fallback title."""
    try:
        title = retrieve_chat_title(thread_id)
        if title:
            return title
    except Exception:
        pass

    return f"Chat {thread_id[:8]}"


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = {}

if "title_saved_for_thread" not in st.session_state:
    st.session_state["title_saved_for_thread"] = set()


# Register the current thread.
add_thread(st.session_state["thread_id"])


# --------------------------------------------------
# Retrieve existing threads
# --------------------------------------------------

try:
    existing_threads = retrieve_all_threads()

    if existing_threads:
        for thread in existing_threads:
            if isinstance(thread, dict):
                thread_id = thread.get("thread_id")
            else:
                thread_id = thread

            if thread_id:
                add_thread(thread_id)

except Exception as exc:
    st.sidebar.warning(f"Could not load chat history: {exc}")


# --------------------------------------------------
# Load messages from a selected thread
# --------------------------------------------------

def load_thread(thread_id):
    """Load the messages stored in the LangGraph checkpoint."""
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        state = chatbot.get_state(config)
        values = state.values or {}
        messages = values.get("messages", [])

        history = []

        for message in messages:
            if isinstance(message, HumanMessage):
                history.append({
                    "role": "user",
                    "content": message.content,
                })

            elif isinstance(message, AIMessage):
                # Avoid showing tool-call-only messages in chat history.
                if getattr(message, "tool_calls", None):
                    if not message.content:
                        continue

                if message.content:
                    if isinstance(message.content, str):
                        content = message.content
                    elif isinstance(message.content, list):
                        content = "\n".join(
                            block.get("text", "")
                            for block in message.content
                            if isinstance(block, dict)
                            and block.get("type") == "text"
                        )
                    else:
                        content = str(message.content)

                    if content.strip():
                        history.append({
                            "role": "assistant",
                            "content": content,
                        })

        return history

    except Exception as exc:
        st.error(f"Could not load this conversation: {exc}")
        return []


# --------------------------------------------------
# Start a new chat
# --------------------------------------------------

def reset_chat():
    new_thread_id = generate_thread_id()

    st.session_state["thread_id"] = new_thread_id
    st.session_state["message_history"] = []

    add_thread(new_thread_id)


# --------------------------------------------------
# Sidebar: chat history
# --------------------------------------------------

with st.sidebar:
    st.header("💬 Chat History")

    if st.button("➕ New Chat", use_container_width=True):
        reset_chat()
        st.rerun()

    st.divider()

    # Display the newest thread first.
    for thread_id in reversed(st.session_state["chat_threads"]):
        title = st.session_state["chat_titles"].get(
            thread_id,
            get_thread_title(thread_id),
        )

        if st.button(
            f"🗨️ {title}",
            key=f"thread_{thread_id}",
            use_container_width=True,
        ):
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = load_thread(thread_id)
            st.rerun()


# --------------------------------------------------
# Ensure the selected thread has its messages loaded
# --------------------------------------------------

current_thread_id = st.session_state["thread_id"]

if not st.session_state["message_history"]:
    st.session_state["message_history"] = load_thread(
        current_thread_id
    )


# --------------------------------------------------
# Display existing conversation
# --------------------------------------------------

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --------------------------------------------------
# Chat input and response streaming
# --------------------------------------------------

user_input = st.chat_input("Ask me anything...")

if user_input and user_input.strip():

    user_input = user_input.strip()
    current_thread_id = st.session_state["thread_id"]

    config = {
        "configurable": {
            "thread_id": current_thread_id
        }
    }

    # Save the user's message in the frontend history.
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # Save a title from the first user message.
    if current_thread_id not in st.session_state["title_saved_for_thread"]:
        title = generate_chat_title(user_input)

        try:
            save_chat_title(current_thread_id, title)
        except Exception as exc:
            st.sidebar.warning(f"Could not save chat title: {exc}")

        st.session_state["chat_titles"][current_thread_id] = title
        st.session_state["title_saved_for_thread"].add(
            current_thread_id
        )

    # Mutable dictionary avoids the nonlocal scope error.
    status_holder = {
        "box": None,
        "tool_used": False,
    }

    assistant_text = ""

    try:
        with st.chat_message("assistant"):

            def ai_only_stream():
                for message_chunk, metadata in chatbot.stream(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ]
                    },
                    config=config,
                    stream_mode="messages",
                ):

                    node_name = metadata.get("langgraph_node")

                    # Show status when the graph enters the tools node.
                    if node_name == "tools":
                        status_holder["tool_used"] = True

                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                "🔧 Using a tool...",
                                expanded=True,
                            )

                        # Do not display raw tool output as assistant text.
                        continue

                    # Ignore messages from unrelated graph nodes.
                    if node_name != "chat_node":
                        continue

                    # The graph has returned to the assistant after tool use.
                    if status_holder["box"] is not None:
                        status_holder["box"].update(
                            label="✅ Tool finished",
                            state="complete",
                            expanded=False,
                        )
                        status_holder["box"] = None

                    if not isinstance(message_chunk, AIMessageChunk):
                        continue

                    # Do not render tool-call payloads.
                    if getattr(message_chunk, "tool_calls", None):
                        continue

                    content = message_chunk.content

                    if isinstance(content, str):
                        if content:
                            yield content

                    elif isinstance(content, list):
                        for block in content:
                            if (
                                isinstance(block, dict)
                                and block.get("type") == "text"
                                and block.get("text")
                            ):
                                yield block["text"]

            assistant_text = st.write_stream(ai_only_stream())

    except Exception as exc:
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="❌ Tool or response failed",
                state="error",
                expanded=True,
            )
            status_holder["box"] = None

        st.error(f"An error occurred: {exc}")

    finally:
        # Complete the status only if a tool was actually used
        # and the status box is still open.
        if (
            status_holder["tool_used"]
            and status_holder["box"] is not None
        ):
            status_holder["box"].update(
                label="✅ Tool finished",
                state="complete",
                expanded=False,
            )
            status_holder["box"] = None

    # Persist the assistant response in the frontend history.
    if assistant_text:
        st.session_state["message_history"].append({
            "role": "assistant",
            "content": assistant_text,
        })

    # Refresh the chat history to show a newly saved title.
    st.rerun()