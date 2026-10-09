import streamlit as st
from LangGraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

# ****************************************
# Utility Functions
# ****************************************

def generate_thread_id():
    return str(uuid.uuid4())


def generate_chat_title(user_message):
    """Generate a short, meaningful title for a conversation."""
    try:
        response = chatbot.invoke({
            "messages": [
                HumanMessage(
                    content=(
                        "Generate a short, meaningful title of 2-5 words "
                        "for a chatbot conversation based on the user's "
                        "message below. Return only the title, without "
                        "quotes or explanations.\n\n"
                        f"User message: {user_message}"
                    )
                )
            ]
        })

        # Extract the final AI response.
        messages = response.get("messages", [])
        if messages:
            title = messages[-1].content

            if isinstance(title, str) and title.strip():
                return title.strip().strip('"').strip("'")[:50]

        return user_message[:30]

    except Exception:
        # Fallback if title generation fails.
        return user_message[:30] or "New Chat"


def add_thread(thread_id):
    """Create a thread only if it does not already exist."""
    for thread in st.session_state["chat_thread"]:
        if thread["thread_id"] == thread_id:
            return

    st.session_state["chat_thread"].append({
        "thread_id": thread_id,
        "title": "New Chat",
        "messages": []
    })


def save_message_to_thread(thread_id, message):
    """Save an individual message to the correct thread."""
    for thread in st.session_state["chat_thread"]:
        if thread["thread_id"] == thread_id:
            thread["messages"].append(message.copy())
            return


def load_thread(thread_id):
    """Load a selected conversation."""
    for thread in st.session_state["chat_thread"]:
        if thread["thread_id"] == thread_id:
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = [
                message.copy() for message in thread["messages"]
            ]
            return


def reset_chat():
    """Start a new conversation."""
    new_thread_id = generate_thread_id()

    st.session_state["thread_id"] = new_thread_id
    st.session_state["message_history"] = []

    add_thread(new_thread_id)


# ****************************************
# Session State Setup
# ****************************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_thread" not in st.session_state:
    st.session_state["chat_thread"] = []

# Ensure the current thread exists.
add_thread(st.session_state["thread_id"])


# ****************************************
# Sidebar UI
# ****************************************

st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.markdown("## Chat History")

if not st.session_state["chat_thread"]:
    st.sidebar.caption("No conversations yet.")

for thread in st.session_state["chat_thread"]:
    tid = thread["thread_id"]

    # Show the conversation topic, not its UUID.
    title = thread["title"]

    if tid == st.session_state["thread_id"]:
        button_label = f"🟢 {title}"
    else:
        button_label = f"💬 {title}"

    st.sidebar.button(
        button_label,
        key=f"thread_button_{tid}",
        use_container_width=True,
        on_click=load_thread,
        args=(tid,)
    )


# ****************************************
# Main Chat UI
# ****************************************

st.title("LangGraph Chatbot")

# Display the selected conversation's messages.
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"])
        else:
            st.write(message["content"])


# ****************************************
# Chat Input and Streaming Response
# ****************************************

user_input = st.chat_input("Type your message here...")

if user_input:
    current_thread_id = st.session_state["thread_id"]

    # Find the active thread.
    active_thread = next(
        (
            thread
            for thread in st.session_state["chat_thread"]
            if thread["thread_id"] == current_thread_id
        ),
        None
    )

    # Generate a topic title only for the first message.
    if active_thread and not active_thread["messages"]:
        active_thread["title"] = generate_chat_title(user_input)

    # Save and display the user message.
    user_message = {
        "role": "user",
        "content": user_input
    }

    st.session_state["message_history"].append(user_message)

    save_message_to_thread(
        current_thread_id,
        user_message
    )

    with st.chat_message("user"):
        st.write(user_input)

    # Pass the thread ID to LangGraph.
    CONFIG = {
        "configurable": {
            "thread_id": current_thread_id
        }
    }

    # Stream the assistant's response.
    with st.chat_message("assistant"):
        try:
            ai_message = st.write_stream(
                message_chunk.content
                for message_chunk, metadata in chatbot.stream(
                    {
                        "messages": [
                            HumanMessage(content=user_input)
                        ]
                    },
                    config=CONFIG,
                    stream_mode="messages"
                )
                if isinstance(message_chunk.content, str)
                and message_chunk.content
            )

            # Save the assistant response.
            assistant_message = {
                "role": "assistant",
                "content": ai_message
            }

            st.session_state["message_history"].append(
                assistant_message
            )

            save_message_to_thread(
                current_thread_id,
                assistant_message
            )

        except Exception as e:
            st.error(f"Something went wrong: {e}")

