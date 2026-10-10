from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
import sqlite3


model = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

conn = sqlite3.connect(database = 'chatbot.db', check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn = conn)


# Persistent storage for chat titles
conn.execute("""
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
    )
""")
conn.commit()


def save_chat_title(thread_id: str, title: str):
    """Save or update a conversation title."""
    conn.execute(
        """
        INSERT INTO chat_titles (thread_id, title)
        VALUES (?, ?)
        ON CONFLICT(thread_id)
        DO UPDATE SET title = excluded.title
        """,
        (thread_id, title)
    )
    conn.commit()


def retrieve_chat_title(thread_id: str):
    """Retrieve a conversation title."""
    cursor = conn.execute(
        "SELECT title FROM chat_titles WHERE thread_id = ?",
        (thread_id,)
    )
    row = cursor.fetchone()

    return row[0] if row else "New Chat"


graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        configurable = checkpoint.config.get("configurable", {})
        thread_id = configurable.get("thread_id")

        if thread_id:
            all_threads.add(thread_id)

    return list(all_threads)    