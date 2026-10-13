# backend.py

import os
import sqlite3
import requests

from typing import TypedDict, Annotated

from dotenv import load_dotenv

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver


# ==========================================================
# 1. Environment Variables
# ==========================================================

load_dotenv()

# Store your API key in the .env file.
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


# ==========================================================
# 2. LLM
# ==========================================================

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)


# ==========================================================
# 3. Tools
# ==========================================================

# DuckDuckGo Search Tool
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform arithmetic operations on two numbers.

    Supported operations:
    add, sub, mul, div
    """

    operation = operation.lower().strip()

    if operation == "add":
        result = first_num + second_num

    elif operation == "sub":
        result = first_num - second_num

    elif operation == "mul":
        result = first_num * second_num

    elif operation == "div":
        if second_num == 0:
            return {
                "error": "Division by zero is not allowed."
            }

        result = first_num / second_num

    else:
        return {
            "error": f"Unsupported operation: {operation}"
        }

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest available stock quote using Alpha Vantage.

    Examples of symbols: IBM, AAPL, TSLA.
    Requires ALPHAVANTAGE_API_KEY in the .env file.
    """

    if not ALPHA_VANTAGE_API_KEY:
        return {
            "error": "Missing ALPHAVANTAGE_API_KEY in the .env file."
        }

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol.strip().upper(),
        "apikey": ALPHA_VANTAGE_API_KEY
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            return {"error": data["Error Message"]}

        if "Note" in data or "Information" in data:
            return {
                "error": data.get(
                    "Note",
                    data.get("Information")
                )
            }

        quote = data.get("Global Quote", {})

        if not quote:
            return {
                "error": (
                    "No stock quote returned. Check the symbol, "
                    "API key, and API usage limits."
                )
            }

        return {
            "symbol": quote.get("01. symbol"),
            "price": quote.get("05. price"),
            "latest_trading_day": quote.get("07. latest trading day"),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent")
        }

    except requests.RequestException as exc:
        return {
            "error": f"Stock API request failed: {exc}"
        }

    except ValueError:
        return {
            "error": "The stock API returned invalid JSON."
        }


# Register all tools
tools = [
    search_tool,
    get_stock_price,
    calculator
]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# ==========================================================
# 4. Chat State
# ==========================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================================
# 5. Chat Node
# ==========================================================

def chat_node(state: ChatState):
    """
    Generate an assistant response or request a tool call.
    """

    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": [response]
    }


# ==========================================================
# 6. SQLite Database and Checkpointer
# ==========================================================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False,
    timeout=30
)

# Enable foreign keys and wait for database locks when needed.
conn.execute("PRAGMA foreign_keys = ON")
conn.execute("PRAGMA busy_timeout = 30000")

checkpointer = SqliteSaver(conn=conn)

# Create LangGraph checkpoint tables if they do not exist.
checkpointer.setup()


# ==========================================================
# 7. Persistent Chat Titles
# ==========================================================

conn.execute("""
    CREATE TABLE IF NOT EXISTS chat_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
    )
""")

conn.commit()


def save_chat_title(thread_id: str, title: str) -> None:
    """
    Save a conversation title or update its existing title.
    """

    conn.execute(
        """
        INSERT INTO chat_titles (thread_id, title)
        VALUES (?, ?)
        ON CONFLICT(thread_id)
        DO UPDATE SET title = excluded.title
        """,
        (thread_id, title.strip() or "New Chat")
    )

    conn.commit()


def retrieve_chat_title(thread_id: str) -> str:
    """
    Return a conversation title.
    """

    cursor = conn.execute(
        """
        SELECT title
        FROM chat_titles
        WHERE thread_id = ?
        """,
        (thread_id,)
    )

    row = cursor.fetchone()

    return row[0] if row else "New Chat"


# ==========================================================
# 8. Build the LangGraph Workflow
# ==========================================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")

# Execute tools when requested by the LLM.
# Otherwise, finish the current turn.
graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

# Return tool results to the LLM.
graph.add_edge("tools", "chat_node")

# Compile with persistent SQLite checkpoints.
chatbot = graph.compile(
    checkpointer=checkpointer
)


# ==========================================================
# 9. Thread / Conversation Helpers
# ==========================================================

def retrieve_all_threads() -> list[str]:
    """
    Retrieve unique thread IDs stored in the checkpoint database.
    """

    all_threads = set()

    for checkpoint in checkpointer.list(None):
        configurable = checkpoint.config.get(
            "configurable", {}
        )

        thread_id = configurable.get("thread_id")

        if thread_id:
            all_threads.add(thread_id)

    return list(all_threads)


def thread_config(thread_id: str) -> dict:
    """
    Return the configuration required to access a conversation.
    """

    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


# ==========================================================
# 10. Optional: Retrieve Messages for a Conversation
# ==========================================================

def retrieve_chat_history(thread_id: str) -> list[BaseMessage]:
    """
    Retrieve the saved messages for a specific conversation.
    """

    state = chatbot.get_state(thread_config(thread_id))

    if not state or not state.values:
        return []

    return state.values.get("messages", [])
