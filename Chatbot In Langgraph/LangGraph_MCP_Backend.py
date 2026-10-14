# LangGraph_MCP_Backend.py

import os
import sys
import asyncio
import threading
import sqlite3
import aiosqlite
from pathlib import Path
from typing import TypedDict, Annotated

import requests
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool, BaseTool
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


# ==========================================================
# 1. Paths and environment
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

load_dotenv(BASE_DIR / ".env")

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

DATABASE_PATH = str(BASE_DIR / "chatbot.db")
TITLES_DATABASE_PATH = str(BASE_DIR / "chat_titles.db")

# Local MCP server path:
# Generative Ai/MCP/mcp-math-server/main.py
MATH_SERVER_PATH = (
    PROJECT_ROOT / "MCP" / "mcp-math-server" / "main.py"
)


# ==========================================================
# 2. Dedicated asynchronous event loop
# ==========================================================

_ASYNC_LOOP = asyncio.new_event_loop()


def _run_event_loop():
    asyncio.set_event_loop(_ASYNC_LOOP)
    _ASYNC_LOOP.run_forever()


_ASYNC_THREAD = threading.Thread(
    target=_run_event_loop,
    name="LangGraphAsyncLoop",
    daemon=True,
)

_ASYNC_THREAD.start()


def submit_async_task(coro):
    """Submit a coroutine to the backend event loop."""
    return asyncio.run_coroutine_threadsafe(
        coro,
        _ASYNC_LOOP,
    )


def run_async(coro):
    """Run a coroutine and wait for its result."""
    return submit_async_task(coro).result()


# ==========================================================
# 3. LLM
# ==========================================================

llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0,
)


# ==========================================================
# 4. Standard tools
# ==========================================================

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Calculate two numbers.

    Supported operations: add, sub, mul, div, mod.
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
            return {"error": "Division by zero is not allowed."}
        result = first_num / second_num

    elif operation in ("mod", "modulus", "remainder"):
        if second_num == 0:
            return {"error": "Modulo by zero is not allowed."}
        result = first_num % second_num

    else:
        return {
            "error": f"Unsupported operation: {operation}"
        }

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get the latest available stock quote from Alpha Vantage.
    Examples: IBM, AAPL, TSLA.
    """

    if not ALPHA_VANTAGE_API_KEY:
        return {
            "error": (
                "Set ALPHAVANTAGE_API_KEY in your .env file."
            )
        }

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol.strip().upper(),
                "apikey": ALPHA_VANTAGE_API_KEY,
            },
            timeout=15,
        )

        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            return {"error": data["Error Message"]}

        if "Note" in data or "Information" in data:
            return {
                "error": data.get(
                    "Note",
                    data.get("Information"),
                )
            }

        quote = data.get("Global Quote", {})

        if not quote:
            return {
                "error": (
                    "No quote returned. Check the symbol, "
                    "API key, and API usage limits."
                )
            }

        return {
            "symbol": quote.get("01. symbol"),
            "price": quote.get("05. price"),
            "latest_trading_day": quote.get(
                "07. latest trading day"
            ),
            "change": quote.get("09. change"),
            "change_percent": quote.get("10. change percent"),
        }

    except requests.RequestException as exc:
        return {"error": f"Stock API request failed: {exc}"}

    except ValueError:
        return {"error": "Invalid JSON returned by stock API."}


# ==========================================================
# 5. MCP client configuration
# ==========================================================

mcp_clients = []

mcp_servers = {}

if MATH_SERVER_PATH.exists():
    mcp_servers["arith"] = {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(MATH_SERVER_PATH)],
    }
else:
    print(
        "WARNING: Local MCP server not found:",
        MATH_SERVER_PATH,
    )

mcp_servers["expense"] = {
    "transport": "streamable_http",
    "url": "https://splendid-gold-dingo.fastmcp.app/mcp",
}


async def load_mcp_tools() -> list[BaseTool]:
    """
    Load tools from each MCP server independently.

    A failure in one server will not prevent the other server
    or the standard tools from being used.
    """

    loaded_tools = []

    for server_name, server_config in mcp_servers.items():
        client = MultiServerMCPClient({
            server_name: server_config
        })

        try:
            server_tools = await client.get_tools()

            loaded_tools.extend(server_tools)
            mcp_clients.append(client)

            print(
                f"MCP server '{server_name}': "
                f"{len(server_tools)} tool(s) loaded."
            )

            for server_tool in server_tools:
                print("  -", server_tool.name)

        except Exception as exc:
            print(
                f"WARNING: MCP server '{server_name}' "
                f"could not be loaded: {exc}"
            )

    return loaded_tools


# Discover tools on the dedicated event loop.
mcp_tools = run_async(load_mcp_tools())

tools = [
    search_tool,
    calculator,
    get_stock_price,
    *mcp_tools,
]

print("\nAvailable tools:")
for registered_tool in tools:
    print(" -", registered_tool.name)

llm_with_tools = llm.bind_tools(tools)


# ==========================================================
# 6. Graph state
# ==========================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================================
# 7. Async graph nodes
# ==========================================================

async def chat_node(state: ChatState):
    """Generate an assistant response or request tool execution."""

    response = await llm_with_tools.ainvoke(
        state["messages"]
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)


# ==========================================================
# 8. Async SQLite checkpointer
# ==========================================================

async def _init_checkpointer():
    conn = await aiosqlite.connect(
        DATABASE_PATH,
        timeout=30,
    )

    await conn.execute("PRAGMA busy_timeout = 30000")

    saver = AsyncSqliteSaver(conn)
    await saver.setup()

    return saver


checkpointer = run_async(_init_checkpointer())


# ==========================================================
# 9. Build and compile graph
# ==========================================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(
    checkpointer=checkpointer,
)


# ==========================================================
# 10. Persistent chat titles
# ==========================================================

def _init_titles_database():
    with sqlite3.connect(TITLES_DATABASE_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_titles (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL
            )
        """)


_init_titles_database()


def save_chat_title(thread_id: str, title: str) -> None:
    """Insert or update a conversation title."""

    with sqlite3.connect(TITLES_DATABASE_PATH) as conn:
        conn.execute(
            """
            INSERT INTO chat_titles (thread_id, title)
            VALUES (?, ?)
            ON CONFLICT(thread_id)
            DO UPDATE SET title = excluded.title
            """,
            (str(thread_id), title.strip() or "New Chat"),
        )


def retrieve_chat_title(thread_id: str) -> str:
    """Return the saved title for a conversation."""

    with sqlite3.connect(TITLES_DATABASE_PATH) as conn:
        row = conn.execute(
            """
            SELECT title
            FROM chat_titles
            WHERE thread_id = ?
            """,
            (str(thread_id),),
        ).fetchone()

    return row[0] if row else "New Chat"


# ==========================================================
# 11. Conversation helpers
# ==========================================================

def thread_config(thread_id: str) -> dict:
    return {
        "configurable": {
            "thread_id": str(thread_id),
        }
    }


async def _retrieve_all_threads():
    thread_ids = set()

    async for checkpoint in checkpointer.alist(None):
        thread_id = (
            checkpoint.config
            .get("configurable", {})
            .get("thread_id")
        )

        if thread_id:
            thread_ids.add(str(thread_id))

    return list(thread_ids)


def retrieve_all_threads() -> list[str]:
    """Return all thread IDs stored in SQLite."""

    return run_async(_retrieve_all_threads())


async def _retrieve_chat_history(thread_id: str):
    state = await chatbot.aget_state(
        thread_config(thread_id)
    )

    if not state or not state.values:
        return []

    return state.values.get("messages", [])


def retrieve_chat_history(thread_id: str):
    """Return messages saved for a particular thread."""

    return run_async(_retrieve_chat_history(thread_id))


# ==========================================================
# 12. Optional startup diagnostics
# ==========================================================

print("\nLangGraph MCP backend initialized.")
print(f"Local math server exists: {MATH_SERVER_PATH.exists()}")
print(f"Registered tool count: {len(tools)}")