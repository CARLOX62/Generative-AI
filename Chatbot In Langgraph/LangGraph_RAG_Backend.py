
# LangGraph_RAG_Backend.py

from __future__ import annotations

import os
import sys
import asyncio
import threading
import sqlite3
import aiosqlite
import tempfile
from pathlib import Path
from typing import Annotated, Any, Optional, TypedDict

import aiosqlite
import requests
from dotenv import load_dotenv

from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool, BaseTool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
    """Submit a coroutine to the dedicated event loop."""
    return asyncio.run_coroutine_threadsafe(
        coro, _ASYNC_LOOP
    )


def run_async(coro):
    """Run a coroutine from synchronous application code."""
    return submit_async_task(coro).result()


# ==========================================================
# 3. LLM and embeddings
# ==========================================================

llm = ChatOllama(
    model=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:1b"),
    temperature=0,
)

embeddings = OllamaEmbeddings(
    model=os.getenv(
        "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
    )
)


# ==========================================================
# 4. RAG storage: one PDF index per conversation
# ==========================================================

_THREAD_RETRIEVERS: dict[str, Any] = {}
_THREAD_METADATA: dict[str, dict] = {}


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Read an uploaded PDF, split it into chunks, create a FAISS
    index, and associate the index with the conversation.
    """

    if not file_bytes:
        raise ValueError("The uploaded PDF is empty.")

    thread_id = str(thread_id)

    if not thread_id:
        raise ValueError("A valid thread ID is required.")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        loader = PyPDFLoader(temp_path)
        documents = loader.load()

        if not documents:
            raise ValueError(
                "No readable pages were found in the PDF."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError(
                "No text could be extracted from the PDF. "
                "If it contains scanned pages, OCR may be required."
            )

        # Build the index using local Ollama embeddings.
        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = {
            "filename": filename or "uploaded.pdf",
            "documents": len(documents),
            "chunks": len(chunks),
        }

        return {
            "success": True,
            "thread_id": thread_id,
            "filename": filename or "uploaded.pdf",
            "documents": len(documents),
            "chunks": len(chunks),
        }

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _get_retriever(thread_id: Optional[str]):
    if thread_id is None:
        return None

    return _THREAD_RETRIEVERS.get(str(thread_id))


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {}).copy()


# ==========================================================
# 5. Standard tools
# ==========================================================

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Perform arithmetic on two numbers.

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
            return {
                "error": "Division by zero is not allowed."
            }
        result = first_num / second_num

    elif operation in ("mod", "modulus", "remainder"):
        if second_num == 0:
            return {
                "error": "Modulo by zero is not allowed."
            }
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
                    "Note", data.get("Information")
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


@tool
def rag_tool(
    query: str,
    config: RunnableConfig,
) -> dict:
    """
    Search the PDF uploaded to the current conversation.

    Use this tool for questions about an uploaded PDF.
    The current thread ID is injected automatically by LangGraph.
    """

    thread_id = (
        config.get("configurable", {}).get("thread_id")
    )

    if not thread_id:
        return {
            "error": (
                "Conversation thread ID is missing. "
                "Run the graph with configurable.thread_id."
            )
        }

    retriever = _get_retriever(str(thread_id))

    if retriever is None:
        return {
            "error": (
                "No PDF is indexed for this conversation. "
                "Upload a PDF in this chat first."
            ),
            "query": query,
        }

    try:
        results = retriever.invoke(query)

        if not results:
            return {
                "query": query,
                "context": [],
                "message": "No relevant passages were found.",
            }

        passages = []

        for document in results:
            passages.append({
                "page_content": document.page_content,
                "page": document.metadata.get("page"),
                "source": document.metadata.get("source"),
            })

        return {
            "query": query,
            "source_file": _THREAD_METADATA.get(
                str(thread_id), {}
            ).get("filename"),
            "context": passages,
            "instruction": (
                "Answer using the retrieved PDF passages. "
                "If they do not contain the answer, say so. "
                "Do not invent information."
            ),
        }

    except Exception as exc:
        return {
            "error": f"PDF retrieval failed: {exc}",
            "query": query,
        }


# ==========================================================
# 6. MCP client configuration
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
    """Load available MCP tools without blocking other tools."""

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


mcp_tools = run_async(load_mcp_tools())

tools = [
    search_tool,
    calculator,
    get_stock_price,
    rag_tool,
    *mcp_tools,
]

print("\nAvailable tools:")

for registered_tool in tools:
    print(" -", registered_tool.name)

llm_with_tools = llm.bind_tools(tools)


# ==========================================================
# 7. Graph state
# ==========================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ==========================================================
# 8. Async graph nodes
# ==========================================================

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Rules:
1. For questions about an uploaded PDF, use rag_tool.
2. Do not claim to have read a PDF unless its passages were retrieved.
3. Answer PDF questions using the retrieved context.
4. If the PDF does not contain the answer, say so clearly.
5. Use web search for current information when appropriate.
6. Use calculator for arithmetic when appropriate.
7. Use get_stock_price for stock quote requests.
8. Use MCP tools when they are appropriate for the request.
9. Explain tool results clearly and accurately.
"""


async def chat_node(
    state: ChatState,
    config: RunnableConfig,
):
    """Generate an answer or request a tool call."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]

    response = await llm_with_tools.ainvoke(
        messages,
        config=config,
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)


# ==========================================================
# 9. Async SQLite checkpointer
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
# 10. Build and compile graph
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
# 11. Persistent chat titles
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
    """Return the saved conversation title."""

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
# 12. Conversation helpers
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
    """Return conversation IDs saved in SQLite."""
    return run_async(_retrieve_all_threads())


async def _retrieve_chat_history(thread_id: str):
    state = await chatbot.aget_state(
        thread_config(thread_id)
    )

    if not state or not state.values:
        return []

    return state.values.get("messages", [])


def retrieve_chat_history(thread_id: str):
    """Retrieve saved messages for a conversation."""
    return run_async(_retrieve_chat_history(thread_id))


# ==========================================================
# 13. Diagnostics
# ==========================================================

print("\nLangGraph MCP + RAG backend initialized.")
print(f"Local math server exists: {MATH_SERVER_PATH.exists()}")
print(f"Registered tool count: {len(tools)}")
print("PDF RAG: enabled")
print("Chat model:", os.getenv(
    "OLLAMA_CHAT_MODEL", "llama3.2:1b"
))
print("Embedding model:", os.getenv(
    "OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"
))
