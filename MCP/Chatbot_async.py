from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio

load_dotenv()

# Initialize the LLM
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)


# Define calculator tool
@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div, mod.
    """
    try:
        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num

        elif operation == "mod":
            if second_num == 0:
                return {"error": "Modulo by zero is not allowed"}
            result = first_num % second_num

        else:
            return {"error": f"Unsupported operation: {operation}"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}


# Register tools
tools = [calculator]

# Bind tools to the LLM
llm_with_tools = llm.bind_tools(tools)


# Define graph state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Build LangGraph
def build_graph():

    async def chat_node(state: ChatState):
        messages = state["messages"]

        # Await the asynchronous LLM call
        response = await llm_with_tools.ainvoke(messages)

        return {"messages": [response]}

    # Tool execution node
    tool_node = ToolNode(tools)

    # Create graph
    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    # Define edges
    graph.add_edge(START, "chat_node")

    graph.add_conditional_edges(
        "chat_node",
        tools_condition
    )

    graph.add_edge("tools", "chat_node")

    # Compile graph
    return graph.compile()


# Run chatbot
async def main():

    chatbot = build_graph()

    result = await chatbot.ainvoke({
        "messages": [
            HumanMessage(
                content=(
                    "Find the modulus of 132354 and 23 "
                    "and give the answer like a cricket commentator."
                )
            )
        ]
    })

    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())