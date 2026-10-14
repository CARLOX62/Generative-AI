from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("Math Server")


# Addition tool
@mcp.tool()
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


# Subtraction tool
@mcp.tool()
def subtract(a: float, b: float) -> float:
    """Subtract the second number from the first number."""
    return a - b


# Multiplication tool
@mcp.tool()
def multiply(a: float, b: float) -> float:
    """Multiply two numbers and return the result."""
    return a * b


# Division tool
@mcp.tool()
def divide(a: float, b: float) -> float:
    """Divide the first number by the second number."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")

    return a / b


# Modulus tool
@mcp.tool()
def modulus(a: float, b: float) -> float:
    """Return the remainder when the first number is divided by the second."""
    if b == 0:
        raise ValueError("Modulus by zero is not allowed.")

    return a % b


# Power tool
@mcp.tool()
def power(a: float, b: float) -> float:
    """Raise the first number to the power of the second number."""
    return a ** b


# Main entry point
if __name__ == "__main__":
    mcp.run(transport="stdio")