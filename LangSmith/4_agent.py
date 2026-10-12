
import os
import requests

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

load_dotenv()

os.environ['LANGCHAIN_PROJECT'] = 'Agent'


# 1. Initialize web search
search_tool = DuckDuckGoSearchRun()


# 2. Create weather tool
@tool
def get_weather_data(city: str) -> str:
    """Fetch the current weather and temperature for a given city."""

    api_key = os.getenv("WEATHERSTACK_API_KEY")

    if not api_key:
        return "Weatherstack API key is missing from the .env file."

    url = "https://api.weatherstack.com/current"

    try:
        response = requests.get(
            url,
            params={
                "access_key": api_key,
                "query": city
            },
            timeout=15
        )

        response.raise_for_status()
        data = response.json()

        if "error" in data:
            return f"Weather API error: {data['error']}"

        current = data.get("current", {})
        location = data.get("location", {})

        return (
            f"City: {location.get('name', city)}\n"
            f"Country: {location.get('country', 'Unknown')}\n"
            f"Temperature: {current.get('temperature', 'Unknown')} °C\n"
            f"Condition: {current.get('weather_descriptions', ['Unknown'])}\n"
            f"Humidity: {current.get('humidity', 'Unknown')}%\n"
            f"Wind speed: {current.get('wind_speed', 'Unknown')} km/h"
        )

    except requests.RequestException as exc:
        return f"Unable to retrieve weather data: {exc}"


# 3. Initialize Ollama model
llm = ChatOllama(
    model="llama3.2:1b",
    temperature=0
)

# 4. Define available tools
tools = [
    search_tool,
    get_weather_data
]

# 5. Create the agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful assistant with web search and weather tools. "
        "Use DuckDuckGoSearchRun to find current or factual information. "
        "Use get_weather_data to obtain current weather conditions. "
        "For questions requiring both search and weather, use both tools. "
        "Do not invent search results or weather information."
    )
)

# 6. Invoke the agent
question = "What is the current temperature in Gurgaon?"

response = agent.invoke({
    "messages": [
        {"role": "user", "content": question}
    ]
})

# 7. Print the final answer
print("\nAssistant:")

for message in reversed(response["messages"]):
    if message.type == "ai" and message.content:
        print(message.content)
        break
