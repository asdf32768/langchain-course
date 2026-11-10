import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient()

@tool
def search(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    # return "Today the weather is sunny"
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-6-luna")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello search-agent!")
    result = agent.invoke({"messages": HumanMessage(content="What is the weather today in random place? Utilize the `search` tool")})
    print(result)

if __name__ == "__main__":
    main()
