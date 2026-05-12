from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from langchain_tavily import TavilySearch


# @tool
# def search(query: str) -> str:
#     """Search for real-time information like weather."""
#     print(f"Searching for {query}")
#     return tavily.search(query=query)

class Source(BaseModel):
    """ Schema for a source used by agent"""

    url:str = Field(description = "The URL of the source")

class AgentResponse(BaseModel):
    """ Schema for agent response with answer and sources"""
    answer:str = Field(description = "Tthe agent's answer to the query")
    sources: List[Source] =Field(default_factory = list, description = "List of sources use to generate the answer")


llm = ChatOpenAI(model='gpt-5')
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format = AgentResponse)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the bay area on linkedin and list their details")]})
    print(result)


if __name__ == "__main__":
    main()
