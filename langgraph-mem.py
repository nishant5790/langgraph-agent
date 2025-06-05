# Import core components 
from langgraph.prebuilt import create_react_agent
from langgraph.store.memory import InMemoryStore
from langmem import create_manage_memory_tool, create_search_memory_tool

from typing import TypedDict , List , Annotated , Sequence
from langgraph.graph import StateGraph , END , START
from langchain_core.messages import BaseMessage
from langchain_core.messages import AIMessage , HumanMessage , ToolMessage , SystemMessage
from langchain_core.tools import tool 
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode , create_react_agent

# Set up storage 
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": "openai:text-embedding-3-small",
    }
) 

# Create an agent with memory capabilities 
agent = create_react_agent(
    "openai:gpt-4o",
    tools=[
        # Memory tools use LangGraph's BaseStore for persistence (4)
        create_manage_memory_tool(namespace=("memories",)),
        create_search_memory_tool(namespace=("memories",)),
    ],
    store=store,
)

from langchain.chat_models import init_chat_model
init_chat_model(model="gpt-4o-mini").bind_tools(tools)