#!/usr/bin/env python3
"""
Action Agent FastAPI Server
Port: 5001
Specializes in filtering data, creating search queries, and finding information
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Action Agent API",
    description="Handles multiple actions like outreach campaigns, CSV exports, Marketo integration, and data processing tasks",
    version="1.0.0"
)

# Pydantic models
class ConversationMessage(BaseModel):
    role: str
    content: str
    agent_id: Optional[str] = None
    timestamp: Optional[str] = None

class SessionInfo(BaseModel):
    session_id: str
    current_agent: Optional[str] = None
    handoff_count: Optional[int] = 0

class AgentInfo(BaseModel):
    tools: List[str] = []
    capabilities: str = ""

class Context(BaseModel):
    conversation_history: List[ConversationMessage] = []
    session_info: SessionInfo
    agent_info: Optional[AgentInfo] = None

class QueryRequest(BaseModel):
    query: str
    # context: Context
    # session_id: str
    # agent_id: str

class AgentResponse(BaseModel):
    response: str
    # handoff_requested: bool = False
    # target_agent: Optional[str] = None
    # metadata: Dict[str, Any] = {}
    # tools_used: List[str] = []


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "action_agent", "timestamp": datetime.now().isoformat()}

@app.post("/process", response_model=AgentResponse)
async def process_query(
    request: QueryRequest,
    # authenticated: bool = Depends(verify_token)
):
    """
    Process filtering queries and return filtered data or handoff to appropriate agent
    """
    logger.info(f"Processing query: {request.query}")
    
    query = request.query.strip()
    query_lower = query.lower()
    
    if query == "add to campaign":
        response_text = "Please provide the details of the campaign you want to add this data to."
        return AgentResponse(response=response_text)
    
    if query == "export to csv":
        response_text = "Please provide the details of the data you want to export to CSV."
        return AgentResponse(response=response_text)
    
    if query == "add to marketo":
        response_text = "Please provide the details of the data you want to add to Marketo."
        return AgentResponse(response=response_text)
    
    if query == "show customers by location":
        response_text = "Here are the customers grouped by location:\n\n"
        # Simulated data
        customers = {
            "California": ["Customer A", "Customer B"],
            "New York": ["Customer C"],
            "Texas": ["Customer D", "Customer E"]
        }
        for location, names in customers.items():
            response_text += f"{location}: {', '.join(names)}\n"

    
    response_text =  f"\n\n💡 Hi I am a action agent . performing : {query}"
    
    return AgentResponse(
        response=response_text
    )

# @app.get("/info")
# async def agent_info():
#     """Get agent information"""
#     return {
#         "name": "Filter Agent",
#         "agent_id": "filter_agent",
#         "description": "Specialized agent for filtering data, creating search queries, and finding specific information from datasets",
#         "keywords": ["filter", "search", "find", "query", "data", "show", "display", "select", "where", "criteria"],
#         "tools": ["data_filter", "search_engine", "query_builder"],
#         "capabilities": [
#             "Filter customers by location, value, industry, status",
#             "Search leads by score, source, company",
#             "Create complex search queries",
#             "Display filtered results with details",
#             "Handoff to Action Agent for exports/campaigns"
#         ],
#         "example_queries": [
#             "Show customers in California",
#             "Find high-value leads",
#             "Filter active customers in tech industry",
#             "Display leads with score above 85",
#             "Show all customers by location"
#         ]
#     }

if __name__ == "__main__":
    print("🔍 Starting Action Agent on port 5002...")
    print("🎯 Capabilities: Data filtering, search queries, dataset analysis")
    print("🔗 API Documentation: http://localhost:5002/docs")
    
    uvicorn.run(
        "action_agent:app",
        host="0.0.0.0",
        port=5002,
        reload=True,
        log_level="info"
    )