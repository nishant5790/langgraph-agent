#!/usr/bin/env python3
"""
Filter Agent FastAPI Server
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
    title="Filter Agent API",
    description="Specialized agent for filtering data, creating search queries, and finding specific information from datasets",
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

# Mock database for demonstration
MOCK_DATA = {
    "customers": [
        {"id": 1, "name": "Acme Corp", "location": "California", "value": "high", "industry": "tech", "status": "active"},
        {"id": 2, "name": "Beta LLC", "location": "New York", "value": "medium", "industry": "finance", "status": "active"},
        {"id": 3, "name": "Gamma Inc", "location": "Texas", "value": "high", "industry": "healthcare", "status": "inactive"},
        {"id": 4, "name": "Delta Co", "location": "California", "value": "low", "industry": "retail", "status": "active"},
        {"id": 5, "name": "Echo Systems", "location": "Florida", "value": "high", "industry": "tech", "status": "active"},
        {"id": 6, "name": "Foxtrot Ltd", "location": "New York", "value": "medium", "industry": "manufacturing", "status": "active"},
        {"id": 7, "name": "Golf Enterprises", "location": "California", "value": "high", "industry": "tech", "status": "active"},
        {"id": 8, "name": "Hotel Corp", "location": "Texas", "value": "low", "industry": "hospitality", "status": "active"},
        {"id": 9, "name": "India Solutions", "location": "Washington", "value": "medium", "industry": "consulting", "status": "active"},
        {"id": 10, "name": "Juliet Group", "location": "California", "value": "high", "industry": "finance", "status": "active"}
    ],
    "leads": [
        {"id": 1, "name": "Alice Johnson", "company": "NewTech", "score": 85, "source": "website"},
        {"id": 2, "name": "Bob Smith", "company": "StartupXYZ", "score": 92, "source": "referral"},
        {"id": 3, "name": "Carol White", "company": "BigCorp", "score": 78, "source": "linkedin"},
        {"id": 4, "name": "David Brown", "company": "MediumBiz", "score": 88, "source": "conference"},
        {"id": 5, "name": "Eva Green", "company": "SmallCo", "score": 95, "source": "website"}
    ]
}

# Authentication middleware
async def verify_token(authorization: str = Header(None)):
    """Verify the authorization token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    if authorization != "Bearer filter-agent-token":
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    return True

def analyze_filter_query(query: str) -> Dict[str, Any]:
    """Analyze the query to determine filter criteria"""
    query_lower = query.lower()
    
    filter_criteria = {}
    
    # Location filters
    locations = ["california", "new york", "texas", "florida", "washington"]
    for location in locations:
        if location in query_lower:
            filter_criteria["location"] = location.title()
            break
    
    # Value filters
    if any(word in query_lower for word in ["high value", "high-value", "premium", "top"]):
        filter_criteria["value"] = "high"
    elif any(word in query_lower for word in ["low value", "low-value", "budget"]):
        filter_criteria["value"] = "low"
    elif any(word in query_lower for word in ["medium value", "mid-tier"]):
        filter_criteria["value"] = "medium"
    
    # Industry filters
    industries = ["tech", "finance", "healthcare", "retail", "manufacturing", "hospitality", "consulting"]
    for industry in industries:
        if industry in query_lower:
            filter_criteria["industry"] = industry
            break
    
    # Status filters
    if any(word in query_lower for word in ["active", "current"]):
        filter_criteria["status"] = "active"
    elif any(word in query_lower for word in ["inactive", "dormant"]):
        filter_criteria["status"] = "inactive"
    
    # Lead score filters
    if any(word in query_lower for word in ["high score", "top score", "best leads"]):
        filter_criteria["min_score"] = 85
    elif any(word in query_lower for word in ["qualified", "good leads"]):
        filter_criteria["min_score"] = 80
    
    return filter_criteria

def apply_filters(data_type: str, criteria: Dict[str, Any]) -> List[Dict]:
    """Apply filter criteria to the specified data type"""
    if data_type == "customers":
        data = MOCK_DATA["customers"]
    elif data_type == "leads":
        data = MOCK_DATA["leads"]
    else:
        return []
    
    filtered_data = data.copy()
    
    for key, value in criteria.items():
        if key == "min_score":
            filtered_data = [item for item in filtered_data if item.get("score", 0) >= value]
        else:
            filtered_data = [item for item in filtered_data if item.get(key, "").lower() == value.lower()]
    
    return filtered_data

def should_handoff_to_action(query: str, context: Context) -> tuple[bool, Optional[str]]:
    """Determine if query should be handed off to action agent"""
    query_lower = query.lower()
    
    # Check for action-related keywords
    action_keywords = ["export", "csv", "download", "campaign", "outreach", "marketo", 
                      "send", "create campaign", "add to", "integrate"]
    
    for keyword in action_keywords:
        if keyword in query_lower:
            return True, "action_agent"
    
    # Check conversation context for sequential requests
    if context.conversation_history:
        last_messages = context.conversation_history[-3:]  # Check last 3 messages
        for msg in last_messages:
            if msg.role == "user" and any(word in msg.content.lower() for word in action_keywords):
                return True, "action_agent"
    
    return False, None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "filter_agent", "timestamp": datetime.now().isoformat()}

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
    
    # # Check if this should be handed off to action agent
    # should_handoff, target_agent = should_handoff_to_action(query, request.context)
    
    # if should_handoff:
    #     return AgentResponse(
    #         response=f"I can help filter the data, but for actions like exporting or campaigns, I'll hand this off to the Action Agent.",
    #         handoff_requested=True,
    #         target_agent=target_agent,
    #         metadata={"reason": "action_required"},
    #         tools_used=["handoff_detector"]
    #     )
    
    # Determine data type from query
    data_type = "customers"
    if any(word in query_lower for word in ["lead", "leads", "prospect", "prospects"]):
        data_type = "leads"
    
    # Analyze and apply filters
    filter_criteria = analyze_filter_query(query)
    
    if not filter_criteria:
        # Generic search/show request
        if any(word in query_lower for word in ["show", "display", "list", "all"]):
            if data_type == "leads":
                results = MOCK_DATA["leads"]
                response_text = f"📋 Showing all leads ({len(results)} total):\n"
                for lead in results[:5]:  # Show first 5
                    response_text += f"• {lead['name']} from {lead['company']} (Score: {lead['score']})\n"
                if len(results) > 5:
                    response_text += f"... and {len(results) - 5} more leads"
            else:
                results = MOCK_DATA["customers"]
                response_text = f"📋 Showing all customers ({len(results)} total):\n"
                for customer in results[:5]:  # Show first 5
                    response_text += f"• {customer['name']} - {customer['location']} ({customer['value']} value)\n"
                if len(results) > 5:
                    response_text += f"... and {len(results) - 5} more customers"
            
            return AgentResponse(
                response=response_text,
                handoff_requested=False,
                metadata={"data_type": data_type, "total_count": len(results), "filter_applied": False},
                tools_used=["data_filter", "query_builder"]
            )
        else:
            return AgentResponse(
                response="I can help you filter data. Try queries like:\n• 'Show customers in California'\n• 'Find high-value leads'\n• 'Filter active customers by industry'\n• 'Display leads with high scores'",
                handoff_requested=False,
                metadata={"suggestion": True},
                tools_used=["query_builder"]
            )
    
    # Apply filters
    filtered_results = apply_filters(data_type, filter_criteria)
    
    if not filtered_results:
        criteria_text = ", ".join([f"{k}={v}" for k, v in filter_criteria.items()])
        return AgentResponse(
            response=f"🔍 No {data_type} found matching criteria: {criteria_text}. Try adjusting your filter criteria.",
            handoff_requested=False,
            metadata={"data_type": data_type, "filter_criteria": filter_criteria, "results_count": 0},
            tools_used=["data_filter", "search_engine"]
        )
    
    # Format response
    criteria_text = ", ".join([f"{k}: {v}" for k, v in filter_criteria.items()])
    
    if data_type == "leads":
        response_text = f"🎯 Found {len(filtered_results)} leads matching criteria ({criteria_text}):\n"
        for lead in filtered_results[:5]:  # Show first 5
            response_text += f"• {lead['name']} from {lead['company']} - Score: {lead['score']} (Source: {lead['source']})\n"
        if len(filtered_results) > 5:
            response_text += f"... and {len(filtered_results) - 5} more leads"
    else:
        response_text = f"🎯 Found {len(filtered_results)} customers matching criteria ({criteria_text}):\n"
        for customer in filtered_results[:5]:  # Show first 5
            response_text += f"• {customer['name']} - {customer['location']}, {customer['industry']} ({customer['value']} value, {customer['status']})\n"
        if len(filtered_results) > 5:
            response_text += f"... and {len(filtered_results) - 5} more customers"
    
    response_text += f"\n\n💡 Would you like me to export this data or create a campaign? I can hand that off to the Action Agent."
    
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
    print("🔍 Starting Filter Agent on port 5001...")
    print("🎯 Capabilities: Data filtering, search queries, dataset analysis")
    print("🔗 API Documentation: http://localhost:5001/docs")
    
    uvicorn.run(
        "filter_agent:app",
        host="0.0.0.0",
        port=5001,
        reload=True,
        log_level="info"
    )