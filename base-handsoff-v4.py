import openai
import json
import uuid
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    MAIN = "main"
    API_AGENT = "api_agent"

class HandoffDecision(Enum):
    CONTINUE = "continue"
    HANDOFF_TO_MAIN = "handoff_to_main"
    HANDOFF_TO_AGENT = "handoff_to_agent"
    DIRECT_RESPONSE = "direct_response"

@dataclass
class Message:
    content: str
    role: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMemory:
    agent_interactions: Dict[str, List[Message]] = field(default_factory=dict)
    agent_capabilities: Dict[str, List[str]] = field(default_factory=dict)
    context_history: List[Dict[str, Any]] = field(default_factory=list)
    current_agent: Optional[str] = None
    handoff_count: int = 0

class BaseAgent(ABC):
    def __init__(self, agent_id: str, name: str, description: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        
    @abstractmethod
    def process_query(self, query: str, context: List[Message]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def should_handoff(self, query: str, context: List[Message]) -> Tuple[HandoffDecision, Optional[str]]:
        pass

class MainAgent(BaseAgent):
    def __init__(self, openai_api_key: str):
        super().__init__(
            agent_id="main_agent",
            name="Main Orchestrator Agent",
            description="Main agent responsible for routing queries to Filter Agent or Action Agent",
            capabilities=["routing", "orchestration", "context_management", "handoff_decision"]
        )
        self.client = openai.OpenAI(api_key=openai_api_key)
        
    def process_query(self, query: str, context: List[Message]) -> Dict[str, Any]:
        """Process query and decide on agent routing"""
        
        system_prompt = f"""
        You are the Main Orchestrator Agent in a multi-agent system with two specialized agents:
        
        1. FILTER AGENT (filter_agent):
           - Capabilities: data filtering, query processing, search operations
           - Use for: "filter", "search", "find", "show me", "get data", "retrieve"
           - Example queries: "Filter customers by location", "Find all leads from last month"
        
        2. ACTION AGENT (action_agent):
           - Capabilities: add_outreach, get_csv, add_marketo, data actions, automation
           - Use for: "add", "create", "execute", "perform", "automate", "export"
           - Example queries: "Add outreach campaign", "Export to CSV", "Add to Marketo"
        
        Available agents and their capabilities:
        {self._get_available_agents_info()}
        
        Current context: {len(context)} previous messages
        
        Analyze the user query and determine:
        - Does it involve FILTERING/SEARCHING data? → route to filter_agent
        - Does it involve ACTIONS/OPERATIONS? → route to action_agent
        - Is it general conversation? → handle yourself
        
        Provide a JSON response with:
        - "target_agent": agent_id to route to (filter_agent, action_agent, or main_agent)
        - "reasoning": explanation of routing decision
        - "context_summary": brief summary of relevant context
        - "response": your response if handling the query yourself
        - "confidence": confidence level (0-1) in routing decision
        """
        
        # Prepare context for LLM
        context_messages = []
        for msg in context[-10:]:
            context_messages.append({
                "role": msg.role,
                "content": f"[{msg.agent_id}] {msg.content}" if msg.agent_id else msg.content
            })
        
        messages = [
            {"role": "system", "content": system_prompt},
            *context_messages,
            {"role": "user", "content": query}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return result
            
        except Exception as e:
            logger.error(f"Main agent processing error: {str(e)}")
            return {
                "target_agent": "main_agent",
                "reasoning": f"Error in processing: {str(e)}",
                "context_summary": "Error occurred",
                "response": "I encountered an error processing your request. Please try again.",
                "confidence": 0.0
            }
    
    def should_handoff(self, query: str, context: List[Message]) -> Tuple[HandoffDecision, Optional[str]]:
        """Main agent always processes queries first"""
        return HandoffDecision.CONTINUE, None
    
    def _get_available_agents_info(self) -> str:
        """Get information about available agents"""
        return "This will be populated by the system with available agents"

class APIAgent(BaseAgent):
    def __init__(self, agent_id: str, name: str, description: str, capabilities: List[str], 
                 api_endpoint: str, api_headers: Dict[str, str] = None):
        super().__init__(agent_id, name, description, capabilities)
        self.api_endpoint = api_endpoint
        self.api_headers = api_headers or {"Content-Type": "application/json"}
        self.request_timeout = 30
        self.max_retries = 3
        
    def process_query(self, query: str, context: List[Message]) -> Dict[str, Any]:
        """Process query through external API"""
        
        # Prepare context for API call
        context_data = []
        for msg in context[-5:]:  # Last 5 messages for context
            context_data.append({
                "role": msg.role,
                "content": msg.content,
                "agent_id": msg.agent_id,
                "timestamp": msg.timestamp.isoformat()
            })
        
        payload = {
            "query": query,
            "context": context_data,
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Make API call with retries
        for attempt in range(self.max_retries):
            try:
                logger.info(f"🌐 Calling {self.name} API (attempt {attempt + 1})")
                response = requests.post(
                    self.api_endpoint,
                    json=payload,
                    headers=self.api_headers,
                    timeout=self.request_timeout
                )
                
                if response.status_code == 200:
                    api_result = response.json()
                    
                    # Analyze response to determine if handoff is needed
                    should_handoff = self._analyze_for_handoff(query, api_result)
                    
                    return {
                        "response": api_result.get("response", "API response received"),
                        "confidence": api_result.get("confidence", 0.8),
                        "should_handoff": should_handoff,
                        "handoff_reason": api_result.get("handoff_reason"),
                        "api_status": "success",
                        "api_metadata": api_result.get("metadata", {}),
                        "tools_used": api_result.get("tools_used", []),
                        "execution_time": api_result.get("execution_time", 0)
                    }
                
                elif response.status_code == 422:
                    # Validation error - likely need different agent
                    logger.warning(f"API validation error: {response.text}")
                    return {
                        "response": "This query seems to be outside my capabilities. Let me handoff to a more suitable agent.",
                        "confidence": 0.1,
                        "should_handoff": True,
                        "handoff_reason": f"API validation error: {response.status_code}",
                        "api_status": "validation_error"
                    }
                
                elif response.status_code >= 500:
                    # Server error - retry
                    logger.warning(f"API server error {response.status_code}, retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                
                else:
                    # Client error - probably wrong agent
                    logger.warning(f"API client error: {response.status_code}")
                    return {
                        "response": "I'm not able to handle this type of request. Let me find a better agent for you.",
                        "confidence": 0.2,
                        "should_handoff": True,
                        "handoff_reason": f"API error: {response.status_code}",
                        "api_status": "client_error"
                    }
                    
            except requests.exceptions.Timeout:
                logger.warning(f"API timeout (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    return {
                        "response": "The request timed out. Please try again or rephrase your query.",
                        "confidence": 0.0,
                        "should_handoff": True,
                        "handoff_reason": "API timeout",
                        "api_status": "timeout"
                    }
                time.sleep(2 ** attempt)
                
            except requests.exceptions.ConnectionError:
                logger.error(f"API connection error (attempt {attempt + 1})")
                if attempt == self.max_retries - 1:
                    return {
                        "response": "Unable to connect to the service. Please try again later.",
                        "confidence": 0.0,
                        "should_handoff": True,
                        "handoff_reason": "API connection error",
                        "api_status": "connection_error"
                    }
                time.sleep(2 ** attempt)
                
            except Exception as e:
                logger.error(f"Unexpected API error: {str(e)}")
                return {
                    "response": f"An unexpected error occurred: {str(e)}",
                    "confidence": 0.0,
                    "should_handoff": True,
                    "handoff_reason": f"Unexpected error: {str(e)}",
                    "api_status": "error"
                }
        
        # All retries failed
        return {
            "response": "Service is currently unavailable. Please try again later.",
            "confidence": 0.0,
            "should_handoff": True,
            "handoff_reason": "API service unavailable",
            "api_status": "unavailable"
        }
    
    def should_handoff(self, query: str, context: List[Message]) -> Tuple[HandoffDecision, Optional[str]]:
        """Determine if API agent should handoff"""
        result = self.process_query(query, context)
        
        if result.get("should_handoff", False):
            return HandoffDecision.HANDOFF_TO_MAIN, "main_agent"
        elif result.get("confidence", 0) < 0.3:
            return HandoffDecision.HANDOFF_TO_MAIN, "main_agent"
        elif result.get("api_status") in ["validation_error", "client_error"]:
            return HandoffDecision.HANDOFF_TO_MAIN, "main_agent"
        else:
            return HandoffDecision.DIRECT_RESPONSE, None
    
    def _analyze_for_handoff(self, query: str, api_result: Dict[str, Any]) -> bool:
        """Analyze API response to determine if handoff is needed"""
        
        # Check explicit handoff indicators from API
        if api_result.get("should_handoff", False):
            return True
        
        # Check confidence level
        if api_result.get("confidence", 1.0) < 0.3:
            return True
        
        # Check for error indicators in response
        response = api_result.get("response", "").lower()
        handoff_indicators = [
            "not my expertise", "can't help", "unable to", "outside my capabilities",
            "don't understand", "unclear", "not sure", "different agent"
        ]
        
        for indicator in handoff_indicators:
            if indicator in response:
                return True
        
        # Check if API returned minimal or error response
        if len(response) < 20 and any(word in response for word in ["error", "fail", "unable"]):
            return True
        
        return False

# ====================
# MOCK API SERVERS FOR TESTING
# ====================

class MockAPIServer:
    """Mock API server to simulate your existing agents"""
    
    @staticmethod
    def create_filter_agent_response(query: str, context: List[Dict]) -> Dict[str, Any]:
        """Simulate Filter Agent API response"""
        
        query_lower = query.lower()
        
        # Successful filter operations
        if any(word in query_lower for word in ["filter", "find", "search", "show", "get", "retrieve"]):
            if "timeout" in query_lower:
                # Simulate timeout scenario
                time.sleep(35)
            
            return {
                "response": f"Successfully filtered data based on your query: '{query}'. Found 23 matching records.",
                "confidence": 0.9,
                "should_handoff": False,
                "metadata": {
                    "records_found": 23,
                    "filter_criteria": query,
                    "execution_time": 1.2
                },
                "tools_used": ["data_filter", "query_processor"]
            }
        
        # Edge case: Action-related queries (should handoff)
        elif any(word in query_lower for word in ["add", "create", "export", "marketo", "outreach"]):
            return {
                "response": "This appears to be an action request. I specialize in filtering and searching data.",
                "confidence": 0.2,
                "should_handoff": True,
                "handoff_reason": "Query requires action capabilities, not filtering"
            }
        
        # Edge case: Unclear queries
        elif len(query.strip()) < 5 or query_lower in ["hello", "hi", "test"]:
            return {
                "response": "I need more specific information about what data you want to filter or search for.",
                "confidence": 0.3,
                "should_handoff": True,
                "handoff_reason": "Query too vague for filtering operations"
            }
        
        # Edge case: Complex queries outside scope
        else:
            return {
                "response": "I'm not sure how to filter data based on this query. Could you be more specific?",
                "confidence": 0.4,
                "should_handoff": True,
                "handoff_reason": "Query unclear or outside filtering scope"
            }
    
    @staticmethod
    def create_action_agent_response(query: str, context: List[Dict]) -> Dict[str, Any]:
        """Simulate Action Agent API response"""
        
        query_lower = query.lower()
        
        # Successful action operations
        if "add_outreach" in query_lower or "outreach" in query_lower:
            return {
                "response": "Successfully created outreach campaign with the specified parameters.",
                "confidence": 0.95,
                "should_handoff": False,
                "metadata": {
                    "campaign_id": "ORC_12345",
                    "recipients": 150,
                    "execution_time": 2.1
                },
                "tools_used": ["add_outreach", "campaign_manager"]
            }
        
        elif "get_csv" in query_lower or "export" in query_lower or "csv" in query_lower:
            return {
                "response": "CSV export completed successfully. Download link: https://example.com/export_12345.csv",
                "confidence": 0.92,
                "should_handoff": False,
                "metadata": {
                    "file_size": "2.3MB",
                    "rows_exported": 1247,
                    "execution_time": 3.5
                },
                "tools_used": ["get_csv", "data_exporter"]
            }
        
        elif "add_marketo" in query_lower or "marketo" in query_lower:
            return {
                "response": "Successfully added contacts to Marketo with the specified tags and segmentation.",
                "confidence": 0.88,
                "should_handoff": False,
                "metadata": {
                    "contacts_added": 89,
                    "marketo_program_id": "MKT_9876",
                    "execution_time": 4.2
                },
                "tools_used": ["add_marketo", "marketo_connector"]
            }
        
        # Edge case: Filter-related queries (should handoff)
        elif any(word in query_lower for word in ["filter", "find", "search", "show me"]):
            return {
                "response": "This looks like a filtering request. I specialize in performing actions and operations.",
                "confidence": 0.25,
                "should_handoff": True,
                "handoff_reason": "Query requires filtering capabilities, not actions"
            }
        
        # Edge case: API errors
        elif "error" in query_lower:
            return {
                "response": "Internal service error occurred",
                "confidence": 0.1,
                "should_handoff": True,
                "handoff_reason": "Service error",
                "error": True
            }
        
        # Edge case: Unsupported actions
        else:
            return {
                "response": "I can perform actions like add_outreach, get_csv, and add_marketo. This request doesn't match my available actions.",
                "confidence": 0.3,
                "should_handoff": True,
                "handoff_reason": "Unsupported action type"
            }

# Mock Flask-like API endpoint simulation
def mock_filter_agent_api(request_data):
    """Mock Filter Agent API endpoint"""
    query = request_data.get("query", "")
    context = request_data.get("context", [])
    
    # Simulate various response scenarios
    if "500_error" in query:
        return {"status": 500, "error": "Internal server error"}
    elif "422_error" in query:
        return {"status": 422, "error": "Validation error"}
    elif "timeout" in query:
        time.sleep(35)  # Simulate timeout
        return {"status": 200, "data": MockAPIServer.create_filter_agent_response(query, context)}
    else:
        return {"status": 200, "data": MockAPIServer.create_filter_agent_response(query, context)}

def mock_action_agent_api(request_data):
    """Mock Action Agent API endpoint"""
    query = request_data.get("query", "")
    context = request_data.get("context", [])
    
    # Simulate various response scenarios
    if "500_error" in query:
        return {"status": 500, "error": "Internal server error"}
    elif "422_error" in query:
        return {"status": 422, "error": "Validation error"}
    elif "connection_error" in query:
        raise requests.exceptions.ConnectionError("Connection failed")
    else:
        return {"status": 200, "data": MockAPIServer.create_action_agent_response(query, context)}

# ====================
# ENHANCED API AGENT WITH MOCK INTEGRATION
# ====================

class MockAPIAgent(APIAgent):
    """API Agent that uses mock responses for testing"""
    
    def __init__(self, agent_id: str, name: str, description: str, capabilities: List[str], 
                 mock_api_function):
        # Use a dummy endpoint since we're mocking
        super().__init__(agent_id, name, description, capabilities, "http://mock-api")
        self.mock_api_function = mock_api_function
    
    def process_query(self, query: str, context: List[Message]) -> Dict[str, Any]:
        """Process query through mock API"""
        
        # Prepare context for API call
        context_data = []
        for msg in context[-5:]:
            context_data.append({
                "role": msg.role,
                "content": msg.content,
                "agent_id": msg.agent_id,
                "timestamp": msg.timestamp.isoformat()
            })
        
        payload = {
            "query": query,
            "context": context_data,
            "agent_id": self.agent_id,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            logger.info(f"🌐 Calling {self.name} Mock API")
            
            # Handle special error cases
            if "connection_error" in query.lower():
                raise requests.exceptions.ConnectionError("Mock connection error")
            elif "timeout" in query.lower():
                raise requests.exceptions.Timeout("Mock timeout")
            
            # Call mock API
            mock_response = self.mock_api_function(payload)
            
            if mock_response["status"] == 200:
                api_result = mock_response["data"]
                should_handoff = self._analyze_for_handoff(query, api_result)
                
                return {
                    "response": api_result.get("response", "API response received"),
                    "confidence": api_result.get("confidence", 0.8),
                    "should_handoff": should_handoff,
                    "handoff_reason": api_result.get("handoff_reason"),
                    "api_status": "success",
                    "api_metadata": api_result.get("metadata", {}),
                    "tools_used": api_result.get("tools_used", []),
                    "execution_time": api_result.get("execution_time", 0)
                }
            
            elif mock_response["status"] == 422:
                return {
                    "response": "This query seems to be outside my capabilities.",
                    "confidence": 0.1,
                    "should_handoff": True,
                    "handoff_reason": f"API validation error: {mock_response.get('error')}",
                    "api_status": "validation_error"
                }
            
            else:
                return {
                    "response": "Service error occurred. Let me find a better agent for you.",
                    "confidence": 0.2,
                    "should_handoff": True,
                    "handoff_reason": f"API error: {mock_response['status']}",
                    "api_status": "server_error"
                }
                
        except requests.exceptions.Timeout:
            return {
                "response": "The request timed out. Please try again.",
                "confidence": 0.0,
                "should_handoff": True,
                "handoff_reason": "API timeout",
                "api_status": "timeout"
            }
        
        except requests.exceptions.ConnectionError:
            return {
                "response": "Unable to connect to the service.",
                "confidence": 0.0,
                "should_handoff": True,
                "handoff_reason": "API connection error",
                "api_status": "connection_error"
            }
        
        except Exception as e:
            return {
                "response": f"An unexpected error occurred: {str(e)}",
                "confidence": 0.0,
                "should_handoff": True,
                "handoff_reason": f"Unexpected error: {str(e)}",
                "api_status": "error"
            }

# ====================
# MULTI-AGENT SYSTEM
# ====================

class MultiAgentSystem:
    def __init__(self, openai_api_key: str):
        """Initialize the multi-agent system"""
        self.openai_api_key = openai_api_key
        
        self.agents: Dict[str, BaseAgent] = {}
        self.memory = AgentMemory()
        self.session_id = str(uuid.uuid4())
        self.active_agent_id = None
        
        # Initialize main agent
        self.main_agent = MainAgent(openai_api_key)
        self.agents[self.main_agent.agent_id] = self.main_agent
        self.memory.current_agent = self.main_agent.agent_id
        self.active_agent_id = self.main_agent.agent_id
        
        # Update main agent with system reference
        self.main_agent._get_available_agents_info = self._get_agents_info
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add a specialized agent to the system"""
        self.agents[agent.agent_id] = agent
        self.memory.agent_capabilities[agent.agent_id] = agent.capabilities
    
    def _get_agents_info(self) -> str:
        """Get formatted information about all available agents"""
        info = []
        for agent_id, agent in self.agents.items():
            if agent_id != "main_agent":
                info.append(f"- {agent.name} ({agent_id}): {agent.description}")
                info.append(f"  Capabilities: {', '.join(agent.capabilities)}")
        return "\n".join(info)
    
    def process_user_query(self, query: str) -> Dict[str, Any]:
        """Process user query in continuous chat session with agent handoffs"""
        
        # Add user message to memory
        user_message = Message(content=query, role="user")
        self._add_message_to_memory(user_message)
        
        # Get current context
        context = self._get_context_messages()
        
        # Use the currently active agent
        current_agent_id = self.active_agent_id
        current_agent = self.agents[current_agent_id]
        
        logger.info(f"🤖 Current active agent: {current_agent.name}")
        
        # If we're with a specialized agent, first check if it can handle the query
        if current_agent_id != "main_agent":
            result = current_agent.process_query(query, context)
            handoff_decision, target_agent = current_agent.should_handoff(query, context)
            
            if handoff_decision == HandoffDecision.DIRECT_RESPONSE:
                # Specialized agent can handle it
                response_message = Message(
                    content=result.get("response", "No response provided"),
                    role="assistant",
                    agent_id=current_agent_id
                )
                self._add_message_to_memory(response_message)
                
                return {
                    "response": result.get("response", "No response provided"),
                    "agent_used": current_agent.name,
                    "agent_id": current_agent_id,
                    "handoff_occurred": False,
                    "confidence": result.get("confidence", 1.0),
                    "session_id": self.session_id,
                    "api_status": result.get("api_status"),
                    "tools_used": result.get("tools_used", []),
                    "execution_time": result.get("execution_time", 0)
                }
            
            elif handoff_decision == HandoffDecision.HANDOFF_TO_MAIN:
                logger.info(f"🔄 {current_agent.name} handing off to Main Agent")
                logger.info(f"📝 Reason: {result.get('handoff_reason', 'Query outside expertise')}")
                
                # Switch to main agent
                self.active_agent_id = "main_agent"
                current_agent_id = "main_agent"
                current_agent = self.agents[current_agent_id]
                
                # Add handoff message
                handoff_message = Message(
                    content=f"[HANDOFF] {self.agents[self.memory.current_agent].name} handing off to Main Agent. Reason: {result.get('handoff_reason', 'Query requires different expertise')}",
                    role="system",
                    agent_id="system"
                )
                self._add_message_to_memory(handoff_message)
        
        # Process with main agent
        if current_agent_id == "main_agent":
            result = current_agent.process_query(query, context)
            target_agent_id = result.get("target_agent")
            
            if target_agent_id and target_agent_id != "main_agent" and target_agent_id in self.agents:
                logger.info(f"🎯 Main Agent routing to: {self.agents[target_agent_id].name}")
                logger.info(f"💭 Reasoning: {result.get('reasoning', 'Best suited for this query')}")
                
                # Switch active agent
                self.active_agent_id = target_agent_id
                specialized_agent = self.agents[target_agent_id]
                
                # Add handoff message
                handoff_message = Message(
                    content=f"[HANDOFF] Main Agent routing to {specialized_agent.name}. User query: {query}",
                    role="system",
                    agent_id="system"
                )
                self._add_message_to_memory(handoff_message)
                
                # Get response from specialized agent
                specialized_context = self._get_context_messages()
                specialized_result = specialized_agent.process_query(query, specialized_context)
                
                # Store response
                response_message = Message(
                    content=specialized_result.get("response", "Hello! I'm now handling your request."),
                    role="assistant",
                    agent_id=target_agent_id
                )
                self._add_message_to_memory(response_message)
                self.memory.current_agent = target_agent_id
                
                return {
                    "response": specialized_result.get("response", "Hello! I'm now handling your request."),
                    "agent_used": specialized_agent.name,
                    "agent_id": target_agent_id,
                    "handoff_occurred": True,
                    "previous_agent": "Main Agent",
                    "handoff_reason": result.get("reasoning", "Specialized expertise needed"),
                    "confidence": specialized_result.get("confidence", 1.0),
                    "session_id": self.session_id,
                    "api_status": specialized_result.get("api_status"),
                    "tools_used": specialized_result.get("tools_used", []),
                    "execution_time": specialized_result.get("execution_time", 0)
                }
            
            else:
                # Main agent handles the query itself
                response_message = Message(
                    content=result.get("response", "How can I help you?"),
                    role="assistant",
                    agent_id=current_agent_id
                )
                self._add_message_to_memory(response_message)
                self.memory.current_agent = current_agent_id
                
                return {
                    "response": result.get("response", "How can I help you?"),
                    "agent_used": current_agent.name,
                    "agent_id": current_agent_id,
                    "handoff_occurred": False,
                    "reasoning": result.get("reasoning", ""),
                    "confidence": result.get("confidence", 1.0),
                    "session_id": self.session_id
                }
    
    def _add_message_to_memory(self, message: Message) -> None:
        """Add message to memory"""
        if message.agent_id:
            if message.agent_id not in self.memory.agent_interactions:
                self.memory.agent_interactions[message.agent_id] = []
            self.memory.agent_interactions[message.agent_id].append(message)
        
        # Add to context history
        self.memory.context_history.append({
            "message": message,
            "timestamp": message.timestamp,
            "agent_id": message.agent_id
        })
    
    def _get_context_messages(self) -> List[Message]:
        """Get recent context messages"""
        recent_messages = []
        for entry in self.memory.context_history[-20:]:
            recent_messages.append(entry["message"])
        return recent_messages
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of current memory state"""
        return {
            "current_agent": self.memory.current_agent,
            "active_agent": self.active_agent_id,
            "active_agent_name": self.agents[self.active_agent_id].name if self.active_agent_id else None,
            "total_messages": len(self.memory.context_history),
            "agents_used": list(self.memory.agent_interactions.keys()),
            "handoff_count": self.memory.handoff_count,
            "session_id": self.session_id
        }
    
    def get_active_agent_info(self) -> Dict[str, Any]:
        """Get information about currently active agent"""
        if self.active_agent_id and self.active_agent_id in self.agents:
            agent = self.agents[self.active_agent_id]
            return {
                "agent_id": agent.agent_id,
                "name": agent.name,
                "description": agent.description,
                "capabilities": agent.capabilities,
                "is_main_agent": agent.agent_id == "main_agent",
                "agent_type": "api_agent" if isinstance(agent, APIAgent) else "main"
            }
        return None
    
    def clear_memory(self) -> None:
        """Clear memory for new session"""
        self.memory = AgentMemory()
        self.memory.current_agent = "main_agent"
        self.active_agent_id = "main_agent"
        self.session_id = str(uuid.uuid4())

# ====================
# COMPREHENSIVE TEST FRAMEWORK
# ====================

class TestFramework:
    def __init__(self, system: MultiAgentSystem):
        self.system = system
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_test(self, test_name: str, query: str, expected_agent: str = None, 
                 expected_handoff: bool = None, expected_keywords: List[str] = None) -> Dict[str, Any]:
        """Run a single test case"""
        print(f"\n🧪 TEST: {test_name}")
        print(f"📝 Query: {query}")
        
        try:
            # Record initial state
            initial_agent = self.system.get_active_agent_info()
            
            # Process query
            start_time = time.time()
            result = self.system.process_user_query(query)
            execution_time = time.time() - start_time
            
            # Analyze results
            test_result = {
                "test_name": test_name,
                "query": query,
                "result": result,
                "execution_time": execution_time,
                "initial_agent": initial_agent["name"] if initial_agent else None,
                "final_agent": result["agent_used"],
                "handoff_occurred": result.get("handoff_occurred", False),
                "api_status": result.get("api_status", "N/A"),
                "tools_used": result.get("tools_used", []),
                "confidence": result.get("confidence", 0),
                "passed": True,
                "errors": []
            }
            
            # Validate expectations
            if expected_agent and result["agent_id"] != expected_agent:
                test_result["passed"] = False
                test_result["errors"].append(f"Expected agent '{expected_agent}', got '{result['agent_id']}'")
            
            if expected_handoff is not None and result.get("handoff_occurred", False) != expected_handoff:
                test_result["passed"] = False
                test_result["errors"].append(f"Expected handoff: {expected_handoff}, got: {result.get('handoff_occurred', False)}")
            
            if expected_keywords:
                response_lower = result["response"].lower()
                for keyword in expected_keywords:
                    if keyword.lower() not in response_lower:
                        test_result["passed"] = False
                        test_result["errors"].append(f"Expected keyword '{keyword}' not found in response")
            
            # Display results
            print(f"🏁 Final agent: {result['agent_used']}")
            if result.get("handoff_occurred"):
                print(f"🔄 Handoff: {result.get('previous_agent', 'Unknown')} → {result['agent_used']}")
                print(f"📄 Reason: {result.get('handoff_reason', 'Not specified')}")
            
            print(f"🤖 Response: {result['response'][:200]}{'...' if len(result['response']) > 200 else ''}")
            print(f"📊 Confidence: {result.get('confidence', 0):.2f}")
            print(f"⏱️ Execution time: {execution_time:.2f}s")
            
            if result.get("tools_used"):
                print(f"🛠️ Tools used: {', '.join(result['tools_used'])}")
            
            if test_result["passed"]:
                print("✅ TEST PASSED")
                self.passed_tests += 1
            else:
                print("❌ TEST FAILED")
                for error in test_result["errors"]:
                    print(f"   ❗ {error}")
                self.failed_tests += 1
            
            self.test_results.append(test_result)
            return test_result
            
        except Exception as e:
            print(f"💥 TEST ERROR: {str(e)}")
            test_result = {
                "test_name": test_name,
                "query": query,
                "passed": False,
                "errors": [f"Exception: {str(e)}"],
                "execution_time": 0
            }
            self.test_results.append(test_result)
            self.failed_tests += 1
            return test_result
    
    def run_all_tests(self):
        """Run comprehensive test suite covering all edge cases"""
        
        print("🚀 STARTING COMPREHENSIVE MULTI-AGENT TEST SUITE")
        print("=" * 80)
        
        # ==================
        # 1. NORMAL OPERATION TESTS
        # ==================
        print("\n📋 SECTION 1: NORMAL OPERATION TESTS")
        
        # Filter Agent tests
        self.run_test(
            "Filter Agent - Basic Query",
            "Filter customers by location California",
            expected_agent="filter_agent",
            expected_handoff=True,
            expected_keywords=["filtered", "records"]
        )
        
        self.run_test(
            "Filter Agent - Search Query",
            "Find all leads from last month",
            expected_agent="filter_agent",
            expected_handoff=True,
            expected_keywords=["found", "matching"]
        )
        
        # Action Agent tests
        self.run_test(
            "Action Agent - Add Outreach",
            "Add outreach campaign for new leads",
            expected_agent="action_agent",
            expected_handoff=True,
            expected_keywords=["outreach", "campaign"]
        )
        
        self.run_test(
            "Action Agent - CSV Export",
            "Export customer data to CSV",
            expected_agent="action_agent",
            expected_handoff=True,
            expected_keywords=["csv", "export"]
        )
        
        self.run_test(
            "Action Agent - Marketo Integration",
            "Add contacts to Marketo with segmentation",
            expected_agent="action_agent",
            expected_handoff=True,
            expected_keywords=["marketo", "contacts"]
        )
        
        # ==================
        # 2. HANDOFF TESTS
        # ==================
        print("\n📋 SECTION 2: HANDOFF LOGIC TESTS")
        
        # Start with one agent, switch to another
        self.run_test(
            "Cross-Agent Handoff - Filter to Action",
            "Show me all California customers",  # Should go to Filter Agent
            expected_agent="filter_agent",
            expected_handoff=True
        )
        
        self.run_test(
            "Cross-Agent Handoff - Action Request",
            "Now add them to outreach campaign",  # Should handoff to Action Agent
            expected_agent="action_agent",
            expected_handoff=True
        )
        
        # Agent recognizes wrong type of request
        self.run_test(
            "Wrong Agent Recognition - Filter Agent",
            "Create a new marketing campaign",  # Filter agent should handoff
            expected_agent="action_agent",  # Should end up with Action Agent
            expected_handoff=True
        )
        
        # ==================
        # 3. ERROR HANDLING TESTS
        # ==================
        print("\n📋 SECTION 3: ERROR HANDLING TESTS")
        
        # API timeout
        self.run_test(
            "API Timeout Handling",
            "Filter data with timeout scenario",
            expected_handoff=True  # Should handoff due to timeout
        )
        
        # API server error
        self.run_test(
            "API Server Error - 500",
            "Filter 500_error data",
            expected_handoff=True  # Should handoff due to server error
        )
        
        # API validation error
        self.run_test(
            "API Validation Error - 422",
            "Filter 422_error data",
            expected_handoff=True  # Should handoff due to validation error
        )
        
        # Connection error
        self.run_test(
            "API Connection Error",
            "Add connection_error outreach",
            expected_handoff=True  # Should handoff due to connection error
        )
        
        # ==================
        # 4. EDGE CASE TESTS
        # ==================
        print("\n📋 SECTION 4: EDGE CASE TESTS")
        
        # Ambiguous queries
        self.run_test(
            "Ambiguous Query",
            "help",
            expected_agent="main_agent",  # Main agent should handle
            expected_handoff=False
        )
        
        # Empty/minimal queries
        self.run_test(
            "Minimal Query",
            "hi",
            expected_handoff=True  # Should trigger handoff due to vagueness
        )
        
        # Complex multi-intent query
        self.run_test(
            "Multi-Intent Query",
            "Filter customers by region and then export to CSV and add to Marketo",
            expected_handoff=True  # Should route to appropriate agent
        )
        
        # Nonsensical query
        self.run_test(
            "Nonsensical Query",
            "purple monkey dishwasher algorithm",
            expected_agent="main_agent",  # Should stay with main agent
            expected_handoff=False
        )
        
        # ==================
        # 5. CONVERSATION CONTINUITY TESTS
        # ==================
        print("\n📋 SECTION 5: CONVERSATION CONTINUITY TESTS")
        
        # Context-dependent follow-up
        self.run_test(
            "Context Follow-up",
            "Show more details about the previous results",
            # Should continue with current agent
            expected_handoff=False
        )
        
        # Agent-specific follow-up
        self.run_test(
            "Agent-Specific Follow-up",
            "Can you export that data?",
            expected_agent="action_agent",  # Should handoff to Action Agent
            expected_handoff=True
        )
        
        # ==================
        # 6. PERFORMANCE TESTS
        # ==================
        print("\n📋 SECTION 6: PERFORMANCE TESTS")
        
        # Multiple rapid queries
        rapid_queries = [
            "Filter leads by score",
            "Add to outreach",
            "Export results",
            "Find high-value customers",
            "Create marketo campaign"
        ]
        
        for i, query in enumerate(rapid_queries):
            self.run_test(
                f"Rapid Query {i+1}",
                query,
                expected_handoff=True
            )
        
        # ==================
        # 7. RECOVERY TESTS
        # ==================
        print("\n📋 SECTION 7: RECOVERY TESTS")
        
        # Recovery after error
        self.run_test(
            "Recovery After Error",
            "Filter customers by location",  # Normal query after errors
            expected_agent="filter_agent",
            expected_handoff=True
        )
        
        # Agent switching after multiple handoffs
        self.run_test(
            "Multiple Handoff Recovery",
            "Add outreach for filtered customers",
            expected_agent="action_agent",
            expected_handoff=True
        )
        
        # ==================
        # TEST SUMMARY
        # ==================
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("🏁 TEST SUITE COMPLETE - COMPREHENSIVE SUMMARY")
        print("=" * 80)
        
        print(f"📊 OVERALL RESULTS:")
        print(f"   ✅ Passed: {self.passed_tests}")
        print(f"   ❌ Failed: {self.failed_tests}")
        print(f"   📈 Success Rate: {(self.passed_tests / (self.passed_tests + self.failed_tests) * 100):.1f}%")
        
        # Categorize results
        categories = {
            "Normal Operation": [],
            "Handoff Logic": [],
            "Error Handling": [],
            "Edge Cases": [],
            "Conversation Continuity": [],
            "Performance": [],
            "Recovery": []
        }
        
        for result in self.test_results:
            test_name = result["test_name"]
            if "Basic" in test_name or "Search" in test_name or "Add" in test_name or "CSV" in test_name or "Marketo" in test_name:
                categories["Normal Operation"].append(result)
            elif "Handoff" in test_name or "Cross-Agent" in test_name or "Recognition" in test_name:
                categories["Handoff Logic"].append(result)
            elif "Error" in test_name or "Timeout" in test_name or "Connection" in test_name:
                categories["Error Handling"].append(result)
            elif "Ambiguous" in test_name or "Minimal" in test_name or "Multi-Intent" in test_name or "Nonsensical" in test_name:
                categories["Edge Cases"].append(result)
            elif "Context" in test_name or "Follow-up" in test_name:
                categories["Conversation Continuity"].append(result)
            elif "Rapid" in test_name:
                categories["Performance"].append(result)
            elif "Recovery" in test_name:
                categories["Recovery"].append(result)
        
        print(f"\n📋 RESULTS BY CATEGORY:")
        for category, results in categories.items():
            if results:
                passed = sum(1 for r in results if r["passed"])
                total = len(results)
                print(f"   {category}: {passed}/{total} passed ({passed/total*100:.1f}%)")
        
        # Detailed failure analysis
        failed_results = [r for r in self.test_results if not r["passed"]]
        if failed_results:
            print(f"\n❌ FAILED TESTS ANALYSIS:")
            for result in failed_results:
                print(f"   🔸 {result['test_name']}")
                for error in result.get("errors", []):
                    print(f"     ❗ {error}")
        
        # Performance analysis
        execution_times = [r.get("execution_time", 0) for r in self.test_results if "execution_time" in r]
        if execution_times:
            avg_time = sum(execution_times) / len(execution_times)
            max_time = max(execution_times)
            print(f"\n⏱️ PERFORMANCE METRICS:")
            print(f"   Average execution time: {avg_time:.2f}s")
            print(f"   Maximum execution time: {max_time:.2f}s")
        
        # Agent usage analysis
        agent_usage = {}
        for result in self.test_results:
            if "result" in result:
                agent = result["result"].get("agent_id", "unknown")
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        print(f"\n🤖 AGENT USAGE STATISTICS:")
        for agent, count in agent_usage.items():
            percentage = count / len(self.test_results) * 100
            print(f"   {agent}: {count} times ({percentage:.1f}%)")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if self.failed_tests > 0:
            print(f"   • Review failed test cases and improve error handling")
            print(f"   • Consider adjusting confidence thresholds for handoff decisions")
            print(f"   • Enhance API error recovery mechanisms")
        
        if any(r.get("execution_time", 0) > 5 for r in self.test_results):
            print(f"   • Optimize API response times and timeout handling")
        
        print(f"   • Monitor agent handoff patterns in production")
        print(f"   • Implement additional edge case handling based on test results")

# ====================
# MAIN EXECUTION
# ====================

if __name__ == "__main__":
    # Initialize the system
    print("🔧 Initializing Multi-Agent System with API Agents...")
    system = MultiAgentSystem(openai_api_key=None)
    
    # Create and add Filter Agent (using mock API)
    filter_agent = MockAPIAgent(
        agent_id="filter_agent",
        name="Filter Agent",
        description="Specialized in data filtering, querying, and search operations",
        capabilities=["data_filtering", "search", "query_processing", "data_retrieval"],
        mock_api_function=mock_filter_agent_api
    )
    system.add_agent(filter_agent)
    
    # Create and add Action Agent (using mock API)
    action_agent = MockAPIAgent(
        agent_id="action_agent",
        name="Action Agent", 
        description="Specialized in performing actions like add_outreach, get_csv, add_marketo",
        capabilities=["add_outreach", "get_csv", "add_marketo", "automation", "data_export"],
        mock_api_function=mock_action_agent_api
    )
    system.add_agent(action_agent)
    
    print("✅ System initialized with agents:")
    for agent_id, agent in system.agents.items():
        print(f"   • {agent.name} ({agent_id})")
        if hasattr(agent, 'capabilities'):
            print(f"     Capabilities: {', '.join(agent.capabilities)}")
    
    # Initialize and run comprehensive test framework
    # print("\n🧪 Starting Comprehensive Test Framework...")
    # test_framework = TestFramework(system)
    # test_framework.run_all_tests()
    
    # print("\n🎉 All tests completed! Check the summary above for detailed results.")
    
    # Optional: Interactive testing mode
    print("\n" + "="*60)
    print("💬 INTERACTIVE TESTING MODE")
    print("Enter queries to test the system manually (type 'quit' to exit):")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n👤 Your query: ").strip()
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input:
                print("🔍 Processing...")
                result = system.process_user_query(user_input)
                
                print(f"🤖 Agent: {result['agent_used']}")
                print(f"📝 Response: {result['response']}")
                if result.get('handoff_occurred'):
                    print(f"🔄 Handoff: {result.get('handoff_reason', 'N/A')}")
                if result.get('tools_used'):
                    print(f"🛠️ Tools: {', '.join(result['tools_used'])}")
                print(f"📊 Confidence: {result.get('confidence', 0):.2f}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
    
    print("\n🏁 Interactive testing complete!")