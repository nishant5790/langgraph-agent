import openai
import json
import uuid
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    content: str
    role: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None

@dataclass
class RegisteredAgent:
    """Simple agent registration"""
    agent_id: str
    name: str
    api_url: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True

class SimpleMultiAgentSystem:
    def __init__(self, openai_api_key: str):
        """Initialize the multi-agent system"""
        self.openai_api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Agent registry
        self.registered_agents: Dict[str, RegisteredAgent] = {}
        
        # Conversation state
        self.conversation_history: List[Message] = []
        self.current_agent_id: Optional[str] = None
        self.session_id = str(uuid.uuid4())
    
    def register_agent(self, 
                      name: str, 
                      api_url: str, 
                      agent_id: str = None,
                      description: str = "",
                      keywords: List[str] = None,
                      headers: Dict[str, str] = None) -> str:
        """
        Register a new agent with minimal configuration
        
        Args:
            name: Human-readable name for the agent
            api_url: API endpoint URL for the agent
            agent_id: Optional custom ID (auto-generated if not provided)
            description: Optional description of what the agent does
            keywords: Optional list of keywords/triggers for this agent
            headers: Optional HTTP headers for API calls
        
        Returns:
            agent_id: The ID assigned to the registered agent
        """
        
        # Generate agent_id if not provided
        if not agent_id:
            agent_id = f"agent_{len(self.registered_agents) + 1}"
        
        # Default headers
        if headers is None:
            headers = {"Content-Type": "application/json"}
        
        # Default keywords from name and description
        if keywords is None:
            keywords = []
            # Extract keywords from name and description
            words = (name + " " + description).lower().split()
            keywords = [word.strip(".,!?") for word in words if len(word) > 3]
        
        # Create registered agent
        agent = RegisteredAgent(
            agent_id=agent_id,
            name=name,
            api_url=api_url,
            description=description,
            keywords=keywords,
            headers=headers
        )
        
        # Test agent connection
        test_result = self._test_agent_connection(agent)
        if not test_result["success"]:
            logger.warning(f"Agent registration warning: {test_result['message']}")
        
        # Register the agent
        self.registered_agents[agent_id] = agent
        
        logger.info(f"✅ Agent '{name}' registered with ID: {agent_id}")
        return agent_id
    
    def unregister_agent(self, agent_id: str) -> bool:
        """Remove an agent from the registry"""
        if agent_id in self.registered_agents:
            agent_name = self.registered_agents[agent_id].name
            del self.registered_agents[agent_id]
            logger.info(f"🗑️ Agent '{agent_name}' unregistered")
            return True
        return False
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        agents_info = []
        for agent_id, agent in self.registered_agents.items():
            agents_info.append({
                "agent_id": agent_id,
                "name": agent.name,
                "description": agent.description,
                "keywords": agent.keywords,
                "enabled": agent.enabled,
                "api_url": agent.api_url
            })
        return agents_info
    
    def process_query(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query through the multi-agent system
        
        Args:
            user_query: The user's input query
            
        Returns:
            Response dict with agent info and response
        """
        
        # Add user message to history
        user_message = Message(content=user_query, role="user")
        self.conversation_history.append(user_message)
        
        # If we have a current agent, try it first
        if self.current_agent_id and self.current_agent_id in self.registered_agents:
            current_agent = self.registered_agents[self.current_agent_id]
            logger.info(f"🔄 Trying current agent: {current_agent.name}")
            
            result = self._call_agent(current_agent, user_query)
            
            # Check if agent wants to continue or handoff
            if result["success"] and not result.get("handoff_requested", False):
                # Agent handled the query successfully
                response_message = Message(
                    content=result["response"],
                    role="assistant", 
                    agent_id=self.current_agent_id
                )
                self.conversation_history.append(response_message)
                
                return {
                    "response": result["response"],
                    "agent_used": current_agent.name,
                    "agent_id": self.current_agent_id,
                    "handoff_occurred": False,
                    "session_id": self.session_id,
                    "execution_time": result.get("execution_time", 0)
                }
        
        # Need to find the right agent (either no current agent or handoff requested)
        logger.info("🔍 Finding appropriate agent for query...")
        
        # Get the best agent for this query
        selected_agent_id = self._select_agent(user_query)
        
        if not selected_agent_id:
            # No suitable agent found
            response = "I don't have a specialized agent for this type of query. Please try rephrasing or register an appropriate agent."
            response_message = Message(content=response, role="assistant")
            self.conversation_history.append(response_message)
            
            return {
                "response": response,
                "agent_used": "System",
                "agent_id": "system",
                "handoff_occurred": False,
                "session_id": self.session_id,
                "error": "No suitable agent found"
            }
        
        # Call the selected agent
        selected_agent = self.registered_agents[selected_agent_id]
        logger.info(f"🎯 Selected agent: {selected_agent.name}")
        
        result = self._call_agent(selected_agent, user_query)
        
        if result["success"]:
            # Update current agent
            previous_agent = self.current_agent_id
            self.current_agent_id = selected_agent_id
            
            # Add response to history
            response_message = Message(
                content=result["response"],
                role="assistant",
                agent_id=selected_agent_id
            )
            self.conversation_history.append(response_message)
            
            return {
                "response": result["response"],
                "agent_used": selected_agent.name,
                "agent_id": selected_agent_id,
                "handoff_occurred": previous_agent != selected_agent_id,
                "previous_agent": previous_agent,
                "session_id": self.session_id,
                "execution_time": result.get("execution_time", 0)
            }
        
        else:
            # Agent call failed
            error_response = f"Sorry, I encountered an issue with the {selected_agent.name}. Please try again."
            response_message = Message(content=error_response, role="assistant")
            self.conversation_history.append(response_message)
            
            return {
                "response": error_response,
                "agent_used": selected_agent.name,
                "agent_id": selected_agent_id,
                "handoff_occurred": False,
                "session_id": self.session_id,
                "error": result.get("error", "Agent call failed")
            }
    
    def _select_agent(self, query: str) -> Optional[str]:
        """Select the best agent for the query using GPT-4o"""
        
        if not self.registered_agents:
            return None
        
        # Prepare agent information for LLM
        agents_info = []
        for agent_id, agent in self.registered_agents.items():
            if agent.enabled:
                agents_info.append({
                    "agent_id": agent_id,
                    "name": agent.name,
                    "description": agent.description,
                    "keywords": agent.keywords
                })
        
        if not agents_info:
            return None
        
        # Create system prompt
        system_prompt = f"""
        You are an intelligent agent router. Your job is to select the best agent to handle a user query.
        
        Available agents:
        {json.dumps(agents_info, indent=2)}
        
        Based on the user query, select the most appropriate agent by analyzing:
        1. Keywords that match the query
        2. Description relevance
        3. Context from the query
        
        Respond with JSON containing:
        - "selected_agent_id": the ID of the best agent (or null if none suitable)
        - "confidence": confidence score 0-1
        - "reasoning": brief explanation of choice
        
        If no agent seems suitable, return null for selected_agent_id.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {query}"}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            selected_id = result.get("selected_agent_id")
            confidence = result.get("confidence", 0)
            reasoning = result.get("reasoning", "")
            
            logger.info(f"🤖 Agent selection - ID: {selected_id}, Confidence: {confidence:.2f}, Reasoning: {reasoning}")
            
            # Return selected agent if confidence is reasonable
            if selected_id and confidence > 0.3:
                return selected_id
            
        except Exception as e:
            logger.error(f"Error in agent selection: {str(e)}")
        
        # Fallback: simple keyword matching
        return self._fallback_agent_selection(query)
    
    def _fallback_agent_selection(self, query: str) -> Optional[str]:
        """Fallback agent selection using keyword matching"""
        query_lower = query.lower()
        best_match = None
        best_score = 0
        
        for agent_id, agent in self.registered_agents.items():
            if not agent.enabled:
                continue
                
            score = 0
            # Check keywords
            for keyword in agent.keywords:
                if keyword.lower() in query_lower:
                    score += 1
            
            # Check name
            if agent.name.lower() in query_lower:
                score += 2
            
            # Check description
            for word in agent.description.lower().split():
                if word in query_lower and len(word) > 3:
                    score += 0.5
            
            if score > best_score:
                best_score = score
                best_match = agent_id
        
        logger.info(f"🎯 Fallback selection: {best_match} (score: {best_score})")
        return best_match if best_score > 0 else None
    
    def _call_agent(self, agent: RegisteredAgent, query: str) -> Dict[str, Any]:
        """Call an agent's API with the query"""
        
        # Prepare context (last few messages)
        context = []
        for msg in self.conversation_history[-5:]:  # Last 5 messages
            context.append({
                "role": msg.role,
                "content": msg.content,
                "agent_id": getattr(msg, 'agent_id', None)
            })
        
        # Prepare payload - simple and standardized
        payload = {
            "query": query,
            "context": context,
            "session_id": self.session_id
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                agent.api_url,
                json=payload,
                headers=agent.headers,
                timeout=30
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle different response formats
                if isinstance(data, str):
                    # Simple string response
                    return {
                        "success": True,
                        "response": data,
                        "execution_time": execution_time
                    }
                elif isinstance(data, dict):
                    # Structured response
                    return {
                        "success": True,
                        "response": data.get("response", str(data)),
                        "handoff_requested": data.get("handoff_requested", False),
                        "execution_time": execution_time,
                        "metadata": data.get("metadata", {})
                    }
                else:
                    return {
                        "success": True,
                        "response": str(data),
                        "execution_time": execution_time
                    }
            
            else:
                logger.warning(f"Agent API returned status {response.status_code}")
                return {
                    "success": False,
                    "error": f"API returned status {response.status_code}",
                    "execution_time": execution_time
                }
                
        except requests.exceptions.Timeout:
            logger.warning(f"Agent API timeout: {agent.name}")
            return {
                "success": False,
                "error": "Request timeout"
            }
        
        except requests.exceptions.ConnectionError:
            logger.warning(f"Agent API connection error: {agent.name}")
            return {
                "success": False,
                "error": "Connection error"
            }
        
        except Exception as e:
            logger.error(f"Agent API error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _test_agent_connection(self, agent: RegisteredAgent) -> Dict[str, Any]:
        """Test if agent API is reachable"""
        test_payload = {
            "query": "test connection",
            "context": [],
            "session_id": "test"
        }
        
        try:
            response = requests.post(
                agent.api_url,
                json=test_payload,
                headers=agent.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return {"success": True, "message": "Connection successful"}
            else:
                return {"success": False, "message": f"API returned status {response.status_code}"}
                
        except Exception as e:
            return {"success": False, "message": f"Connection failed: {str(e)}"}
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        history = []
        for msg in self.conversation_history:
            history.append({
                "role": msg.role,
                "content": msg.content,
                "agent_id": getattr(msg, 'agent_id', None),
                "timestamp": msg.timestamp.isoformat()
            })
        return history
    
    def clear_conversation(self) -> None:
        """Clear conversation history and reset session"""
        self.conversation_history = []
        self.current_agent_id = None
        self.session_id = str(uuid.uuid4())
        logger.info("🧹 Conversation cleared")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return {
            "session_id": self.session_id,
            "current_agent": self.current_agent_id,
            "message_count": len(self.conversation_history),
            "registered_agents": len(self.registered_agents)
        }

# ====================
# MOCK AGENTS FOR TESTING
# ====================

def create_mock_filter_agent():
    """Mock filter agent API"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/process', methods=['POST'])
    def process():
        data = request.json
        query = data.get('query', '')
        
        if any(word in query.lower() for word in ['filter', 'search', 'find', 'show']):
            return jsonify({
                "response": f"✅ Filtered data based on: '{query}'. Found 15 matching records.",
                "handoff_requested": False,
                "metadata": {"records_found": 15}
            })
        else:
            return jsonify({
                "response": "This query doesn't seem to be about filtering data. Try a different agent.",
                "handoff_requested": True
            })
    
    return app

def create_mock_action_agent():
    """Mock action agent API"""
    from flask import Flask, request, jsonify
    
    app = Flask(__name__)
    
    @app.route('/process', methods=['POST'])
    def process():
        data = request.json
        query = data.get('query', '')
        
        if any(word in query.lower() for word in ['add', 'create', 'export', 'csv', 'marketo', 'outreach']):
            return jsonify({
                "response": f"✅ Action completed: '{query}'. Operation successful!",
                "handoff_requested": False,
                "metadata": {"action_type": "automated"}
            })
        else:
            return jsonify({
                "response": "This query doesn't seem to be about performing actions. Try a different agent.",
                "handoff_requested": True
            })
    
    return app

# ====================
# SIMPLE USAGE EXAMPLES
# ====================

def example_usage():
    """Show how easy it is to use the system"""
    
    print("🚀 Simple Multi-Agent System Demo")
    print("=" * 50)
    
    # Initialize system
    system = SimpleMultiAgentSystem(openai_api_key="your-openai-api-key-here")
    
    # Register agents with minimal configuration
    print("\n📝 Registering agents...")
    
    # Method 1: Minimal registration (just name and URL)
    filter_agent_id = system.register_agent(
        name="Data Filter Agent",
        api_url="https://your-filter-api.com/process"
    )
    
    # Method 2: With description and keywords
    action_agent_id = system.register_agent(
        name="Action Agent",
        api_url="https://your-action-api.com/process",
        description="Handles outreach campaigns, CSV exports, and Marketo integration",
        keywords=["outreach", "export", "csv", "marketo", "campaign"]
    )
    
    # Method 3: With custom headers (for authentication)
    analytics_agent_id = system.register_agent(
        name="Analytics Agent",
        api_url="https://your-analytics-api.com/process",
        description="Provides data insights and analytics",
        keywords=["analytics", "insights", "reports", "dashboard"],
        headers={"Authorization": "Bearer your-token-here"}
    )
    
    print("✅ All agents registered!")
    
    # List registered agents
    print("\n📋 Registered agents:")
    agents = system.list_agents()
    for agent in agents:
        print(f"  • {agent['name']} (ID: {agent['agent_id']})")
        print(f"    Keywords: {', '.join(agent['keywords'])}")
    
    # Process queries
    print("\n💬 Processing queries...")
    
    test_queries = [
        "Filter customers by location",
        "Add outreach campaign for new leads", 
        "Generate analytics report",
        "Export data to CSV",
        "Show me insights from last month"
    ]
    
    for query in test_queries:
        print(f"\n👤 User: {query}")
        result = system.process_query(query)
        print(f"🤖 {result['agent_used']}: {result['response']}")
        if result.get('handoff_occurred'):
            print(f"🔄 Handoff occurred from {result.get('previous_agent', 'None')}")
    
    # Show session info
    print(f"\n📊 Session info: {system.get_session_info()}")

# ====================
# MAIN EXECUTION
# ====================

if __name__ == "__main__":
    print("🔧 Simple Multi-Agent System")
    print("=" * 40)
    
    # Quick start example
    system = SimpleMultiAgentSystem(openai_api_key="your-api-key")
    
    print("\n📝 QUICK START - Register your first agent:")
    print("system.register_agent(")
    print("    name='My Agent',")
    print("    api_url='https://my-api.com/process'")
    print(")")
    
    # Example with mock data
    print("\n🧪 Demo with mock agents...")
    
    # Register demo agents
    filter_id = system.register_agent(
        name="Filter Agent",
        api_url="" \
        "",  # Mock URL
        description="Filters and searches data",
        keywords=["filter", "search", "find", "data"]
    )
    
    action_id = system.register_agent(
        name="Action Agent", 
        api_url="http://localhost:5002/process",  # Mock URL
        description="Performs actions like exports and campaigns",
        keywords=["add", "create", "export", "outreach", "marketo"]
    )
    
    print(f"✅ Registered {len(system.registered_agents)} agents")
    
    # Interactive demo
    print("\n💬 Try these example queries:")
    examples = [
        "Filter customers by region",
        "Add new outreach campaign",
        "Export data to CSV", 
        "Find high-value leads"
    ]
    
    for example in examples:
        print(f"  • {example}")
    
    print("\n" + "="*60)
    print("🎯 KEY FEATURES:")
    print("✅ Simple agent registration - just name and API URL")
    print("✅ Automatic agent selection using GPT-4o")
    print("✅ Intelligent handoff between agents")
    print("✅ Conversation context preservation")
    print("✅ Error handling and fallbacks")
    print("✅ Session management")
    print("=" * 60)
    
    print("\n📚 INTEGRATION GUIDE:")
    print("1. Initialize: system = SimpleMultiAgentSystem(api_key)")
    print("2. Register: system.register_agent(name, api_url)")
    print("3. Process: system.process_query(user_input)")
    print("4. That's it! The system handles everything else.")
    
    # Optional: Start interactive mode
    print("\n💡 Want to test interactively? Uncomment the code below:")
    print("""
    # Interactive testing
    while True:
        query = input("\\n👤 Your query (or 'quit'): ")
        if query.lower() == 'quit':
            break
        result = system.process_query(query)
        print(f"🤖 {result['agent_used']}: {result['response']}")
    """)