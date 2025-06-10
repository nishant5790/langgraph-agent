import openai
import json
import uuid
import requests
import time
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Message:
    content: str
    role: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None

@dataclass
class RegisteredAgent:
    """Agent registration with enhanced configuration"""
    agent_id: str
    name: str
    api_url: str
    description: str = ""
    keywords: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    tools: List[str] = field(default_factory=list)
    handoff_conditions: List[str] = field(default_factory=list)
    priority: int = 1  # Higher number = higher priority
    timeout: int = 30

class MultiAgentOrchestrator:
    def __init__(self, openai_api_key: str):
        """Initialize the multi-agent orchestration system"""
        self.openai_api_key = openai_api_key
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # Agent registry
        self.registered_agents: Dict[str, RegisteredAgent] = {}
        
        # Conversation state
        self.conversation_history: List[Message] = []
        self.current_agent_id: Optional[str] = None
        self.session_id = str(uuid.uuid4())
        self.agent_memory: Dict[str, Any] = {}  # Memory for agent context
        
        # Handoff configuration
        self.max_handoffs = 5  # Prevent infinite loops
        self.current_handoff_count = 0
    
    def load_agents_from_yaml(self, yaml_path: str) -> Dict[str, str]:
        """
        Load and register agents from YAML configuration file
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            Dict mapping agent names to their IDs
        """
        try:
            with open(yaml_path, 'r') as file:
                config = yaml.safe_load(file)
            
            registered_ids = {}
            
            for agent_config in config.get('agents', []):
                agent_id = self.register_agent_from_config(agent_config)
                registered_ids[agent_config['name']] = agent_id
                
            logger.info(f"✅ Loaded {len(registered_ids)} agents from {yaml_path}")
            return registered_ids
            
        except Exception as e:
            logger.error(f"❌ Failed to load agents from YAML: {str(e)}")
            raise
    
    def register_agent_from_config(self, config: Dict[str, Any]) -> str:
        """Register agent from configuration dictionary"""
        
        # Required fields
        name = config['name']
        api_url = config['api_url']
        agent_id = config['agent_id']
        print(f"Registering agent: {name} with API URL: {api_url} , agent_id: {agent_id}")
        
        # Optional fields with defaults
        agent_id = config.get('agent_id', f"agent_{len(self.registered_agents) + 1}")
        print(f"Using agent ID: {agent_id}")
        description = config.get('description', "")
        keywords = config.get('keywords', [])
        headers = config.get('headers', {"Content-Type": "application/json"})
        if not isinstance(headers, dict):
            logger.warning("Headers should be a dictionary. Using default headers.")
            headers = {"Content-Type": "application/json","Authorization": f"test-key"}

        tools = config.get('tools', [])
        handoff_conditions = config.get('handoff_conditions', [])
        priority = config.get('priority', 1)
        timeout = config.get('timeout', 30)
        enabled = config.get('enabled', True)
        
        # Auto-generate keywords if not provided
        if not keywords:
            words = (name + " " + description).lower().split()
            keywords = [word.strip(".,!?") for word in words if len(word) > 3]
            # Add tool names as keywords
            keywords.extend([tool.lower() for tool in tools])
        
        # Create registered agent
        agent = RegisteredAgent(
            agent_id=agent_id,
            name=name,
            api_url=api_url,
            description=description,
            keywords=keywords,
            headers=headers,
            tools=tools,
            handoff_conditions=handoff_conditions,
            priority=priority,
            timeout=timeout,
            enabled=enabled
        )
        
        # Test agent connection
        test_result = self._test_agent_connection(agent)
        if not test_result["success"]:
            logger.warning(f"⚠️  Agent '{name}' connection test failed: {test_result['message']}")
        
        # Register the agent
        self.registered_agents[agent_id] = agent
        
        # Initialize agent memory
        self.agent_memory[agent_id] = {
            "usage_count": 0,
            "success_rate": 1.0,
            "last_used": None,
            "common_queries": []
        }
        
        logger.info(f"✅ Agent '{name}' registered with ID: {agent_id}")
        return agent_id
    
    def register_agent(self, 
                      name: str, 
                      api_url: str, 
                      agent_id: str = None,
                      description: str = "",
                      keywords: List[str] = None,
                      headers: Dict[str, str] = None,
                      tools: List[str] = None,
                      priority: int = 1) -> str:
        """
        Register a new agent programmatically
        
        Args:
            name: Human-readable name for the agent
            api_url: API endpoint URL for the agent
            agent_id: Optional custom ID (auto-generated if not provided)
            description: Optional description of what the agent does
            keywords: Optional list of keywords/triggers for this agent
            headers: Optional HTTP headers for API calls
            tools: List of tools/capabilities this agent has
            priority: Agent priority (higher = more likely to be selected)
        
        Returns:
            agent_id: The ID assigned to the registered agent
        """
        
        config = {
            'name': name,
            'api_url': api_url,
            'agent_id': agent_id,
            'description': description,
            'keywords': keywords or [],
            'headers': headers,
            'tools': tools or [],
            'priority': priority
        }
        
        return self.register_agent_from_config(config)
    
    def process_query(self, user_query: str, force_agent_id: str = None) -> Dict[str, Any]:
        """
        Process user query through the multi-agent system with enhanced handoff logic
        
        Args:
            user_query: The user's input query
            force_agent_id: Optional - force selection of specific agent
            
        Returns:
            Response dict with agent info and response
        """
        
        # Reset handoff counter for new queries
        if not hasattr(self, '_processing_query'):
            self.current_handoff_count = 0
            self._processing_query = True
        
        # Add user message to history
        user_message = Message(content=user_query, role="user")
        self.conversation_history.append(user_message)
        
        # Check for handoff prevention
        if self.current_handoff_count >= self.max_handoffs:
            error_response = "Maximum handoffs reached. Please start a new conversation."
            self._processing_query = False
            return self._create_error_response(error_response)
        
        # Determine which agent to use
        selected_agent_id = force_agent_id or self._intelligent_agent_selection(user_query)
        print(f"Selected agent ID: {selected_agent_id}")
        
        if not selected_agent_id:
            self._processing_query = False
            return self._create_error_response("No suitable agent found for this query.")
        
        # Call the selected agent
        selected_agent = self.registered_agents[selected_agent_id]
        logger.info(f"🎯 Selected agent: {selected_agent.name} (ID: {selected_agent_id})")
        
        result = self._call_agent_with_context(selected_agent, user_query)
        
        # Update agent memory
        self._update_agent_memory(selected_agent_id, result["success"])
        
        if result["success"]:
            # Handle potential handoff request
            handoff_info = self._process_handoff_request(result, user_query)
            
            if handoff_info["handoff_requested"]:
                self.current_handoff_count += 1
                logger.info(f"🔄 Handoff requested to: {handoff_info['target_agent']}")
                
                # Recursive call for handoff
                return self.process_query(user_query, handoff_info['target_agent'])
            
            # Successful response without handoff
            previous_agent = self.current_agent_id
            self.current_agent_id = selected_agent_id
            
            # Add response to history
            response_message = Message(
                content=result["response"],
                role="assistant",
                agent_id=selected_agent_id
            )
            self.conversation_history.append(response_message)
            
            self._processing_query = False
            return {
                "response": result["response"],
                "agent_used": selected_agent.name,
                "agent_id": selected_agent_id,
                "handoff_occurred": previous_agent != selected_agent_id,
                "previous_agent": previous_agent,
                "session_id": self.session_id,
                "execution_time": result.get("execution_time", 0),
                "handoff_count": self.current_handoff_count,
                "metadata": result.get("metadata", {})
            }
        
        else:
            # Agent call failed
            self._processing_query = False
            return self._create_error_response(
                f"The {selected_agent.name} encountered an issue: {result.get('error', 'Unknown error')}"
            )
    
    def _intelligent_agent_selection(self, query: str) -> Optional[str]:
        """Enhanced agent selection with memory and context awareness"""

        print(f"Intelligent agent selection for query: {query}")
        
        if not self.registered_agents:
            return None
        
        # Get enabled agents with their memory scores
        available_agents = []
        for agent_id, agent in self.registered_agents.items():
            if agent.enabled:
                memory = self.agent_memory.get(agent_id, {})
                available_agents.append({
                    "agent_id": agent_id,
                    "name": agent.name,
                    "description": agent.description,
                    "keywords": agent.keywords,
                    "tools": agent.tools,
                    "priority": agent.priority,
                    "success_rate": memory.get("success_rate", 1.0),
                    "usage_count": memory.get("usage_count", 0)
                })
        print(f"Available agents for selection:{available_agents}")
        if not available_agents:
            return None
        
        # Try LLM-based selection first
        llm_selection = self._llm_agent_selection(query, available_agents)
        if llm_selection:
            return llm_selection
        
        # Fallback to enhanced keyword matching
        return self._enhanced_fallback_selection(query, available_agents)
    
    def _llm_agent_selection(self, query: str, agents_info: List[Dict]) -> Optional[str]:
        """Use GPT-4o for intelligent agent selection"""
        
        # Include conversation context for better selection
        recent_context = []
        for msg in self.conversation_history[-3:]:
            recent_context.append({
                "role": msg.role,
                "content": msg.content[:200],  # Truncate long messages
                "agent_id": getattr(msg, 'agent_id', None)
            })
        
        system_prompt = f"""
        You are an intelligent agent router for a multi-agent system. Your job is to select the best agent to handle a user query based on:
        
        1. Agent capabilities (description, keywords, tools)
        2. Agent performance (success rate, priority)
        3. Recent conversation context
        4. Query semantics and intent
        
        Available agents:
        {json.dumps(agents_info, indent=2)}
        
        Recent conversation context:
        {json.dumps(recent_context, indent=2)}
        
        Current agent: {self.current_agent_id}
        
        Selection criteria:
        - Match query intent with agent capabilities
        - Consider conversation flow and context
        - Prefer agents with higher success rates for similar queries
        - Consider agent priority scores
        - If current agent can handle the query, prefer continuity
        
        Respond with JSON containing:
        - "selected_agent_id": the ID of the best agent (or null if none suitable)
        - "confidence": confidence score 0-1
        - "reasoning": brief explanation of choice
        - "handoff_needed": true if changing from current agent
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"User query: {query}"}
                ],
                temperature=0.2,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            selected_id = result.get("selected_agent_id")
            confidence = result.get("confidence", 0)
            reasoning = result.get("reasoning", "")
            
            logger.info(f"🤖 LLM Selection - Agent: {selected_id}, Confidence: {confidence:.2f}")
            logger.info(f"💭 Reasoning: {reasoning}")
            
            # Return selected agent if confidence is reasonable
            if selected_id and confidence > 0.4:
                return selected_id
                
        except Exception as e:
            logger.error(f"❌ LLM agent selection failed: {str(e)}")
        
        return None
    
    def _enhanced_fallback_selection(self, query: str, agents_info: List[Dict]) -> Optional[str]:
        """Enhanced fallback selection with scoring algorithm"""
        
        query_lower = query.lower()
        scored_agents = []
        
        for agent in agents_info:
            score = 0
            
            # Keyword matching (weighted)
            for keyword in agent["keywords"]:
                if keyword.lower() in query_lower:
                    score += 2
            
            # Tool matching (higher weight)
            for tool in agent["tools"]:
                if tool.lower() in query_lower:
                    score += 3
            
            # Name matching
            if agent["name"].lower() in query_lower:
                score += 4
            
            # Description word matching
            desc_words = agent["description"].lower().split()
            for word in desc_words:
                if word in query_lower and len(word) > 3:
                    score += 1
            
            # Apply priority and success rate multipliers
            score *= agent["priority"]
            score *= agent["success_rate"]
            
            # Bonus for current agent (continuity)
            if agent["agent_id"] == self.current_agent_id:
                score *= 1.2
            
            scored_agents.append((agent["agent_id"], score, agent["name"]))
        
        # Sort by score and return best match
        scored_agents.sort(key=lambda x: x[1], reverse=True)
        
        if scored_agents and scored_agents[0][1] > 0:
            best_agent = scored_agents[0]
            logger.info(f"🎯 Fallback selection: {best_agent[2]} (score: {best_agent[1]:.2f})")
            return best_agent[0]
        
        return None
    
    def _call_agent_with_context(self, agent: RegisteredAgent, query: str) -> Dict[str, Any]:
        """Call agent with enhanced context and error handling"""
        
        # Prepare enhanced context
        context = {
            "conversation_history": [],
            "session_info": {
                "session_id": self.session_id,
                "current_agent": self.current_agent_id,
                "handoff_count": self.current_handoff_count
            },
            "agent_info": {
                "tools": agent.tools,
                "capabilities": agent.description
            }
        }
        
        # Add recent conversation history
        for msg in self.conversation_history[-8:]:  # Last 8 messages for context
            context["conversation_history"].append({
                "role": msg.role,
                "content": msg.content,
                "agent_id": getattr(msg, 'agent_id', None),
                "timestamp": msg.timestamp.isoformat()
            })
        
        # Prepare payload
        payload = {
            "query": query
            # "context": context,
            # "session_id": self.session_id,
            # "agent_id": agent.agent_id
        }
        
        try:
            start_time = time.time()
            
            response = requests.post(
                agent.api_url,
                json=payload,
                headers=agent.headers,
                timeout=agent.timeout
            )
            
            execution_time = time.time() - start_time
            
            if response.status_code == 200:
                return self._parse_agent_response(response.json(), execution_time)
            else:
                logger.warning(f"⚠️  Agent {agent.name} returned status {response.status_code}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text[:200]}",
                    "execution_time": execution_time
                }
                
        except requests.exceptions.Timeout:
            logger.warning(f"⏰ Agent {agent.name} timed out")
            return {"success": False, "error": "Request timeout"}
        
        except requests.exceptions.ConnectionError:
            logger.warning(f"🔌 Connection error for agent {agent.name}")
            return {"success": False, "error": "Connection error"}
        
        except Exception as e:
            logger.error(f"❌ Agent call error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _parse_agent_response(self, data: Any, execution_time: float) -> Dict[str, Any]:
        """Parse and normalize agent response"""
        
        if isinstance(data, str):
            return {
                "success": True,
                "response": data,
                "execution_time": execution_time
            }
        
        elif isinstance(data, dict):
            return {
                "success": True,
                "response": data.get("response", str(data)),
                "handoff_requested": data.get("handoff_requested", False),
                "target_agent": data.get("target_agent"),
                "execution_time": execution_time,
                "metadata": data.get("metadata", {}),
                "tools_used": data.get("tools_used", [])
            }
        
        else:
            return {
                "success": True,
                "response": str(data),
                "execution_time": execution_time
            }
    
    def _process_handoff_request(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Process handoff requests from agents"""
        
        if not result.get("handoff_requested", False):
            return {"handoff_requested": False}
        
        target_agent = result.get("target_agent")
        
        if target_agent and target_agent in self.registered_agents:
            # Explicit target agent specified
            return {
                "handoff_requested": True,
                "target_agent": target_agent
            }
        
        # Need to determine target agent based on query
        # Use LLM to find appropriate agent for handoff
        selected_agent_id = self._intelligent_agent_selection(query)
        
        if selected_agent_id and selected_agent_id != self.current_agent_id:
            return {
                "handoff_requested": True,
                "target_agent": selected_agent_id
            }
        
        return {"handoff_requested": False}
    
    def _update_agent_memory(self, agent_id: str, success: bool) -> None:
        """Update agent memory with usage statistics"""
        
        if agent_id not in self.agent_memory:
            self.agent_memory[agent_id] = {
                "usage_count": 0,
                "success_rate": 1.0,
                "last_used": None,
                "common_queries": []
            }
        
        memory = self.agent_memory[agent_id]
        memory["usage_count"] += 1
        memory["last_used"] = datetime.now()
        
        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        if success:
            memory["success_rate"] = memory["success_rate"] * (1 - alpha) + alpha
        else:
            memory["success_rate"] = memory["success_rate"] * (1 - alpha)
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        
        response_message = Message(content=error_message, role="assistant")
        self.conversation_history.append(response_message)
        
        return {
            "response": error_message,
            "agent_used": "System",
            "agent_id": "system",
            "handoff_occurred": False,
            "session_id": self.session_id,
            "error": True,
            "handoff_count": self.current_handoff_count
        }
    
    def _test_agent_connection(self, agent: RegisteredAgent) -> Dict[str, Any]:
        """Test agent connectivity with enhanced diagnostics"""
        
        test_payload = {
            "query": "test connection"
            # "context": {"conversation_history": [], "session_info": {"session_id": "test"}},
            # "session_id": "test",
            # "agent_id": agent.agent_id
        }
        
        try:
            response = requests.post(
                agent.api_url,
                json=test_payload,
                headers=agent.headers,
                timeout=10
            )
            print(f" Test payload: {json.dumps(test_payload, indent=2)}")
            if response.status_code == 200:
                return {"success": True, "message": "Connection successful"}
            else:
                return {
                    "success": False, 
                    "message": f"HTTP {response.status_code}: {response.text[:100]}"
                }
                
        except Exception as e:
            print(f"❌ Error testing agent connection: {str(e)}")
            return {"success": False, "message": f"Connection failed: {str(e)}"}
    
    # Utility methods
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents with enhanced info"""
        agents_info = []
        for agent_id, agent in self.registered_agents.items():
            memory = self.agent_memory.get(agent_id, {})
            agents_info.append({
                "agent_id": agent_id,
                "name": agent.name,
                "description": agent.description,
                "keywords": agent.keywords,
                "tools": agent.tools,
                "enabled": agent.enabled,
                "api_url": agent.api_url,
                "priority": agent.priority,
                "usage_count": memory.get("usage_count", 0),
                "success_rate": memory.get("success_rate", 1.0),
                "last_used": memory.get("last_used")
            })
        return agents_info
    
    def get_agent_performance(self) -> Dict[str, Any]:
        """Get performance statistics for all agents"""
        performance = {}
        for agent_id, memory in self.agent_memory.items():
            agent = self.registered_agents.get(agent_id)
            if agent:
                performance[agent.name] = {
                    "usage_count": memory.get("usage_count", 0),
                    "success_rate": memory.get("success_rate", 1.0),
                    "last_used": memory.get("last_used"),
                    "enabled": agent.enabled
                }
        return performance
    
    def clear_conversation(self) -> None:
        """Clear conversation history and reset session"""
        self.conversation_history = []
        self.current_agent_id = None
        self.session_id = str(uuid.uuid4())
        self.current_handoff_count = 0
        logger.info("🧹 Conversation cleared")
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get current session information"""
        return {
            "session_id": self.session_id,
            "current_agent": self.current_agent_id,
            "message_count": len(self.conversation_history),
            "registered_agents": len(self.registered_agents),
            "handoff_count": self.current_handoff_count,
            "agent_performance": self.get_agent_performance()
        }
    
    def export_configuration(self, output_path: str) -> None:
        """Export current agent configuration to YAML"""
        config = {"agents": []}
        
        for agent in self.registered_agents.values():
            agent_config = {
                "name": agent.name,
                "agent_id": agent.agent_id,
                "api_url": agent.api_url,
                "description": agent.description,
                "keywords": agent.keywords,
                "headers": dict(agent.headers) if agent.headers else {},
                "tools": agent.tools,
                "handoff_conditions": agent.handoff_conditions,
                "priority": agent.priority,
                "timeout": agent.timeout,
                "enabled": agent.enabled
            }
            config["agents"].append(agent_config)
        
        with open(output_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        
        logger.info(f"✅ Configuration exported to {output_path}")

# Alias for backward compatibility
SimpleMultiAgentSystem = MultiAgentOrchestrator