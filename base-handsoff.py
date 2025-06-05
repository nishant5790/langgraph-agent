import json
import re
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum

# Mock LLM client - replace with your actual LLM implementation
class MockLLMClient:
    """Mock LLM client - replace with your actual LLM implementation (OpenAI, Anthropic, etc.)"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
    
    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """
        Mock LLM response generation
        Replace this with actual LLM API calls
        """
        # Simple keyword-based routing for demonstration
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['filter', 'search', 'find', 'query', 'where', 'select']):
            return "filter_agent"
        elif any(word in prompt_lower for word in ['add', 'create', 'outreach', 'csv', 'marketo', 'action', 'execute']):
            return "action_agent"
        else:
            return "main_agent"

class AgentType(Enum):
    MAIN = "main_agent"
    FILTER = "filter_agent"
    ACTION = "action_agent"

@dataclass
class HandoffResult:
    """Result of an agent handoff operation"""
    success: bool
    agent_type: AgentType
    response: str
    data: Optional[Dict[str, Any]] = None
    next_agent: Optional[AgentType] = None

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, llm_client: MockLLMClient):
        self.name = name
        self.llm_client = llm_client
    
    @abstractmethod
    def process(self, user_input: str, context: Dict[str, Any] = None) -> HandoffResult:
        """Process user input and return result"""
        pass
    
    def can_handle(self, user_input: str) -> bool:
        """Check if this agent can handle the given input"""
        return True

class MainAgent(BaseAgent):
    """Main orchestrator agent that routes requests to appropriate agents"""
    
    def __init__(self, llm_client: MockLLMClient):
        super().__init__("MainAgent", llm_client)
        self.routing_prompt = """
        You are an intelligent routing agent. Based on the user query, determine which agent should handle the request.
        
        Available agents:
        1. filter_agent - Handles filtering, searching, querying data
        2. action_agent - Handles actions like add_outreach, get_csv, add_marketo
        
        User Query: {query}
        
        Respond with only the agent name (filter_agent or action_agent):
        """
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> HandoffResult:
        # Use LLM to determine which agent should handle the request
        routing_prompt = self.routing_prompt.format(query=user_input)
        agent_decision = self.llm_client.generate_response(routing_prompt)
        
        # Clean up the response
        agent_decision = agent_decision.strip().lower()
        
        if "filter_agent" in agent_decision:
            target_agent = AgentType.FILTER
        elif "action_agent" in agent_decision:
            target_agent = AgentType.ACTION
        else:
            # Default fallback
            target_agent = AgentType.FILTER
        
        return HandoffResult(
            success=True,
            agent_type=AgentType.MAIN,
            response=f"Routing request to {target_agent.value}",
            next_agent=target_agent,
            data={"original_query": user_input, "context": context or {}}
        )

class FilterAgent(BaseAgent):
    """Agent that handles filtering and query operations"""
    
    def __init__(self, llm_client: MockLLMClient):
        super().__init__("FilterAgent", llm_client)
        self.filter_prompt = """
        You are a filter query generator. Convert the user's natural language request into a structured filter query.
        
        User Request: {query}
        
        Generate a filter query in JSON format with the following structure:
        {{
            "filters": [
                {{"field": "field_name", "operator": "equals|contains|greater_than|less_than", "value": "filter_value"}}
            ],
            "sort": {{"field": "field_name", "order": "asc|desc"}},
            "limit": 100
        }}
        
        Filter Query:
        """
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> HandoffResult:
        # Generate filter query using LLM
        filter_prompt = self.filter_prompt.format(query=user_input)
        filter_response = self.llm_client.generate_response(filter_prompt, max_tokens=200)
        
        # Try to parse the JSON response
        try:
            # Extract JSON from response if it's embedded in text
            json_match = re.search(r'\{.*\}', filter_response, re.DOTALL)
            if json_match:
                filter_query = json.loads(json_match.group())
            else:
                # Fallback filter query
                filter_query = {
                    "filters": [{"field": "name", "operator": "contains", "value": user_input}],
                    "sort": {"field": "created_date", "order": "desc"},
                    "limit": 100
                }
        except json.JSONDecodeError:
            # Fallback filter query
            filter_query = {
                "filters": [{"field": "name", "operator": "contains", "value": user_input}],
                "sort": {"field": "created_date", "order": "desc"},
                "limit": 100
            }
        
        return HandoffResult(
            success=True,
            agent_type=AgentType.FILTER,
            response=f"Generated filter query for: {user_input}",
            data={"filter_query": filter_query, "original_query": user_input}
        )
    
    def can_handle(self, user_input: str) -> bool:
        filter_keywords = ['filter', 'search', 'find', 'query', 'where', 'select', 'show', 'list']
        return any(keyword in user_input.lower() for keyword in filter_keywords)

class ActionAgent(BaseAgent):
    """Agent that handles various actions like add_outreach, get_csv, add_marketo"""
    
    def __init__(self, llm_client: MockLLMClient):
        super().__init__("ActionAgent", llm_client)
        self.available_actions = {
            "add_outreach": self._add_outreach,
            "get_csv": self._get_csv,
            "add_marketo": self._add_marketo
        }
        
        self.action_prompt = """
        You are an action executor. Based on the user request, determine which action to perform and extract the parameters.
        
        Available actions:
        1. add_outreach - Add outreach campaign (parameters: campaign_name, target_audience, message)
        2. get_csv - Export data to CSV (parameters: data_type, filters, filename)
        3. add_marketo - Add Marketo integration (parameters: campaign_id, email_template, segment)
        
        User Request: {query}
        
        Respond with JSON format:
        {{
            "action": "action_name",
            "parameters": {{"param1": "value1", "param2": "value2"}}
        }}
        
        Action:
        """
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> HandoffResult:
        # Use LLM to determine action and parameters
        action_prompt = self.action_prompt.format(query=user_input)
        action_response = self.llm_client.generate_response(action_prompt, max_tokens=200)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', action_response, re.DOTALL)
            if json_match:
                action_data = json.loads(json_match.group())
            else:
                raise json.JSONDecodeError("No JSON found", action_response, 0)
        except json.JSONDecodeError:
            # Fallback action determination
            action_data = self._fallback_action_detection(user_input)
        
        action_name = action_data.get("action")
        parameters = action_data.get("parameters", {})
        
        if action_name in self.available_actions:
            result = self.available_actions[action_name](parameters)
            return HandoffResult(
                success=True,
                agent_type=AgentType.ACTION,
                response=f"Executed {action_name}: {result}",
                data={"action": action_name, "parameters": parameters, "result": result}
            )
        else:
            return HandoffResult(
                success=False,
                agent_type=AgentType.ACTION,
                response=f"Unknown action: {action_name}",
                data={"error": f"Action {action_name} not available"}
            )
    
    def _fallback_action_detection(self, user_input: str) -> Dict[str, Any]:
        """Fallback action detection based on keywords"""
        user_input_lower = user_input.lower()
        
        if "outreach" in user_input_lower:
            return {
                "action": "add_outreach",
                "parameters": {"campaign_name": "New Campaign", "target_audience": "All", "message": user_input}
            }
        elif "csv" in user_input_lower or "export" in user_input_lower:
            return {
                "action": "get_csv",
                "parameters": {"data_type": "contacts", "filename": "export.csv"}
            }
        elif "marketo" in user_input_lower:
            return {
                "action": "add_marketo",
                "parameters": {"campaign_id": "default", "email_template": "template1"}
            }
        else:
            return {"action": "unknown", "parameters": {}}
    
    def _add_outreach(self, parameters: Dict[str, Any]) -> str:
        """Add outreach campaign"""
        campaign_name = parameters.get("campaign_name", "Default Campaign")
        target_audience = parameters.get("target_audience", "All")
        message = parameters.get("message", "Default message")
        
        # Simulate outreach creation
        return f"Outreach campaign '{campaign_name}' created for {target_audience}"
    
    def _get_csv(self, parameters: Dict[str, Any]) -> str:
        """Export data to CSV"""
        data_type = parameters.get("data_type", "contacts")
        filename = parameters.get("filename", "export.csv")
        
        # Simulate CSV export
        return f"CSV export '{filename}' created for {data_type} data"
    
    def _add_marketo(self, parameters: Dict[str, Any]) -> str:
        """Add Marketo integration"""
        campaign_id = parameters.get("campaign_id", "default")
        email_template = parameters.get("email_template", "template1")
        
        # Simulate Marketo integration
        return f"Marketo campaign {campaign_id} created with template {email_template}"
    
    def can_handle(self, user_input: str) -> bool:
        action_keywords = ['add', 'create', 'outreach', 'csv', 'marketo', 'action', 'execute', 'export']
        return any(keyword in user_input.lower() for keyword in action_keywords)

class MultiAgentSystem:
    """Main multi-agent system orchestrator"""
    
    def __init__(self, llm_client: MockLLMClient):
        self.llm_client = llm_client
        self.agents = {
            AgentType.MAIN: MainAgent(llm_client),
            AgentType.FILTER: FilterAgent(llm_client),
            AgentType.ACTION: ActionAgent(llm_client)
        }
        self.current_agent = AgentType.MAIN
        self.conversation_history = []
    
    def process_request(self, user_input: str, context: Dict[str, Any] = None) -> HandoffResult:
        """Process user request through the multi-agent system"""
        
        # Start with main agent for routing
        current_agent = self.agents[AgentType.MAIN]
        result = current_agent.process(user_input, context)
        
        # If main agent suggests handoff, execute it
        if result.next_agent and result.next_agent in self.agents:
            target_agent = self.agents[result.next_agent]
            result = target_agent.process(user_input, result.data)
        
        # Store conversation history
        self.conversation_history.append({
            "user_input": user_input,
            "agent_used": result.agent_type.value,
            "response": result.response,
            "data": result.data
        })
        
        return result
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

# Example usage and testing
def main():
    # Initialize the multi-agent system
    llm_client = MockLLMClient()
    mas = MultiAgentSystem(llm_client)
    
    # Test cases
    test_queries = [
        "Filter contacts by name containing 'John'",
        "Add a new outreach campaign for marketing",
        "Export customer data to CSV",
        "Create a Marketo campaign",
        "Show me all contacts from last month",
        "Add outreach for product launch"
    ]
    
    print("Multi-Agent System Demo")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nUser Query: {query}")
        print("-" * 30)
        
        result = mas.process_request(query)
        
        print(f"Agent Used: {result.agent_type.value}")
        print(f"Response: {result.response}")
        if result.data:
            print(f"Data: {json.dumps(result.data, indent=2)}")
        
        print()
    
    # Show conversation history
    print("\nConversation History:")
    print("=" * 50)
    for i, entry in enumerate(mas.get_conversation_history(), 1):
        print(f"{i}. Query: {entry['user_input']}")
        print(f"   Agent: {entry['agent_used']}")
        print(f"   Response: {entry['response']}")
        print()

if __name__ == "__main__":
    main()