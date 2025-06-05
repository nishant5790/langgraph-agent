import json
import re
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
from datetime import datetime
import openai
import json
from datetime import datetime

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)




# OpenAI GPT-4o Client
class OpenAIClient:
    """OpenAI GPT-4o client for LLM operations"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        openai.api_key = self.api_key
        self.model = model
    
    def generate_response(self, prompt: str, max_tokens: int = 150, temperature: float = 0.1) -> str:
        """Generate response using OpenAI GPT-4o"""
        try:

            openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._fallback_response(prompt)
    
    def generate_chat_response(self, messages: List[Dict[str, str]], max_tokens: int = 150, temperature: float = 0.1) -> str:
        """Generate response using conversation history"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return self._fallback_response(messages[-1]["content"])
    
    def _fallback_response(self, prompt: str) -> str:
        """Fallback response when API fails"""
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['filter', 'search', 'find', 'query', 'where', 'select']):
            return "filter_agent"
        elif any(word in prompt_lower for word in ['add', 'create', 'outreach', 'csv', 'marketo', 'action', 'execute']):
            return "action_agent"
        else:
            return "main_agent"

@dataclass
class MemoryEntry:
    """Individual memory entry"""
    timestamp: datetime
    user_input: str
    agent_type: str
    response: str
    data: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "user_input": self.user_input,
            "agent_type": self.agent_type,
            "response": self.response,
            "data": self.data,
            "context": self.context
        }

class MemoryManager:
    """Manages conversation memory and context"""
    
    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.short_term_memory: List[MemoryEntry] = []
        self.long_term_memory: List[MemoryEntry] = []
        self.context_cache: Dict[str, Any] = {}
        self.user_preferences: Dict[str, Any] = {}
    
    def add_memory(self, entry: MemoryEntry):
        """Add new memory entry"""
        self.short_term_memory.append(entry)
        
        # Move to long-term memory if short-term is full
        if len(self.short_term_memory) > self.max_entries:
            self.long_term_memory.extend(self.short_term_memory[:-self.max_entries//2])
            self.short_term_memory = self.short_term_memory[-self.max_entries//2:]
    
    def get_recent_context(self, limit: int = 5) -> List[MemoryEntry]:
        """Get recent conversation context"""
        return self.short_term_memory[-limit:]
    
    def get_relevant_memories(self, query: str, limit: int = 3) -> List[MemoryEntry]:
        """Get memories relevant to current query"""
        query_lower = query.lower()
        relevant = []
        
        # Search through short-term memory first
        for entry in reversed(self.short_term_memory):
            if any(word in entry.user_input.lower() for word in query_lower.split()):
                relevant.append(entry)
                if len(relevant) >= limit:
                    break
        
        # Search long-term memory if needed
        if len(relevant) < limit:
            for entry in reversed(self.long_term_memory):
                if any(word in entry.user_input.lower() for word in query_lower.split()):
                    relevant.append(entry)
                    if len(relevant) >= limit:
                        break
        
        return relevant
    
    def update_context(self, key: str, value: Any):
        """Update context cache"""
        self.context_cache[key] = value
    
    def get_context(self, key: str = None) -> Any:
        """Get context information"""
        if key:
            return self.context_cache.get(key)
        return self.context_cache.copy()
    
    def update_user_preferences(self, preferences: Dict[str, Any]):
        """Update user preferences"""
        self.user_preferences.update(preferences)
    
    def get_conversation_summary(self) -> str:
        """Generate a summary of recent conversation"""
        recent_entries = self.get_recent_context(10)
        if not recent_entries:
            return "No recent conversation history."
        
        summary_parts = []
        for entry in recent_entries:
            summary_parts.append(f"User asked: {entry.user_input[:50]}...")
            summary_parts.append(f"Agent ({entry.agent_type}) responded: {entry.response[:50]}...")
        
        return "\n".join(summary_parts)
    
    def clear_memory(self):
        """Clear all memory"""
        self.short_term_memory.clear()
        self.long_term_memory.clear()
        self.context_cache.clear()

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
    memory_context: Optional[Dict[str, Any]] = None

class BaseAgent(ABC):
    """Base class for all agents with memory support"""
    
    def __init__(self, name: str, llm_client: OpenAIClient, memory_manager: MemoryManager):
        self.name = name
        self.llm_client = llm_client
        self.memory_manager = memory_manager
    
    @abstractmethod
    def process(self, user_input: str, context: Dict[str, Any] = None) -> HandoffResult:
        """Process user input and return result"""
        pass
    
    def can_handle(self, user_input: str) -> bool:
        """Check if this agent can handle the given input"""
        return True
    
    def get_context_prompt(self, user_input: str) -> str:
        """Get context-aware prompt with memory"""
        recent_context = self.memory_manager.get_recent_context(3)
        relevant_memories = self.memory_manager.get_relevant_memories(user_input, 2)
        
        context_str = ""
        if recent_context:
            context_str += "Recent conversation:\n"
            for entry in recent_context:
                context_str += f"User: {entry.user_input}\nAgent: {entry.response}\n"
        
        if relevant_memories:
            context_str += "\nRelevant past interactions:\n"
            for entry in relevant_memories:
                context_str += f"User: {entry.user_input}\nAgent: {entry.response}\n"
        
        return context_str

class MainAgent(BaseAgent):
    """Main orchestrator agent with memory-aware routing"""
    
    def __init__(self, llm_client: OpenAIClient, memory_manager: MemoryManager):
        super().__init__("MainAgent", llm_client, memory_manager)
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> HandoffResult:
        # Get context from memory
        context_prompt = self.get_context_prompt(user_input)
        
        routing_prompt = f"""
        You are an intelligent routing agent for a multi-agent system. Based on the user query and conversation context, determine which specialized agent should handle the request.

        {context_prompt}

        Available agents:
        1. filter_agent - Handles filtering, searching, querying data, finding information
        2. action_agent - Handles actions like add_outreach, get_csv, add_marketo, creating campaigns, executing tasks

        Current user query: {user_input}

        Consider the conversation context and user's intent. Respond with only the agent name (filter_agent or action_agent):
        """
        
        # Create messages for chat completion
        messages = [
            {"role": "system", "content": "You are a routing agent that determines which specialized agent should handle user requests."},
            {"role": "user", "content": routing_prompt}
        ]
        
        agent_decision = self.llm_client.generate_chat_response(messages, max_tokens=50)
        
        # Clean up the response
        agent_decision = agent_decision.strip().lower()
        
        if "filter_agent" in agent_decision:
            target_agent = AgentType.FILTER
        elif "action_agent" in agent_decision:
            target_agent = AgentType.ACTION
        else:
            # Use memory to make a better decision
            recent_context = self.memory_manager.get_recent_context(1)
            if recent_context and recent_context[-1].agent_type == "filter_agent":
                target_agent = AgentType.FILTER
            else:
                target_agent = AgentType.ACTION
        
        # Update context cache
        self.memory_manager.update_context("last_routing_decision", {
            "target_agent": target_agent.value,
            "user_input": user_input,
            "timestamp":datetime.now()
        })
        
        return HandoffResult(
            success=True,
            agent_type=AgentType.MAIN,
            response=f"Routing request to {target_agent.value}",
            next_agent=target_agent,
            data={"original_query": user_input, "context": context or {}},
            memory_context={"routing_decision": target_agent.value}
        )

class FilterAgent(BaseAgent):
    """Agent that handles filtering and query operations with memory"""
    
    def __init__(self, llm_client: OpenAIClient, memory_manager: MemoryManager):
        super().__init__("FilterAgent", llm_client, memory_manager)
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> HandoffResult:
        # Get context from memory
        context_prompt = self.get_context_prompt(user_input)
        
        filter_prompt = f"""
        You are a filter query generator. Convert the user's natural language request into a structured filter query.
        
        {context_prompt}
        
        Current user request: {user_input}
        
        Generate a filter query in JSON format with the following structure:
        {{
            "filters": [
                {{"field": "field_name", "operator": "equals|contains|greater_than|less_than|in", "value": "filter_value"}}
            ],
            "sort": {{"field": "field_name", "order": "asc|desc"}},
            "limit": 100,
            "offset": 0
        }}
        
        Consider the conversation context when determining field names and values. Return only the JSON:
        """
        
        messages = [
            {"role": "system", "content": "You are a filter query generator that converts natural language to structured queries."},
            {"role": "user", "content": filter_prompt}
        ]
        
        filter_response = self.llm_client.generate_chat_response(messages, max_tokens=300)
        
        # Try to parse the JSON response
        try:
            # Extract JSON from response if it's embedded in text
            json_match = re.search(r'\{.*\}', filter_response, re.DOTALL)
            if json_match:
                filter_query = json.loads(json_match.group())
            else:
                raise json.JSONDecodeError("No JSON found", filter_response, 0)
        except json.JSONDecodeError:
            # Use memory to create a better fallback
            recent_filters = self.memory_manager.get_relevant_memories("filter", 1)
            if recent_filters and recent_filters[0].data and "filter_query" in recent_filters[0].data:
                base_filter = recent_filters[0].data["filter_query"]
                filter_query = {
                    "filters": [{"field": "name", "operator": "contains", "value": user_input}],
                    "sort": base_filter.get("sort", {"field": "created_date", "order": "desc"}),
                    "limit": 100,
                    "offset": 0
                }
            else:
                filter_query = {
                    "filters": [{"field": "name", "operator": "contains", "value": user_input}],
                    "sort": {"field": "created_date", "order": "desc"},
                    "limit": 100,
                    "offset": 0
                }
        
        # Update memory with successful filter
        self.memory_manager.update_context("last_filter_query", filter_query)
        
        return HandoffResult(
            success=True,
            agent_type=AgentType.FILTER,
            response=f"Generated filter query for: {user_input}",
            data={"filter_query": filter_query, "original_query": user_input},
            memory_context={"filter_created": True, "query_type": "filter"}
        )

class ActionAgent(BaseAgent):
    """Agent that handles various actions with memory support"""
    
    def __init__(self, llm_client: OpenAIClient, memory_manager: MemoryManager):
        super().__init__("ActionAgent", llm_client, memory_manager)
        self.available_actions = {
            "add_outreach": self._add_outreach,
            "get_csv": self._get_csv,
            "add_marketo": self._add_marketo
        }
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> HandoffResult:
        # Get context from memory
        context_prompt = self.get_context_prompt(user_input)
        
        action_prompt = f"""
        You are an action executor with memory of past interactions. Based on the user request and conversation context, determine which action to perform and extract the parameters.
        
        {context_prompt}
        
        Available actions:
        1. add_outreach - Add outreach campaign (parameters: campaign_name, target_audience, message, channel)
        2. get_csv - Export data to CSV (parameters: data_type, filters, filename, format)
        3. add_marketo - Add Marketo integration (parameters: campaign_id, email_template, segment, automation_type)
        
        Current user request: {user_input}
        
        Consider the conversation context when determining parameters. Respond with JSON format:
        {{
            "action": "action_name",
            "parameters": {{"param1": "value1", "param2": "value2"}},
            "reasoning": "Brief explanation of why this action was chosen"
        }}
        
        Return only the JSON:
        """
        
        messages = [
            {"role": "system", "content": "You are an action executor that determines actions and parameters from user requests."},
            {"role": "user", "content": action_prompt}
        ]
        
        action_response = self.llm_client.generate_chat_response(messages, max_tokens=400)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', action_response, re.DOTALL)
            if json_match:
                action_data = json.loads(json_match.group())
            else:
                raise json.JSONDecodeError("No JSON found", action_response, 0)
        except json.JSONDecodeError:
            # Use memory for better fallback
            action_data = self._memory_aware_fallback(user_input)
        
        action_name = action_data.get("action")
        parameters = action_data.get("parameters", {})
        reasoning = action_data.get("reasoning", "")
        
        if action_name in self.available_actions:
            result = self.available_actions[action_name](parameters)
            
            # Update memory with successful action
            self.memory_manager.update_context("last_action", {
                "action": action_name,
                "parameters": parameters,
                "result": result,
                "timestamp": datetime.now()
            })
            
            return HandoffResult(
                success=True,
                agent_type=AgentType.ACTION,
                response=f"Executed {action_name}: {result}",
                data={
                    "action": action_name, 
                    "parameters": parameters, 
                    "result": result,
                    "reasoning": reasoning
                },
                memory_context={"action_executed": action_name, "success": True}
            )
        else:
            return HandoffResult(
                success=False,
                agent_type=AgentType.ACTION,
                response=f"Unknown action: {action_name}",
                data={"error": f"Action {action_name} not available"},
                memory_context={"action_failed": action_name, "success": False}
            )
    
    def _memory_aware_fallback(self, user_input: str) -> Dict[str, Any]:
        """Memory-aware fallback action detection"""
        user_input_lower = user_input.lower()
        
        # Check recent actions for context
        recent_actions = self.memory_manager.get_relevant_memories("add_outreach", 1)
        
        if "outreach" in user_input_lower:
            # Use previous outreach parameters if available
            base_params = {"campaign_name": "New Campaign", "target_audience": "All", "message": user_input}
            if recent_actions and recent_actions[0].data:
                last_params = recent_actions[0].data.get("parameters", {})
                base_params.update({k: v for k, v in last_params.items() if k in ["target_audience", "channel"]})
            
            return {
                "action": "add_outreach",
                "parameters": base_params,
                "reasoning": "Detected outreach request"
            }
        elif "csv" in user_input_lower or "export" in user_input_lower:
            return {
                "action": "get_csv",
                "parameters": {"data_type": "contacts", "filename": "export.csv", "format": "csv"},
                "reasoning": "Detected export request"
            }
        elif "marketo" in user_input_lower:
            return {
                "action": "add_marketo",
                "parameters": {"campaign_id": "default", "email_template": "template1"},
                "reasoning": "Detected Marketo request"
            }
        else:
            return {
                "action": "unknown", 
                "parameters": {},
                "reasoning": "Could not determine action from input"
            }
    
    def _add_outreach(self, parameters: Dict[str, Any]) -> str:
        """Add outreach campaign"""
        campaign_name = parameters.get("campaign_name", "Default Campaign")
        target_audience = parameters.get("target_audience", "All")
        message = parameters.get("message", "Default message")
        channel = parameters.get("channel", "email")
        
        return f"Outreach campaign '{campaign_name}' created for {target_audience} via {channel}"
    
    def _get_csv(self, parameters: Dict[str, Any]) -> str:
        """Export data to CSV"""
        data_type = parameters.get("data_type", "contacts")
        filename = parameters.get("filename", "export.csv")
        format_type = parameters.get("format", "csv")
        
        return f"{format_type.upper()} export '{filename}' created for {data_type} data"
    
    def _add_marketo(self, parameters: Dict[str, Any]) -> str:
        """Add Marketo integration"""
        campaign_id = parameters.get("campaign_id", "default")
        email_template = parameters.get("email_template", "template1")
        segment = parameters.get("segment", "all")
        
        return f"Marketo campaign {campaign_id} created with template {email_template} for segment {segment}"

class MultiAgentSystem:
    """Main multi-agent system orchestrator with comprehensive memory"""
    
    def __init__(self, openai_api_key: str = None, model: str = "gpt-4o"):
        self.llm_client = OpenAIClient(openai_api_key, model)
        self.memory_manager = MemoryManager()
        self.agents = {
            AgentType.MAIN: MainAgent(self.llm_client, self.memory_manager),
            AgentType.FILTER: FilterAgent(self.llm_client, self.memory_manager),
            AgentType.ACTION: ActionAgent(self.llm_client, self.memory_manager)
        }
        self.current_agent = AgentType.MAIN
        self.session_id = str(datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    def process_request(self, user_input: str, context: Dict[str, Any] = None) -> HandoffResult:
        """Process user request through the multi-agent system with memory"""
        
        # Start with main agent for routing
        current_agent = self.agents[AgentType.MAIN]
        result = current_agent.process(user_input, context)
        
        # If main agent suggests handoff, execute it
        if result.next_agent and result.next_agent in self.agents:
            target_agent = self.agents[result.next_agent]
            result = target_agent.process(user_input, result.data)
        
        # Store in memory
        memory_entry = MemoryEntry(
            timestamp=datetime.now(),
            user_input=user_input,
            agent_type=result.agent_type.value,
            response=result.response,
            data=result.data,
            context=result.memory_context
        )
        self.memory_manager.add_memory(memory_entry)
        
        return result
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history from memory"""
        return [entry.to_dict() for entry in self.memory_manager.short_term_memory]
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation"""
        return self.memory_manager.get_conversation_summary()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            "short_term_entries": len(self.memory_manager.short_term_memory),
            "long_term_entries": len(self.memory_manager.long_term_memory),
            "context_cache_size": len(self.memory_manager.context_cache),
            "user_preferences": self.memory_manager.user_preferences,
            "session_id": self.session_id
        }
    
    def clear_memory(self):
        """Clear all memory"""
        self.memory_manager.clear_memory()
    
    def export_memory(self, filename: str = None):
        """Export memory to JSON file"""
        if not filename:
            filename = f"conversation_memory_{self.session_id}.json"
        
        memory_data = {
            "session_id": self.session_id,
            "short_term_memory": [entry.to_dict() for entry in self.memory_manager.short_term_memory],
            "long_term_memory": [entry.to_dict() for entry in self.memory_manager.long_term_memory],
            "context_cache": self.memory_manager.context_cache,
            "user_preferences": self.memory_manager.user_preferences
        }
        
        # with open(filename, 'w') as f:
        #     json.dump(memory_data, f, indent=2)

        with open(filename, 'w') as f:
            json.dump(memory_data, f, indent=2, cls=EnhancedJSONEncoder)
        

        return filename

# Example usage and testing
def main():
    # Initialize the multi-agent system with OpenAI GPT-4o
    # Make sure to set your OPENAI_API_KEY environment variable
    try:
        mas = MultiAgentSystem()
    except ValueError as e:
        print(f"Error: {e}")
        print("Please set your OPENAI_API_KEY environment variable")
        return
    
    # Test cases with memory awareness
    test_queries = [
        "Filter contacts by name containing 'John'",
        "Add a new outreach campaign for the John contacts",
        "Export those filtered contacts to CSV",
        "Create a Marketo campaign for the same audience",
        "Show me all contacts from last month",
        "Add another outreach campaign similar to the previous one"
    ]
    
    print("Multi-Agent System with Memory Demo")
    print("=" * 50)
    print(f"Using OpenAI GPT-4o model")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 30)
        
        result = mas.process_request(query)
        
        print(f"Agent Used: {result.agent_type.value}")
        print(f"Response: {result.response}")
        if result.data:
            print(f"Data: {json.dumps(result.data, indent=2)}")
        
        # Show memory stats periodically
        if i % 3 == 0:
            print(f"\nMemory Stats: {mas.get_memory_stats()}")
        
        print()
    
    # Show conversation summary
    print("\nConversation Summary:")
    print("=" * 50)
    print(mas.get_conversation_summary())
    
    # Export memory
    print(f"\nExporting memory to file...")
    filename = mas.export_memory()
    print(f"Memory exported to: {filename}")

if __name__ == "__main__":
    main()