import json
import os
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import openai
from enum import Enum


class HandoffDecision(Enum):
    """Enum for handoff decisions"""
    MAIN_AGENT_HANDLE = "main_agent_handle"
    HANDOFF_TO_SUBAGENT = "handoff_to_subagent"
    NO_SUITABLE_AGENT = "no_suitable_agent"


@dataclass
class Message:
    """Represents a message in the conversation"""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    agent_id: Optional[str] = None


@dataclass
class AgentContext:
    """Stores context and memory for each agent"""
    agent_id: str
    conversation_history: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_active: datetime = field(default_factory=datetime.now)


class MemorySystem:
    """Manages memory and context for all agents"""
    
    def __init__(self, storage_path: str = "agent_memory.json"):
        self.storage_path = storage_path
        self.contexts: Dict[str, AgentContext] = {}
        self.load_memory()
    
    def add_message(self, agent_id: str, message: Message):
        """Add a message to an agent's conversation history"""
        if agent_id not in self.contexts:
            self.contexts[agent_id] = AgentContext(agent_id=agent_id)
        
        self.contexts[agent_id].conversation_history.append(message)
        self.contexts[agent_id].last_active = datetime.now()
        self.save_memory()
    
    def get_context(self, agent_id: str) -> Optional[AgentContext]:
        """Retrieve context for a specific agent"""
        return self.contexts.get(agent_id)
    
    def get_conversation_history(self, agent_id: str, limit: int = 10) -> List[Message]:
        """Get recent conversation history for an agent"""
        context = self.get_context(agent_id)
        if context:
            return context.conversation_history[-limit:]
        return []
    
    def update_metadata(self, agent_id: str, metadata: Dict[str, Any]):
        """Update metadata for an agent"""
        if agent_id not in self.contexts:
            self.contexts[agent_id] = AgentContext(agent_id=agent_id)
        
        self.contexts[agent_id].metadata.update(metadata)
        self.save_memory()
    
    def save_memory(self):
        """Persist memory to disk"""
        data = {}
        for agent_id, context in self.contexts.items():
            data[agent_id] = {
                "agent_id": context.agent_id,
                "conversation_history": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "agent_id": msg.agent_id
                    }
                    for msg in context.conversation_history
                ],
                "metadata": context.metadata,
                "last_active": context.last_active.isoformat()
            }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_memory(self):
        """Load memory from disk"""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                
                for agent_id, context_data in data.items():
                    messages = [
                        Message(
                            role=msg["role"],
                            content=msg["content"],
                            timestamp=datetime.fromisoformat(msg["timestamp"]),
                            agent_id=msg.get("agent_id")
                        )
                        for msg in context_data["conversation_history"]
                    ]
                    
                    self.contexts[agent_id] = AgentContext(
                        agent_id=agent_id,
                        conversation_history=messages,
                        metadata=context_data["metadata"],
                        last_active=datetime.fromisoformat(context_data["last_active"])
                    )
            except Exception as e:
                print(f"Error loading memory: {e}")


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_id: str, name: str, description: str, 
                 capabilities: List[str], openai_client: openai.OpenAI):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.client = openai_client
        self.is_active = False
    
    @abstractmethod
    def can_handle(self, user_message: str, context: List[Message]) -> float:
        """
        Determine if this agent can handle the user message.
        Returns a confidence score between 0 and 1.
        """
        pass
    
    @abstractmethod
    def process(self, user_message: str, context: List[Message]) -> str:
        """Process the user message and return a response"""
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "is_active": self.is_active
        }


class SubAgent(BaseAgent):
    """Specialized sub-agent implementation"""
    
    def __init__(self, agent_id: str, name: str, description: str,
                 capabilities: List[str], openai_client: openai.OpenAI,
                 system_prompt: str):
        super().__init__(agent_id, name, description, capabilities, openai_client)
        self.system_prompt = system_prompt
    
    def can_handle(self, user_message: str, context: List[Message]) -> float:
        """Use LLM to determine if this agent can handle the message"""
        prompt = f"""
        You are evaluating if the agent '{self.name}' should handle a user message.
        
        Agent Description: {self.description}
        Agent Capabilities: {', '.join(self.capabilities)}
        
        User Message: {user_message}
        
        Recent Context: {self._format_context(context[-3:])}
        
        Rate from 0.0 to 1.0 how well this agent can handle this message.
        Consider:
        - Does the message align with the agent's capabilities?
        - Does the context suggest this agent's expertise is needed?
        - Is this agent the best fit compared to a general assistant?
        
        Respond with ONLY a number between 0.0 and 1.0.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.3,
                max_tokens=10
            )
            
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.0
    
    def process(self, user_message: str, context: List[Message]) -> str:
        """Process the message using the agent's specialized knowledge"""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add relevant context
        for msg in context[-5:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add the current user message
        messages.append({"role": "user", "content": user_message})
        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    
    def _format_context(self, context: List[Message]) -> str:
        """Format context messages for prompt"""
        if not context:
            return "No recent context"
        
        formatted = []
        for msg in context:
            formatted.append(f"{msg.role}: {msg.content[:100]}...")
        return "\n".join(formatted)


class MainAgent:
    """Main orchestration agent that manages sub-agents and handoffs"""
    
    def __init__(self, openai_api_key: str, memory_system: Optional[MemorySystem] = None):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.sub_agents: Dict[str, SubAgent] = {}
        self.memory = memory_system or MemorySystem()
        self.agent_id = "main_agent"
        self.current_active_agent: Optional[str] = None
        self.system_prompt = """
        You are the main orchestration agent in a multi-agent system.
        Your role is to:
        1. Understand user requests
        2. Decide whether to handle them yourself or delegate to a specialized sub-agent
        3. Manage conversations and maintain context
        4. Provide helpful, accurate responses
        
        Be conversational, helpful, and clear in your communication.
        """
    
    def register_sub_agent(self, sub_agent: SubAgent):
        """Register a new sub-agent"""
        self.sub_agents[sub_agent.agent_id] = sub_agent
        print(f"Registered sub-agent: {sub_agent.name} ({sub_agent.agent_id})")
    
    def unregister_sub_agent(self, agent_id: str):
        """Unregister a sub-agent"""
        if agent_id in self.sub_agents:
            del self.sub_agents[agent_id]
            print(f"Unregistered sub-agent: {agent_id}")
    
    def decide_handoff(self, user_message: str) -> tuple[HandoffDecision, Optional[str]]:
        """Decide whether to handoff to a sub-agent or handle directly"""
        
        # Get context for decision making
        context = self.memory.get_conversation_history(self.agent_id, limit=5)
        
        # If we're already in a handoff, check if we should continue
        if self.current_active_agent:
            active_agent = self.sub_agents.get(self.current_active_agent)
            if active_agent:
                # Check if the conversation should continue with the current agent
                continuation_score = active_agent.can_handle(user_message, context)
                if continuation_score > 0.6:
                    return HandoffDecision.HANDOFF_TO_SUBAGENT, self.current_active_agent
        
        # Evaluate all sub-agents
        agent_scores = {}
        for agent_id, agent in self.sub_agents.items():
            score = agent.can_handle(user_message, context)
            agent_scores[agent_id] = score
        
        # Find the best agent
        if agent_scores:
            best_agent_id = max(agent_scores, key=agent_scores.get)
            best_score = agent_scores[best_agent_id]
            
            # If best score is high enough, handoff
            if best_score > 0.7:
                return HandoffDecision.HANDOFF_TO_SUBAGENT, best_agent_id
        
        # Otherwise, main agent handles it
        return HandoffDecision.MAIN_AGENT_HANDLE, None
    
    def process_message(self, user_message: str) -> str:
        """Process a user message through the orchestration system"""
        
        # Add user message to memory
        self.memory.add_message(
            self.agent_id,
            Message(role="user", content=user_message)
        )
        
        # Decide on handoff
        decision, selected_agent_id = self.decide_handoff(user_message)
        
        response = ""
        responding_agent = self.agent_id
        
        if decision == HandoffDecision.HANDOFF_TO_SUBAGENT and selected_agent_id:
            # Handoff to sub-agent
            sub_agent = self.sub_agents[selected_agent_id]
            
            # Notify about handoff if switching agents
            if self.current_active_agent != selected_agent_id:
                self.current_active_agent = selected_agent_id
                handoff_msg = f"[Handing off to {sub_agent.name}]"
                print(handoff_msg)
            
            # Get sub-agent's context
            sub_context = self.memory.get_conversation_history(selected_agent_id, limit=5)
            
            # Process with sub-agent
            response = sub_agent.process(user_message, sub_context)
            responding_agent = selected_agent_id
            
            # Add messages to sub-agent's memory
            self.memory.add_message(
                selected_agent_id,
                Message(role="user", content=user_message)
            )
            self.memory.add_message(
                selected_agent_id,
                Message(role="assistant", content=response, agent_id=selected_agent_id)
            )
            
        else:
            # Main agent handles
            if self.current_active_agent:
                print("[Returning control to main agent]")
                self.current_active_agent = None
            
            # Get main agent context
            main_context = self.memory.get_conversation_history(self.agent_id, limit=5)
            
            # Process with main agent
            messages = [{"role": "system", "content": self.system_prompt}]
            for msg in main_context:
                messages.append({"role": msg.role, "content": msg.content})
            messages.append({"role": "user", "content": user_message})
            
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7
            )
            
            response = completion.choices[0].message.content
        
        # Add assistant response to main agent memory
        self.memory.add_message(
            self.agent_id,
            Message(role="assistant", content=response, agent_id=responding_agent)
        )
        
        return response
    
    def get_active_agent_info(self) -> Dict[str, Any]:
        """Get information about the currently active agent"""
        if self.current_active_agent and self.current_active_agent in self.sub_agents:
            return self.sub_agents[self.current_active_agent].get_info()
        return {
            "agent_id": self.agent_id,
            "name": "Main Orchestration Agent",
            "description": "General purpose agent that orchestrates sub-agents",
            "is_active": True
        }
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        agents = [self.get_active_agent_info()]
        for agent in self.sub_agents.values():
            agents.append(agent.get_info())
        return agents


# Example usage and helper functions

def create_customer_support_agent(openai_client: openai.OpenAI) -> SubAgent:
    """Create a customer support specialized agent"""
    return SubAgent(
        agent_id="customer_support_agent",
        name="Customer Support Specialist",
        description="Handles customer inquiries, complaints, and support tickets",
        capabilities=[
            "Answer customer questions",
            "Handle complaints",
            "Process refunds and returns",
            "Provide product information",
            "Troubleshoot common issues"
        ],
        openai_client=openai_client,
        system_prompt="""
        You are a friendly and helpful customer support specialist.
        Your goal is to assist customers with their inquiries and resolve their issues.
        
        Guidelines:
        - Be empathetic and understanding
        - Provide clear and accurate information
        - Offer solutions when possible
        - Escalate complex issues appropriately
        - Always maintain a professional and positive tone
        """
    )


def create_technical_agent(openai_client: openai.OpenAI) -> SubAgent:
    """Create a technical support specialized agent"""
    return SubAgent(
        agent_id="technical_agent",
        name="Technical Expert",
        description="Handles technical questions, coding help, and system issues",
        capabilities=[
            "Debug code",
            "Explain technical concepts",
            "Provide coding solutions",
            "System architecture advice",
            "API integration help"
        ],
        openai_client=openai_client,
        system_prompt="""
        You are a technical expert specializing in software development and system architecture.
        
        Guidelines:
        - Provide accurate technical information
        - Include code examples when relevant
        - Explain complex concepts clearly
        - Consider best practices and security
        - Offer multiple solutions when applicable
        """
    )


def create_sales_agent(openai_client: openai.OpenAI) -> SubAgent:
    """Create a sales specialized agent"""
    return SubAgent(
        agent_id="sales_agent",
        name="Sales Representative",
        description="Handles product recommendations, pricing inquiries, and sales processes",
        capabilities=[
            "Product recommendations",
            "Pricing information",
            "Discount eligibility",
            "Order processing",
            "Upselling and cross-selling"
        ],
        openai_client=openai_client,
        system_prompt="""
        You are a knowledgeable sales representative.
        Your goal is to help customers find the right products and make informed purchasing decisions.
        
        Guidelines:
        - Understand customer needs
        - Recommend appropriate products
        - Explain features and benefits
        - Be helpful but not pushy
        - Provide accurate pricing and availability information
        """
    )


# Example usage
if __name__ == "__main__":
    # Initialize the system
    OPENAI_API_KEY = None  # Replace with your actual API key
    
    # Create main agent with memory system
    memory_system = MemorySystem("multi_agent_memory.json")
    main_agent = MainAgent(OPENAI_API_KEY, memory_system)
    
    # Create and register sub-agents
    customer_support = create_customer_support_agent(main_agent.client)
    technical_expert = create_technical_agent(main_agent.client)
    sales_rep = create_sales_agent(main_agent.client)
    
    main_agent.register_sub_agent(customer_support)
    main_agent.register_sub_agent(technical_expert)
    main_agent.register_sub_agent(sales_rep)
    
    # Interactive loop
    print("\nMulti-Agent Orchestration System")
    print("================================")
    print("Type 'exit' to quit, 'agents' to list agents, or enter your message:\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            break
        elif user_input.lower() == 'agents':
            agents = main_agent.list_agents()
            print("\nRegistered Agents:")
            for agent in agents:
                status = "ACTIVE" if agent.get("is_active") or agent["agent_id"] == main_agent.current_active_agent else ""
                print(f"- {agent['name']} ({agent['agent_id']}) {status}")
            print()
            continue
        
        # Process the message
        response = main_agent.process_message(user_input)
        
        # Show which agent responded
        current_agent = main_agent.get_active_agent_info()
        print(f"\n[{current_agent['name']}]: {response}\n")