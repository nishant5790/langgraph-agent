import openai
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import logging
from datetime import datetime
import os
api_key = os.getenv("OPENAI_API_KEY")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    MAIN = "main"
    FILTER = "filter"
    ACTION = "action"

@dataclass
class Message:
    role: str
    content: str
    timestamp: datetime
    agent_type: AgentType
    
@dataclass
class HandoffDecision:
    should_handoff: bool
    target_agent: Optional[AgentType]
    reason: str
    maintain_context: bool = True

class AgentMemory:
    def __init__(self):
        self.conversation_history: List[Message] = []
        self.current_agent: AgentType = AgentType.MAIN
        self.agent_context: Dict[AgentType, Dict[str, Any]] = {
            AgentType.MAIN: {},
            AgentType.FILTER: {},
            AgentType.ACTION: {}
        }
        self.handoff_history: List[Dict[str, Any]] = []
    
    def add_message(self, role: str, content: str, agent_type: AgentType):
        """Add a message to conversation history"""
        message = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            agent_type=agent_type
        )
        self.conversation_history.append(message)
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get recent messages for context"""
        return self.conversation_history[-count:]
    
    def get_agent_context(self, agent_type: AgentType) -> Dict[str, Any]:
        """Get context specific to an agent"""
        return self.agent_context[agent_type]
    
    def update_agent_context(self, agent_type: AgentType, context: Dict[str, Any]):
        """Update agent-specific context"""
        self.agent_context[agent_type].update(context)
    
    def record_handoff(self, from_agent: AgentType, to_agent: AgentType, reason: str):
        """Record handoff for tracking"""
        self.handoff_history.append({
            'from': from_agent.value,
            'to': to_agent.value,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        })

class BaseAgent:
    def __init__(self, client: openai.OpenAI, memory: AgentMemory):
        self.client = client
        self.memory = memory
        self.agent_type = AgentType.MAIN
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent"""
        raise NotImplementedError
    
    def should_handoff(self, user_input: str) -> HandoffDecision:
        """Determine if this agent should handoff to another agent"""
        return HandoffDecision(False, None, "No handoff needed")
    
    def process_message(self, user_input: str) -> str:
        """Process a user message and return response"""
        raise NotImplementedError
    
    def format_conversation_history(self) -> List[Dict[str, str]]:
        """Format conversation history for OpenAI API"""
        messages = [{"role": "system", "content": self.get_system_prompt()}]
        
        # Add recent conversation history
        for msg in self.memory.get_recent_messages():
            messages.append({
                "role": msg.role,
                "content": f"[{msg.agent_type.value}] {msg.content}"
            })
        
        return messages

class MainAgent(BaseAgent):
    def __init__(self, client: openai.OpenAI, memory: AgentMemory):
        super().__init__(client, memory)
        self.agent_type = AgentType.MAIN
    
    def get_system_prompt(self) -> str:
        return """You are the Main Agent in a multi-agent system. Your role is to:
1. Analyze user queries and decide which specialized agent should handle them
2. Route conversations to the appropriate agent (Filter Agent or Action Agent)
3. Maintain context and facilitate smooth handoffs between agents
4. Handle general queries that don't require specialized agents

Agent Types:
- Filter Agent: Handles queries related to filtering data, creating filter queries, search operations
- Action Agent: Handles action-based queries like add_outreach, get_csv, add_marketo, data manipulation

When deciding on handoffs, consider:
- Filter-related keywords: filter, search, query, find, select, where, criteria
- Action-related keywords: add, create, execute, run, process, outreach, marketo, csv

Always explain your routing decisions and maintain conversational context."""
    
    def analyze_query_intent(self, user_input: str) -> HandoffDecision:
        """Analyze user intent and decide on handoff"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze the user query and determine which agent should handle it.
                        
                        Return a JSON response with:
                        {
                            "should_handoff": true/false,
                            "target_agent": "filter"/"action"/null,
                            "reason": "explanation of decision",
                            "confidence": 0.0-1.0
                        }
                        
                        Filter Agent handles: filtering, searching, querying data, creating filter conditions
                        Action Agent handles: adding outreach, processing CSV, Marketo operations, data actions
                        Main Agent handles: general conversation, routing, context switching"""
                    },
                    {"role": "user", "content": user_input}
                ],
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            
            target_agent = None
            if result.get("target_agent") == "filter":
                target_agent = AgentType.FILTER
            elif result.get("target_agent") == "action":
                target_agent = AgentType.ACTION
            
            return HandoffDecision(
                should_handoff=result["should_handoff"],
                target_agent=target_agent,
                reason=result["reason"]
            )
            
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return HandoffDecision(False, None, "Error in analysis")
    
    def process_message(self, user_input: str) -> str:
        """Process message and handle routing"""
        # Analyze if we need to handoff
        handoff_decision = self.analyze_query_intent(user_input)
        
        if handoff_decision.should_handoff and handoff_decision.target_agent:
            # Record the handoff decision
            self.memory.record_handoff(
                self.agent_type, 
                handoff_decision.target_agent, 
                handoff_decision.reason
            )
            
            return f"I'll route your query to the {handoff_decision.target_agent.value} agent. {handoff_decision.reason}"
        
        # Handle the query directly
        try:
            messages = self.format_conversation_history()
            messages.append({"role": "user", "content": user_input})
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in main agent processing: {e}")
            return "I apologize, but I encountered an error processing your request."

class FilterAgent(BaseAgent):
    def __init__(self, client: openai.OpenAI, memory: AgentMemory):
        super().__init__(client, memory)
        self.agent_type = AgentType.FILTER
    
    def get_system_prompt(self) -> str:
        return """You are the Filter Agent, specialized in creating filter queries and handling search operations.
        
Your capabilities include:
- Creating SQL-like filter conditions
- Building search queries
- Data filtering and selection criteria
- Query optimization suggestions

When you receive a request that's outside your domain (like actions, outreach, or CSV operations), 
you should recommend handoff back to the main agent for proper routing.

Always provide clear, structured filter queries and explain your logic."""
    
    def should_handoff(self, user_input: str) -> HandoffDecision:
        """Check if query should be handed off to another agent"""
        action_keywords = ['add', 'create', 'execute', 'run', 'outreach', 'marketo', 'csv', 'process']
        
        if any(keyword in user_input.lower() for keyword in action_keywords):
            return HandoffDecision(
                should_handoff=True,
                target_agent=AgentType.MAIN,
                reason="This appears to be an action request that should be handled by the Action Agent"
            )
        
        return HandoffDecision(False, None, "Query is within filter agent scope")
    
    def process_message(self, user_input: str) -> str:
        """Process filter-related queries"""
        # Check if we should handoff
        handoff_decision = self.should_handoff(user_input)
        if handoff_decision.should_handoff:
            return f"This query seems to be outside my expertise. I'll hand it back to the main agent for proper routing. {handoff_decision.reason}"
        
        try:
            messages = self.format_conversation_history()
            messages.append({"role": "user", "content": user_input})
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3
            )
            
            # Update context with filter-related information
            self.memory.update_agent_context(self.agent_type, {
                'last_filter_query': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in filter agent processing: {e}")
            return "I encountered an error processing your filter request."

class ActionAgent(BaseAgent):
    def __init__(self, client: openai.OpenAI, memory: AgentMemory):
        super().__init__(client, memory)
        self.agent_type = AgentType.ACTION
    
    def get_system_prompt(self) -> str:
        return """You are the Action Agent, specialized in executing actions and operations.
        
Your capabilities include:
- add_outreach: Adding outreach campaigns
- get_csv: Retrieving and processing CSV data
- add_marketo: Marketo platform operations
- Data processing and manipulation actions

When you receive a request that's outside your domain (like filtering or general queries), 
you should recommend handoff back to the main agent for proper routing.

Always confirm actions before executing and provide clear status updates."""
    
    def should_handoff(self, user_input: str) -> HandoffDecision:
        """Check if query should be handed off to another agent"""
        filter_keywords = ['filter', 'search', 'query', 'find', 'select', 'where', 'criteria']
        
        if any(keyword in user_input.lower() for keyword in filter_keywords) and \
           not any(action_word in user_input.lower() for action_word in ['add', 'create', 'execute']):
            return HandoffDecision(
                should_handoff=True,
                target_agent=AgentType.MAIN,
                reason="This appears to be a filter/search request that should be handled by the Filter Agent"
            )
        
        return HandoffDecision(False, None, "Query is within action agent scope")
    
    def process_message(self, user_input: str) -> str:
        """Process action-related queries"""
        # Check if we should handoff
        handoff_decision = self.should_handoff(user_input)
        if handoff_decision.should_handoff:
            return f"This query seems to be outside my expertise. I'll hand it back to the main agent for proper routing. {handoff_decision.reason}"
        
        try:
            messages = self.format_conversation_history()
            messages.append({"role": "user", "content": user_input})
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.3
            )
            
            # Update context with action-related information
            self.memory.update_agent_context(self.agent_type, {
                'last_action_request': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in action agent processing: {e}")
            return "I encountered an error processing your action request."

class MultiAgentSystem:
    def __init__(self, openai_api_key: str):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.memory = AgentMemory()
        
        # Initialize agents
        self.main_agent = MainAgent(self.client, self.memory)
        self.filter_agent = FilterAgent(self.client, self.memory)
        self.action_agent = ActionAgent(self.client, self.memory)
        
        self.agents = {
            AgentType.MAIN: self.main_agent,
            AgentType.FILTER: self.filter_agent,
            AgentType.ACTION: self.action_agent
        }
    
    def get_current_agent(self) -> BaseAgent:
        """Get the currently active agent"""
        return self.agents[self.memory.current_agent]
    
    def process_user_input(self, user_input: str) -> str:
        """Main entry point for processing user input"""
        try:
            # Add user message to memory
            self.memory.add_message("user", user_input, self.memory.current_agent)
            
            # Get current agent
            current_agent = self.get_current_agent()
            
            # Process the message
            if self.memory.current_agent == AgentType.MAIN:
                # Main agent decides on routing
                handoff_decision = current_agent.analyze_query_intent(user_input)
                
                if handoff_decision.should_handoff and handoff_decision.target_agent:
                    # Switch to target agent
                    self.memory.current_agent = handoff_decision.target_agent
                    self.memory.record_handoff(
                        AgentType.MAIN, 
                        handoff_decision.target_agent, 
                        handoff_decision.reason
                    )
                    
                    # Process with new agent
                    new_agent = self.get_current_agent()
                    response = new_agent.process_message(user_input)
                    
                    response = f"[Switched to {handoff_decision.target_agent.value} agent]\n{response}"
                else:
                    # Process with main agent
                    response = current_agent.process_message(user_input)
            else:
                # Check if specialized agent wants to handoff
                handoff_decision = current_agent.should_handoff(user_input)
                
                if handoff_decision.should_handoff:
                    # Switch back to main agent for re-routing
                    self.memory.current_agent = AgentType.MAIN
                    self.memory.record_handoff(
                        current_agent.agent_type,
                        AgentType.MAIN,
                        handoff_decision.reason
                    )
                    
                    # Process with main agent
                    response = self.process_user_input(user_input)
                else:
                    # Process with current specialized agent
                    response = current_agent.process_message(user_input)
            
            # Add assistant response to memory
            self.memory.add_message("assistant", response, self.memory.current_agent)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in multi-agent system: {e}")
            return "I apologize, but I encountered an error processing your request."
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "current_agent": self.memory.current_agent.value,
            "conversation_length": len(self.memory.conversation_history),
            "handoff_count": len(self.memory.handoff_history),
            "recent_handoffs": self.memory.handoff_history[-3:] if self.memory.handoff_history else []
        }
    
    def reset_conversation(self):
        """Reset the conversation and return to main agent"""
        self.memory = AgentMemory()
        self.main_agent.memory = self.memory
        self.filter_agent.memory = self.memory
        self.action_agent.memory = self.memory

# Example usage
if __name__ == "__main__":
    # Initialize the system
    system = MultiAgentSystem(openai_api_key=api_key)
    
    # Example conversation
    test_queries = [
        "Hello, I need help with my data",
        "I want to filter customers by age > 25 and location = 'New York'",
        "Now I need to add an outreach campaign for these filtered customers",
        "Can you help me get a CSV export of the results?",
        "I also need to add these contacts to Marketo",
        "What's the status of our conversation?"
    ]
    
    print("Multi-Agent System Demo")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nUser: {query}")
        response = system.process_user_input(query)
        print(f"System: {response}")
        print(f"Current Agent: {system.memory.current_agent.value}")
        print("-" * 30)
            # Show system status
        print("\nSystem Status:")
        print(json.dumps(system.get_system_status(), indent=2))

        print("==" * 30)
        print("\nEnd of conversation.\n")
    # Show system status
    # print("\nSystem Status:")
    # print(json.dumps(system.get_system_status(), indent=2))