#!/usr/bin/env python3
"""
Simple Usage Example for Multi-Agent System
Shows how easy it is to set up and use the system
"""

from AgentOrchestrator import MultiAgentOrchestrator
import time

def simple_example():
    """Simple example showing basic usage"""
    
    print("🚀 Simple Multi-Agent System Example")
    print("=" * 50)
    
    # 1. Initialize the system
    print("\n1️⃣ Initializing system...")
    system = MultiAgentOrchestrator(openai_api_key=None)
    
    # 2. Register agents - Method 1: From YAML
    print("\n2️⃣ Loading agents from YAML...")
    try:
        registered_agents = system.load_agents_from_yaml("agents_config.yaml")
        print(f"✅ Loaded {len(registered_agents)} agents from YAML")
        for name, agent_id in registered_agents.items():
            print(f"  • {name} (ID: {agent_id})")
    except FileNotFoundError:
        print("⚠️  YAML file not found, registering agents manually...")
        
        # Method 2: Register manually
        system.register_agent(
            name="Filter Agent",
            api_url="http://localhost:5001/process",
            description="Filters and searches data",
            keywords=["filter", "search", "find", "show"]
        )
        
        system.register_agent(
            name="Action Agent", 
            api_url="http://localhost:5002/process",
            description="Handles exports, campaigns, and integrations",
            keywords=["export", "csv", "outreach", "campaign", "marketo"]
        )
        
        print("✅ Agents registered manually")
    
    # 3. List all agents
    print("\n3️⃣ Available agents:")
    agents = system.list_agents()
    for agent in agents:
        print(f"  • {agent['name']}: {agent['description']}")
        print(f"    Keywords: {', '.join(agent['keywords'][:5])}...")
        print(f"    Tools: {', '.join(agent['tools'][:3])}...")
    
    # 4. Process queries
    print("\n4️⃣ Processing queries...")
    
    test_queries = [
        "Filter customers by location",
        "Show me high-value prospects",
        "Export the filtered data to CSV", 
        "Create an outreach campaign",
        "Add these leads to Marketo"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n  Query {i}: {query}")
        
        try:
            # This is all you need to do!
            result = system.process_query(query)
            
            print(f"  🤖 {result['agent_used']}: {result['response'][:100]}...")
            
            if result.get('handoff_occurred'):
                print(f"  🔄 Handed off from {result.get('previous_agent', 'None')}")
                
            print(f"  ⏱️  Response time: {result.get('execution_time', 0):.2f}s")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
    
    # 5. Show session statistics
    print("\n5️⃣ Session Statistics:")
    session_info = system.get_session_info()
    print(f"  • Session ID: {session_info['session_id'][:8]}...")
    print(f"  • Total messages: {session_info['message_count']}")
    print(f"  • Current agent: {session_info.get('current_agent', 'None')}")
    
    # 6. Show agent performance
    print("\n6️⃣ Agent Performance:")
    performance = system.get_agent_performance()
    for agent_name, stats in performance.items():
        print(f"  • {agent_name}:")
        print(f"    - Uses: {stats['usage_count']}")
        print(f"    - Success rate: {stats['success_rate']:.1%}")
        print(f"    - Last used: {stats['last_used'] or 'Never'}")


def interactive_demo():
    """Interactive demo where user can type queries"""
    
    print("🎮 Interactive Multi-Agent Demo")
    print("=" * 40)
    print("Type your queries and see the system in action!")
    print("Type 'quit' to exit, 'agents' to list agents, 'clear' to clear conversation")
    print()
    
    # Initialize system
    system = MultiAgentOrchestrator(openai_api_key=None)
    
    # Register demo agents (these would be your real agents)
    system.register_agent(
        name="Filter Agent",
        agent_id="filter_agent",
        api_url="http://localhost:5001/process",
        description="Filters and searches data",
        keywords=["filter", "search", "find", "show", "where"]
    )
    
    system.register_agent(
        name="Action Agent",
        agent_id="action_agent",
        api_url="http://localhost:5002/process", 
        description="Handles exports, campaigns, and integrations",
        keywords=["export", "csv", "outreach", "campaign", "marketo", "add", "create"]
    )
    
    print(f"✅ System ready with {len(system.registered_agents)} agents")
    print()
    
    while True:
        try:
            # Get user input
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                continue
                
            # Handle special commands
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'agents':
                print("\n📋 Available agents:")
                for agent in system.list_agents():
                    print(f"  • {agent['name']}: {agent['description']}")
                print()
                continue
            elif user_input.lower() == 'clear':
                system.clear_conversation()
                print("🧹 Conversation cleared!")
                print()
                continue
            elif user_input.lower() == 'stats':
                session_info = system.get_session_info()
                print(f"\n📊 Session Stats:")
                print(f"  • Messages: {session_info['message_count']}")
                print(f"  • Current agent: {session_info.get('current_agent', 'None')}")
                # print(f"  • All data: {system.get_all_data()}")
                print()
                continue
            
            # Process the query
            print("🤔 Processing...")
            start_time = time.time()
            
            result = system.process_query(user_input)
            
            processing_time = time.time() - start_time
            
            # Display result
            agent_name = result.get('agent_used', 'System')
            response = result.get('response', 'No response')
            
            print(f"🤖 {agent_name}: {response}")
            
            # Show additional info
            if result.get('handoff_occurred'):
                print(f"🔄 Handed off from {result.get('previous_agent', 'None')}")
            
            if result.get('error'):
                print(f"❌ Error occurred: {result.get('error')}")
            
            print(f"⏱️  Processing time: {processing_time:.2f}s")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print()


def yaml_config_example():
    """Show how to use YAML configuration"""
    
    print("📄 YAML Configuration Example")
    print("=" * 40)
    
    yaml_content = """
# agents_config.yaml
agents:
  - name: "Filter Agent"
    agent_id: "filter_agent"
    api_url: "http://localhost:5001/process"
    description: "Filters and searches data"
    keywords: ["filter", "search", "find", "show"]
    tools: ["data_filter", "search_engine"]
    priority: 2
    
  - name: "Action Agent"
    agent_id: "action_agent" 
    api_url: "http://localhost:5002/process"
    description: "Handles exports and campaigns"
    keywords: ["export", "csv", "outreach", "campaign"]
    tools: ["add_outreach", "get_csv", "add_marketo"]
    priority: 3
"""
    
    print("1️⃣ Create your YAML configuration file:")
    print(yaml_content)
    
    print("2️⃣ Load agents from YAML:")
    print("""
from mult_agent_enhanced import MultiAgentOrchestrator

# Initialize system
system = MultiAgentOrchestrator(openai_api_key="your-api-key")

# Load all agents from YAML - that's it!
registered_agents = system.load_agents_from_yaml("agents_config.yaml")

# Start processing queries
result = system.process_query("Filter customers by location")
""")
    
    print("3️⃣ Key benefits of YAML configuration:")
    print("  ✅ Easy to manage multiple agents")
    print("  ✅ Version control friendly")
    print("  ✅ Share configurations between environments")
    print("  ✅ No code changes needed for agent updates")
    print("  ✅ Supports all agent features (tools, priorities, etc.)")


def integration_guide():
    """Show how to integrate with existing systems"""
    
    print("🔧 Integration Guide")
    print("=" * 30)
    
    print("""
📋 STEP 1: Prepare Your Agents
────────────────────────────────
Each agent should expose a REST API endpoint that accepts:

POST /process
{
  "query": "user's query string",
  "context": {
    "conversation_history": [...],
    "session_info": {...}
  },
  "session_id": "session-uuid",
  "agent_id": "your-agent-id"
}

And returns:
{
  "response": "agent's response to user",
  "handoff_requested": false,  // optional
  "target_agent": "other_agent_id",  // optional
  "metadata": {...},  // optional
  "tools_used": [...]  // optional
}

📋 STEP 2: Register Your Agents
──────────────────────────────────
Option A - YAML Configuration (Recommended):
1. Create agents_config.yaml with your agent details
2. Load with: system.load_agents_from_yaml("agents_config.yaml")

Option B - Programmatic Registration:
system.register_agent(
    name="Your Agent",
    api_url="https://your-api.com/process",
    description="What your agent does",
    keywords=["relevant", "keywords"],
    tools=["tool1", "tool2"]
)

📋 STEP 3: Process User Queries
──────────────────────────────────
# That's it! The system handles everything else
result = system.process_query(user_input)

📋 STEP 4: Handle Responses
──────────────────────────────
response = result['response']  # Show to user
agent_used = result['agent_used']  # Which agent responded
handoff_occurred = result.get('handoff_occurred', False)  # Did handoff happen

📋 ADVANCED: Agent Handoffs
──────────────────────────────────
Your agents can request handoffs by returning:
{
  "response": "I'll hand this off to the export agent",
  "handoff_requested": true,
  "target_agent": "export_agent"  // optional - system will choose if not specified
}

The main system will automatically:
✅ Select the best agent for handoff
✅ Maintain conversation context
✅ Prevent infinite loops
✅ Track performance metrics
""")


def main_menu():
    """Main menu for examples"""
    
    print("🎯 Multi-Agent System Examples")
    print("=" * 40)
    print("Choose an example to run:")
    print()
    print("1️⃣  Simple Example - Basic usage demonstration")
    print("2️⃣  Interactive Demo - Chat with the system")
    print("3️⃣  YAML Configuration - Show configuration setup")
    print("4️⃣  Integration Guide - How to integrate with your agents")
    print("5️⃣  Exit")
    print()
    
    while True:
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == '1':
                simple_example()
                break
            elif choice == '2':
                interactive_demo()
                break
            elif choice == '3':
                yaml_config_example()
                break
            elif choice == '4':
                integration_guide()
                break
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break


if __name__ == "__main__":
    main_menu()