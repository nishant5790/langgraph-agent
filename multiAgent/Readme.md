# 🤖 Multi-Agent Orchestration System

A powerful, flexible multi-agent system that intelligently routes user queries to specialized agents and manages handoffs between them. Built with OpenAI GPT-4o for smart agent selection and designed for easy integration with any agent APIs.

## ✨ Key Features

- **🧠 Intelligent Agent Selection** - Uses GPT-4o to select the best agent for each query
- **🔄 Smart Handoffs** - Agents can seamlessly handoff to other agents when needed
- **💾 Memory & Context** - Maintains conversation context and agent performance memory
- **📄 YAML Configuration** - Easy agent management through configuration files
- **🛡️ Error Handling** - Robust error handling with fallbacks and timeout protection
- **📊 Performance Tracking** - Built-in analytics for agent usage and success rates
- **🔁 Loop Prevention** - Prevents infinite handoff loops
- **⚡ Fast & Scalable** - Efficient agent selection and concurrent request handling

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install openai requests pyyaml flask
```

### 2. Create Agent Configuration

Create `agents_config.yaml`:

```yaml
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
```

### 3. Initialize and Use

```python
from mult_agent_enhanced import MultiAgentOrchestrator

# Initialize system
system = MultiAgentOrchestrator(openai_api_key="your-openai-api-key")

# Load agents from YAML
system.load_agents_from_yaml("agents_config.yaml")

# Process user queries - that's it!
result = system.process_query("Filter customers by location")
print(result['response'])
```

## 📋 Agent API Requirements

Your agents should expose a REST API endpoint that:

### Accepts POST requests to `/process`:
```json
{
  "query": "user's query string",
  // "context": {
  //   "conversation_history": [...],
  //   "session_info": {"session_id": "uuid", "current_agent": "agent_id"}
  // },
  // "session_id": "session-uuid",
  // "agent_id": "your-agent-id"
}
```

### Returns responses in format:
```json
{
  "response": "agent's response to user",
  // "handoff_requested": false,
  // "target_agent": "other_agent_id",
  // "metadata": {"any": "additional data"},
  // "tools_used": ["tool1", "tool2"]
}
```

## 🔄 Handoff System

Agents can request handoffs by setting `handoff_requested: true` in their response:

```json
{
  "response": "I found the data. Handing off to Action Agent for export.",
  "handoff_requested": true,
  "target_agent": "action_agent"
}
```

The system automatically:
- ✅ Selects the best target agent (if not specified)
- ✅ Maintains conversation context
- ✅ Prevents infinite loops
- ✅ Tracks handoff performance

## 📊 Agent Configuration Options

| Field | Required | Description | Example |
|-------|----------|-------------|---------|
| `name` | ✅ | Human-readable agent name | "Filter Agent" |
| `api_url` | ✅ | Agent's API endpoint | "http://localhost:5001/process" |
| `agent_id` | ❌ | Unique identifier | "filter_agent" |
| `description` | ❌ | What the agent does | "Filters and searches data" |
| `keywords` | ❌ | Trigger words | ["filter", "search"] |
| `tools` | ❌ | Agent capabilities | ["data_filter", "search_engine"] |
| `priority` | ❌ | Selection priority (1-10) | 2 |
| `headers` | ❌ | HTTP headers | {"Authorization": "Bearer token"} |
| `timeout` | ❌ | Request timeout (seconds) | 30 |
| `enabled` | ❌ | Whether agent is active | true |

## 🧪 Testing Your System

Run comprehensive tests:

```bash
# Run all tests
python test_system.py test

# Run interactive demo
python test_system.py demo

# Run usage examples
python usage_example.py
```

### Test Categories:
- ✅ **Basic Functionality** - Agent registration, selection, and responses
- ✅ **Handoff Scenarios** - Complex multi-agent conversations
- ✅ **Error Handling** - Timeouts, connection errors, malformed responses
- ✅ **Edge Cases** - Empty queries, special characters, very long inputs
- ✅ **Performance** - Large numbers of agents, concurrent requests
- ✅ **Configuration** - YAML loading, export, validation

## 🎯 Usage Examples

### Example 1: Filter → Action Handoff
```
User: "Filter high-value customers and export to CSV"

Filter Agent: "Found 150 high-value customers. Handing off to Action Agent for export."
↓ [Automatic Handoff]
Action Agent: "✅ CSV export completed. 150 records exported to customers_highvalue.csv"
```

### Example 2: Context-Aware Conversation
```
User: "Show me customers in California"
Filter Agent: "Found 45 customers in California."

User: "Export those to CSV"
Action Agent: "✅ Exported 45 California customers to CSV."

User: "Create outreach campaign for them"
Action Agent: "✅ Outreach campaign created for 45 California customers."
```

## 📈 Performance Monitoring

Track agent performance:

```python
# Get performance stats
performance = system.get_agent_performance()
for agent_name, stats in performance.items():
    print(f"{agent_name}: {stats['usage_count']} uses, {stats['success_rate']:.1%} success")

# Get session info
session_info = system.get_session_info()
print(f"Messages: {session_info['message_count']}")
print(f"Handoffs: {session_info['handoff_count']}")
```

## 🔧 Advanced Configuration

### Memory Management
```python
# Clear conversation history
system.clear_conversation()

# Get conversation history
history = system.get_conversation_history()
```

### Export/Import Configuration
```python
# Export current configuration
system.export_configuration("backup_config.yaml")

# Load from different environment
system.load_agents_from_yaml("production_config.yaml")
```

### Manual Agent Registration
```python
# Register agent programmatically
system.register_agent(
    name="Custom Agent",
    api_url="https://api.example.com/process",
    description="Custom functionality",
    keywords=["custom", "special"],
    tools=["custom_tool"],
    headers={"Authorization": "Bearer token"},
    priority=5
)
```

## 🛡️ Error Handling

The system handles various error scenarios:

- **🔌 Connection Errors** - Graceful fallback when agents are unreachable
- **⏰ Timeouts** - Configurable timeouts with proper error messages
- **🔄 Infinite Loops** - Automatic detection and prevention of handoff loops
- **📄 Invalid Responses** - Handles malformed JSON and unexpected formats
- **❌ Agent Failures** - Continues operation when individual agents fail

## 🏗️ Architecture

```
User Query
    ↓
Main System (GPT-4o Selection)
    ↓
Selected Agent (API Call)
    ↓
Response / Handoff Request
    ↓
[If Handoff] → New Agent Selection
    ↓
Final Response to User
```

### Key Components:
- **🎯 Agent Router** - Intelligent selection using GPT-4o + fallback algorithms
- **💾 Memory Manager** - Conversation history and agent performance tracking
- **🔄 Handoff Controller** - Manages agent transitions and prevents loops
- **📊 Performance Monitor** - Tracks usage statistics and success rates
- **⚙️ Configuration Manager** - YAML-based agent configuration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request


## 🆘 Support

- 📚 **Documentation**: See the examples and test files
- 🐛 **Issues**: Report bugs in the issue tracker
- 📧 **Contact**: Reach out for enterprise support

---

## 🎉 Ready to Get Started?

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your `agents_config.yaml`
4. Run the examples: `python usage_example.py`
5. Start building your multi-agent system!

**🚀 Happy orchestrating!**


✅ Key Features Implemented:
1. YAML Configuration Support

Easy agent registration through YAML files
No code changes needed to add/modify agents
Supports all agent properties (tools, priorities, headers, etc.)

2. Intelligent Agent Selection

Uses GPT-4o for smart agent routing
Fallback keyword matching system
Memory-based performance tracking
Priority-based selection

3. Robust Handoff System

Automatic handoff detection and execution
Context preservation across handoffs
Loop prevention (max handoffs limit)
Support for explicit target agent specification

4. Memory & Context Management

Conversation history maintenance
Agent performance tracking (usage, success rates)
Session management
Context-aware decision making

5. Comprehensive Testing Suite

Basic functionality tests: Agent registration, selection, responses
Handoff scenario tests: Complex multi-agent conversations
Error handling tests: Timeouts, connection errors, malformed responses
Edge case tests: Empty queries, special characters, very long inputs
Performance tests: Large numbers of agents, concurrent requests
Configuration tests: YAML loading, export, validation

📁 Files Created:

mult_agent_enhanced.py - Enhanced multi-agent system with YAML support
agents_config.yaml - Sample configuration for your Filter and Action agents
test_system.py - Comprehensive test suite covering all edge cases
usage_example.py - Simple examples showing how to use the system
README.md - Complete documentation and integration guide

🎯 For Your Use Case:
Filter Agent Integration:
```yaml- 
  name: "Filter Agent"
  api_url: "http://your-filter-api.com/process"
  keywords: ["filter", "search", "find", "query"]
  tools: ["data_filter", "search_engine"] 
 Action Agent Integration: 
 ```


```yaml- 
  name: "Action Agent"  
  api_url: "http://your-action-api.com/process"
  keywords: ["export", "csv", "outreach", "marketo", "campaign"]
  tools: ["add_outreach", "get_csv", "add_marketo"]
```
# 🚀 Getting Started: 

###  Set up your agents in 
```
agents_config.yaml
```
### Initialize the system: 
```python
system = MultiAgentOrchestrator(api_key)
```

### Load agents: 
```python
system.load_agents_from_yaml("agents_config.yaml")
Process queries: result = system.process_query("user input")
```

### 🧪 Running Tests:

 ``` bash 
# Run comprehensive tests
python test_system.py test 


# Run interactive demo  
python test_system.py demo

# Try usage examples
python usage_example.py

```

The system is designed to be production-ready with robust error handling, performance monitoring, and comprehensive testing. It will intelligently route queries between your Filter and Action agents, handle handoffs seamlessly, and maintain conversation context throughout multi-agent interactions.RetryClaude can make mistakes. Please double-check responses.