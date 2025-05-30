from dotenv import load_dotenv
load_dotenv()
import random

from langchain_openai import ChatOpenAI
import time

from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o")

# Create specialized agents

def add(a: float, b: float) -> float:
    """Add two numbers."""
    # print(f"Adding {a} and {b}")
    # time.sleep(50)  # Simulate a long-running task
    # print("Addition complete")
    
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

math_agent = create_react_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert",
    prompt="You are a math expert."
)

research_agent = create_react_agent(
    model=model,
    tools=[web_search],
    name="research_expert",
    prompt="You are a world class researcher with access to web search."
)

# Create supervisor workflow
workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You are a team supervisor managing a research expert and a math expert. "
        "For current events, use research_agent. "
        "For math problems, use math_agent."
    )
)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "can you apply filter to outreach sequence? for filtering email sequences, marketing campaigns"
        }
    ]
})

print(result)


class FilterAgent:
    """
    Mock FilterAgent to simulate data filtering.
    Modified to sometimes return ambiguous results for Analysis Paralysis.
    """
    def invoke(self, query: str):
        print(f"FilterAgent: Filtering data based on query: '{query}'")
        mock_data = [
            {"id": 1, "name": "Alice", "age": 30, "city": "New York", "salary": 70000, "department": "Marketing"},
            {"id": 2, "name": "Bob", "age": 25, "city": "London", "salary": 60000, "department": "Sales"},
            {"id": 3, "name": "Charlie", "age": 35, "city": "New York", "salary": 80000, "department": "Engineering"},
            {"id": 4, "name": "David", "age": 40, "city": "New York", "salary": 90000, "department": "Marketing"}
        ]        
        if random.random() < 0.3: # 30% chance of ambiguity
            print("FilterAgent: Result is ambiguous, suggests further research.")
            return {"status": "ambiguous_result", "data": [], "reason": "More specific criteria needed."}
        elif random.random() < 0.3: # 30% chance of error
            print(f"FilterAgent: Filtering FAILED due to a simulated error!")
            return {"status": "failure", "message": f"Filter '{query}' encountered an unexpected error."}
        elif "name" in query.lower():
            return {"status": "success", "data": [d for d in mock_data if "alice" in d["name"].lower() or "david" in d["name"].lower()]}
        else:
            return {"status": "success", "data": mock_data}
        
class ActionAgent:
    """
    Mock ActionAgent to simulate performing actions on data.
    Modified to sometimes fail for Premature Disengagement.
    """
    def invoke(self, action_type: str, data: list):
        print(f"ActionAgent: Attempting action '{action_type}' on {len(data)} records.")
        if random.random() < 0.2 and action_type != "export_csv": # 20% chance of failure for non-export actions
            print(f"ActionAgent: Action '{action_type}' FAILED due to a simulated error!")
            return {"status": "failure", "message": f"Action '{action_type}' encountered an unexpected error."}        
        if action_type == "send_email":
            print(f"  Sending email to: {[d.get('name', 'N/A') for d in data]}")
            return {"status": "success", "message": f"Email sent to {len(data)} recipients."}
        elif action_type == "export_csv":
            print(f"  Exporting {len(data)} records to CSV.")
            return {"status": "success", "message": f"Data exported to CSV for {len(data)} records."}
        elif action_type == "update_record":
            print(f"  Updating records: {data}")
            return {"status": "success", "message": f"Records updated for {len(data)} entries."}
        else:
            return {"status": "failure", "message": f"Unknown action type: {action_type}"}