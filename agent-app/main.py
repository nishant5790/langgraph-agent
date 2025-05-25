from langgraph.graph import StateGraph , START,END
from typing import TypedDict, Dict,List

class AgentState(TypedDict):
    num1:int
    num2:int
    result:int

def add_val(state:AgentState)->AgentState:
    state['result'] = state['num1']+state['num2']
    return state

graph = StateGraph(AgentState)

graph.add_node('add_number',add_val)
graph.set_entry_point('add_number')
graph.set_finish_point('add_number')
app = graph.compile()
