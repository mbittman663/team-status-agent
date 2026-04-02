from typing import TypedDict, List
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
import json
import pandas as pd

# =========================
# 0. Load environment vars
# =========================
load_dotenv()  # loads OPENAI_API_KEY from .env

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env")

# =========================
# 1. Define State
# =========================
class AgentState(TypedDict):
    input: str
    plan: str
    research: str
    output: str
    history: List[str]

# =========================
# 2. Initialize LLM
# =========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

# =========================
# 3. Persistent Memory Setup
# =========================
MEMORY_FILE = "agent_memory.json"
EXCEL_FILE = "BankingTeam.xlsx"

def load_memory() -> List[str]:
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_memory(memory: List[str]):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)

# =========================
# 4. Load Excel Tasks
# =========================
def load_task_memory() -> List[str]:
    """Load tasks from Excel into memory-friendly strings."""
    if not os.path.exists(EXCEL_FILE):
        return []

    df = pd.read_excel(EXCEL_FILE)
    tasks_memory = []
    for _, row in df.iterrows():
        tasks_memory.append(f"Team: {row['Team']}, Task: {row['Task']}, Owner: {row['Owner']}, Status: {row['Status']}, Notes: {row['Notes']}")
    return tasks_memory

# =========================
# 5. Agent Nodes
# =========================
def planner(state: AgentState) -> AgentState:
    prompt = f"""
Break this task into a clear step-by-step plan considering the user query:

User query:
{state['input']}

Task:
{state['input']}
"""
    response = llm.invoke(prompt).content
    state["plan"] = response
    state["history"].append(f"PLAN:\n{response}")
    return state

def researcher(state: AgentState) -> AgentState:
    prompt = f"""
Use the following plan to gather insights:

Plan:
{state['plan']}
"""
    response = llm.invoke(prompt).content
    state["research"] = response
    state["history"].append(f"RESEARCH:\n{response}")
    return state

def writer(state: AgentState) -> AgentState:
    # Load past memory
    memory = load_memory()
    memory_context = "\n".join(memory[-5:])  # last 5 entries
    # Load Excel tasks
    task_memory = load_task_memory()
    tasks_context = "\n".join(task_memory)

    prompt = f"""
You are an assistant helping analyze tasks for multiple teams in a concise manner for the senior manager leading all of the teams.

Past agent memory:
{memory_context}

Known tasks from Excel:
{tasks_context}

Research / Plan:
{state['research']}

Answer the user query as specifically and concisely as possible.
"""
    response = llm.invoke(prompt).content
    state["output"] = response
    state["history"].append(f"OUTPUT:\n{response}")

    # Save output to memory
    memory.append(response)
    save_memory(memory)
    return state

# =========================
# 6. Build Graph
# =========================
def build_graph():
    builder = StateGraph(AgentState)
    builder.add_node("planner", planner)
    builder.add_node("researcher", researcher)
    builder.add_node("writer", writer)
    builder.set_entry_point("planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    return builder.compile()

# =========================
# 7. Run Agent
# =========================
def run_agent(user_input: str):
    graph = build_graph()
    initial_state: AgentState = {
        "input": user_input,
        "plan": "",
        "research": "",
        "output": "",
        "history": []
    }
    result = graph.invoke(initial_state)
    return result

# =========================
# 8. CLI Entry Point
# =========================
if __name__ == "__main__":
    print("=== TEAM TASK AGENT ===")
    print("You can query completed/in progress/not started tasks for multiple teams.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter your query: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        result = run_agent(user_input)
        print("\n=== FINAL OUTPUT ===\n")
        print(result["output"])
        print("\n=== DEBUG HISTORY ===\n")
        for step in result["history"]:
            print(step)
            print("-" * 50)
