
import os
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

# --------------------------------------------------
# Project setup
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from src.agent_tools import (
    query_telemetry_db,
    get_temperature_breaches,
    fetch_corridor_conditions,
    search_compliance_sop,
)

# --------------------------------------------------
# LLM configuration
# --------------------------------------------------

AGENT_LLM_SETTING = os.getenv("Agent_llm", "OLLAMA").strip().upper()

if AGENT_LLM_SETTING == "OPENAI":
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
    )

elif AGENT_LLM_SETTING == "DEEPSEEK":
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        temperature=0,
        max_tokens=1024,
    )

else:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:3b"),
        temperature=0,
        num_predict=1024,
    )

# --------------------------------------------------
# Tools
# --------------------------------------------------

fde_tools = [
    query_telemetry_db,
    get_temperature_breaches,
    fetch_corridor_conditions,
    search_compliance_sop,
]

llm_with_tools = llm.bind_tools(fde_tools)

# --------------------------------------------------
# Agent state
# --------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --------------------------------------------------
# Temperature breach handling
# --------------------------------------------------

def is_temperature_breach_question(question: str) -> bool:
    question = question.lower()

    breach_terms = [
        "temperature breach",
        "temperature breaches",
        "temperature violation",
        "temperature violations",
        "above 4",
        "exceed 4",
        "exceeds 4",
        "cold chain breach",
        "cold-chain breach",
        "overheated cargo",
    ]

    return any(term in question for term in breach_terms)


def handle_temperature_breach_question(user_input: str) -> str:
    print("\nRetrieving verified temperature records from SQL Server...")

    telemetry = get_temperature_breaches.invoke({})

    print("Retrieving applicable cold-chain SOP rules...")

    sop_rules = search_compliance_sop.invoke({
        "query": (
            "Fresh perishables temperature limits, critical temperature "
            "breach above 4.0 degrees Celsius, auxiliary cooling unit, "
            "actual ETA delay greater than one hour, and Tier 2 escalation."
        )
    })

    answer = f"""
COLD-CHAIN TEMPERATURE BREACH ANALYSIS

Question:
{user_input}

1. VERIFIED DATABASE RESULTS

{telemetry}

2. RETRIEVED SOP RULES

{sop_rules}

3. OPERATIONAL RULES

- Fresh perishables must remain between 0.0°C and 4.0°C.
- A temperature strictly greater than 4.0°C is a critical breach.
- Exactly 4.0°C is not a breach.
- The dispatcher must contact the driver and instruct them to restart
  the auxiliary cooling unit.
- If the actual ETA delay exceeds one hour, divert the vehicle to the
  nearest emergency cold-storage facility.
- A shipment classified as High Risk with delay probability greater
  than 0.65 must be escalated to the Tier 2 Logistics Manager.

IMPORTANT DATA LIMITATIONS

- The database results are historical records, not confirmed live alerts.
- Delay probability is not the same as actual ETA delay.
- A port congestion value alone does not confirm that a shipment is
  at the Port of Long Beach or Los Angeles.
- Operational actions must be based on the relevant verified shipment
  location and current conditions.
"""

    return answer
# --------------------------------------------------
# Tier 2 escalation handling
# --------------------------------------------------

def is_tier2_question(question: str) -> bool:
    question = question.lower()

    tier2_terms = [
        "tier 2",
        "tier2",
        "tier-two",
        "escalation criteria",
        "escalation candidates",
        "high risk shipments",
    ]

    return any(term in question for term in tier2_terms)


def handle_tier2_question(user_input: str) -> str:
    print("\nRetrieving Tier 2 escalation candidates from SQL Server...")

    sql_query = """
        SELECT TOP 10
            [Timestamp],
            [Latitude],
            [Longitude],
            [Current_Temperature_C],
            [Risk_Classification],
            [Delay_Probability],
            [Port_Congestion_Level]
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        WHERE [Risk_Classification] = 'High Risk'
          AND [Delay_Probability] > 0.65
        ORDER BY [Delay_Probability] DESC;
    """

    telemetry = query_telemetry_db.invoke({
        "sql_query": sql_query
    })

    print("Retrieving Tier 2 escalation rule from SOP...")

    sop_rules = search_compliance_sop.invoke({
        "query": (
            "Any shipment classified as High Risk with delay "
            "probability greater than 0.65 must be escalated "
            "to the Tier 2 Logistics Manager."
        )
    })

    return f"""
TIER 2 ESCALATION ANALYSIS

Question:
{user_input}

1. VERIFIED DATABASE RESULTS

{telemetry}

2. RETRIEVED SOP RULE

{sop_rules}

3. ESCALATION CRITERIA

A shipment must be escalated when BOTH conditions are satisfied:

- Risk_Classification = High Risk
- Delay_Probability > 0.65

4. REQUIRED ACTION

Escalate qualifying shipments to the Tier 2 Logistics Manager.

IMPORTANT:
- These database records may be historical, not live alerts.
- A delay probability greater than 0.65 is a risk indicator,
  not confirmation of an actual delay.
- A temperature breach is not required for Tier 2 escalation.
"""

# --------------------------------------------------
# LangGraph reasoning node
# --------------------------------------------------

def reasoning_node(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# --------------------------------------------------
# Build agent graph
# --------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("reasoner", reasoning_node)
workflow.add_node("tools", ToolNode(fde_tools))

workflow.add_edge(START, "reasoner")

workflow.add_conditional_edges(
    "reasoner",
    tools_condition,
)

workflow.add_edge("tools", "reasoner")
workflow.add_edge("reasoner", END)

memory = MemorySaver()

fde_agent = workflow.compile(checkpointer=memory)


# --------------------------------------------------
# Main application
# --------------------------------------------------

def main():

    prompt_path = PROJECT_ROOT / "src" / "prompts" / "system_prompt.txt"

    if not prompt_path.exists():
        print(f"System prompt file not found: {prompt_path}")
        return

    system_prompt_text = prompt_path.read_text(encoding="utf-8")

    print("\n======================================")
    print(" COLD CHAIN LOGISTICS AI ASSISTANT")
    print("======================================")
    print(f"LLM mode: {AGENT_LLM_SETTING}")
    print("Type 'exit' to stop.\n")

    thread_config = {
        "configurable": {
            "thread_id": "cold-chain-session-1"
        }
    }

    first_turn = True

    while True:

        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\nAssistant: Goodbye!")
                break

            # Use verified SQL and SOP data for breach questions.
            if is_temperature_breach_question(user_input):
                answer = handle_temperature_breach_question(user_input)
                print("\nAssistant:")
                print(answer)
                continue
            if is_tier2_question(user_input):
                answer = handle_tier2_question(user_input)
            print("\nAssistant:")
            print(answer)
            continue

            # Normal agent conversation through LangGraph.
            if first_turn:
                messages = [
                    SystemMessage(content=system_prompt_text),
                    HumanMessage(content=user_input),
                ]
                first_turn = False
            else:
                messages = [HumanMessage(content=user_input)]

            print("\nAssistant: Thinking...")

            result = fde_agent.invoke(
                {"messages": messages},
                config=thread_config,
            )

            final_message = result["messages"][-1]

            print("\nAssistant:")
            print(final_message.content)

        except KeyboardInterrupt:
            print("\n\nAssistant: Session stopped.")
            break

        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
