
import os
import sys
import uuid
import json
import urllib
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_core.messages import HumanMessage, ToolMessage

# ==========================================
# 1. IMMEDIATE PATH & ENVIRONMENT RESOLUTION
# ==========================================

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

# Import orchestrator, graph and deterministic handlers
from src.orchestrator import (
    fde_agent,
    is_temperature_breach_question,
    handle_temperature_breach_question,
    is_tier2_question,
    handle_tier2_question
)

# ==========================================
# 2. SQL CREDENTIALS MAPPING FROM .ENV
# ==========================================

db_host = os.getenv("SQL_SERVER_HOST", "127.0.0.1")
db_port = os.getenv("SQL_SERVER_PORT", "1433")
db_user = os.getenv("SQL_AGENT_USER", "USR_FDE_RO")
db_password = os.getenv("SQL_AGENT_PASSWORD")

connection_string = (
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={db_host},{db_port};"
    f"DATABASE=master;"
    f"UID={db_user};"
    f"PWD={db_password};"
    f"Encrypt=no;"
    f"TrustServerCertificate=yes;"
)

log_params = urllib.parse.quote_plus(connection_string)

log_engine = create_engine(
    f"mssql+pyodbc:///?odbc_connect={log_params}"
)


def write_audit_log(session_id, node_name, tool_name, content):
    """Writes agent execution traces to the SQL audit table."""

    try:
        with log_engine.connect() as conn:
            conn.execute(
                text("""
                    INSERT INTO FDE_VIEWS.AgentAuditLog
                    (SessionID, NodeExecuted, ToolName, Content)
                    VALUES (:session_id, :node_name, :tool_name, :content)
                """),
                {
                    "session_id": session_id,
                    "node_name": node_name,
                    "tool_name": tool_name,
                    "content": content
                }
            )
            conn.commit()

    except Exception as e:
        print(f"Audit Log Failed (Silent): {e}")


# ==========================================
# 3. PAGE CONFIGURATION & ENTERPRISE THEME
# ==========================================

st.set_page_config(
    page_title="FDE Supply Chain Dispatch Console",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0B0E14;
        color: #E2E8F0;
    }

    div[data-testid="stSidebar"] {
        background-color: #111622;
        border-right: 1px solid #1E293B;
    }

    .stMarkdown code {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
    }
   /* Make typed chat input text clearly visible */
div[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
    background-color: #1E293B !important;
}

/* Chat input placeholder */
div[data-testid="stChatInput"] textarea::placeholder {
    color: #CBD5E1 !important;
    opacity: 1 !important;
}
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 4. MULTI-USER STATE & THREAD MANAGEMENT
# ==========================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

thread_config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}


# ==========================================
# 5. SIDEBAR NAVIGATION & METADATA
# ==========================================

with st.sidebar:

    image_path = script_dir / "image_L25X5q.png"

    if image_path.exists():
        st.image(str(image_path), width=65)
    else:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/2830/2830305.png",
            width=65
        )

    st.title("FDE Command Center")

    app_mode = st.radio(
        "System Mode",
        [
            "🧊 Dispatch Console",
            "🛡️ Security & Audit Logs"
        ]
    )

    st.markdown("---")

    st.caption(
        f"Session Token: `{st.session_state.thread_id[:8]}...`"
    )

    st.markdown(
        f"**Reasoning Architecture:** "
        f"`{os.getenv('Agent_llm', 'DEEPSEEK')}`"
    )

    st.markdown("---")

    if st.button(
        "🗑️ Purge Dispatch Workspace Session",
        use_container_width=True
    ):
        st.session_state.ui_messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ==========================================
# 6. VIEW ROUTING (DISPATCH VS AUDIT)
# ==========================================

if app_mode == "🧊 Dispatch Console":

    # ------------------------------------------
    # TAB 1: CHAT UI & AGENT EXECUTION
    # ------------------------------------------

    st.title("Cold-Chain Incident Control Dashboard")

    st.caption(
        "Production Data Engineering Pipeline • "
        "Real-Time Decision Optimization Platform"
    )

    # Display previous conversation messages
    for entry in st.session_state.ui_messages:

        with st.chat_message(
            entry["role"],
            avatar="👤" if entry["role"] == "user" else "🤖"
        ):

            if "traces" in entry:

                for trace in entry["traces"]:

                    if trace["type"] == "tool_input":

                        st.markdown(
                            f"**⚡ Intent Recognized:** `{trace['name']}`"
                        )

                        with st.expander(
                            f"📥 View Generated Input ({trace['name']})",
                            expanded=False
                        ):
                            st.json(trace["args"])

                    elif trace["type"] == "tool_output":

                        with st.expander(
                            f"📤 View Raw Output ({trace['name']})",
                            expanded=False
                        ):
                            st.code(
                                trace["content"],
                                language="text"
                            )

            st.markdown(entry["content"])


    # ------------------------------------------
    # USER CHAT INPUT
    # ------------------------------------------

    if user_input := st.chat_input(
        "Query fleet telemetry, corridor updates, or compliance thresholds..."
    ):

        st.session_state.ui_messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)


        # ------------------------------------------
        # ASSISTANT RESPONSE
        # ------------------------------------------

        with st.chat_message("assistant", avatar="🤖"):

            final_response = ""
            current_traces = []

            try:

                # ==========================================
                # ROUTE 1: TEMPERATURE BREACH QUESTIONS
                # ==========================================

                if is_temperature_breach_question(user_input):

                    with st.status(
                        "🌡️ Checking temperature breaches and SOP rules...",
                        expanded=True
                    ) as status:

                        final_response = handle_temperature_breach_question(
                            user_input
                        )

                        status.update(
                            label="Temperature Breach Evaluation Complete",
                            state="complete",
                            expanded=False
                        )


                # ==========================================
                # ROUTE 2: TIER 2 ESCALATION QUESTIONS
                # ==========================================

                elif is_tier2_question(user_input):

                    with st.status(
                        "🚨 Checking Tier 2 escalation criteria...",
                        expanded=True
                    ) as status:

                        final_response = handle_tier2_question(
                            user_input
                        )

                        status.update(
                            label="Tier 2 Escalation Evaluation Complete",
                            state="complete",
                            expanded=False
                        )


                # ==========================================
                # ROUTE 3: NORMAL LANGGRAPH AGENT EXECUTION
                # ==========================================

                else:

                    with st.status(
                        "🧠 Initializing Core Reasoner Node...",
                        expanded=True
                    ) as status:

                        events = fde_agent.stream(
                            {
                                "messages": [
                                    HumanMessage(content=user_input)
                                ]
                            },
                            config=thread_config,
                            stream_mode="updates"
                        )

                        for event in events:

                            for node_name, node_state in event.items():

                                # ----------------------------------
                                # A. REASONER NODE
                                # ----------------------------------

                                if node_name == "reasoner":

                                    latest_msg = node_state["messages"][-1]

                                    # Intercept tool calls
                                    if (
                                        hasattr(latest_msg, "tool_calls")
                                        and latest_msg.tool_calls
                                    ):

                                        status.update(
                                            label="🧠 Agent generated tool parameters..."
                                        )

                                        for tool_call in latest_msg.tool_calls:

                                            st.markdown(
                                                f"**⚡ Intent Recognized:** "
                                                f"`{tool_call['name']}`"
                                            )

                                            with st.expander(
                                                f"📥 View Generated Input "
                                                f"({tool_call['name']})",
                                                expanded=False
                                            ):
                                                st.json(tool_call["args"])

                                            current_traces.append({
                                                "type": "tool_input",
                                                "name": tool_call["name"],
                                                "args": tool_call["args"]
                                            })

                                            write_audit_log(
                                                session_id=st.session_state.thread_id,
                                                node_name="reasoner",
                                                tool_name=tool_call["name"],
                                                content=json.dumps(
                                                    tool_call["args"]
                                                )
                                            )


                                    # Intercept final generation
                                    if latest_msg.content:

                                        final_response = latest_msg.content

                                        status.update(
                                            label="📝 Generating Operational Resolution Report..."
                                        )

                                        write_audit_log(
                                            session_id=st.session_state.thread_id,
                                            node_name="reasoner_final",
                                            tool_name="LLM Text Synthesis",
                                            content=final_response
                                        )


                                # ----------------------------------
                                # B. TOOL EXECUTION NODE
                                # ----------------------------------

                                elif node_name == "tools":

                                    status.update(
                                        label="🔧 Executing Enterprise Subsystem Tools..."
                                    )

                                    for msg in node_state.get("messages", []):

                                        if isinstance(msg, ToolMessage):

                                            with st.expander(
                                                f"📤 View Raw Output ({msg.name})",
                                                expanded=False
                                            ):

                                                st.code(
                                                    msg.content,
                                                    language="text"
                                                )

                                            current_traces.append({
                                                "type": "tool_output",
                                                "name": msg.name,
                                                "content": msg.content
                                            })

                                            write_audit_log(
                                                session_id=st.session_state.thread_id,
                                                node_name="tools",
                                                tool_name=msg.name,
                                                content=msg.content
                                            )


                        status.update(
                            label="Incident Matrix Evaluation Complete",
                            state="complete",
                            expanded=False
                        )


                # ==========================================
                # DISPLAY AND SAVE FINAL RESPONSE
                # ==========================================

                if final_response:

                    st.markdown(final_response)

                    st.session_state.ui_messages.append({
                        "role": "assistant",
                        "content": final_response,
                        "traces": current_traces
                    })

                else:

                    error_fallback = (
                        "⚠️ Execution Timeout: System engine "
                        "encountered an unresolved processing edge case."
                    )

                    st.error(error_fallback)


            except Exception as e:

                error_message = (
                    f"⚠️ Agent execution failed: {str(e)}"
                )

                st.error(error_message)

                st.session_state.ui_messages.append({
                    "role": "assistant",
                    "content": error_message,
                    "traces": current_traces
                })


# ==========================================
# 7. SECURITY & AUDIT LOGS
# ==========================================

elif app_mode == "🛡️ Security & Audit Logs":

    st.title("🛡️ Enterprise Agent Audit Trail")

    st.caption(
        "Secure database inspection of FDE_VIEWS.AgentAuditLog"
    )

    st.markdown("### Database Authorization Gate")

    st.markdown(
        "Enter high-privilege administrative credentials "
        "(defined in `.env` as `SQL_ADMIN_USER`) to query audit logs."
    )

    with st.form("admin_auth_form"):

        col1, col2 = st.columns(2)

        with col1:

            input_user = st.text_input(
                "Admin Username",
                value=os.getenv("SQL_ADMIN_USER", "")
            )

        with col2:

            input_pass = st.text_input(
                "Admin Password",
                type="password",
                value=""
            )

        submit_admin = st.form_submit_button(
            "Authenticate & Load Logs",
            use_container_width=True
        )


    if submit_admin:

        expected_admin_user = os.getenv("SQL_ADMIN_USER")
        expected_admin_pass = os.getenv("SQL_ADMIN_PASSWORD")

        if (
            expected_admin_user
            and expected_admin_pass
            and input_user == expected_admin_user
            and input_pass == expected_admin_pass
        ):

            admin_engine = None

            try:

                # Build isolated administrator connection
                admin_params = urllib.parse.quote_plus(
                    "DRIVER={ODBC Driver 18 for SQL Server};"
                    f"SERVER={db_host},{db_port};"
                    "DATABASE=master;"
                    f"UID={input_user};"
                    f"PWD={input_pass};"
                    "Encrypt=no;"
                    "TrustServerCertificate=yes;"
                )

                admin_engine = create_engine(
                    f"mssql+pyodbc:///?odbc_connect={admin_params}"
                )

                with admin_engine.connect() as conn:

                    query = """
                        SELECT
                            LogID,
                            Timestamp,
                            SessionID,
                            NodeExecuted,
                            ToolName,
                            Content
                        FROM FDE_VIEWS.AgentAuditLog
                        ORDER BY Timestamp DESC
                    """

                    df = pd.read_sql(query, conn)


                st.success(
                    "✅ Authenticated successfully as Admin."
                )

                if not df.empty:

                    st.dataframe(
                        df,
                        column_config={
                            "LogID": st.column_config.NumberColumn(
                                "ID",
                                format="%d"
                            ),

                            "Timestamp": st.column_config.DatetimeColumn(
                                "Execution Time",
                                format="DD/MM/YYYY-h:mm a"
                            ),

                            "SessionID": "Session Token",
                            "NodeExecuted": "Graph Node",
                            "ToolName": "Tool Triggered",
                            "Content": "Raw Payload Data"
                        },
                        hide_index=True,
                        use_container_width=True,
                        height=600
                    )

                else:

                    st.info(
                        "No audit logs found in the database. "
                        "Run a query in the Dispatch Console first."
                    )

            except Exception as e:

                st.error(
                    f"Database Query Failed: {e}"
                )

            finally:

                if admin_engine is not None:
                    admin_engine.dispose()

        else:

            st.error(
                "❌ Invalid Administrator Credentials."
            )
