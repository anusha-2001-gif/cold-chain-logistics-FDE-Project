
import os
import urllib.parse
import requests
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


# ==========================================
# 1. ENVIRONMENT & DYNAMIC INDEX ATTACHMENT
# ==========================================

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent

load_dotenv(dotenv_path=project_root / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

def get_cached_huggingface_embeddings(model_name: str):
    """
    Loads and caches the HuggingFace embedding model.
    """
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _load_model(name: str):
        print(
            f"MEMORY SEED: Caching local model [{name}]..."
        )

        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=name,
            model_kwargs={"device": "cpu"}
        )

    return _load_model(model_name)


EMBEDDINGS_MODEL_SETTING = os.getenv(
    "Embeddings_model", "LOCAL"
).strip().upper()

# SQL Server configuration
db_host = os.getenv("SQL_SERVER_HOST", "127.0.0.1")
db_port = os.getenv("SQL_SERVER_PORT", "1433")
db_user = os.getenv("SQL_AGENT_USER", "USR_FDE_RO")
db_password = os.getenv("SQL_AGENT_PASSWORD")


if not PINECONE_API_KEY:
    raise ValueError(
        "CRITICAL: Ensure PINECONE_API_KEY is present in your active .env profile."
    )


# Select embedding model
if EMBEDDINGS_MODEL_SETTING == "OPENAI":

    print("Mode: Connecting to Cloud OpenAI Index (1536 Dim Space)...")

    embeddings = OpenAIEmbeddings()
    INDEX_NAME = "fde-sop-index-openai"

else:

    local_model_target = os.getenv(
        "Local_Embedding_Model", "BAAI/bge-m3"
    ).strip()

    print(
        f"Mode: Connecting to Local Fallback "
        f"[{local_model_target}] Index (1024 Dim Space)..."
    )

    try:
        import streamlit as st

        if st.runtime.exists():
            embeddings = get_cached_huggingface_embeddings(
                local_model_target
            )
        else:
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=local_model_target,
                model_kwargs={"device": "cpu"}
            )

    except ImportError:

        from langchain_huggingface import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=local_model_target,
            model_kwargs={"device": "cpu"}
        )

    INDEX_NAME = "fde-sop-index-local"


# Connect to Pinecone
vector_store = PineconeVectorStore(
    index_name=INDEX_NAME,
    embedding=embeddings
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# ==========================================
# 2. CORE FDE AGENT TOOLS
# ==========================================

@tool
def query_telemetry_db(sql_query: str) -> str:
    """
    Executes a SQL SELECT query against the
    FDE_VIEWS.VW_ACTIVE_FLEET view.

    Available columns:
    Timestamp, Latitude, Longitude, Current_Temperature_C,
    Cargo_Condition_Code, Risk_Classification,
    Delay_Probability, Port_Congestion_Level,
    Route_Risk_Index.
    """

    engine = None

    try:

        # Build SQL Server connection string
        connection_string = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={db_host},{db_port};"
            f"DATABASE=master;"
            f"UID={db_user};"
            f"PWD={db_password};"
            f"Encrypt=no;"
            f"TrustServerCertificate=yes;"
        )

        params = urllib.parse.quote_plus(connection_string)

        engine = create_engine(
            f"mssql+pyodbc:///?odbc_connect={params}"
        )

        # Allow only SELECT queries
        if not sql_query.strip().upper().startswith("SELECT"):
            return (
                "SECURITY BLOCK: Only SELECT operations "
                "are authorized on this view."
            )

        # Execute query
        with engine.connect() as conn:

            cursor = conn.execute(text(sql_query))

            columns = list(cursor.keys())
            rows = cursor.fetchmany(10)

            if not rows:
                return "No records matched the query criteria."

            formatted_output = (
                f"COLUMNS: {', '.join(columns)}\n"
            )

            for row in rows:
                formatted_output += str(tuple(row)) + "\n"

            return formatted_output

    except Exception as e:

        return f"Database Error: {str(e)}"

    finally:

        if engine is not None:
            engine.dispose()


@tool
def get_temperature_breaches() -> str:
    """
    Retrieves the 10 highest temperature-breach records
    from FDE_VIEWS.VW_ACTIVE_FLEET.
    A breach occurs only when temperature is greater than 4.0°C.
    Results are sorted from highest to lowest temperature.
    """

    engine = None

    try:
        connection_string = (
            f"DRIVER={{ODBC Driver 18 for SQL Server}};"
            f"SERVER={db_host},{db_port};"
            f"DATABASE=master;"
            f"UID={db_user};"
            f"PWD={db_password};"
            f"Encrypt=no;"
            f"TrustServerCertificate=yes;"
        )

        params = urllib.parse.quote_plus(connection_string)

        engine = create_engine(
            f"mssql+pyodbc:///?odbc_connect={params}"
        )

        sql_query = """
            SELECT TOP 10
                Timestamp,
                Latitude,
                Longitude,
                Current_Temperature_C,
                Risk_Classification,
                Delay_Probability,
                Port_Congestion_Level
            FROM FDE_VIEWS.VW_ACTIVE_FLEET
            WHERE Current_Temperature_C > 4.0
            ORDER BY Current_Temperature_C DESC;
        """

        with engine.connect() as conn:
            result = conn.execute(text(sql_query))
            rows = result.fetchall()

            if not rows:
                return "No temperature breaches above 4.0°C were found."

            output = [
                "VERIFIED DATABASE RESULTS",
                "Condition: Current_Temperature_C > 4.0°C",
                f"Number of records returned: {len(rows)}",
                "Sorted by temperature: Highest to lowest",
                ""
            ]

            for row in rows:
                output.append(str(dict(row._mapping)))

            return "\n".join(output)

    except Exception as e:
        return f"Temperature breach database error: {str(e)}"

    finally:
        if engine is not None:
            engine.dispose()

@tool
def fetch_corridor_conditions(
    latitude: float,
    longitude: float
) -> str:
    """
    Fetches real-time weather and corridor conditions
    from a live REST API for given GPS coordinates.

    Provides temperature, wind speed, and computed
    corridor congestion index.
    """

    try:

        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            f"&current_weather=true"
        )

        response = requests.get(url, timeout=6)
        response.raise_for_status()

        payload = response.json().get(
            "current_weather", {}
        )

        temp = payload.get("temperature", "N/A")
        wind = payload.get("windspeed", 0.0)

        congestion_index = (
            8.5 if wind > 10.0 else 2.5
        )

        status_note = (
            "High Transit Disruption"
            if wind > 10.0
            else "Corridor Normal"
        )

        return (
            f"--- LIVE CORRIDOR TELEMETRY ---\n"
            f"Target GPS: {latitude}, {longitude}\n"
            f"External Temp: {temp}°C | "
            f"Wind Speed: {wind} km/h\n"
            f"Corridor Risk: {status_note} "
            f"(Congestion Index: {congestion_index}/10)\n"
            f"-------------------------------"
        )

    except Exception as e:

        return (
            f"Corridor API Communication Failure: {str(e)}"
        )


@tool
def search_compliance_sop(query: str) -> str:
    """
    Searches enterprise Standard Operating Procedures
    indexed in the Pinecone Vector DB.

    Retrieves regulatory thresholds, cold-chain breach
    mitigations, and rerouting rules.
    """

    try:

        matched_docs = retriever.invoke(query)

        if not matched_docs:
            return "No matching compliance clauses found."

        formatted_context = "\n\n".join(
            [
                (
                    f"[Source: "
                    f"{doc.metadata.get('source_file', 'SOP')} "
                    f"| Format: "
                    f"{doc.metadata.get('file_format', 'RAW')}]\n"
                    f"{doc.page_content}"
                )
                for doc in matched_docs
            ]
        )

        return (
            f"--- COMPLIANCE SOP CONTEXT ---\n"
            f"{formatted_context}\n"
            f"------------------------------"
        )

    except Exception as e:

        return f"Vector Store Retrieval Error: {str(e)}"


# ==========================================
# 3. LOCAL VERIFICATION
# ==========================================

if __name__ == "__main__":

    print("\n--- Testing Tool 1: SQL Telemetry View ---")

    print(
        query_telemetry_db.invoke(
            "SELECT TOP 2 Latitude, Longitude, "
            "Current_Temperature_C "
            "FROM FDE_VIEWS.VW_ACTIVE_FLEET"
        )
    )

    print("\n--- Testing Tool 2: Live Corridor API ---")

    print(
        fetch_corridor_conditions.invoke({
            "latitude": 33.77,
            "longitude": -118.19
        })
    )

    print("\n--- Testing Tool 3: Pinecone Vector Retrieval ---")

    print(
        search_compliance_sop.invoke(
            "What are the temperature rules for fresh perishables?"
        )
    )
