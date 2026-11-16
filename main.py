# main.py
"""
Magic Data Assistant - Autonomous LangGraph Analyst API
Production-ready backend for intelligent data querying and analysis.
"""

import os
import io
import re
import json
import logging
from typing import Any, List, Dict, Optional, Tuple

import pandas as pd
import psycopg
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

# --- Langchain & LangGraph Imports ---
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv(override=True)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("data_assistant")

def get_db_url() -> str:
    """Dynamically retrieves the latest database connection string from environment."""
    load_dotenv(override=True)
    return (os.getenv("DATABASE_URL") or os.getenv("NEON_CONNECTION_STRING") or "").strip()

def get_google_api_key() -> str:
    """Dynamically retrieves the latest Google API key from environment."""
    load_dotenv(override=True)
    return os.getenv("GOOGLE_API_KEY", "").strip()

def get_gemini_model() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()

def extract_text_content(content: Any) -> str:
    """Extracts plain text string from LLM message content (handles str, list of blocks/dicts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(str(part["text"]))
            elif hasattr(part, "text"):
                text_parts.append(str(part.text))
        return "\n".join(text_parts)
    return str(content) if content is not None else ""

# --- Security: SQL Safeguards ---
FORBIDDEN_SQL_KEYWORDS = {
    "DROP", "DELETE", "TRUNCATE", "ALTER", "UPDATE", "INSERT",
    "GRANT", "REVOKE", "CREATE", "REPLACE"
}


def is_query_safe(sql_query: str) -> Tuple[bool, List[str]]:
    """Checks whether the SQL query contains any destructive operations."""
    tokens = set(re.findall(r'\b[A-Za-z]+\b', sql_query.upper()))
    violations = list(tokens.intersection(FORBIDDEN_SQL_KEYWORDS))
    return (len(violations) == 0, violations)


# --- Helper Functions ---
def pandas_to_sql_type(dtype) -> str:
    """Maps pandas Series data type to PostgreSQL column data type."""
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    elif pd.api.types.is_float_dtype(dtype):
        return "DOUBLE PRECISION"
    elif pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    else:
        return "TEXT"


def clean_dataframe(df: pd.DataFrame, table_name: str = "") -> pd.DataFrame:
    """
    Cleans numeric currency columns (e.g. '$90,000', '₹50,000') while
    strictly preserving regular text columns that happen to contain commas
    (e.g. 'San Francisco, CA').
    """
    cleaned = df.copy()
    for col in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[col]) or cleaned[col].dtype == 'object':
            try:
                s = cleaned[col].dropna().astype(str).str.strip()
                if len(s) == 0:
                    continue
                currency_pattern = r'^[\$₹€£]\s*[\d,]+(?:\.\d+)?$'
                matches = s.str.match(currency_pattern)
                # If over 50% match currency format, clean and convert to numeric
                if (matches.sum() / len(s)) > 0.5:
                    cleaned_col = cleaned[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                    cleaned[col] = pd.to_numeric(cleaned_col, errors='coerce')
            except Exception as e:
                logger.warning(f"Could not clean column {col} in table {table_name}: {e}")
    return cleaned


def process_dataframe_to_db(df: pd.DataFrame, table_name: str, db_url: str) -> str:
    """
    Sanitizes DataFrame schema and inserts all rows into PostgreSQL using high-performance COPY.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"Empty or invalid DataFrame for table '{table_name}'.")
    if not db_url:
        raise ValueError("Database connection URL is required.")

    # Sanitize table name
    sanitized_table = re.sub(r'[^a-zA-Z0-9_]+', '_', table_name).lower().strip('_')
    if not sanitized_table:
        sanitized_table = f"dataset_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

    logger.info(f"Processing table: '{sanitized_table}'")
    cleaned_df = clean_dataframe(df, sanitized_table)

    # Sanitize column names and map SQL types
    columns_with_types = []
    valid_columns_for_copy = []
    final_col_mapping = {}

    for col_orig in cleaned_df.columns:
        col_str = str(col_orig).strip()
        sanitized_col = re.sub(r'[^a-zA-Z0-9_]+', '_', col_str).lower().strip('_')
        if not sanitized_col:
            sanitized_col = f"col_{len(final_col_mapping) + 1}"

        suffix = 1
        final_col = sanitized_col
        while final_col in final_col_mapping.values():
            final_col = f"{sanitized_col}_{suffix}"
            suffix += 1
        final_col_mapping[col_orig] = final_col

        sql_type = pandas_to_sql_type(cleaned_df[col_orig].dtype)
        columns_with_types.append(f'"{final_col}" {sql_type}')
        valid_columns_for_copy.append(f'"{final_col}"')

    if not columns_with_types:
        raise ValueError(f"No valid columns found for table '{sanitized_table}'.")

    # Rename DataFrame columns to match sanitized SQL column names
    cleaned_df.columns = [final_col_mapping[orig] for orig in df.columns]

    # Execute table recreation and fast COPY in PostgreSQL
    with psycopg.connect(db_url, autocommit=True, connect_timeout=15) as conn:
        with conn.cursor() as cur:
            create_query = (
                f'DROP TABLE IF EXISTS "{sanitized_table}";\n'
                f'CREATE TABLE "{sanitized_table}" (id SERIAL PRIMARY KEY, {", ".join(columns_with_types)});'
            )
            cur.execute(create_query)

            # High-performance CSV COPY streaming
            output = io.StringIO()
            cleaned_df.to_csv(output, sep=',', header=False, index=False, na_rep='\\N')
            output.seek(0)

            copy_command = f'COPY "{sanitized_table}" ({", ".join(valid_columns_for_copy)}) FROM STDIN WITH (FORMAT CSV, NULL \'\\N\');'
            with cur.copy(copy_command) as copy:
                while True:
                    chunk = output.read(16384)
                    if not chunk:
                        break
                    copy.write(chunk)

    logger.info(f"Successfully processed and stored table '{sanitized_table}' ({len(cleaned_df)} rows).")
    return sanitized_table


# --- Initialize FastAPI App ---
app = FastAPI(
    title="Magic Data Assistant API",
    description="Autonomous LangGraph Analyst Agent for multi-table data analysis.",
    version="3.0.0"
)

# CORS setup
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class HistoryMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str


class ChatRequest(BaseModel):
    question: str
    history: Optional[List[HistoryMessage]] = None
    table_names: Optional[List[str]] = None


# --- LangGraph Agent Implementation ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    db_url: str
    combined_schema: str
    route: str


@tool
def run_sql_query(sql_query: str = Field(description="The PostgreSQL read-only query to be executed.")):
    """A tool to execute a read-only SQL query against the database."""
    pass


def get_llm_instances():
    """Lazily initializes and validates LLM instances."""
    api_key = get_google_api_key()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not configured.")
    model = get_gemini_model()
    llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
    llm_router = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.0)
    llm_sql = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.0)
    llm_sql_tools = llm_sql.bind_tools([run_sql_query])
    llm_general = ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.7)
    return llm, llm_router, llm_sql_tools, llm_general


# --- Graph Nodes ---
def router_agent(state: AgentState):
    """Decides the execution path: SQL, Analytical, or General Chat based on user query and schema."""
    logger.info("Routing query...")
    last_user_question = ""
    if state["messages"] and isinstance(state["messages"][-1], HumanMessage):
        last_user_question = state["messages"][-1].content
    else:
        return {"route": "general_path"}

    previous_ai_message = ""
    if len(state["messages"]) > 1:
        for msg in reversed(state["messages"][:-1]):
            if isinstance(msg, AIMessage):
                previous_ai_message = msg.content
                break

    router_prompt = f"""
    Analyze the user's LATEST question and the database schema to determine the best path.
    Consider the previous assistant message for context if the user's reply is brief.

    Paths:
    1. `sql_path`: If the question requires specific calculations, aggregation, filtering, or data retrieval via SQL.
    2. `analytical_path`: If the question is abstract or conceptual *about the data* shown in the schema (e.g. what analyses can be done, schema design, dataset purpose).
    3. `general_path`: For greetings, conversational replies ("yes", "thanks"), or questions unrelated to the database schemas.

    Schema:
    {state['combined_schema']}

    Previous Assistant Message: "{previous_ai_message}"
    Latest User Question: "{last_user_question}"

    Respond with exactly one word: `sql_path`, `analytical_path`, or `general_path`.
    """
    _, llm_router, _, _ = get_llm_instances()
    response = llm_router.invoke([HumanMessage(content=router_prompt)])
    decision = extract_text_content(response.content).strip().lower()

    if "sql_path" in decision:
        route = "sql_path"
    elif "analytical_path" in decision:
        route = "analytical_path"
    else:
        route = "general_path"

    logger.info(f"Router Decision: {route}")
    return {"route": route}


def sql_generation_agent(state: AgentState):
    """Generates the PostgreSQL query based on history, schemas, and user question."""
    logger.info("Generating SQL query...")
    last_user_question = state["messages"][-1].content

    system_message = f"""
    You are an expert PostgreSQL data analyst. Generate a valid PostgreSQL query to answer the user request, then call the `run_sql_query` tool.

    **Available Tables & Schemas:**
    {state['combined_schema']}

    **CRITICAL RULES:**
    - Enclose ALL table and column identifiers in double quotes (e.g. "employees", "salary").
    - ONLY generate read-only SELECT queries or CTEs (WITH ... SELECT). Never generate DROP, DELETE, INSERT, or UPDATE.
    - If calculating on possible string columns, clean in query: CAST(REPLACE("col", '$', '') AS NUMERIC).
    - If joining tables, match foreign keys logically.
    - If a previous SQL attempt failed (see history), analyze the error and fix the query.
    """

    messages_for_llm = [SystemMessage(content=system_message)]
    history_to_include = state["messages"][-7:-1]
    messages_for_llm.extend(history_to_include)
    messages_for_llm.append(HumanMessage(content=f"Based on history and schemas, generate SQL for: \"{last_user_question}\" and call run_sql_query."))

    _, _, llm_sql_tools, _ = get_llm_instances()
    response = llm_sql_tools.invoke(messages_for_llm)
    return {"messages": state["messages"] + [response]}


def execute_sql(state: AgentState):
    """Executes the generated SQL query against PostgreSQL with safety validation."""
    logger.info("Executing SQL...")
    if not state.get("messages"):
        return {"messages": [HumanMessage(content="Internal Error: State lost before SQL execution.")]}

    last_message = state["messages"][-1]
    tool_calls = getattr(last_message, 'tool_calls', [])

    if not tool_calls or not isinstance(tool_calls, list) or len(tool_calls) == 0:
        content = getattr(last_message, 'content', 'No content')
        logger.warning(f"Agent did not generate a tool call: {content}")
        tool_call_id = getattr(last_message, 'id', 'missing_tool_call_id')
        error_content = "SQL Agent did not produce a tool call. If answering directly, please present the answer."
        return {"messages": state["messages"] + [ToolMessage(content=error_content, tool_call_id=tool_call_id)]}

    tool_call = tool_calls[0]
    sql_query = tool_call.get("args", {}).get("sql_query", "").strip()
    tool_call_id = tool_call.get("id", "tool_call_id")

    if not sql_query:
        return {"messages": state["messages"] + [ToolMessage(content="Error: Empty SQL query provided.", tool_call_id=tool_call_id)]}

    # Security verification
    safe, violations = is_query_safe(sql_query)
    if not safe:
        err = f"Security Violation: Query contains prohibited operation ({', '.join(violations)}). Only read-only queries are permitted."
        logger.warning(err)
        return {"messages": state["messages"] + [ToolMessage(content=err, tool_call_id=tool_call_id)]}

    db_url = state["db_url"]
    logger.info(f"Executing query: {sql_query}")

    try:
        with psycopg.connect(db_url, autocommit=True, connect_timeout=10) as conn:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                if cur.description:
                    columns = [desc[0] for desc in cur.description]
                    rows = cur.fetchmany(100)
                    results = [dict(zip(columns, row)) for row in rows]
                    tool_result = json.dumps(results, default=str)
                    if cur.fetchone() is not None:
                        tool_result += "\n... (truncated: additional rows omitted)"
                else:
                    tool_result = f"Query executed successfully. {cur.rowcount} rows affected."
        logger.info("SQL execution successful.")
        return {"messages": state["messages"] + [ToolMessage(content=tool_result, tool_call_id=tool_call_id)]}
    except Exception as e:
        error_message = f"Error executing SQL: {e}. Query: {sql_query}"
        logger.warning(f"SQL Execution Failed: {error_message}")
        return {"messages": state["messages"] + [ToolMessage(content=error_message, tool_call_id=tool_call_id)]}


def summarize_result(state: AgentState):
    """Summarizes the SQL query result or error context clearly in Markdown."""
    logger.info("Summarizing results...")
    if not state.get("messages"):
        return {"messages": [HumanMessage(content="Error: State lost before summarization.")]}

    system_message = (
        "You are a helpful and precise data analyst. Summarize the SQL query result clearly for the user. "
        "Format numeric amounts and key statistics neatly using Markdown formatting (bold, bullet points, or tables). "
        "If an error occurred, explain what went wrong and how the user might rephrase or clarify their question."
    )
    messages_for_llm = [SystemMessage(content=system_message)]
    history_to_include = state["messages"][-7:]
    messages_for_llm.extend(history_to_include)

    llm, _, _, _ = get_llm_instances()
    response = llm.invoke(messages_for_llm)
    return {"messages": state["messages"] + [response]}


def analytical_agent(state: AgentState):
    """Provides high-level analytical guidance without executing queries."""
    logger.info("Running analytical agent...")
    system_message = f"""
    You are an expert data analyst. Based on the database schemas and conversation history, provide a high-level, logical answer to the user's question.
    Do not generate SQL code. Offer insights, suggested analyses, or data relationships.
    Schemas:
    {state['combined_schema']}
    """
    messages_for_llm = [SystemMessage(content=system_message)]
    history_to_include = state["messages"][-7:]
    messages_for_llm.extend(history_to_include)

    llm, _, _, _ = get_llm_instances()
    response = llm.invoke(messages_for_llm)
    return {"messages": state["messages"] + [response]}


def general_chat_agent(state: AgentState):
    """Handles polite conversational interaction and guides the user toward data analysis."""
    logger.info("Running general chat agent...")
    latest_user_message = ""
    previous_ai_message = ""
    if state["messages"]:
        latest_user_message = extract_text_content(state["messages"][-1].content)
        for msg in reversed(state["messages"][:-1]):
            if isinstance(msg, AIMessage):
                previous_ai_message = extract_text_content(msg.content)
                break

    general_prompt = f"""
    You are Magic Data Assistant.
    Your previous message: "{previous_ai_message}"
    User latest message: "{latest_user_message}"
    Available tables schema summary: {state.get('combined_schema', 'No schema loaded')}

    Respond politely and concisely. If the user is responding with conversational acknowledgments,
    acknowledge warmly and remind them what data or questions you can help analyze.
    """
    messages_for_llm = [
        SystemMessage(content="You are Magic Data Assistant, a helpful AI data analysis companion."),
        HumanMessage(content=general_prompt)
    ]
    _, _, _, llm_general = get_llm_instances()
    response = llm_general.invoke(messages_for_llm)
    return {"messages": state["messages"] + [response]}


# --- Self-correction loop logic ---
MAX_RETRIES = 2


def check_for_sql_error(state: AgentState):
    """Checks SQL execution result and loops back to sql_generator on error up to MAX_RETRIES."""
    if not state.get("messages"):
        return "summarize"
    last_message = state["messages"][-1]

    execution_error_attempts = 0
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage) and "Error executing SQL" in msg.content:
            execution_error_attempts += 1
        elif isinstance(msg, ToolMessage) and "Error executing SQL" not in msg.content:
            break
        elif isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
            break

    if isinstance(last_message, ToolMessage) and ("Error executing SQL" in last_message.content or "SQL Agent Error" in last_message.content):
        logger.warning(f"SQL Error Detected (Attempt {execution_error_attempts})")
        if execution_error_attempts >= MAX_RETRIES or "Security Violation" in last_message.content:
            logger.info("Max retries reached or security violation. Proceeding to summarizer.")
            return "summarize"
        else:
            logger.info("Retrying query generation...")
            return "retry"
    else:
        return "summarize"


def route_logic(state: AgentState):
    return state.get("route", "general_path")


# --- Construct and Compile the LangGraph Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_agent)
workflow.add_node("sql_generator", sql_generation_agent)
workflow.add_node("executor", execute_sql)
workflow.add_node("summarizer", summarize_result)
workflow.add_node("analyst", analytical_agent)
workflow.add_node("general_chat", general_chat_agent)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    route_logic,
    {
        "sql_path": "sql_generator",
        "analytical_path": "analyst",
        "general_path": "general_chat",
    }
)
workflow.add_edge("sql_generator", "executor")
workflow.add_conditional_edges(
    "executor",
    check_for_sql_error,
    {"retry": "sql_generator", "summarize": "summarizer"}
)
workflow.add_edge("summarizer", END)
workflow.add_edge("analyst", END)
workflow.add_edge("general_chat", END)

graph = workflow.compile()


# --- Database Schema Helper ---
def fetch_database_schemas(db_url: str, filter_tables: Optional[List[str]] = None) -> Tuple[str, List[str]]:
    """Retrieves formatted column schemas for active tables in public schema."""
    if not db_url:
        return ("Database connection string not configured.", [])

    all_schemas = []
    found_tables = []
    try:
        with psycopg.connect(db_url, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public' ORDER BY tablename;")
                all_table_names = [row[0] for row in cur.fetchall()]

                target_tables = [t for t in all_table_names if not filter_tables or t in filter_tables]
                found_tables = target_tables

                for table_name in target_tables:
                    try:
                        cur.execute(
                            "SELECT column_name, data_type FROM information_schema.columns "
                            "WHERE table_name = %s AND table_schema = 'public' ORDER BY ordinal_position;",
                            (table_name,)
                        )
                        cols = cur.fetchall()
                        if cols:
                            cols_str = ", ".join([f'"{str(c[0])}" ({str(c[1])})' for c in cols])
                            all_schemas.append(f'Table "{table_name}": {cols_str}')
                    except Exception as err:
                        logger.warning(f"Error fetching schema for {table_name}: {err}")
                        all_schemas.append(f'Table "{table_name}": Error fetching schema.')
        combined = "\n".join(all_schemas) if all_schemas else "No tables found in public schema."
        return (combined, found_tables)
    except Exception as db_err:
        logger.error(f"Database error fetching schemas: {db_err}")
        return (f"Error connecting to database: {db_err}", [])


# --- API Endpoints ---

@app.get("/")
async def read_root():
    """Serves the web client index.html."""
    script_dir = os.path.dirname(__file__)
    index_path = os.path.join(script_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to Magic Data Assistant API. Visit /docs for OpenAPI documentation."}


@app.get("/health")
async def health_check():
    """Production health check verifying database and LLM configuration."""
    db_status = "unconfigured"
    db_error = None
    table_count = 0
    db_url = get_db_url()
    api_key = get_google_api_key()
    model = get_gemini_model()

    if db_url:
        try:
            with psycopg.connect(db_url, connect_timeout=4) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT count(*) FROM pg_catalog.pg_tables WHERE schemaname = 'public';")
                    table_count = cur.fetchone()[0]
            db_status = "connected"
        except Exception as e:
            db_status = "disconnected"
            db_error = str(e)

    llm_status = "configured" if bool(api_key) else "unconfigured"

    is_healthy = db_status == "connected" and llm_status == "configured"
    return JSONResponse(
        status_code=status.HTTP_200_OK if is_healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "status": "healthy" if is_healthy else "degraded",
            "database": {
                "status": db_status,
                "tables_count": table_count,
                "error": db_error,
            },
            "llm": {
                "provider": "google-gemini",
                "model": model,
                "status": llm_status,
            },
            "version": "3.0.0"
        }
    )


@app.get("/tables/")
async def list_tables():
    """Returns the names of all active tables currently stored in the database."""
    db_url = get_db_url()
    if not db_url:
        return {"tables": []}
    try:
        with psycopg.connect(db_url, connect_timeout=4) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT tablename FROM pg_catalog.pg_tables WHERE schemaname = 'public' ORDER BY tablename;")
                tables = [row[0] for row in cur.fetchall()]
                return {"tables": tables}
    except Exception as e:
        logger.warning(f"Could not list tables: {e}")
        return {"tables": [], "warning": f"Database temporarily unavailable: {str(e)}"}


@app.post("/upload-file/")
async def upload_file(file: UploadFile = File(...)):
    """Uploads and ingests a CSV or multi-sheet Excel file into PostgreSQL tables."""
    db_url = get_db_url()
    if not db_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection string (DATABASE_URL / NEON_CONNECTION_STRING) is not configured in .env."
        )
    if not file or not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded.")

    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".csv", ".xlsx"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only .csv and .xlsx files are supported.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

        created_tables = []

        if ext == ".csv":
            try:
                df = pd.read_csv(io.BytesIO(contents), low_memory=False)
                if df.empty:
                    raise ValueError("CSV file contains no data.")
                base_name = os.path.splitext(filename)[0]
                tbl = process_dataframe_to_db(df, base_name, db_url)
                created_tables.append(tbl)
            except Exception as csv_err:
                logger.error(f"CSV ingestion error: {csv_err}")
                raise HTTPException(status_code=400, detail=f"CSV processing error: {str(csv_err)}")

        elif ext == ".xlsx":
            try:
                all_sheets = pd.read_excel(io.BytesIO(contents), sheet_name=None)
                if not all_sheets:
                    raise ValueError("No sheets found in Excel file.")
                for sheet_name, sheet_df in all_sheets.items():
                    if sheet_df.empty:
                        continue
                    tbl = process_dataframe_to_db(sheet_df, str(sheet_name), db_url)
                    created_tables.append(tbl)
                if not created_tables:
                    raise ValueError("No data found in any sheet of the Excel file.")
            except Exception as xlsx_err:
                logger.error(f"XLSX ingestion error: {xlsx_err}")
                raise HTTPException(status_code=400, detail=f"Excel processing error: {str(xlsx_err)}")

        return {
            "message": f"Successfully processed '{filename}'. Created/updated: {', '.join(created_tables)}",
            "created_tables": created_tables
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected upload failure: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Server error processing uploaded file.")


@app.post("/chat/")
async def chat_with_data(request: ChatRequest):
    """Handles multi-turn conversational data querying through the LangGraph agent."""
    api_key = get_google_api_key()
    db_url = get_db_url()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="GOOGLE_API_KEY is not configured in .env."
        )
    if not db_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="DATABASE_URL / NEON_CONNECTION_STRING is not configured in .env."
        )

    # Fetch database schema
    combined_schema, active_tables = fetch_database_schemas(db_url, request.table_names)

    # Convert request history into LangChain messages
    incoming_history = request.history or []
    langchain_messages: List[BaseMessage] = []
    for msg in incoming_history:
        if msg.role == "user":
            langchain_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            langchain_messages.append(AIMessage(content=msg.content))

    langchain_messages.append(HumanMessage(content=request.question))

    initial_state: AgentState = {
        "messages": langchain_messages,
        "db_url": db_url,
        "combined_schema": combined_schema,
        "route": "general_path",
    }

    try:
        final_state = graph.invoke(initial_state, {"recursion_limit": 10})
        final_answer = "I was unable to complete the analysis."
        response_history: List[Dict[str, str]] = []

        if final_state and isinstance(final_state.get("messages"), list) and final_state["messages"]:
            final_ai_message = final_state["messages"][-1]
            if hasattr(final_ai_message, 'content') and final_ai_message.content:
                final_answer = extract_text_content(final_ai_message.content)

            for m in final_state["messages"]:
                if isinstance(m, HumanMessage) and m.content:
                    response_history.append({"role": "user", "content": extract_text_content(m.content)})
                elif isinstance(m, AIMessage) and m.content:
                    response_history.append({"role": "assistant", "content": extract_text_content(m.content)})
        else:
            response_history = [{"role": h.role, "content": h.content} for h in incoming_history]
            response_history.append({"role": "user", "content": request.question})
            response_history.append({"role": "assistant", "content": final_answer})

        return {"answer": final_answer, "history": response_history}

    except Exception as err:
        logger.error(f"Error during agent execution: {err}", exc_info=True)
        fallback_answer = "Sorry, an internal error occurred while processing your query. Please verify your database connection and try again."
        history_out = [{"role": h.role, "content": h.content} for h in incoming_history]
        history_out.append({"role": "user", "content": request.question})
        history_out.append({"role": "assistant", "content": fallback_answer})
        return {"answer": fallback_answer, "history": history_out}


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
