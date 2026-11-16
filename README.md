# Magic Data Assistant 📊

An enterprise-ready, autonomous data analysis assistant powered by **FastAPI**, **LangGraph**, **Google Gemini**, and **PostgreSQL (Neon)**.

Magic Data Assistant ingests multi-sheet Excel workbooks and CSV files directly into PostgreSQL with high-performance `COPY` streaming, inspects database schemas dynamically, and answers natural language analytical queries via an autonomous agentic reasoning graph with built-in self-correction.

---

## 🚀 Key Features

- **Multi-File & Multi-Sheet Ingestion**: Uploads CSV and `.xlsx` workbooks directly into relational PostgreSQL tables with automatic type inference, column sanitization, and currency parsing.
- **Autonomous LangGraph Agent**:
  - **Router Agent**: Dynamically classifies intent (`sql_path`, `analytical_path`, or `general_path`).
  - **SQL Generator**: Generates syntactically correct PostgreSQL queries matching active schemas.
  - **Self-Correction Loop**: Catches SQL execution errors and automatically refines queries up to 2 retry attempts.
  - **Security Guardrails**: Enforces read-only querying (`SELECT`, `WITH`) and strictly blocks destructive operations (`DROP`, `DELETE`, `TRUNCATE`, `ALTER`, etc.).
  - **Summarizer**: Explains query results cleanly in Markdown with bold highlights, bullets, and tables.
- **Multi-Turn Conversation Memory**: Tracks contextual history across dialogue turns.
- **Modern Responsive Web UI**: Drag-and-drop file ingestion, live active tables overview, Markdown rendering via `marked.js`, and real-time query responses.
- **Production Observability**: Built-in `/health` check verifying database connectivity, active tables, and LLM configuration.

---

## 🏗️ Architecture

```
                  ┌───────────────────────────────┐
                  │   Client / Browser Frontend   │
                  │ (HTML5 / Tailwind / MarkedJS) │
                  └───────────────┬───────────────┘
                                  │ HTTP REST
                                  ▼
                    ┌───────────────────────────┐
                    │     FastAPI Application   │
                    └─────────────┬─────────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
┌──────────────────┐                            ┌───────────────────┐
│ /upload-file/    │                            │ /chat/            │
│ Ingest CSV/Excel │                            │ LangGraph Agent   │
└─────────┬────────┘                            └─────────┬─────────┘
          │ (COPY Streaming)                              │
          ▼                                               ▼
┌──────────────────────────────┐              ┌─────────────────────────┐
│ PostgreSQL / Neon Database   │ ◄─────────── │ Router -> SQL -> Run -> │
│ (Relational Data & Schemas)  │   Execute    │ Summarizer / Self-Fix   │
└──────────────────────────────┘              └─────────────────────────┘
```

---

## 🛠️ Prerequisites

- Python 3.10+
- PostgreSQL or [Neon](https://neon.tech/) Serverless Database
- [Google AI Studio API Key](https://aistudio.google.com/)

---

## ⚙️ Installation & Setup

1. **Clone or navigate to the repository**:
   ```bash
   git clone <repo-url>
   cd data_assistant
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and fill in your credentials:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   DATABASE_URL=postgresql://user:password@host/dbname?sslmode=require
   GEMINI_MODEL=gemini-3.8-flash
   HOST=0.0.0.0
   PORT=8000
   ```

---

## 🏃 Running the Application

### Development Server
```bash
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

### Production Server
For production environments, run with multiple worker processes:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI frontend |
| `GET` | `/health` | Production health check (database connection & LLM status) |
| `GET` | `/tables/` | List all active tables currently stored in the database |
| `POST` | `/upload-file/` | Upload `.csv` or `.xlsx` file for automatic database ingestion |
| `POST` | `/chat/` | Ask questions with conversation history and table context |

### Example Chat Request:
```json
POST /chat/
Content-Type: application/json

{
  "question": "What is the total revenue by department?",
  "history": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hello! How can I help you analyze your data?"}
  ],
  "table_names": ["employees", "salaries"]
}
```

---

## 🧪 Testing

Run the test suite:

### 1. Offline Unit Tests (Runs without external server or keys)
```bash
python -m unittest tests/test_api.py
```
Tests:
- Currency detection and data cleaning preserving regular text commas.
- SQL security guardrails blocking destructive commands.
- Pandas to PostgreSQL datatype mapping.

### 2. Live Integration Tests (Against running server)
```bash
python tests/test_api.py --live
```
Tests:
- `/health` endpoint verification.
- File upload (CSV and XLSX).
- Multi-table queries and table lifecycle.

---

## 📁 Repository Structure

```
data_assistant/
├── main.py              # FastAPI server, LangGraph workflow, DB ingestion & agent
├── index.html           # Client interface with Markdown rendering & chat history
├── requirements.txt     # Clean, pinned production dependencies
├── .env.example         # Environment template for deployment
├── .gitignore           # Production ignore rules
├── README.md            # Comprehensive documentation
├── samples/             # Sample datasets for quick testing
│   └── Walmart_Sales.csv
└── tests/               # Test suites
    └── test_api.py      # Unit tests & live integration tests
```

---

## 🔒 Security & Best Practices

- **Strict Read-Only SQL**: Direct query execution disallows keywords such as `DROP`, `DELETE`, `TRUNCATE`, `ALTER`, `UPDATE`, `INSERT`, preventing prompt injection data modifications.
- **SQL Injection Prevention**: Double quotes applied to column and table names, sanitized against non-alphanumeric characters.
- **Secrets Management**: Secrets loaded exclusively via `.env`, prevented from being checked into version control.
