import uvicorn
from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from pydantic import BaseModel
import sqlite3
import pandas as pd
import os
from pathlib import Path
import json
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
import re
import subprocess
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime

# Try to import optional dependencies, with graceful fallback
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("dotenv not installed. Environment variables must be set manually.")

try:
    import google.generativeai as genai
except ImportError:
    print("WARNING: google.generativeai not installed. LLM features will not work.")
    genai = None

try:
    import kagglehub
except ImportError:
    print("WARNING: kagglehub not installed. Kaggle dataset loading will not work.")
    kagglehub = None

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    import pickle
except ImportError:
    print("WARNING: sentence-transformers or faiss not installed. RAG features will not work.")
    SentenceTransformer = None
    faiss = None

# --- Configuration ---
DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "analytics_db.sqlite")
DEFAULT_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
VECTOR_STORE_PATH = os.path.join(DATA_DIR, "schema_vector_store")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Lightweight but effective model

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

class VectorStore:
    def __init__(self, store_path: str, model_name: str = EMBEDDING_MODEL):
        self.store_path = store_path
        self.index_path = os.path.join(store_path, "faiss_index.bin")
        self.metadata_path = os.path.join(store_path, "metadata.pkl")
        self.model = SentenceTransformer(model_name) if SentenceTransformer else None
        self.index = None
        self.metadata = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load()
        else:
            self.initialize()

    def initialize(self):
        """Initialize a new FAISS index."""
        if not self.model:
            raise ImportError("SentenceTransformer not available")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []

    def load(self):
        """Load existing index and metadata."""
        if not self.model:
            raise ImportError("SentenceTransformer not available")
        self.index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)

    def save(self):
        """Save index and metadata to disk."""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)

    def add_schema_entries(self, schema: dict):
        """Add schema entries to the vector store."""
        if not self.model:
            raise ImportError("SentenceTransformer not available")
        
        # Clear existing data
        self.initialize()
        
        # Process schema into text chunks
        entries = []
        for table_name, table_info in schema.items():
            # Create table-level entry
            table_desc = f"Table {table_name}: {table_info.get('description', 'No description')}"
            entries.append(table_desc)
            
            # Create column-level entries
            for column in table_info.get('columns', []):
                col_desc = f"Column {table_name}.{column['name']}: {column.get('description', 'No description')} (Type: {column['type']})"
                entries.append(col_desc)
        
        # Generate embeddings and add to index
        embeddings = self.model.encode(entries)
        self.index.add(np.array(embeddings).astype('float32'))
        self.metadata = entries
        self.save()

    def search(self, query: str, k: int = 5) -> List[str]:
        """Search for relevant schema entries."""
        if not self.model or not self.index:
            raise ImportError("Vector store not properly initialized")
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search the index
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), 
            min(k, len(self.metadata))
        )
        
        # Return relevant entries
        return [self.metadata[i] for i in indices[0]]

def get_relevant_schema_from_rag(question: str, db_schema: dict) -> str:
    """
    Retrieve relevant schema information using RAG.
    If vector store is empty, it will be populated with the current schema.
    """
    if not SentenceTransformer or not faiss:
        print("RAG dependencies not available. Returning full schema.")
        return f"DATABASE SCHEMA:\n{json.dumps(db_schema, indent=2)}"
    
    try:
        # Initialize vector store
        vector_store = VectorStore(VECTOR_STORE_PATH)
        
        # If vector store is empty, populate it
        if len(vector_store.metadata) == 0:
            print("Populating vector store with schema information...")
            vector_store.add_schema_entries(db_schema)
        
        # Search for relevant schema entries
        relevant_entries = vector_store.search(question)
        
        if not relevant_entries:
            return f"DATABASE SCHEMA:\n{json.dumps(db_schema, indent=2)}"
        
        # Format the relevant entries
        schema_context = "RELEVANT SCHEMA INFORMATION:\n"
        schema_context += "\n".join(relevant_entries)
        schema_context += "\n\nFULL SCHEMA:\n"
        schema_context += json.dumps(db_schema, indent=2)
        
        return schema_context
        
    except Exception as e:
        print(f"Error in RAG schema retrieval: {e}")
        return f"DATABASE SCHEMA:\n{json.dumps(db_schema, indent=2)}"

# --- FastAPI App ---
app = FastAPI(title="Enhanced Analytics Bot API")

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    semantic_schema_json: str  # Expecting JSON string from frontend
    model_name: str = "gemini-1.5-flash"
    user_api_key: str | None = None
    use_rag_for_schema: bool = False # For RAG toggle

class SemanticSchemaRequest(BaseModel):
    user_api_key: str | None = None
    model_name: str = "gemini-1.5-flash"

class LoadKaggleRequest(BaseModel):
    dataset_name: str # e.g., "manjeetsingh/retaildataset"

class ExecutePythonRequest(BaseModel):
    script: str
    user_api_key: str | None = None # For consistency if needed by sandbox

class FixCodeRequest(BaseModel):
    code: str
    error: str
    user_api_key: str | None = None
    model_name: str = "gemini-1.5-flash"

# --- Helper Functions ---
def get_db_connection(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def get_db_schema(db_path: str = DB_PATH):
    """Retrieves the schema of the specified SQLite database."""
    if not os.path.exists(db_path):
        return {} # Return empty schema if DB doesn't exist
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    schema = {}
    for table in tables:
        cursor.execute(f"PRAGMA table_info('{table}');")
        columns = []
        for row in cursor.fetchall():
            columns.append({
                "name": row[1],
                "type": row[2],
                "notnull": bool(row[3]),
                "primary_key": bool(row[5])
            })
        schema[table] = columns
    conn.close()
    return schema

def execute_sql_query(sql_query: str, db_path: str = DB_PATH):
    """Executes a SQL query and returns a Pandas DataFrame."""
    if not os.path.exists(db_path) and "FROM" in sql_query.upper(): # Basic check
        raise HTTPException(status_code=400, detail=f"Database at {db_path} not found or not initialized. Load data first.")
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query(sql_query, conn)
        return df
    except Exception as e:
        print(f"SQL Execution Error: {e}")
        raise HTTPException(status_code=400, detail=f"Error executing SQL: {str(e)}")
    finally:
        conn.close()

def configure_gemini(api_key: str | None, model_name: str):
    """Configures and returns a Gemini GenerativeModel."""
    if genai is None:
        raise HTTPException(status_code=500, detail="Gemini API module not installed. Install with 'pip install google-generativeai'")
    
    resolved_api_key = api_key or DEFAULT_GEMINI_API_KEY
    if not resolved_api_key:
        raise HTTPException(status_code=401, detail="Gemini API key not provided or configured on server.")
    
    genai.configure(api_key=resolved_api_key)
    return genai.GenerativeModel(model_name)

def generate_plot(df, chart_type, x_column=None, y_column=None, title=None):
    """Generates a plot and returns a base64 encoded image string."""
    if df.empty:
        return None
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("viridis", 8)
    plt.clf() # Clear any previous plots

    # Auto-detect columns if not provided
    if x_column is None and len(df.columns) > 0:
        x_column = df.columns[0]
    if y_column is None:
        if len(df.columns) > 1:
            y_column = df.columns[1]
        elif len(df.columns) == 1: # Use index for x if only one column for y
            y_column = df.columns[0]
            df = df.reset_index()
            x_column = 'index'
        else: # No columns to plot
            return None

    try:
        if chart_type == 'bar':
            sns.barplot(x=df[x_column], y=df[y_column], data=df, palette=palette, alpha=0.8)
        elif chart_type == 'line':
            sns.lineplot(
                x=df[x_column], 
                y=df[y_column], 
                data=df,
                markers=True,  # Add markers at data points
                dashes=False,  # Solid lines
                marker='o',    # Circle markers
                markersize=8,  # Marker size
                markeredgecolor='white',  # White edge for contrast
                markeredgewidth=1.5,      # Edge width
                linewidth=2.5,            # Line thickness
                color=palette[0]          # Line color
            )
            plt.grid(True, linestyle='--', alpha=0.7)
        elif chart_type == 'scatter':
            size_col = df[y_column] if pd.api.types.is_numeric_dtype(df[y_column]) else None
            
            sns.scatterplot(
                x=df[x_column], 
                y=df[y_column], 
                data=df,
                hue=df.columns[2] if len(df.columns) > 2 else None,  # Use third column for color if available
                size=size_col,  # Dynamic sizing based on y-value
                sizes=(50, 200),  # Min and max point size
                alpha=0.7,  # Transparency
                palette=palette,
                edgecolor='white',  # White edges for contrast
                linewidth=0.5
            )
            if pd.api.types.is_numeric_dtype(df[x_column]) and pd.api.types.is_numeric_dtype(df[y_column]):
                sns.regplot(x=df[x_column], y=df[y_column], scatter=False, line_kws={"color": "red", "alpha": 0.7, "lw": 2, "ls": "--"})
                
        elif chart_type == 'box':
            sns.boxplot(
                x=df[x_column], 
                y=df[y_column], 
                data=df,
                palette=palette,
                width=0.6,  # Box width
                fliersize=5,  # Outlier point size
                linewidth=1.5  # Line width for boxes
            )
            # Add strip plot on top of boxplot for actual data distribution
            sns.stripplot(
                x=df[x_column], 
                y=df[y_column], 
                data=df,
                size=4, 
                color='black', 
                alpha=0.5
            )
        elif chart_type == 'pie' and len(df) <= 10:
            # Ensure y_column is numeric for pie chart values
            if pd.api.types.is_numeric_dtype(df[y_column]):
                 plt.pie(df[y_column], labels=df[x_column], autopct='%1.1f%%', startangle=90)
            else: # Fallback or error if y_column is not numeric
                sns.barplot(x=df[x_column], y=df[y_column], data=df, palette=palette) # Fallback to bar
                chart_type = 'bar (fallback from pie due to non-numeric y-axis)'
        elif chart_type == 'hist':
            sns.histplot(
                df[y_column if y_column in df.columns else x_column],
                kde=True,
                color=palette[0],
                alpha=0.7,
                edgecolor='white',
                linewidth=1
            )
            # Add mean line
            mean_val = df[y_column if y_column in df.columns else x_column].mean()
            plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, 
                        label=f'Mean: {mean_val:.2f}')
            plt.legend()
        else: # Default to bar chart
            ax = sns.barplot(x=df[x_column], y=df[y_column], data=df, palette=palette)
            chart_type = f'bar (defaulted from {chart_type})'

        plt.title(title if title else f"{chart_type.capitalize()} of {y_column} by {x_column}")
        plt.xticks(rotation=45, ha='right')
        plt.xlabel(x_column, fontsize=12, labelpad=10)
        plt.ylabel(y_column, fontsize=12, labelpad=10)
        plt.xticks(rotation=45 if len(df) > 5 else 0, ha='right' if len(df) > 5 else 'center')
        plt.tight_layout()
        plt.figtext(0.9, 0.05, 'Data Insights', fontstyle='italic', alpha=0.5)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return plot_base64
    except Exception as e:
        print(f"Error generating plot: {e}\n{traceback.format_exc()}")
        plt.close()
        return None

# --- Data Loading Functions ---
def _clean_column_name(col_name):
    """Cleans a column name for DB compatibility."""
    return str(col_name).strip().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('.', '_').replace('/', '_')

def _load_df_to_db(df: pd.DataFrame, table_name: str, db_path: str = DB_PATH, if_exists: str = 'replace'):
    """Loads a DataFrame into a specified SQLite table."""
    conn = get_db_connection(db_path)
    cleaned_table_name = _clean_column_name(table_name) # Clean table name as well
    df.columns = [_clean_column_name(col) for col in df.columns]
    try:
        df.to_sql(cleaned_table_name, conn, if_exists=if_exists, index=False)
        print(f"DataFrame loaded into table '{cleaned_table_name}' in {db_path}")
    except Exception as e:
        print(f"Error loading DataFrame to DB table {cleaned_table_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error writing to DB table {cleaned_table_name}: {str(e)}")
    finally:
        conn.close()

# --- API Endpoints ---
@app.post("/load-kaggle-dataset/")
async def load_kaggle_dataset_endpoint(request: LoadKaggleRequest):
    """Downloads a dataset from Kaggle and loads CSVs into the database."""
    if kagglehub is None:
        raise HTTPException(status_code=500, detail="Kagglehub not installed. Install with 'pip install kagglehub'")
        
    dataset_name = request.dataset_name
    target_db_path = DB_PATH
    
    try:
        print(f"Attempting to download Kaggle dataset: {dataset_name} to {DATA_DIR}")
        
        try:
            # Try to download without authentication - works for public datasets
            download_path = kagglehub.dataset_download(dataset_name)
            print(f"Dataset {dataset_name} downloaded to raw path: {download_path}")
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg:
                raise HTTPException(status_code=404, detail=f"Dataset '{dataset_name}' not found. Check the name and ensure it's public.")
            elif "403" in error_msg or "401" in error_msg or "authentication" in error_msg.lower():
                raise HTTPException(status_code=403, detail=f"Authentication error. This dataset may require authentication. Please use a public dataset.")
            else:
                raise HTTPException(status_code=500, detail=f"Error downloading dataset: {error_msg}")

        processed_files = []
        
        potential_csv_files = []
        if os.path.isdir(download_path):
            for root, _, files in os.walk(download_path):
                for file in files:
                    if file.lower().endswith('.csv'):
                        potential_csv_files.append(os.path.join(root, file))
        elif os.path.isfile(download_path) and download_path.lower().endswith('.csv'):
            potential_csv_files.append(download_path)

        # Handle ZIP files if no CSVs found directly
        if not potential_csv_files:
            # Check if it's a zip file
            if os.path.isfile(download_path) and download_path.lower().endswith('.zip'):
                import zipfile
                extracted_folder_name = Path(download_path).stem
                extraction_path = os.path.join(DATA_DIR, extracted_folder_name)
                os.makedirs(extraction_path, exist_ok=True)
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
                print(f"Extracted zip file to {extraction_path}")
                # Now search for CSVs in the extracted folder
                for root, _, files in os.walk(extraction_path):
                    for file in files:
                        if file.lower().endswith('.csv'):
                            potential_csv_files.append(os.path.join(root, file))
            else:
                 return {"message": f"Kaggle dataset {dataset_name} downloaded to {download_path}, but no CSV files found directly or in a zip archive."}

        if not potential_csv_files:
            return {"message": f"No CSV files found for dataset {dataset_name} at {download_path} (after attempting zip extraction)."}

        for csv_file_path in potential_csv_files:
            try:
                df = pd.read_csv(csv_file_path, encoding='utf-8', on_bad_lines='warn')
            except UnicodeDecodeError:
                df = pd.read_csv(csv_file_path, encoding='latin-1', on_bad_lines='warn')
            except Exception as read_e:
                print(f"Could not read CSV {csv_file_path}: {read_e}")
                continue # Skip this file

            table_name = Path(csv_file_path).stem
            _load_df_to_db(df, table_name, target_db_path)
            processed_files.append(Path(csv_file_path).name)

        if not processed_files:
             return {"message": f"Kaggle dataset {dataset_name} downloaded, but no CSV files could be processed into the database."}

        return {"message": f"Kaggle dataset '{dataset_name}' processed. Tables created/updated for: {', '.join(processed_files)} in {target_db_path}."}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error loading Kaggle dataset: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error loading Kaggle dataset '{dataset_name}': {str(e)}")

@app.post("/upload-csv/")
async def upload_csv_endpoint(file: UploadFile = File(...)):
    """Uploads a CSV file and loads it into a database table."""
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")
    
    temp_file_path = os.path.join(DATA_DIR, f"temp_{file.filename}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        try:
            df = pd.read_csv(temp_file_path, encoding='utf-8', on_bad_lines='warn')
        except UnicodeDecodeError:
            df = pd.read_csv(temp_file_path, encoding='latin-1', on_bad_lines='warn')
        
        table_name = Path(file.filename).stem
        _load_df_to_db(df, table_name, DB_PATH)
        
        return {"message": f"CSV '{file.filename}' uploaded and loaded into table '{_clean_column_name(table_name)}' in {DB_PATH}."}
    except Exception as e:
        print(f"Error processing uploaded CSV: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing uploaded CSV '{file.filename}': {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path) # Clean up temp file
        await file.close()

@app.get("/db-schema/")
async def get_database_schema_endpoint(db_path: str = DB_PATH):
    """Returns the schema of the SQLite database."""
    try:
        return get_db_schema(db_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-initial-semantic-schema/")
async def generate_initial_semantic_schema_endpoint(request: SemanticSchemaRequest):
    """Uses LLM to generate an initial semantic schema based on the DB schema."""
    try:
        if genai is None:
            raise HTTPException(status_code=500, detail="Gemini API module not installed. Install with 'pip install google-generativeai'")
            
        # Check for API key
        resolved_api_key = request.user_api_key or DEFAULT_GEMINI_API_KEY
        if not resolved_api_key:
            raise HTTPException(
                status_code=401, 
                detail="No Gemini API key provided. Please provide an API key in the sidebar or set GEMINI_API_KEY environment variable."
            )
            
        model = configure_gemini(resolved_api_key, request.model_name)
        db_schema = get_db_schema(DB_PATH) # Use the global DB_PATH
        if not db_schema:
            raise HTTPException(status_code=404, detail=f"Database schema is empty or database at {DB_PATH} not found. Load data first.")

        prompt = f"""
        You are a data analyst creating a semantic layer for a database.
        Given the database schema:
        {json.dumps(db_schema, indent=2)}

        For each table and column, provide a concise, human-readable description.
        Identify potential business concepts or metrics.
        Format as a JSON object where keys are table names. Each table object has "description" and "columns" (object of column_name: description).
        Example: {{ "users": {{ "description": "User info.", "columns": {{ "user_id": "Unique ID.", "signup_date": "Registration date." }} }} }}
        Generate the semantic schema. Respond ONLY with the JSON object.
        """
        response = model.generate_content(prompt)
        generated_schema_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        
        try:
            semantic_schema = json.loads(generated_schema_text)
            return semantic_schema
        except json.JSONDecodeError as e:
            print(f"Error decoding LLM response for semantic schema: {e}. Raw: {generated_schema_text}")
            raise HTTPException(status_code=500, detail=f"Could not parse LLM response for semantic schema. Raw output: {generated_schema_text}")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /generate-initial-semantic-schema: {e}\n{traceback.format_exc()}")
        if "API key" in str(e).lower():
            raise HTTPException(status_code=401, detail="Invalid or missing Gemini API key. Please check your API key in the sidebar.")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.post("/query-analyzer/")
async def analyze_query_endpoint(request: QueryRequest):
    """
    Analyzes a natural language question, generates SQL or Python, executes SQL if applicable,
    and suggests visualizations.
    """
    try:
        if genai is None:
            raise HTTPException(status_code=500, detail="Gemini API module not installed. Install with 'pip install google-generativeai'")
            
        model = configure_gemini(request.user_api_key, request.model_name)
        db_schema = get_db_schema(DB_PATH)

        if not db_schema: # Check if DB is empty or not initialized
             return {
                "thinking_steps": "Database is empty or not found. Please load data first.",
                "analysis_type": "error",
                "generated_sql_query": None,
                "generated_python_script": None,
                "required_packages": [],
                "reason_if_not_sql_or_python": "Database schema is not available. Load data before asking questions.",
                "results_table": None,
                "plot_base64": None,
                "simple_summary_inference": "Cannot process query as the database is not initialized or empty.",
                "chart_type": "table",
                "input_tokens_used": 0, # Placeholder
                "output_tokens_used": 0 # Placeholder
            }

        try:
            semantic_schema = json.loads(request.semantic_schema_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for semantic_schema_json.")

        schema_context_for_llm = ""
        if request.use_rag_for_schema:
            schema_context_for_llm = get_relevant_schema_from_rag(request.question, db_schema)
        else:
            schema_context_for_llm = f"DATABASE SCHEMA:\n{json.dumps(db_schema, indent=2)}"

        prompt_parts = [
            "You are an expert data analyst. Your task is to convert natural language questions into executable SQL queries for SQLite or Python scripts for more complex analysis.",
            "You will be given a database schema (or relevant parts of it), a semantic schema (business context), and a user question.",
            "Follow these instructions:",
            "1. Understand the question, database schema, and semantic schema.",
            "2. ALWAYS try to solve the problem using SQL first. Only use Python if SQL is absolutely insufficient.",
            "3. Determine the best analysis type based on these strict criteria:",
            "   - Use 'sql' if ANY of these are true:",
            "     * The question can be answered with standard SQL operations (SELECT, JOIN, GROUP BY, etc.)",
            "     * The data can be aggregated and filtered using SQL functions",
            "     * The visualization can be created from the SQL results directly",
            "     * The analysis involves basic data retrieval or filtering",
            "     * The question asks for counts, sums, averages, or other basic aggregations",
            "     * The question involves joining tables or filtering data",
            "   - Use 'python' ONLY if ALL of these are true:",
            "     * SQL cannot handle the required data transformations",
            "     * Complex statistical analysis is needed",
            "     * Advanced visualization requirements beyond basic charts",
            "     * Machine learning or statistical modeling is required",
            "     * Complex data cleaning or preprocessing is needed",
            "     * Time series analysis with advanced features is required",
            "     * Custom calculations that cannot be done in SQL are needed",
            "4. If 'sql':",
            "   - Generate a valid SQLite query.",
            "   - ALWAYS suggest a visualization (chart type, x-column, y-column, and title).",
            "   - Choose from these chart types: 'bar', 'line', 'scatter', 'pie', 'hist', 'table', 'box'.",
            "   - Ensure the query returns data suitable for the suggested visualization.",
            "5. If 'python':",
            "   - State that Python is needed and explain why SQL cannot handle the task.",
            "   - Generate a Python script that:",
            "     * Uses ONLY these packages: pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, nltk, scipy, regex, pillow",
            "     * ALWAYS includes visualization unless explicitly not needed",
            "     * Uses the available helper functions instead of writing custom code:",
            "       - get_db_connection_from_script() -> sqlite3.Connection",
            "       - execute_sql_with_validation(query: str) -> pd.DataFrame",
            "       - save_plot_with_validation(fig: plt.Figure, title: str = None) -> None",
            "       - print_error_summary(error: Exception) -> None",
            "     * Prints results in a clear, structured format",
            "     * Saves plots using save_plot_with_validation()",
            "     * If there are multiple plots, then make all of them as sub plots and then save a single plot using save_plot_validation() function",
            "     * Uses execute_sql_with_validation() for any SQL operations",
            "     * OUTPUT CODE MUST NOT GIVE THE DECLARATIONS OF THE HELPER FUNCTIONS MENTIONED ABOVE. JUST THINK AS IT IS ALREADY PRESENT IN EXECUTION ENVIRONMENT.",
            "6. If the question is too ambiguous or beyond reasonable analytical scope, set analysis_type to 'complex' and explain.",
            "7. Output your response as a JSON object with these keys:",
            "   - 'thinking_steps': Your detailed reasoning about why SQL or Python was chosen.",
            "   - 'analysis_type': 'sql', 'python', or 'complex'.",
            "   - 'generated_sql_query': SQL query string (if 'sql'), else null.",
            "   - 'generated_python_script': Python script string (if 'python'), else null.",
            "   - 'required_packages': List of strings for Python packages (if 'python'), else [].",
            "   - 'reason_if_not_sql_or_python': Explanation (if 'complex' or if Python is chosen over SQL), else null.",
            "   - 'chart_type': Suggested chart type (if 'sql' and applicable).",
            "   - 'chart_x_column': Suggested x-axis column (if 'sql' and applicable).",
            "   - 'chart_y_column': Suggested y-axis column (if 'sql' and applicable).",
            "   - 'chart_title': Suggested chart title (if 'sql' and applicable).",
            "\n",
            schema_context_for_llm,
            "\nSEMANTIC SCHEMA (Business Context):\n",
            json.dumps(semantic_schema, indent=2),
            "\nUSER QUESTION:\n",
            request.question,
            "\nRespond ONLY with the JSON object. Do not include any other text, markdown, or explanations outside the JSON structure."
        ]
        llm_prompt = "\n".join(prompt_parts)


        input_tokens_used = len(llm_prompt) // 4 # Rough estimate
        output_tokens_used = 0 # Will be estimated after getting response

        generation_config = genai.types.GenerationConfig(candidate_count=1)
        model_response = model.generate_content(llm_prompt, generation_config=generation_config)
        
        llm_output_json = model_response.text
        output_tokens_used = len(llm_output_json) // 4 # Rough estimate

        try:
            llm_response_data = json.loads(llm_output_json)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError from LLM output: {e}. Raw: {llm_output_json}")
            # Try multiple strategies to extract valid JSON
            json_strategies = [
                # Strategy 1: Look for JSON within ```json ... ```
                lambda text: re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL),
                # Strategy 2: Look for JSON within ``` ... ```
                lambda text: re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL),
                # Strategy 3: Look for JSON between curly braces
                lambda text: re.search(r"(\{.*\})", text, re.DOTALL),
                # Strategy 4: Try to fix common JSON formatting issues
                lambda text: re.sub(r'(\w+):', r'"\1":', text)  # Add quotes to keys
            ]
            
            for strategy in json_strategies:
                try:
                    if callable(strategy):
                        # For regex strategies
                        match = strategy(llm_output_json)
                        if match:
                            json_str = match.group(1)
                            # Clean up the JSON string
                            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
                            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                            llm_response_data = json.loads(json_str)
                            print("Successfully parsed JSON using strategy")
                            break
                    else:
                        # For string manipulation strategies
                        modified_json = strategy(llm_output_json)
                        llm_response_data = json.loads(modified_json)
                        print("Successfully parsed JSON using string manipulation")
                        break
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            if not llm_response_data:
                # If all strategies fail, try to construct a minimal valid response
                print("All JSON parsing strategies failed. Constructing minimal valid response.")
                llm_response_data = {
                    "thinking_steps": "Error parsing LLM response. Please try again.",
                    "analysis_type": "error",
                    "generated_sql_query": None,
                    "generated_python_script": None,
                    "required_packages": [],
                    "reason_if_not_sql_or_python": "Failed to parse LLM response. Raw output: " + llm_output_json[:200] + "...",
                    "results_table": None,
                    "plot_base64": None,
                    "simple_summary_inference": "Error in processing the request. Please try again.",
                    "chart_type": "table",
                    "input_tokens_used": input_tokens_used,
                    "output_tokens_used": output_tokens_used
                }

        # Validate required fields in the response
        required_fields = ["thinking_steps", "analysis_type"]
        missing_fields = [field for field in required_fields if field not in llm_response_data]
        if missing_fields:
            print(f"Warning: Missing required fields in LLM response: {missing_fields}")
            # Add missing fields with default values
            for field in missing_fields:
                llm_response_data[field] = "Not provided by LLM"

        # Validate analysis_type
        valid_analysis_types = ["sql", "python", "complex", "error"]
        if llm_response_data.get("analysis_type") not in valid_analysis_types:
            print(f"Warning: Invalid analysis_type: {llm_response_data.get('analysis_type')}")
            llm_response_data["analysis_type"] = "error"
            llm_response_data["reason_if_not_sql_or_python"] = f"Invalid analysis type returned by LLM: {llm_response_data.get('analysis_type')}"

        # Ensure consistent null values (use None instead of null strings)
        for key in ["generated_sql_query", "generated_python_script", "reason_if_not_sql_or_python"]:
            if llm_response_data.get(key) == "null" or llm_response_data.get(key) == "":
                llm_response_data[key] = None

        # Validate and clean up chart-related fields
        valid_chart_types = ["bar", "line", "scatter", "pie", "hist", "table", "box"]
        chart_type = llm_response_data.get("chart_type", "table")
        if chart_type not in valid_chart_types:
            print(f"Warning: Invalid chart_type: {chart_type}")
            llm_response_data["chart_type"] = "table"

        analysis_type = llm_response_data.get("analysis_type")
        sql_query = llm_response_data.get("generated_sql_query")
        python_script = llm_response_data.get("generated_python_script")
        
        results_df = None
        results_data = None
        plot_base64 = None
        analysis_summary = llm_response_data.get("thinking_steps", "No thinking steps provided.")
        executed_python_output = None # For results from Python if executed by backend (not in this version)

        if analysis_type == "sql" and sql_query:
            try:
                results_df = execute_sql_query(sql_query, DB_PATH)
                results_data = results_df.to_dict(orient="records") if results_df is not None else []
                
                if not results_data:
                    analysis_summary += "\nSQL query executed successfully but returned no data."
                else:
                    analysis_summary += f"\nSuccessfully executed SQL. Found {len(results_data)} records."
                    chart_type = llm_response_data.get("chart_type", "table")
                    if chart_type != "table" and not results_df.empty:
                        plot_base64 = generate_plot(
                            results_df,
                            chart_type,
                            x_column=llm_response_data.get("chart_x_column"),
                            y_column=llm_response_data.get("chart_y_column"),
                            title=llm_response_data.get("chart_title", "Query Results")
                        )
                        if plot_base64:
                            analysis_summary += " Chart generated."
                        else:
                            analysis_summary += " Could not generate chart from SQL results."
            except HTTPException as e: # Catch SQL execution errors specifically
                analysis_summary += f"\nError executing SQL: {e.detail}"
                # Potentially change analysis_type to 'error' or similar
                llm_response_data["analysis_type"] = "error_sql_execution" # So frontend knows
                llm_response_data["reason_if_not_sql_or_python"] = f"SQL Execution Failed: {e.detail}"


        elif analysis_type == "python":
            analysis_summary += "\nPython script generated. Ready for execution in the frontend editor."
            # In this version, we don't execute Python on the backend automatically.
            # The script is sent to the frontend.
        
        elif analysis_type == "complex":
            analysis_summary = llm_response_data.get("reason_if_not_sql_or_python", "The question is too complex for standard SQL/Python analysis with current setup.")

        return {
            "thinking_steps": llm_response_data.get("thinking_steps"),
            "analysis_type": llm_response_data.get("analysis_type"),
            "generated_sql_query": sql_query,
            "generated_python_script": python_script,
            "required_packages": llm_response_data.get("required_packages", []),
            "reason_if_not_sql_or_python": llm_response_data.get("reason_if_not_sql_or_python"),
            "results_table": results_data, # From SQL execution
            "plot_base64": plot_base64,   # From SQL execution
            "simple_summary_inference": analysis_summary,
            "chart_type": llm_response_data.get("chart_type", "table"), # From LLM suggestion for SQL
            "chart_x_column": llm_response_data.get("chart_x_column"),
            "chart_y_column": llm_response_data.get("chart_y_column"),
            "chart_title": llm_response_data.get("chart_title"),
            "input_tokens_used": input_tokens_used,   # Placeholder
            "output_tokens_used": output_tokens_used, # Placeholder
            "executed_python_output": None, # Placeholder for direct python output from backend
            "python_plot_base64": None    # Placeholder for direct python plot from backend
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /query-analyzer: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@app.post("/execute-python/")
async def execute_python_script_endpoint(request: ExecutePythonRequest):
    """
    Execute a Python script in a controlled environment.
    WARNING: Direct execution of arbitrary code is a security risk in production.
    """
    script_to_execute = request.script
    
    # Create a temporary directory for the script to run and save files
    temp_run_dir = os.path.abspath(os.path.join(DATA_DIR, "py_run_temp"))
    os.makedirs(temp_run_dir, exist_ok=True)
    script_file_path = os.path.join(temp_run_dir, "script.py")
    db_file_path_for_script = os.path.abspath(DB_PATH)

    # Basic security check - prevent obvious dangerous operations
    dangerous_patterns = [
        r"import\s+os\s*;?\s*os\.system",
        r"import\s+subprocess\s*;?\s*subprocess\.run",
        r"import\s+subprocess\s*;?\s*subprocess\.Popen",
        r"__import__\s*\(",
        r"eval\s*\(",
        r"exec\s*\(",
        r"open\s*\([^)]*[wax]\+?",
        r"shutil\.rmtree",
        r"os\.remove",
        r"os\.unlink",
        r"os\.rmdir",
        r"os\.removedirs",
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, script_to_execute, re.IGNORECASE):
            raise HTTPException(
                status_code=400,
                detail="Script contains potentially dangerous operations. Please modify the script to use only safe operations."
            )

    # Add a preamble to the script
    script_preamble = f"""

import os
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use('Agg') # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import json
import traceback
from typing import Optional, Dict, Any, List
import numpy as np
from datetime import datetime

# Configure plot styling
plt.style.use('default')  # Use default matplotlib style
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Make the database path available to the script
DB_PATH_INSIDE_SCRIPT = f'{db_file_path_for_script}'
PLOT_OUTPUT_PATH = f'{os.path.join(temp_run_dir, "plot.png")}'

def get_db_connection_from_script() -> sqlite3.Connection:
    \"\"\"Get a connection to the SQLite database with error handling.\"\"\"
    try:
        return sqlite3.connect(DB_PATH_INSIDE_SCRIPT)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {{e}}")
        raise

def execute_sql_with_validation(query: str) -> pd.DataFrame:
    \"\"\"Execute SQL query with validation and error handling.\"\"\"
    try:
        with get_db_connection_from_script() as conn:
            df = pd.read_sql_query(query, conn)
            if df.empty:
                print("Warning: Query returned no data")
            return df
    except Exception as e:
        print(f"Error executing SQL query: {{e}}")
        raise

def save_plot_with_validation(fig: plt.Figure, title: str = None) -> None:
    \"\"\"Save plot with validation and error handling.\"\"\"
    try:
        if title:
            fig.suptitle(title, fontsize=16, y=1.02)
        fig.tight_layout()
        fig.savefig(PLOT_OUTPUT_PATH, dpi=300, bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"Error saving plot: {{e}}")
        raise

def print_error_summary(error: Exception) -> None:
    \"\"\"Print a formatted error summary.\"\"\"
    print(f"\\nError Summary:")
    print(f"Type: {{type(error).__name__}}")
    print(f"Message: {{str(error)}}")
    print(f"\\nTraceback:")
    print(traceback.format_exc())

# Main execution wrapper
try:
    # Your script code will be inserted here
    pass
except Exception as e:
    print_error_summary(e)
    raise

"""
    full_script_content = script_preamble + "\n" + script_to_execute

    # Ensure the directory exists and is clean
    if os.path.exists(temp_run_dir):
        for file in os.listdir(temp_run_dir):
            file_path = os.path.join(temp_run_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error cleaning up existing file {file_path}: {e}")
    else:
        os.makedirs(temp_run_dir)

    # Write the script file
    with open(script_file_path, "w") as f:
        f.write(full_script_content)

    try:
        # Using subprocess for a degree of isolation (still not a full sandbox)
        import subprocess
        process = subprocess.run(
            ["python", script_file_path],
            capture_output=True,
            text=True,
            timeout=30, # Add a timeout
            cwd=temp_run_dir, # Run script from its own directory
            env={**os.environ, "PYTHONPATH": temp_run_dir} # Isolate Python path
        )
        captured_stdout = process.stdout
        error_output = process.stderr

        if process.returncode != 0:
            error_output = f"Script execution failed with code {process.returncode}:\n{process.stderr}"
            print(error_output)
        else:
            print("Python script executed via subprocess successfully.")
            # Check if a plot was saved
            plot_file_path = os.path.join(temp_run_dir, "plot.png")
            if os.path.exists(plot_file_path):
                with open(plot_file_path, "rb") as pf:
                    plot_b64_from_python = base64.b64encode(pf.read()).decode('utf-8')
                print("Plot generated by Python script and encoded.")
            
            # Check for structured DataFrame output
            df_json_match = re.search(r"PYTHON_DF_RESULT_JSON_START>>>(.*?)<<<PYTHON_DF_RESULT_JSON_END", captured_stdout, re.DOTALL)
            if df_json_match:
                df_json_str = df_json_match.group(1)
                try:
                    # Replace the matched part with a marker or remove it to clean up stdout
                    captured_stdout = captured_stdout.replace(df_json_match.group(0), "[DataFrame result below]\n")
                    print(f"DataFrame JSON extracted from Python script output.")
                except json.JSONDecodeError:
                    print("Could not parse DataFrame JSON from script output.")

    except subprocess.TimeoutExpired:
        error_output = "Python script execution timed out after 30 seconds."
        print(error_output)
    except Exception as e:
        error_output = f"Error during Python script execution attempt: {str(e)}\n{traceback.format_exc()}"
        print(error_output)
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(script_file_path):
                os.remove(script_file_path)
            plot_file_path = os.path.join(temp_run_dir, "plot.png")
            if os.path.exists(plot_file_path):
                os.remove(plot_file_path)
            # Clean up any other temporary files that might have been created
            for file in os.listdir(temp_run_dir):
                file_path = os.path.join(temp_run_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(f"Error cleaning up temporary file {file_path}: {e}")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")

    if error_output:
        return {
            "status": "error",
            "output": captured_stdout,
            "error": error_output,
            "plot_base64": None
        }
    
    return {
        "status": "success",
        "output": captured_stdout.strip(),
        "error": None,
        "plot_base64": plot_b64_from_python if 'plot_b64_from_python' in locals() else None
    }

@app.post("/clear-database/")
async def clear_database_endpoint():
    """Deletes all tables from the database."""
    try:
        conn = get_db_connection(DB_PATH)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Drop each table
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table};")
        
        conn.commit()
        conn.close()
        
        return {"message": f"Successfully cleared database. Removed {len(tables)} tables."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.post("/fix-python-code/")
async def fix_python_code_endpoint(request: FixCodeRequest):
    """
    Uses LLM to fix Python code based on the error message.
    """
    try:
        if genai is None:
            raise HTTPException(status_code=500, detail="Gemini API module not installed. Install with 'pip install google-generativeai'")
            
        model = configure_gemini(request.user_api_key, request.model_name)
        
        prompt = f"""
        You are an expert Python developer. Fix the following Python code that has an error.
        The code should use only these helper functions:
        - get_db_connection_from_script() -> sqlite3.Connection
        - execute_sql_with_validation(query: str) -> pd.DataFrame
        - save_plot_with_validation(fig: plt.Figure, title: str = None) -> None
        - print_error_summary(error: Exception) -> None

        Do not redefine these functions, just use them.
        Always include visualization unless explicitly not needed.
        Use proper error handling.

        Original code:
        ```python
        {request.code}
        ```

        Error message:
        {request.error}

        Provide the fixed code. Respond ONLY with the fixed Python code, no explanations or markdown.
        """
        
        response = model.generate_content(prompt)
        fixed_code = response.text.strip()
        
        # Clean up the response to ensure it's just the code
        fixed_code = re.sub(r'```python\s*', '', fixed_code)
        fixed_code = re.sub(r'\s*```', '', fixed_code)
        
        return {"fixed_code": fixed_code}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fixing code: {str(e)}")

# --- To run the app (from terminal) ---
# uvicorn main:app --reload --host 0.0.0.0 --port 8002
if __name__ == "__main__":
    # This part is for direct execution of main.py (e.g. python main.py)
    # It's not strictly necessary if you always run with uvicorn from the command line.
    # However, it's good practice for discoverability.
    print("Starting Uvicorn server for Analytics Bot API...")
    print(f"API will be available at http://127.0.0.1:8002")
    print(f"Access OpenAPI docs at http://127.0.0.1:8002/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8002, reload=True)

