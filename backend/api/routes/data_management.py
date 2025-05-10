from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import os
import json
import pandas as pd
from pathlib import Path
import traceback
import re
import base64

# Optional dependencies
try:
    import kagglehub
except ImportError:
    kagglehub = None
try:
    import google.generativeai as genai
except ImportError:
    genai = None

from core.database import get_db_schema, get_db_connection
from core.config import DATA_DIR, DB_PATH, DEFAULT_GEMINI_API_KEY

router = APIRouter()

# --- Helper functions ---
def _clean_column_name(col_name):
    return str(col_name).strip().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('.', '_').replace('/', '_')

def _load_df_to_db(df: pd.DataFrame, table_name: str, db_path: str = DB_PATH, if_exists: str = 'replace'):
    conn = get_db_connection(db_path)
    cleaned_table_name = _clean_column_name(table_name)
    df.columns = [_clean_column_name(col) for col in df.columns]
    try:
        df.to_sql(cleaned_table_name, conn, if_exists=if_exists, index=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error writing to DB table {cleaned_table_name}: {str(e)}")
    finally:
        conn.close()

# --- Models ---
class LoadKaggleRequest(BaseModel):
    dataset_name: str

class SemanticSchemaRequest(BaseModel):
    user_api_key: Optional[str] = None
    model_name: str = "gemini-1.5-flash"

class ExecutePythonRequest(BaseModel):
    script: str
    user_api_key: Optional[str] = None

@router.post("/load-kaggle-dataset/")
async def load_kaggle_dataset_endpoint(request: LoadKaggleRequest):
    if kagglehub is None:
        raise HTTPException(status_code=500, detail="Kagglehub not installed. Install with 'pip install kagglehub'")
    dataset_name = request.dataset_name
    target_db_path = DB_PATH
    try:
        try:
            download_path = kagglehub.dataset_download(dataset_name)
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
        if not potential_csv_files:
            if os.path.isfile(download_path) and download_path.lower().endswith('.zip'):
                import zipfile
                extracted_folder_name = Path(download_path).stem
                extraction_path = os.path.join(DATA_DIR, extracted_folder_name)
                os.makedirs(extraction_path, exist_ok=True)
                with zipfile.ZipFile(download_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)
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
                continue
            table_name = Path(csv_file_path).stem
            _load_df_to_db(df, table_name, target_db_path)
            processed_files.append(Path(csv_file_path).name)
        if not processed_files:
            return {"message": f"Kaggle dataset {dataset_name} downloaded, but no CSV files could be processed into the database."}
        return {"message": f"Kaggle dataset '{dataset_name}' processed. Tables created/updated for: {', '.join(processed_files)} in {target_db_path}."}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading Kaggle dataset '{dataset_name}': {str(e)}")

@router.post("/upload-csv/")
async def upload_csv_endpoint(file: UploadFile = File(...)):
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
        raise HTTPException(status_code=500, detail=f"Error processing uploaded CSV '{file.filename}': {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        await file.close()

@router.get("/db-schema/")
async def get_database_schema_endpoint():
    try:
        return get_db_schema(DB_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-initial-semantic-schema/")
async def generate_initial_semantic_schema_endpoint(request: SemanticSchemaRequest):
    try:
        if genai is None:
            raise HTTPException(status_code=500, detail="Gemini API module not installed. Install with 'pip install google-generativeai'")
        resolved_api_key = request.user_api_key or DEFAULT_GEMINI_API_KEY
        if not resolved_api_key:
            raise HTTPException(
                status_code=401,
                detail="No Gemini API key provided. Please provide an API key in the sidebar or set GEMINI_API_KEY environment variable."
            )
        from core.llm import configure_gemini
        model = configure_gemini(resolved_api_key, request.model_name)
        db_schema = get_db_schema(DB_PATH)
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
            raise HTTPException(status_code=500, detail=f"Could not parse LLM response for semantic schema. Raw output: {generated_schema_text}")
    except HTTPException:
        raise
    except Exception as e:
        if "API key" in str(e).lower():
            raise HTTPException(status_code=401, detail="Invalid or missing Gemini API key. Please check your API key in the sidebar.")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.post("/execute-python/")
async def execute_python_script_endpoint(request: ExecutePythonRequest):
    script_to_execute = request.script
    temp_run_dir = os.path.abspath(os.path.join(DATA_DIR, "py_run_temp"))
    os.makedirs(temp_run_dir, exist_ok=True)
    script_file_path = os.path.join(temp_run_dir, "script.py")
    db_file_path_for_script = os.path.abspath(DB_PATH)
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
DB_PATH_INSIDE_SCRIPT = r'{db_file_path_for_script}'
PLOT_OUTPUT_PATH = r'{os.path.join(temp_run_dir, "plot.png")}'

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
    if os.path.exists(temp_run_dir):
        for file in os.listdir(temp_run_dir):
            file_path = os.path.join(temp_run_dir, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                pass
    else:
        os.makedirs(temp_run_dir)
    with open(script_file_path, "w") as f:
        f.write(full_script_content)
    try:
        import subprocess
        process = subprocess.run(
            ["python", script_file_path],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=temp_run_dir,
            env={**os.environ, "PYTHONPATH": temp_run_dir}
        )
        captured_stdout = process.stdout
        error_output = process.stderr
        if process.returncode != 0:
            error_output = f"Script execution failed with code {process.returncode}:\n{process.stderr}"
        else:
            plot_file_path = os.path.join(temp_run_dir, "plot.png")
            if os.path.exists(plot_file_path):
                with open(plot_file_path, "rb") as pf:
                    plot_b64_from_python = base64.b64encode(pf.read()).decode('utf-8')
            df_json_match = re.search(r"PYTHON_DF_RESULT_JSON_START>>>(.*?)<<<PYTHON_DF_RESULT_JSON_END", captured_stdout, re.DOTALL)
            if df_json_match:
                df_json_str = df_json_match.group(1)
                try:
                    captured_stdout = captured_stdout.replace(df_json_match.group(0), "[DataFrame result below]\n")
                except json.JSONDecodeError:
                    pass
    except subprocess.TimeoutExpired:
        error_output = "Python script execution timed out after 30 seconds."
    except Exception as e:
        error_output = f"Error during Python script execution attempt: {str(e)}\n{traceback.format_exc()}"
    finally:
        try:
            if os.path.exists(script_file_path):
                os.remove(script_file_path)
            plot_file_path = os.path.join(temp_run_dir, "plot.png")
            if os.path.exists(plot_file_path):
                os.remove(plot_file_path)
            for file in os.listdir(temp_run_dir):
                file_path = os.path.join(temp_run_dir, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    pass
        except Exception as cleanup_error:
            pass
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

@router.post("/clear-database/")
async def clear_database_endpoint():
    try:
        conn = get_db_connection(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table};")
        conn.commit()
        conn.close()
        return {"message": f"Successfully cleared database. Removed {len(tables)} tables."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}") 