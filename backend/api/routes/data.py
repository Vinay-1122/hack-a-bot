from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import json
import pandas as pd
from pathlib import Path
import sqlite3
import traceback
import re
import base64
import numpy as np

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

# --- Models ---
class LoadKaggleRequest(BaseModel):
    dataset_name: str

class SemanticSchemaRequest(BaseModel):
    user_api_key: Optional[str] = None
    model_name: str = "gemini-1.5-flash"

class ExecutePythonRequest(BaseModel):
    script: str
    user_api_key: Optional[str] = None

class GenerateInferenceRequest(BaseModel):
    data: List[Dict[str, Any]]
    max_rows: Optional[int] = 50
    user_api_key: Optional[str] = None
    question: Optional[str] = None

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

@router.post("/load-data/")
async def load_default_data():
    """Load the default retail analytics dataset."""
    try:
        # Get the path to the default data files
        base_dir = Path(__file__).parent.parent.parent
        data_dir = base_dir / "retail-analysis-dataset"
        
        # Map the actual filenames
        stores_file = data_dir / "stores data-set.csv"
        features_file = data_dir / "Features data set.csv"
        sales_file = data_dir / "sales data-set.csv"

        # Verify files exist
        for file_path in [stores_file, features_file, sales_file]:
            if not file_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Required data file not found: {file_path}"
                )

        # Read the CSV files
        stores_df = pd.read_csv(stores_file)
        features_df = pd.read_csv(features_file)
        sales_df = pd.read_csv(sales_file)

        # Create data directory if it doesn't exist
        db_dir = base_dir / "data"
        db_dir.mkdir(exist_ok=True)

        # Connect to SQLite database using the configured path
        conn = sqlite3.connect(DB_PATH)

        # Write data to SQLite
        stores_df.to_sql("stores", conn, if_exists="replace", index=False)
        features_df.to_sql("features", conn, if_exists="replace", index=False)
        sales_df.to_sql("sales", conn, if_exists="replace", index=False)

        # Load default schema
        schema_path = base_dir / "schema.json"
        with open(schema_path, "r") as f:
            default_schema = json.load(f)

        conn.close()

        return {
            "message": "Default retail analytics dataset loaded successfully",
            "schema": default_schema
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

def print_data(dataframes) -> None:
    output = {{}}
    for i, df in enumerate(dataframes):
        if isinstance(df['df'], pd.DataFrame):
            output[df['title']] = df['df'].to_dict(orient='records')

    print("DATAFRAME_OUTPUT_START")
    print(json.dumps(output))
    print("DATAFRAME_OUTPUT_END")
        

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
        else:
            print("Python script executed via subprocess successfully.")
            # Check if a plot was saved
            plot_file_path = os.path.join(temp_run_dir, "plot.png")
            if os.path.exists(plot_file_path):
                with open(plot_file_path, "rb") as pf:
                    plot_b64_from_python = base64.b64encode(pf.read()).decode('utf-8')
            
            # Check for structured DataFrame output
            df_json_match = re.search(r'DATAFRAME_OUTPUT_START\n(.*?)\nDATAFRAME_OUTPUT_END', captured_stdout, re.DOTALL)
            if df_json_match:
                df_json_str = df_json_match.group(1)
                try:
                    # Replace the matched part with a marker or remove it to clean up stdout
                    df_json = json.loads(df_json_str)
                except json.JSONDecodeError:
                    print("Could not parse DataFrame JSON from script output.")

    except subprocess.TimeoutExpired:
        error_output = "Python script execution timed out after 30 seconds."
    except Exception as e:
        error_output = f"Error during Python script execution attempt: {str(e)}\n{traceback.format_exc()}"
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
            print(f"Error cleaning up temporary files: {cleanup_error}")

    if error_output:
        return {
            "status": "error",
            "output": captured_stdout,
            "error": error_output,
            "plot_base64": None
        }
    
    return {
        "status": "success",
        "output": df_json_str if 'df_json' in locals() else None,
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

@router.post("/generate-inference/")
async def generate_inference_endpoint(request: GenerateInferenceRequest):
    """Generate a summary inference from the provided data using LLM."""
    try:
        print(f"[Inference] Starting inference generation for {len(request.data)} records")
        
        # Convert data to DataFrame
        try:
            df = pd.DataFrame(request.data)
            print(f"[Inference] Successfully created DataFrame with shape {df.shape}")
        except Exception as e:
            print(f"[Inference] Error creating DataFrame: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
        
        # Limit rows if needed
        if len(df) > request.max_rows:
            print(f"[Inference] Limiting data from {len(df)} to {request.max_rows} rows")
            df = df.head(request.max_rows)
        
        # Get basic data info for the prompt
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Generate basic statistics for the prompt
        stats_info = {}
        for col in numeric_cols:
            try:
                stats = df[col].describe()
                stats_info[col] = {
                    "mean": float(stats['mean']),
                    "min": float(stats['min']),
                    "max": float(stats['max']),
                    "std": float(stats['std'])
                }
            except Exception as e:
                print(f"[Inference] Error calculating stats for {col}: {str(e)}")
        
        # Get value counts for categorical columns
        cat_info = {}
        for col in categorical_cols:
            try:
                value_counts = df[col].value_counts().head(5).to_dict()
                cat_info[col] = {
                    "unique_count": len(df[col].unique()),
                    "top_values": value_counts
                }
            except Exception as e:
                print(f"[Inference] Error calculating value counts for {col}: {str(e)}")
        
        # Prepare data for LLM
        data_info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "numeric_stats": stats_info,
            "categorical_stats": cat_info,
            "sample_data": df.head(5).to_dict(orient='records')
        }
        print(request.question)
        
        # Configure Gemini
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
            model = configure_gemini(resolved_api_key, "gemini-1.5-flash")
            
            # Generate prompt for LLM
            prompt = f"""
            You are a data analyst tasked with generating insights from a dataset. Your primary goal is to answer the specific question asked, followed by relevant supporting analysis.

            Original Question:
            {request.question if request.question else "No specific question provided"}

            Ex: If the question is "What is the average price of the products?", and lets say the df contains value 20, then the answer should be "The average price of the products is 20".

            Dataset Overview:
            - Total Records: {data_info['total_rows']}
            - Total Columns: {data_info['total_columns']}
            - Numeric Columns: {', '.join(data_info['numeric_columns'])}
            - Categorical Columns: {', '.join(data_info['categorical_columns'])}

            Numeric Statistics:
            {json.dumps(data_info['numeric_stats'], indent=2)}

            Categorical Statistics:
            {json.dumps(data_info['categorical_stats'], indent=2)}

            Sample Data (First 5 rows):
            {json.dumps(data_info['sample_data'], indent=2)}

            Please provide a concise analysis following these guidelines:

            1. Structure your response as follows:
               ### Direct Answer
               - Start with a clear, direct answer to the original question
               - Use `code` blocks for specific numbers and metrics
               - Use **bold** for key findings
               - Keep this section focused on answering the question

               ### Supporting Analysis
               - Provide relevant statistics that support your answer
               - Use markdown tables for comparing metrics
               - Include trends that relate to the question
               Example:
               ```
               Metric    | Value
               ---------|--------
               Mean     | `123.45`
               Median   | `120.00`
               ```

               ### Additional Insights
               - Share any other relevant patterns or insights
               - Focus on insights that provide context to the answer
               - Use > for important notes
               - Use `code` for specific numbers

            2. Formatting Guidelines:
               - Use ### for main headers
               - Use #### for subheaders
               - Use proper markdown lists with - or *
               - Use `code` blocks for numbers and metrics
               - Use **bold** for emphasis on key findings
               - Use > for important notes
               - Use proper markdown tables
               - Use proper markdown code blocks for formulas

            3. For numeric analysis:
               - Focus on metrics that directly relate to the question
               - Use markdown tables for relevant correlations
               - Use `code` blocks for formulas
               - Format percentages with % symbol
               Example:
               ```
               Correlation = `0.85`
               Growth Rate = `12.5%`
               ```

            4. Keep the analysis focused:
               - Maximum 2-3 bullet points per section
               - Prioritize information that answers the question
               - Include only relevant supporting data
               - Use consistent formatting

            5. Special formatting rules:
               - Use proper markdown line breaks
               - Use proper markdown horizontal rules (---)
               - Use proper markdown blockquotes (>)
               - Use proper markdown code blocks (```)
               - Use proper markdown tables
               - Use proper markdown lists

            Format your response in clean, well-structured markdown.
            Focus on answering the original question first, then provide supporting analysis.
            Keep the overall length short and scannable.
            Ensure all special characters and formatting render properly.
            Make sure your analysis directly addresses the original question and provides clear, actionable insights.
            """
            
            print("[Inference] Sending request to LLM")
            response = model.generate_content(prompt)
            summary = response.text.strip()
            
            print("[Inference] Successfully generated summary")
            return {
                "status": "success",
                "summary": summary
            }
            
        except HTTPException:
            raise
        except Exception as e:
            print(f"[Inference] Error in LLM processing: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Inference] Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error in inference generation: {str(e)}") 