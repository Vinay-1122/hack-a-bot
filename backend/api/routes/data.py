from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
import os
import json
import pandas as pd
from pathlib import Path
import sqlite3

from core.database import get_db_schema, get_db_connection
from core.config import DATA_DIR, DB_PATH, DEFAULT_GEMINI_API_KEY

# Optional dependencies
try:
    import kagglehub
except ImportError:
    kagglehub = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

router = APIRouter()

# --- Models ---
class LoadKaggleRequest(BaseModel):
    dataset_name: str

class SemanticSchemaRequest(BaseModel):
    user_api_key: Optional[str] = None
    model_name: str = "gemini-1.5-flash"

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

@router.post("/clear-database/")
async def clear_database_endpoint():
    try:
        conn = get_db_connection(DB_PATH)
        cursor = conn.cursor()
        
        # Get list of all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        # Drop each table
        for table in tables:
            cursor.execute(f"DROP TABLE IF EXISTS {table[0]};")
        
        conn.commit()
        conn.close()
        
        return {"message": "Database cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}") 