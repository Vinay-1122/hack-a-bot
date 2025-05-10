import sqlite3
import pandas as pd
import os
import json
from typing import Dict, List, Any
from .config import DB_PATH

def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Access columns by name
    return conn

def get_db_schema(db_path: str = DB_PATH) -> Dict[str, List[Dict[str, Any]]]:
    """Retrieves the schema of the specified SQLite database."""
    if not os.path.exists(db_path):
        return {}  # Return empty schema if DB doesn't exist
    
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

def execute_sql_query(sql_query: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """Executes a SQL query and returns a Pandas DataFrame."""
    if not os.path.exists(db_path) and "FROM" in sql_query.upper():  # Basic check
        raise ValueError(f"Database at {db_path} not found or not initialized. Load data first.")
    
    conn = get_db_connection(db_path)
    try:
        df = pd.read_sql_query(sql_query, conn)
        return df
    except Exception as e:
        raise ValueError(f"Error executing SQL: {str(e)}")
    finally:
        conn.close()

def _clean_column_name(col_name: str) -> str:
    """Cleans a column name for DB compatibility."""
    return str(col_name).strip().lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('.', '_').replace('/', '_')

def load_df_to_db(df: pd.DataFrame, table_name: str, db_path: str = DB_PATH, if_exists: str = 'replace') -> None:
    """Loads a DataFrame into a specified SQLite table."""
    conn = get_db_connection(db_path)
    cleaned_table_name = _clean_column_name(table_name)  # Clean table name as well
    df.columns = [_clean_column_name(col) for col in df.columns]
    
    try:
        df.to_sql(cleaned_table_name, conn, if_exists=if_exists, index=False)
        print(f"DataFrame loaded into table '{cleaned_table_name}' in {db_path}")
    except Exception as e:
        raise ValueError(f"Error writing to DB table {cleaned_table_name}: {str(e)}")
    finally:
        conn.close()

def clear_database(db_path: str = DB_PATH) -> int:
    """Deletes all tables from the database. Returns number of tables cleared."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # Get list of all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    # Drop each table
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table};")
    
    conn.commit()
    conn.close()
    
    return len(tables) 