import re
from typing import List

def check_dangerous_operations(script: str) -> List[str]:
    """
    Checks for potentially dangerous operations in Python code.
    Returns a list of dangerous patterns found.
    """
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
    
    found_patterns = []
    for pattern in dangerous_patterns:
        if re.search(pattern, script, re.IGNORECASE):
            found_patterns.append(pattern)
    
    return found_patterns

def sanitize_python_script(script: str) -> str:
    """
    Sanitizes a Python script by removing or replacing dangerous operations.
    Returns the sanitized script.
    """
    # Add a preamble to the script with safe imports and configurations
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
    
    # Remove any dangerous operations
    dangerous_patterns = check_dangerous_operations(script)
    if dangerous_patterns:
        raise ValueError(f"Script contains potentially dangerous operations: {', '.join(dangerous_patterns)}")
    
    # Return the sanitized script
    return script_preamble + "\n" + script 