import os
import json
from typing import Optional
import google.generativeai as genai
from .config import DEFAULT_GEMINI_API_KEY

def configure_gemini(api_key: Optional[str], model_name: str):
    """Configures and returns a Gemini GenerativeModel."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError("Gemini API module not installed. Install with 'pip install google-generativeai'")
    
    resolved_api_key = api_key or DEFAULT_GEMINI_API_KEY
    if not resolved_api_key:
        raise ValueError("Gemini API key not provided or configured on server.")
    
    genai.configure(api_key=resolved_api_key)
    return genai.GenerativeModel(model_name)

def generate_semantic_schema_prompt(db_schema: dict) -> str:
    """Generates the prompt for semantic schema generation."""
    return f"""
    You are a data analyst creating a semantic layer for a database.
    Given the database schema:
    {json.dumps(db_schema, indent=2)}

    For each table and column, provide a concise, human-readable description.
    Identify potential business concepts or metrics.
    Format as a JSON object where keys are table names. Each table object has "description" and "columns" (object of column_name: description).
    Example: {{ "users": {{ "description": "User info.", "columns": {{ "user_id": "Unique ID.", "signup_date": "Registration date." }} }} }}
    Generate the semantic schema. Respond ONLY with the JSON object.
    """

def generate_query_analysis_prompt(question: str, schema_context: str, semantic_schema: dict) -> str:
    """Generates the prompt for query analysis."""
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
        schema_context,
        "\nSEMANTIC SCHEMA (Business Context):\n",
        json.dumps(semantic_schema, indent=2),
        "\nUSER QUESTION:\n",
        question,
        "\nRespond ONLY with the JSON object. Do not include any other text, markdown, or explanations outside the JSON structure."
    ]
    return "\n".join(prompt_parts)

def generate_code_fix_prompt(code: str, error: str) -> str:
    """Generates the prompt for fixing Python code."""
    
    return f"""
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
        {code}
        ```

        Error message:
        {error}

        Provide the fixed code. Respond ONLY with the fixed Python code, no explanations or markdown.
        """