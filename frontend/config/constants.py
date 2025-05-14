# API Configuration
API_URL = "http://127.0.0.1:8001/api"

# Model Pricing Configuration
MODEL_PRICING_PER_MILLION_TOKENS = {
    "gemini-1.5-flash": {"input": 0.35, "output": 0.70},  # USD per 1M tokens
    "gemini-1.5-pro": {"input": 3.50, "output": 7.00},    # USD per 1M tokens
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},    # USD per 1M tokens
}

# Session State Defaults
SESSION_STATE_DEFAULTS = {
    "chat_history": [],
    "current_semantic_schema": "{}",
    "gemini_api_key": "",
    "selected_model": "gemini-1.5-flash",  # Default model
    "total_cost": 0.0,
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "use_rag_for_schema": False,
    "python_code_editable": "",  # For Python editor
    "show_python_editor": False,
    "user_question_input_key": 0,  # To help reset input field
    "db_initialized": False,  # Track if DB has data
    "fixed_code_available": False,  # Track if fixed code is available
    "current_error_message": None,  # Track current error message
    "fix_button_clicked": False,  # Track if fix button was clicked
    "editor_key": 0,  # Track editor updates
    "new_code": "",  # Store new code temporarily
    "new_code_available": False,  # Flag for new code
    "advanced_mode": False  # Toggle for basic/advanced mode
} 
