import streamlit as st
from config.constants import SESSION_STATE_DEFAULTS

def init_session_state():
    """Initialize session state variables."""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "user_question_input_key" not in st.session_state:
        st.session_state.user_question_input_key = 0
    
    if "current_question_to_process" not in st.session_state:
        st.session_state.current_question_to_process = ""
    
    if "db_initialized" not in st.session_state:
        st.session_state.db_initialized = False
    
    if "current_semantic_schema" not in st.session_state:
        st.session_state.current_semantic_schema = "{}"
    
    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = None
    
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemini-1.5-flash"
    
    if "use_rag_for_schema" not in st.session_state:
        st.session_state.use_rag_for_schema = False
    
    if "show_thinking_steps" not in st.session_state:
        st.session_state.show_thinking_steps = False
    
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False
    
    if "python_code_editable" not in st.session_state:
        st.session_state.python_code_editable = ""
    
    if "show_python_editor" not in st.session_state:
        st.session_state.show_python_editor = False
    
    if "editor_key" not in st.session_state:
        st.session_state.editor_key = 0
    
    if "auto_run_enabled" not in st.session_state:
        st.session_state.auto_run_enabled = False
    
    if "advanced_mode" not in st.session_state:
        st.session_state.advanced_mode = False
    
    if "use_rag_mode" not in st.session_state:
        st.session_state.use_rag_mode = False

def update_session_state(key, value):
    """Update a specific session state value."""
    st.session_state[key] = value

def get_session_state(key, default=None):
    """Get a specific session state value."""
    return st.session_state.get(key, default)

def reset_session_state():
    """Reset all session state values to defaults."""
    for key, value in SESSION_STATE_DEFAULTS.items():
        st.session_state[key] = value 