import streamlit as st
from config.constants import SESSION_STATE_DEFAULTS

def init_session_state():
    """Initialize session state with default values."""
    for key, value in SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value

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