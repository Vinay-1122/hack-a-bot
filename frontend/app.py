import streamlit as st
from utils.session_state import init_session_state
from components.sidebar import render_sidebar
from components.main import render_main_page

# Set page config
st.set_page_config(layout="wide", page_title="HackBot Analytics")

# Initialize session state
init_session_state()

# Render the sidebar
render_sidebar()

# Render the main page
render_main_page() 