import streamlit as st
import requests
from config.constants import API_URL

def load_default_data():
    """Load the default retail analytics dataset."""
    try:
        response = requests.post(f"{API_URL}/load-data/")
        response.raise_for_status()
        st.toast("Default retail analytics dataset loaded successfully!", icon="✅")
        st.session_state.db_initialized = True
        return response.json()
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "Connection refused" in error_msg:
            st.error("Could not connect to the backend server. Please ensure it's running.")
        else:
            st.error(f"Failed to load default data: {error_msg}")
        st.session_state.db_initialized = False
        return None

def load_csv_data(uploaded_file):
    """Load data from uploaded CSV file."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        response = requests.post(f"{API_URL}/upload-csv/", files=files, timeout=120)
        response.raise_for_status()
        st.toast("CSV data loaded successfully!", icon="✅")
        st.session_state.db_initialized = True
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading CSV data: {e.response.text if e.response else str(e)}")
        return False

def load_kaggle_data(dataset_id):
    """Load data from Kaggle dataset."""
    try:
        payload = {"dataset_name": dataset_id}
        response = requests.post(f"{API_URL}/load-kaggle-dataset/", json=payload, timeout=300)
        response.raise_for_status()
        st.toast("Kaggle dataset loaded successfully!", icon="✅")
        st.session_state.db_initialized = True
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error loading Kaggle dataset: {e.response.text if e.response else str(e)}")
        return False 