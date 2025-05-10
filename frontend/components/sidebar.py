import streamlit as st
import requests
import json
import time
from config.constants import API_URL, MODEL_PRICING_PER_MILLION_TOKENS
from utils.cost_utils import get_cost_summary
from utils.data_utils import load_default_data, load_csv_data, load_kaggle_data

def delay_and_rerun(seconds=1):
    """Add a delay before rerunning the app."""
    time.sleep(seconds)
    st.rerun()

def check_db_status():
    """Checks if the database has any tables, indicating it's initialized."""
    try:
        response = requests.get(f"{API_URL}/db-schema/")
        response.raise_for_status()
        schema = response.json()
        is_initialized = bool(schema)  # True if schema is not empty
        st.session_state.db_initialized = is_initialized
        if is_initialized:
            st.toast("Database is initialized and ready!", icon="✅")
        else:
            st.warning("Database is empty. Please load data to get started.")
    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        if "Connection refused" in error_msg:
            st.error("Could not connect to the backend server. Please ensure it's running.")
        else:
            st.error(f"Failed to check database status: {error_msg}")
        st.session_state.db_initialized = False

def render_sidebar():
    """Render the sidebar with all its components."""
    with st.sidebar:
        st.title("⚙️ Settings & Context")

        # Basic/Advanced Mode Toggle
        st.session_state.advanced_mode = st.toggle(
            "Advanced Mode",
            value=st.session_state.advanced_mode,
            help="Toggle between basic and advanced settings"
        )

        # Cost Metrics (visible in both modes)
        st.subheader("💰 Cost Estimation")
        cost_summary = get_cost_summary()
        cost_col1, cost_col2 = st.columns(2)
        cost_col1.metric("Total Input Tokens", f"{cost_summary['total_input_tokens']:,}")
        cost_col2.metric("Total Output Tokens", f"{cost_summary['total_output_tokens']:,}")
        st.metric("Estimated Total Cost", f"${cost_summary['total_cost']:.4f} USD")
        st.caption("Costs are estimates based on token counts and model pricing.")

        st.divider()

        if st.session_state.advanced_mode:
            # API Key Input (only in advanced mode)
            st.session_state.gemini_api_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.gemini_api_key,
                help="If not provided here, the server's backup key will be used (if configured)."
            )

            # Model Selection (only in advanced mode)
            available_models = list(MODEL_PRICING_PER_MILLION_TOKENS.keys())
            if st.session_state.selected_model not in available_models and available_models:
                st.session_state.selected_model = available_models[0]
            elif not available_models:
                st.error("No models configured for pricing.")
                st.stop()

            st.session_state.selected_model = st.selectbox(
                "Select LLM Model",
                options=available_models,
                index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
            )

        st.subheader("📊 Data Management")
        
        # Custom CSS for buttons
        st.markdown("""
            <style>
            div[data-testid="stButton"] button {
                width: 100%;
                height: 3em;
                font-size: 1.2em;
            }
            div[data-testid="stButton"] button.primary {
                background-color: #FF4B4B;
                color: white;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Always show the load data button in basic mode
        if not st.session_state.advanced_mode:
            if st.button("📥 Load Default Retail Analytics Dataset", use_container_width=True):
                with st.spinner("Loading default dataset..."):
                    load_default_data()
                    delay_and_rerun()
        
        # Check DB status button
        if st.button("🔄 Check/Refresh DB Status"):
            check_db_status()
            delay_and_rerun()

        if st.session_state.advanced_mode:
            # Clear DB option (only in advanced mode)
            clear_db = st.toggle("Clear existing data before loading new data", value=False, 
                               help="When enabled, all existing data will be deleted before loading new data")
            
            if clear_db:
                if st.button("🗑️ Clear All Data", type="secondary"):
                    with st.spinner("Clearing database..."):
                        try:
                            response = requests.post(f"{API_URL}/clear-database/")
                            response.raise_for_status()
                            st.toast("Database cleared successfully!", icon="✅")
                            st.session_state.current_semantic_schema = "{}"  # Reset schema
                            check_db_status()  # Update DB status
                            delay_and_rerun()
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error clearing database: {e.response.text if e.response else str(e)}")

            # Data loading options (only in advanced mode)
            data_source = st.radio(
                "Select Data Source",
                ["Upload CSV", "Kaggle Dataset"],
                horizontal=True
            )

            if data_source == "Upload CSV":
                uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
                if uploaded_file is not None:
                    if st.button("Load CSV Data"):
                        load_csv_data(uploaded_file)
                        delay_and_rerun()

            elif data_source == "Kaggle Dataset":
                kaggle_dataset = st.text_input("Enter Kaggle Dataset ID")
                if kaggle_dataset and st.button("Load Kaggle Data"):
                    load_kaggle_data(kaggle_dataset)
                    delay_and_rerun()

        # Schema Management (only in advanced mode)
        if st.session_state.advanced_mode:
            if st.button("✨ Auto-generate Initial Schema", disabled=not st.session_state.db_initialized):
                if not st.session_state.db_initialized:
                    st.warning("Cannot generate schema. Database is empty. Please load data first.")
                else:
                    with st.spinner("Generating initial semantic schema with LLM..."):
                        try:
                            payload = {
                                "user_api_key": st.session_state.gemini_api_key or None,
                                "model_name": st.session_state.selected_model
                            }
                            response = requests.post(f"{API_URL}/generate-initial-semantic-schema/", json=payload)
                            response.raise_for_status()
                            generated_schema = response.json()
                            st.session_state.current_semantic_schema = json.dumps(generated_schema, indent=2)
                            st.toast("Initial schema generated successfully!", icon="✅")
                            delay_and_rerun()
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error generating schema: {e.response.json().get('detail') if e.response else str(e)}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred: {str(e)}")
            
            # Schema Editor (only in advanced mode)
            edited_schema = st.text_area(
                "Edit Semantic Schema (JSON):",
                value=st.session_state.current_semantic_schema,
                height=250,
                key="semantic_schema_editor_area"
            )
            if edited_schema != st.session_state.current_semantic_schema:
                try:
                    json.loads(edited_schema)  # Validate JSON
                    st.session_state.current_semantic_schema = edited_schema
                    st.toast("Semantic schema updated successfully!", icon="✅")
                    delay_and_rerun()
                except json.JSONDecodeError:
                    st.warning("Invalid JSON format. Please correct it.", icon="⚠️")

            st.subheader("🔬 Schema Retrieval Method")
            st.session_state.use_rag_for_schema = st.toggle(
                "Use RAG for relevant schema context (experimental)",
                value=st.session_state.use_rag_for_schema,
                key="rag_toggle_key",
                help="If your database schema is very large, RAG might help the LLM focus on relevant parts. Requires backend RAG setup."
            )

        st.divider()
        st.subheader("📜 History")
        if st.session_state.chat_history:
            for i, entry in enumerate(reversed(st.session_state.chat_history)):
                if st.button(f"Re-run: {entry['question'][:30]}...", key=f"history_q_{i}_{entry['question'][:5]}"):
                    # Set the question first
                    st.session_state.current_question_to_process = entry['question']
                    # Force input field update
                    st.session_state.user_question_input_key += 1
                    # Rerun to update UI
                    delay_and_rerun()
        else:
            st.caption("No questions asked yet.") 