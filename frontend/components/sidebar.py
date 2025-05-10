import streamlit as st
import requests
import json
from config.constants import API_URL, MODEL_PRICING_PER_MILLION_TOKENS
from utils.cost_utils import get_cost_summary

def check_db_status():
    """Checks if the database has any tables, indicating it's initialized."""
    try:
        response = requests.get(f"{API_URL}/db-schema/")
        response.raise_for_status()
        schema = response.json()
        st.session_state.db_initialized = bool(schema)  # True if schema is not empty
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API to check DB status: {e}")
        st.session_state.db_initialized = False  # Assume not initialized on error

def render_sidebar():
    """Render the sidebar with all its components."""
    with st.sidebar:
        st.title("⚙️ Settings & Context")

        # API Key Input
        st.session_state.gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.gemini_api_key,
            help="If not provided here, the server's backup key will be used (if configured)."
        )

        # Model Selection
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

        # Cost Metrics
        st.subheader("💰 Cost Estimation")
        cost_summary = get_cost_summary()
        cost_col1, cost_col2 = st.columns(2)
        cost_col1.metric("Total Input Tokens", f"{cost_summary['total_input_tokens']:,}")
        cost_col2.metric("Total Output Tokens", f"{cost_summary['total_output_tokens']:,}")
        st.metric("Estimated Total Cost", f"${cost_summary['total_cost']:.4f} USD")
        st.caption("Costs are estimates based on token counts and model pricing.")

        st.divider()
        st.subheader("📚 Data Management")
        if not st.session_state.db_initialized:
            st.warning("Database appears empty. Please load data.")
        
        if st.button("🔄 Check/Refresh DB Status"):
            check_db_status()
            st.rerun()

        # Data Loading Options
        clear_db = st.toggle("Clear existing data before loading new data", value=False, 
                           help="When enabled, all existing data will be deleted before loading new data")
        
        if clear_db:
            if st.button("🗑️ Clear All Data", type="secondary"):
                with st.spinner("Clearing database..."):
                    try:
                        response = requests.post(f"{API_URL}/clear-database/")
                        response.raise_for_status()
                        st.success(response.json().get("message", "Database cleared successfully."))
                        st.session_state.current_semantic_schema = "{}"  # Reset schema
                        check_db_status()  # Update DB status
                        st.rerun()
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error clearing database: {e.response.text if e.response else str(e)}")

        # Kaggle Dataset Loading
        with st.expander("Load from Kaggle"):
            kaggle_dataset_name = st.text_input("Kaggle Dataset (e.g., manjeetsingh/retaildataset)", key="kaggle_name_input")
            if st.button("Load Kaggle Dataset", key="load_kaggle_button"):
                if kaggle_dataset_name:
                    with st.spinner(f"Loading {kaggle_dataset_name}... This may take a moment."):
                        try:
                            payload = {"dataset_name": kaggle_dataset_name}
                            response = requests.post(f"{API_URL}/load-kaggle-dataset/", json=payload, timeout=300)
                            response.raise_for_status()
                            st.success(response.json().get("message", "Dataset loading initiated."))
                            st.session_state.current_semantic_schema = "{}"  # Reset schema
                            check_db_status()  # Update DB status
                            st.toast("Kaggle data loaded! Regenerate semantic schema if needed.", icon="📚")
                        except requests.exceptions.Timeout:
                            st.error(f"Timeout: Loading Kaggle dataset '{kaggle_dataset_name}' took too long.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error loading Kaggle dataset: {e.response.text if e.response and e.response.text else str(e)}")
                else:
                    st.warning("Please enter a Kaggle dataset name.")

        # CSV Upload
        with st.expander("Upload your own CSV"):
            uploaded_csv_file = st.file_uploader("Choose a CSV file", type=["csv"], key="csv_uploader_input")
            if uploaded_csv_file is not None:
                if st.button("Load Uploaded CSV", key="upload_csv_button"):
                    with st.spinner(f"Uploading and loading {uploaded_csv_file.name}..."):
                        files = {"file": (uploaded_csv_file.name, uploaded_csv_file.getvalue(), "text/csv")}
                        try:
                            response = requests.post(f"{API_URL}/upload-csv/", files=files, timeout=120)
                            response.raise_for_status()
                            st.success(response.json().get("message", "CSV processed."))
                            st.session_state.current_semantic_schema = "{}"  # Reset schema
                            check_db_status()  # Update DB status
                            st.toast("CSV data loaded! Regenerate semantic schema if needed.", icon="📄")
                        except requests.exceptions.Timeout:
                            st.error(f"Timeout: Uploading CSV '{uploaded_csv_file.name}' took too long.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error uploading CSV: {e.response.text if e.response and e.response.text else str(e)}")

        st.divider()

        # Semantic Schema Section
        st.subheader("🧠 Semantic Schema (Business Context)")
        st.caption("Define business terms, column descriptions, and relationships. (JSON format)")

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
                        response = requests.post(f"{API_URL}/semantic-schema/", json=payload)
                        response.raise_for_status()
                        generated_schema = response.json()
                        st.session_state.current_semantic_schema = json.dumps(generated_schema, indent=2)
                        st.success("Initial schema generated!")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error generating schema: {e.response.json().get('detail') if e.response else str(e)}")
                    except Exception as e:
                        st.error(f"An unexpected error occurred: {str(e)}")
        
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
                st.toast("Semantic schema updated locally.", icon="✅")
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
                    st.session_state.user_question_input_key += 1
                    st.session_state.current_question_to_process = entry['question']
                    st.rerun()
        else:
            st.caption("No questions asked yet.") 