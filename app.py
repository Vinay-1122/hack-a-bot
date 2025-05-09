import streamlit as st
import requests
import json
import pandas as pd
import base64
import re
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---
API_URL = process.env.BACKEND_URL

# --- Initialize session state ---
def init_session_state():
    defaults = {
        "chat_history": [],
        "current_semantic_schema": "{}",
        "gemini_api_key": "",
        "selected_model": "gemini-1.5-flash", # Default model
        "total_cost": 0.0,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "use_rag_for_schema": False,
        "python_code_editable": "", # For Python editor
        "show_python_editor": False,
        "user_question_input_key": 0, # To help reset input field
        "db_initialized": False, # Track if DB has data
        "fixed_code_available": False, # Track if fixed code is available
        "current_error_message": None, # Track current error message
        "fix_button_clicked": False, # Track if fix button was clicked
        "editor_key": 0, # Track editor updates
        "new_code": "", # Store new code temporarily
        "new_code_available": False # Flag for new code
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def handle_fix_button_click():
    """Callback function for the Fix with AI button"""
    try:
        with st.spinner("Hackbot is analyzing and fixing the code..."):
            fix_payload = {
                "code": st.session_state.python_code_editable,
                "error": st.session_state.current_error_message,
                "user_api_key": st.session_state.gemini_api_key or None,
                "model_name": st.session_state.selected_model
            }
            
            # Make the POST request
            fix_response = requests.post(
                f"{API_URL}/fix-python-code/",
                json=fix_payload,
                timeout=60
            )
            
            if fix_response.status_code == 200:
                fixed_code = fix_response.json().get("fixed_code")
                if fixed_code:
                    # Store the new code and set flag
                    st.session_state.new_code = fixed_code
                    st.session_state.new_code_available = True
                    # Increment editor key to force update
                    st.session_state.editor_key += 1
                    st.success("Code has been fixed by Hackbot. You can review and run it again.")
                    # Reset the fix button state
                    st.session_state.fix_button_clicked = False
                    st.session_state.current_error_message = None
                    # Force a rerun to update the editor
                    st.rerun()
                else:
                    st.error("Hackbot couldn't generate a fix for the code.")
                    st.session_state.fix_button_clicked = False
            else:
                st.error(f"Error from server: {fix_response.text}")
                st.session_state.fix_button_clicked = False
                
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting Hackbot fix: {e.response.text if e.response else str(e)}")
        st.session_state.fix_button_clicked = False
    except Exception as e:
        st.error(f"Error during Hackbot code fixing: {str(e)}")
        st.session_state.fix_button_clicked = False

init_session_state()


# --- Pricing (Example, replace with actuals from Gemini) ---
# Prices per 1 Million tokens for easier calculation
MODEL_PRICING_PER_MILLION_TOKENS = {
    "gemini-1.5-flash": {"input": 0.35, "output": 0.70}, # USD per 1M tokens
    "gemini-1.5-pro": {"input": 3.50, "output": 7.00},   # USD per 1M tokens
    "gemini-1.0-pro": {"input": 0.50, "output": 1.50},   # USD per 1M tokens
    # Add other models you support
}

# --- Helper Functions ---
def update_costs(input_tokens, output_tokens, model_name):
    """Updates total cost and token counts in session state."""
    st.session_state.total_input_tokens += input_tokens
    st.session_state.total_output_tokens += output_tokens
    
    pricing = MODEL_PRICING_PER_MILLION_TOKENS.get(model_name)
    if pricing:
        cost_for_this_call = (input_tokens / 1_000_000 * pricing["input"]) + \
                               (output_tokens / 1_000_000 * pricing["output"])
        st.session_state.total_cost += cost_for_this_call
    else:
        st.warning(f"Pricing for model '{model_name}' not found. Cost calculation may be inaccurate.")

def check_db_status():
    """Checks if the database has any tables, indicating it's initialized."""
    try:
        response = requests.get(f"{API_URL}/db-schema/")
        response.raise_for_status()
        schema = response.json()
        st.session_state.db_initialized = bool(schema) # True if schema is not empty
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API to check DB status: {e}")
        st.session_state.db_initialized = False # Assume not initialized on error

# --- UI Layout ---
st.set_page_config(layout="wide", page_title="HackBot Analytics")

# Check DB status on first load or if explicitly asked
if "db_initialized" not in st.session_state:
    check_db_status()


# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings & Context")

    st.session_state.gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key,
        help="If not provided here, the server's backup key will be used (if configured)."
    )

    available_models = list(MODEL_PRICING_PER_MILLION_TOKENS.keys())
    # Ensure current selected model is in the list, if not, default to first one
    if st.session_state.selected_model not in available_models and available_models:
        st.session_state.selected_model = available_models[0]
    elif not available_models: # Should not happen if MODEL_PRICING is populated
        st.error("No models configured for pricing.")
        st.stop()

    st.session_state.selected_model = st.selectbox(
        "Select LLM Model",
        options=available_models,
        index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
    )

    st.subheader("💰 Cost Estimation")
    cost_col1, cost_col2 = st.columns(2)
    cost_col1.metric("Total Input Tokens", f"{st.session_state.total_input_tokens:,}")
    cost_col2.metric("Total Output Tokens", f"{st.session_state.total_output_tokens:,}")
    st.metric("Estimated Total Cost", f"${st.session_state.total_cost:.4f} USD")
    st.caption("Costs are estimates based on token counts and model pricing.")

    st.divider()
    st.subheader("📚 Data Management")
    if not st.session_state.db_initialized:
        st.warning("Database appears empty. Please load data.")
    
    if st.button("🔄 Check/Refresh DB Status"):
        check_db_status()
        st.rerun()

    # Add clear database toggle and button
    clear_db = st.toggle("Clear existing data before loading new data", value=False, 
                        help="When enabled, all existing data will be deleted before loading new data")
    
    if clear_db:
        if st.button("🗑️ Clear All Data", type="secondary"):
            with st.spinner("Clearing database..."):
                try:
                    response = requests.post(f"{API_URL}/clear-database/")
                    response.raise_for_status()
                    st.success(response.json().get("message", "Database cleared successfully."))
                    st.session_state.current_semantic_schema = "{}" # Reset schema
                    check_db_status() # Update DB status
                    st.rerun()
                except requests.exceptions.RequestException as e:
                    st.error(f"Error clearing database: {e.response.text if e.response else str(e)}")

    with st.expander("Load from Kaggle"):
        kaggle_dataset_name = st.text_input("Kaggle Dataset (e.g., manjeetsingh/retaildataset)", key="kaggle_name_input")
        if st.button("Load Kaggle Dataset", key="load_kaggle_button"):
            if kaggle_dataset_name:
                with st.spinner(f"Loading {kaggle_dataset_name}... This may take a moment."):
                    try:
                        payload = {"dataset_name": kaggle_dataset_name}
                        response = requests.post(f"{API_URL}/load-kaggle-dataset/", json=payload, timeout=300) # Increased timeout
                        response.raise_for_status()
                        st.success(response.json().get("message", "Dataset loading initiated."))
                        st.session_state.current_semantic_schema = "{}" # Reset schema
                        check_db_status() # Update DB status
                        st.toast("Kaggle data loaded! Regenerate semantic schema if needed.", icon="📚")
                    except requests.exceptions.Timeout:
                        st.error(f"Timeout: Loading Kaggle dataset '{kaggle_dataset_name}' took too long.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error loading Kaggle dataset: {e.response.text if e.response and e.response.text else str(e)}")
            else:
                st.warning("Please enter a Kaggle dataset name.")

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
                        st.session_state.current_semantic_schema = "{}" # Reset schema
                        check_db_status() # Update DB status
                        st.toast("CSV data loaded! Regenerate semantic schema if needed.", icon="📄")
                    except requests.exceptions.Timeout:
                        st.error(f"Timeout: Uploading CSV '{uploaded_csv_file.name}' took too long.")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error uploading CSV: {e.response.text if e.response and e.response.text else str(e)}")
    st.divider()

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
                    response = requests.post(f"{API_URL}/generate-initial-semantic-schema/", json=payload)
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
            json.loads(edited_schema) # Validate JSON
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
            if st.button(f"Re-run: {entry['question'][:30]}...", key=f"history_q_{i}_{entry['question'][:5]}"): # Make key more unique
                # When re-running, preserve the settings of that run if possible, or use current
                st.session_state.user_question_input_key += 1 # Force re-render of input
                # For simplicity, we'll use current settings for re-run.
                # To use historical settings, you'd need to store them with chat history.
                st.session_state.current_question_to_process = entry['question']
                st.rerun()
    else:
        st.caption("No questions asked yet.")


# --- Main Page ---
st.title("🤖 Welcome to HackBot Analytics")
st.markdown("Ask questions about your data in natural language. Load data and configure settings in the sidebar.")

if not st.session_state.db_initialized:
    st.info("Welcome! Please load some data using the sidebar to get started with your analysis.")

# User input
# Use a key that changes to allow programmatic reset/update
user_question = st.text_input("Your question:", 
                              key=f"user_question_input_{st.session_state.user_question_input_key}", 
                              placeholder="e.g., What are the total sales per product category?",
                              value=st.session_state.get("current_question_to_process", ""))

# Clear the temporary question holder after using it
if "current_question_to_process" in st.session_state:
    del st.session_state["current_question_to_process"]


if st.button("🔍 Ask Gemini", use_container_width=True, type="primary", disabled=not st.session_state.db_initialized):
    if not st.session_state.db_initialized:
        st.error("Please load data into the database before asking questions.")
    elif user_question:
        # Validate semantic schema JSON before sending
        try:
            json.loads(st.session_state.current_semantic_schema)
        except json.JSONDecodeError:
            st.error("The semantic schema in the sidebar is not valid JSON. Please fix it before asking a question.")
            st.stop()

        with st.spinner(f"Thinking with {st.session_state.selected_model}..."):
            payload = {
                "question": user_question,
                "semantic_schema_json": st.session_state.current_semantic_schema,
                "model_name": st.session_state.selected_model,
                "user_api_key": st.session_state.gemini_api_key or None,
                "use_rag_for_schema": st.session_state.use_rag_for_schema
            }
            api_response_data = None
            try:
                response = requests.post(f"{API_URL}/query-analyzer/", json=payload, timeout=120)
                response.raise_for_status()
                api_response_data = response.json()

                # Update costs
                update_costs(
                    api_response_data.get("input_tokens_used", 0),
                    api_response_data.get("output_tokens_used", 0),
                    st.session_state.selected_model
                )
                # Store in history
                st.session_state.chat_history.append({
                    "question": user_question,
                    "response": api_response_data
                })
                # Reset Python editor state for new question
                st.session_state.python_code_editable = api_response_data.get("generated_python_script", "")
                st.session_state.show_python_editor = bool(st.session_state.python_code_editable)


            except requests.exceptions.Timeout:
                st.error(f"API request timed out. The server might be busy or the query too complex.")
            except requests.exceptions.HTTPError as e:
                err_detail = "No specific error detail from server."
                try:
                    err_detail = e.response.json().get("detail", err_detail)
                except: # pylint: disable=bare-except
                    err_detail = e.response.text if e.response is not None else str(e)
                st.error(f"API Error: {e.response.status_code} - {err_detail}")
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to API: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
        
        # Clear the input field after processing by changing its key
        st.session_state.user_question_input_key += 1
        st.rerun() # Rerun to display results and clear input

    elif not user_question:
        st.warning("Please enter a question.")

# Display latest response from chat history
if st.session_state.chat_history:
    latest_entry = st.session_state.chat_history[-1]
    api_response_data = latest_entry["response"]
    question_asked = latest_entry["question"]

    st.markdown(f"### Results for: *\"{question_asked}\"*")

    st.subheader("🧠 Thinking Steps (from LLM)")
    st.markdown(f"> {api_response_data.get('thinking_steps', 'N/A')}")

    analysis_type = api_response_data.get("analysis_type")

    if analysis_type == "sql":
        st.subheader("Generated SQL Query")
        st.code(api_response_data.get('generated_sql_query', 'No SQL query generated.'), language="sql")

        st.subheader("📊 Results / Plot (from SQL)")
        results_table = api_response_data.get('results_table')
        plot_b64 = api_response_data.get('plot_base64')
        
        if plot_b64:
            try:
                if ',' in plot_b64: header, encoded = plot_b64.split(",", 1)
                else: encoded = plot_b64
                plot_image_bytes = base64.b64decode(encoded)
                st.image(plot_image_bytes, caption=f"Generated Plot (Type: {api_response_data.get('chart_type', 'N/A')})", use_container_width=True)
            except Exception as img_e:
                st.error(f"Error displaying SQL plot: {img_e}")
                st.text("Base64 data for plot might be corrupted.")

        if results_table is not None: # Check for None explicitly, as empty list is valid
            if results_table:
                st.subheader("📋 Summary Table (from SQL)")
                try:
                    df_results = pd.DataFrame(results_table)
                    st.dataframe(df_results, use_container_width=True)
                except Exception as df_e:
                    st.error(f"Error displaying results table: {df_e}")
                    st.json(results_table)
            elif plot_b64 is None : # No table and no plot
                 st.info("SQL query executed, but no tabular data was returned and no plot was generated.")
        elif plot_b64 is None : # No plot and results_table is None (e.g. query error before execution)
            st.info("No tabular data or plot to display for SQL execution.")


    elif analysis_type == "python":
        st.subheader("🐍 Python Code for Advanced Analysis")
        st.info(api_response_data.get("reason_if_not_sql_or_python", "Python script generated for analysis."))
        
        generated_python_script = api_response_data.get("generated_python_script")
        if generated_python_script:
            st.session_state.show_python_editor = st.toggle(
                "Show/Edit Python Code", 
                value=st.session_state.show_python_editor, 
                key="python_editor_toggle_main"
            )

            if st.session_state.show_python_editor:
                required_packages = api_response_data.get("required_packages", [])
                if required_packages:
                    st.caption(f"Suggested packages for script: `{'`, `'.join(required_packages)}`")
                
                # Create a container for the code editor
                editor_container = st.container()
                
                # Update the code editor within the container
                with editor_container:
                    # Force a rerun if we have new code
                    if st.session_state.get("new_code_available", False):
                        st.session_state.python_code_editable = st.session_state.get("new_code", "")
                        st.session_state.new_code_available = False
                        st.rerun()
                    
                    st.session_state.python_code_editable = st.text_area(
                        "Python Code Editor:",
                        value=st.session_state.python_code_editable,
                        height=400,
                        key=f"python_code_editor_main_area_{st.session_state.editor_key}",
                        help="Edit the script if needed. Ensure the script prints results to stdout or saves a plot as 'plot.png'."
                    )

                if st.button("▶️ Run Python Script in Sandbox", key="run_python_button_main"):
                    if st.session_state.python_code_editable:
                        with st.spinner("Executing Python script in sandbox... This may take a moment."):
                            try:
                                py_payload = {
                                    "script": st.session_state.python_code_editable,
                                    "user_api_key": st.session_state.gemini_api_key or None
                                }
                                py_response = requests.post(f"{API_URL}/execute-python/", json=py_payload, timeout=60) # Timeout for execution
                                py_response.raise_for_status()
                                python_run_results = py_response.json()

                                st.subheader("🐍 Python Script Execution Output")
                                if python_run_results.get("status") == "success":
                                    st.success("Python script executed successfully.")
                                else:
                                    st.error("Python script execution encountered an error.")

                                if python_run_results.get("output"):
                                    st.markdown("##### Script Output (stdout):")
                                    
                                    # Attempt to parse and display DataFrame if present
                                    output_text = python_run_results.get("output")
                                    df_json_match = re.search(r"PYTHON_DF_RESULT_JSON_START>>>(.*?)<<<PYTHON_DF_RESULT_JSON_END", output_text, re.DOTALL)
                                    
                                    if df_json_match:
                                        df_json_str = df_json_match.group(1)
                                        # Remove the matched part from the main output to avoid duplication
                                        display_output_text = output_text.replace(df_json_match.group(0), "").strip()
                                        if display_output_text: # Show remaining stdout if any
                                             st.code(display_output_text, language="text")
                                        
                                        st.markdown("##### DataFrame Result from Python:")
                                        try:
                                            py_df = pd.read_json(df_json_str, orient='records')
                                            st.dataframe(py_df, use_container_width=True)
                                        except Exception as parse_df_e:
                                            st.error(f"Could not parse DataFrame from Python script output: {parse_df_e}")
                                            st.text("Raw DataFrame JSON (if any):\n" + df_json_str)
                                    else: # No special DataFrame output found
                                        st.code(output_text, language="text")

                                if python_run_results.get("error"):
                                    st.markdown("##### Script Error Output (stderr):")
                                    st.code(python_run_results.get("error"), language="text")
                                    st.session_state.current_error_message = python_run_results.get("error")
                                else:
                                    st.session_state.current_error_message = None

                                # Check for errors in the script output (from print_error_summary)
                                output_text = python_run_results.get("output", "")
                                if "Error Summary:" in output_text:
                                    st.markdown("##### Error from Script Output:")
                                    st.code(output_text, language="text")
                                    if not st.session_state.current_error_message:  # Only set if we don't have a direct error
                                        st.session_state.current_error_message = output_text

                                # Add Fix with AI button if there's any error
                                if st.session_state.current_error_message:
                                    if st.button("🤖 Fix with AI", 
                                               key=f"fix_with_ai_button_{st.session_state.user_question_input_key}",
                                               on_click=handle_fix_button_click):
                                        pass  # The callback will handle everything

                                python_plot_b64 = python_run_results.get("plot_base64")
                                if python_plot_b64:
                                    st.markdown("##### Plot from Python Script:")
                                    try:
                                        if ',' in python_plot_b64: header, encoded = python_plot_b64.split(",", 1)
                                        else: encoded = python_plot_b64
                                        py_plot_bytes = base64.b64decode(encoded)
                                        st.image(py_plot_bytes, caption="Plot generated by Python script", use_container_width=True)
                                    except Exception as py_img_e:
                                        st.error(f"Error displaying Python plot: {py_img_e}")
                                # Update chat history with python execution results (optional)
                                # latest_entry["response"]["executed_python_output"] = python_run_results.get("output")
                                # latest_entry["response"]["python_plot_base64"] = python_plot_b64

                            except requests.exceptions.Timeout:
                                st.error("Python script execution timed out.")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Error calling Python execution API: {e.response.text if e.response else str(e)}")
                            except Exception as e_py:
                                st.error(f"Error during Python script execution: {str(e_py)}")
                    else:
                        st.warning("Python code editor is empty.")
            else: # If toggle is off, show the initially generated script (read-only)
                 st.code(generated_python_script, language="python")


    elif analysis_type == "complex":
        st.warning(f"Complex Analysis: {api_response_data.get('reason_if_not_sql_or_python', 'This question requires a more complex approach than currently supported.')}")
    
    elif analysis_type == "error" or api_response_data.get("reason_if_not_sql_or_python"): # Catch all for other errors from LLM
        st.error(f"Analysis Error: {api_response_data.get('reason_if_not_sql_or_python', 'Could not process the request.')}")


    st.subheader("📝 LLM's Simple Summary / Inference")
    st.markdown(api_response_data.get('simple_summary_inference', 'N/A'))

    st.divider()
