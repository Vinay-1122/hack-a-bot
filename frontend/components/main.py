import streamlit as st
import requests
import json
import pandas as pd
import base64
from config.constants import API_URL
from utils.cost_utils import update_costs
from components.python_editor import render_python_editor

def render_main_page():
    """Render the main page with all its components."""
    st.title("🤖 Welcome to HackBot Analytics")
    st.markdown("Ask questions about your data in natural language. Load data and configure settings in the sidebar.")

    if not st.session_state.db_initialized:
        st.info("Welcome! Please load some data using the sidebar to get started with your analysis.")

    # User input
    user_question = st.text_input(
        "Your question:", 
        key=f"user_question_input_{st.session_state.user_question_input_key}", 
        placeholder="e.g., What are the total sales per product category?",
        value=st.session_state.get("current_question_to_process", "")
    )

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
                    except:
                        err_detail = e.response.text if e.response is not None else str(e)
                    st.error(f"API Error: {e.response.status_code} - {err_detail}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to API: {e}")
                except Exception as e:
                    st.error(f"An unexpected error occurred: {str(e)}")
            
            # Clear the input field after processing by changing its key
            st.session_state.user_question_input_key += 1
            st.rerun()  # Rerun to display results and clear input

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

            if results_table is not None:  # Check for None explicitly, as empty list is valid
                if results_table:
                    st.subheader("📋 Summary Table (from SQL)")
                    try:
                        df_results = pd.DataFrame(results_table)
                        st.dataframe(df_results, use_container_width=True)
                    except Exception as df_e:
                        st.error(f"Error displaying results table: {df_e}")
                        st.json(results_table)
                elif plot_b64 is None:  # No table and no plot
                     st.info("SQL query executed, but no tabular data was returned and no plot was generated.")
            elif plot_b64 is None:  # No plot and results_table is None (e.g. query error before execution)
                st.info("No tabular data or plot to display for SQL execution.")

        elif analysis_type == "python":
            st.subheader("🐍 Python Code for Advanced Analysis")
            st.info(api_response_data.get("reason_if_not_sql_or_python", "Python script generated for analysis."))
            
            generated_python_script = api_response_data.get("generated_python_script")
            if generated_python_script:
                render_python_editor(generated_python_script)

        elif analysis_type == "complex":
            st.warning(f"Complex Analysis: {api_response_data.get('reason_if_not_sql_or_python', 'This question requires a more complex approach than currently supported.')}")
        
        elif analysis_type == "error" or api_response_data.get("reason_if_not_sql_or_python"):  # Catch all for other errors from LLM
            st.error(f"Analysis Error: {api_response_data.get('reason_if_not_sql_or_python', 'Could not process the request.')}")

        st.subheader("📝 LLM's Simple Summary / Inference")
        st.markdown(api_response_data.get('simple_summary_inference', 'N/A'))

        st.divider() 