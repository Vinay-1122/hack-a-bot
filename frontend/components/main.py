import streamlit as st
import requests
import json
import pandas as pd
import base64
from config.constants import API_URL
from utils.cost_utils import update_costs
from components.python_editor import render_python_editor
from utils.session_state import init_session_state

def truncate_text(text, max_length=50):
    """Truncate text to a maximum length."""
    return text[:max_length] + "..." if len(text) > max_length else text

def render_main_page():
    """Render the main page with all its components."""
    init_session_state()
    st.title("🤖 Welcome to HackBot Analytics")
    st.markdown("Ask questions about your data in natural language. Load data and configure settings in the sidebar.")

    if not st.session_state.db_initialized:
        st.info("Welcome! Please load some data using the sidebar to get started with your analysis.")

    # Add RAG mode toggle at the top
    use_rag_mode = st.toggle("Additional Conversation Mode", 
                           help="Enable to use conversation history for better context-aware responses",
                           value=st.session_state.get("use_rag_mode", False))
    
    # Update session state
    st.session_state.use_rag_mode = use_rag_mode

    # User input
    user_question = st.text_input(
        "Your question:", 
        key=f"user_question_input_{st.session_state.user_question_input_key}", 
        placeholder="e.g., What are the total sales per product category?",
        value=st.session_state.get("current_question_to_process", "")
    )

    # Create a single row for button and checkboxes with adjusted proportions
    col1, col2, col3 = st.columns([0.8, 1, 1])
    
    with col1:
        ask_button = st.button("🔍 Ask Gemini", use_container_width=True, type="primary", disabled=not st.session_state.db_initialized)
    
    with col2:
        use_advanced_mode = st.checkbox("Advanced", help="Forces Python analysis with enhanced schema awareness")
    
    with col3:
        use_sql_only = st.checkbox("SQL Only", help="Forces SQL query generation with strict schema validation")

    if ask_button:
        if not st.session_state.db_initialized:
            st.error("Please load data into the database before asking questions.")
        elif user_question:
            # Reset UI state when asking a new question
            st.session_state.show_thinking_steps = False
            st.session_state.show_debug = False
            
            # Set auto-run to true for Python analysis
            st.session_state.auto_run_enabled = True
            
            # Modify the question based on mode
            processed_question = user_question
            if use_advanced_mode:
                processed_question += " Use python for this analysis, please be careful while generating the python code and use the helper functions that are available and make sure to use the schema details for proper references."
            elif use_sql_only:
                processed_question += " Use SQL for this analysis, please be careful while generating the SQL query and ensure it strictly follows the database schema. Do not use Python for this analysis."
            
            # Validate semantic schema JSON before sending
            try:
                json.loads(st.session_state.current_semantic_schema)
            except json.JSONDecodeError:
                st.error("The semantic schema in the sidebar is not valid JSON. Please fix it before asking a question.")
                st.stop()

            with st.spinner(f"Thinking with {st.session_state.selected_model}..."):
                payload = {
                    "question": processed_question,
                    "semantic_schema_json": st.session_state.current_semantic_schema,
                    "model_name": st.session_state.selected_model,
                    "user_api_key": st.session_state.gemini_api_key or None,
                    "use_rag_for_schema": st.session_state.use_rag_for_schema,
                    "use_rag_mode": use_rag_mode  # Add RAG mode to payload
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
                        "processed_question": processed_question,
                        "response": api_response_data,
                        "advanced_mode": use_advanced_mode,
                        "sql_only": use_sql_only,
                        "use_rag_mode": use_rag_mode
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
            st.rerun()

        elif not user_question:
            st.warning("Please enter a question.")

    # Display latest response from chat history
    if st.session_state.chat_history:
        latest_entry = st.session_state.chat_history[-1]
        api_response_data = latest_entry["response"]
        question_asked = latest_entry["question"]
        
        # Display follow-up questions at the top if available
        follow_up_questions = api_response_data.get("follow_up_questions", [])
        if follow_up_questions:
            st.markdown("#### 💡 Suggested Questions")
            cols = st.columns(3)  # Use 3 columns for better space utilization
            for i, question in enumerate(follow_up_questions):
                with cols[i % 3]:
                    truncated_question = truncate_text(question)
                    if st.button(f"Q{i+1}: {truncated_question}", 
                               key=f"follow_up_{i}", 
                               use_container_width=True,
                               help=question):  # Show full question on hover
                        st.session_state.current_question_to_process = question
                        st.session_state.user_question_input_key += 1
                        st.rerun()
            st.markdown("---")  # Add a separator

        # Create a more compact question display
        st.markdown(f"### 🔍 Question: {question_asked}")
        advanced_indicator = " (Advanced Mode)" if latest_entry.get("advanced_mode", False) else ""
        sql_only_indicator = " (SQL Only Mode)" if latest_entry.get("sql_only", False) else ""
        rag_indicator = " (RAG Mode)" if latest_entry.get("use_rag_mode", False) else ""
        st.markdown(f"*{advanced_indicator}{sql_only_indicator}{rag_indicator}*")

        # Debug section (collapsed by default)
        if st.toggle("Show Debug Information", value=st.session_state.get("show_debug", False)):
            st.json(api_response_data)

        analysis_type = api_response_data.get("analysis_type")

        if analysis_type == "sql":
            st.markdown("#### 📊 SQL Analysis")
            st.code(api_response_data.get('generated_sql_query', 'No SQL query generated.'), language="sql")

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

            if results_table is not None:
                if results_table:
                    st.markdown("#### 📋 Results Table")
                    try:
                        df_results = pd.DataFrame(results_table)
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Generate inference automatically
                        with st.spinner("Generating data insights..."):
                            try:
                                inference_payload = {
                                    "data": results_table,
                                    "max_rows": 50,
                                    "user_api_key": st.session_state.gemini_api_key or None,
                                    "question": question_asked  # Add the question to the payload
                                }
                                inference_response = requests.post(
                                    f"{API_URL}/generate-inference/",
                                    json=inference_payload,
                                    timeout=60
                                )
                                inference_response.raise_for_status()
                                inference_data = inference_response.json()
                                
                                if inference_data["status"] == "success":
                                    st.markdown("#### 📊 Data Insights")
                                    st.markdown(inference_data["summary"])
                            except Exception as e:
                                st.error(f"Error generating data insights: {str(e)}")
                    except Exception as df_e:
                        st.error(f"Error displaying results table: {df_e}")
                        st.json(results_table)
                elif plot_b64 is None:
                    st.info("SQL query executed, but no tabular data was returned and no plot was generated.")
            elif plot_b64 is None:
                st.info("No tabular data or plot to display for SQL execution.")

        elif analysis_type == "python":
            st.markdown("#### 🐍 Python Analysis")
            generated_python_script = api_response_data.get("generated_python_script")
            if generated_python_script:
                render_python_editor(generated_python_script)

        elif analysis_type == "complex":
            st.warning(f"Complex Analysis: {api_response_data.get('reason_if_not_sql_or_python', 'This question requires a more complex approach than currently supported.')}")
        
        elif analysis_type == "error" or api_response_data.get("reason_if_not_sql_or_python"):
            st.error(f"Analysis Error: {api_response_data.get('reason_if_not_sql_or_python', 'Could not process the request.')}")

        st.divider()