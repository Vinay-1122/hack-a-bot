import streamlit as st
import requests
import re
import pandas as pd
import base64
import time
from config.constants import API_URL

def handle_fix_button_click():
    """Callback function for the Fix with AI button"""
    try:
        with st.spinner("Hackbot is analyzing and fixing the code..."):
            fix_payload = {
                "code": st.session_state.python_code_editable,
                "error_message": st.session_state.current_error_message,
                "user_api_key": st.session_state.gemini_api_key or None,
                "model_name": st.session_state.selected_model
            }
            
            # Make the POST request
            fix_response = requests.post(
                f"{API_URL}/code-fix/",
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
                    time.sleep(1)  # Add delay to prevent keyboard interrupt
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

def execute_python_script(code, max_retries=4):
    """Execute Python script with auto-retry functionality."""
    retry_count = 0
    while retry_count < max_retries:
        try:
            py_payload = {
                "script": code,
                "user_api_key": st.session_state.gemini_api_key or None
            }
            py_response = requests.post(f"{API_URL}/execute-python/", json=py_payload, timeout=60)
            py_response.raise_for_status()
            python_run_results = py_response.json()

            # Check for error summary in output
            output_text = python_run_results.get("output", "")
            has_error_summary = "Error Summary:" in output_text
            error_message = python_run_results.get("error")

            # KeyboardInterrupt: re-run without incrementing retry
            if error_message and "KeyboardInterrupt" in error_message:
                st.warning("KeyboardInterrupt detected. Re-running script...")
                time.sleep(1)
                continue

            if python_run_results.get("status") == "success" and not has_error_summary:
                return python_run_results, None
            else:
                if not error_message and has_error_summary:
                    error_message = output_text
                if retry_count < max_retries - 1:
                    st.warning(f"Attempt {retry_count + 1} failed. Retrying with AI fix...")
                    fix_payload = {
                        "code": code,
                        "error_message": error_message,
                        "user_api_key": st.session_state.gemini_api_key or None,
                        "model_name": st.session_state.selected_model
                    }
                    fix_response = requests.post(f"{API_URL}/code-fix/", json=fix_payload, timeout=60)
                    if fix_response.status_code == 200:
                        fixed_code = fix_response.json().get("fixed_code")
                        if fixed_code:
                            code = fixed_code
                            time.sleep(1)
                            retry_count += 1
                            continue
                return python_run_results, error_message
        except Exception as e:
            if retry_count < max_retries - 1:
                st.warning(f"Attempt {retry_count + 1} failed. Retrying...")
                time.sleep(1)
                retry_count += 1
                continue
            return None, str(e)
    return None, "Maximum retry attempts reached"

def execute_python_script_debug(code):
    """Execute Python script in debug mode - no auto-retries."""
    try:
        py_payload = {
            "script": code,
            "user_api_key": st.session_state.gemini_api_key or None
        }
        py_response = requests.post(f"{API_URL}/execute-python/", json=py_payload, timeout=60)
        py_response.raise_for_status()
        python_run_results = py_response.json()
        error_message = python_run_results.get("error")
        if error_message and "KeyboardInterrupt" in error_message:
            st.warning("KeyboardInterrupt detected. Re-running script...")
            time.sleep(1)
            return execute_python_script_debug(code)
        return python_run_results, None
    except Exception as e:
        return None, str(e)

def render_python_editor(generated_python_script):
    """Render the Python code editor with execution functionality."""
    if generated_python_script:
        # Add edit/debug mode toggle
        st.session_state.edit_mode = st.toggle(
            "Edit/Debug Mode", 
            value=st.session_state.get("edit_mode", False),
            help="Toggle between automatic execution with AI fixes and manual edit/debug mode"
        )

        if st.session_state.edit_mode:
            # Show code editor in edit mode
            st.session_state.show_python_editor = st.toggle(
                "Show/Edit Python Code", 
                value=st.session_state.show_python_editor, 
                key="python_editor_toggle_main"
            )

            if st.session_state.show_python_editor:
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

                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button("▶️ Run Python Script", key="run_python_button_main"):
                        if st.session_state.python_code_editable:
                            with st.spinner("Executing Python script..."):
                                python_run_results, error = execute_python_script_debug(st.session_state.python_code_editable)
                                
                                if error:
                                    st.error(f"Error executing Python script: {error}")
                                    st.session_state.current_error_message = error
                                else:
                                    display_python_results(python_run_results)
                        else:
                            st.warning("Python code editor is empty.")
                
                with col2:
                    if st.session_state.current_error_message:
                        if st.button("🤖 Fix with AI", 
                                   key=f"fix_with_ai_button_{st.session_state.user_question_input_key}",
                                   on_click=handle_fix_button_click):
                            pass
        else:
            # Automatic execution mode
            st.info("Running in automatic mode. The code will be executed and automatically fixed if needed.")
            with st.spinner("Executing Python script with automatic fixes..."):
                python_run_results, error = execute_python_script(generated_python_script)
                
                if error:
                    st.error(f"Failed to execute script after maximum retries: {error}")
                else:
                    display_python_results(python_run_results)

def display_python_results(python_run_results):
    """Display the results of Python script execution."""
    st.markdown(
        """
        <style>
        .full-width-output .stDataFrame, .full-width-output .stMarkdown, .full-width-output .stCodeBlock, .full-width-output .stImage {
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.container():
        st.subheader("🐍 Python Script Execution Output")
        if python_run_results.get("status") == "success":
            st.success("Python script executed successfully.")
        else:
            st.error("Python script execution encountered an error.")

        output_text = python_run_results.get("output", "")
        # Remove 'None' output
        if output_text is not None and output_text.strip().lower() == "none":
            output_text = ""

        # Find all DataFrame JSON blocks
        df_json_matches = list(re.finditer(r"PYTHON_DF_RESULT_JSON_START>>>(.*?)<<<PYTHON_DF_RESULT_JSON_END", output_text, re.DOTALL))
        shown_any_table = False
        for match in df_json_matches:
            df_json_str = match.group(1)
            try:
                py_df = pd.read_json(df_json_str, orient='records')
                st.dataframe(py_df, use_container_width=True, height=400)
                shown_any_table = True
            except Exception as parse_df_e:
                st.error(f"Could not parse DataFrame from Python script output: {parse_df_e}")
                st.text("Raw DataFrame JSON (if any):\n" + df_json_str)
        # Remove all DataFrame JSON blocks from output_text
        for match in df_json_matches:
            output_text = output_text.replace(match.group(0), "")

        # Show printed output behind a toggle if any
        if output_text.strip():
            with st.expander("Show Raw Output (stdout)", expanded=False):
                st.code(output_text.strip(), language="text")

        # Show error output if any
        if python_run_results.get("error"):
            st.markdown("##### Script Error Output (stderr):")
            st.code(python_run_results.get("error"), language="text")
            st.session_state.current_error_message = python_run_results.get("error")
        else:
            st.session_state.current_error_message = None

        # Show error summary if present in output
        if "Error Summary:" in output_text:
            st.markdown("##### Error from Script Output:")
            st.code(output_text, language="text")
            if not st.session_state.current_error_message:
                st.session_state.current_error_message = output_text

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