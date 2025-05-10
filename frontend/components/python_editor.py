import streamlit as st
import requests
import re
import pandas as pd
import base64
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

def render_python_editor(generated_python_script):
    """Render the Python code editor with execution functionality."""
    if generated_python_script:
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

            if st.button("▶️ Run Python Script in Sandbox", key="run_python_button_main"):
                if st.session_state.python_code_editable:
                    with st.spinner("Executing Python script in sandbox... This may take a moment."):
                        try:
                            py_payload = {
                                "script": st.session_state.python_code_editable,
                                "user_api_key": st.session_state.gemini_api_key or None
                            }
                            py_response = requests.post(f"{API_URL}/execute-python/", json=py_payload, timeout=60)
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
                                    if display_output_text:  # Show remaining stdout if any
                                         st.code(display_output_text, language="text")
                                    
                                    st.markdown("##### DataFrame Result from Python:")
                                    try:
                                        py_df = pd.read_json(df_json_str, orient='records')
                                        st.dataframe(py_df, use_container_width=True)
                                    except Exception as parse_df_e:
                                        st.error(f"Could not parse DataFrame from Python script output: {parse_df_e}")
                                        st.text("Raw DataFrame JSON (if any):\n" + df_json_str)
                                else:  # No special DataFrame output found
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

                        except requests.exceptions.Timeout:
                            st.error("Python script execution timed out.")
                        except requests.exceptions.RequestException as e:
                            st.error(f"Error calling Python execution API: {e.response.text if e.response else str(e)}")
                        except Exception as e_py:
                            st.error(f"Error during Python script execution: {str(e_py)}")
                else:
                    st.warning("Python code editor is empty.")
        else:  # If toggle is off, show the initially generated script (read-only)
             st.code(generated_python_script, language="python") 