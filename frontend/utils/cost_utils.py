import streamlit as st
from config.constants import MODEL_PRICING_PER_MILLION_TOKENS

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

def get_cost_summary():
    """Returns a dictionary with cost summary information."""
    return {
        "total_input_tokens": st.session_state.total_input_tokens,
        "total_output_tokens": st.session_state.total_output_tokens,
        "total_cost": st.session_state.total_cost
    } 