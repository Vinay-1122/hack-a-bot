from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import re
import google.generativeai as genai
from core.llm import configure_gemini, generate_query_analysis_prompt
from core.database import get_db_schema, execute_sql_query
from core.vector_store import get_relevant_schema_from_rag, store_conversation, get_relevant_conversations, clear_conversation_history
from utils.plotting import generate_plot

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    semantic_schema_json: str  # Expecting JSON string from frontend
    model_name: str = "gemini-1.5-flash"
    user_api_key: Optional[str] = None
    use_rag_for_schema: bool = False  # For RAG toggle
    use_rag_mode: bool = False  # New field for RAG mode

@router.post("/query-analyzer/")
async def analyze_query_endpoint(request: QueryRequest):
    """
    Analyzes a natural language question, generates SQL or Python,
    executes SQL if applicable, and suggests visualizations.
    """
    try:
        # Configure Gemini
        model = configure_gemini(request.user_api_key, request.model_name)
        db_schema = get_db_schema()

        if not db_schema:  # Check if DB is empty or not initialized
            return {
                "thinking_steps": "Database is empty or not found. Please load data first.",
                "analysis_type": "error",
                "generated_sql_query": None,
                "generated_python_script": None,
                "required_packages": [],
                "reason_if_not_sql_or_python": "Database schema is not available. Load data before asking questions.",
                "results_table": None,
                "plot_base64": None,
                "simple_summary_inference": "Cannot process query as the database is not initialized or empty.",
                "chart_type": "table",
                "input_tokens_used": 0,
                "output_tokens_used": 0,
                "follow_up_questions": []
            }

        try:
            semantic_schema = json.loads(request.semantic_schema_json)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format for semantic_schema_json.")

        # Get schema context
        schema_context_for_llm = ""
        if request.use_rag_for_schema:
            schema_context_for_llm = get_relevant_schema_from_rag(request.question, db_schema)
        else:
            schema_context_for_llm = f"DATABASE SCHEMA:\n{json.dumps(db_schema, indent=2)}"

        # Get relevant conversation history if RAG mode is enabled
        conversation_context = ""
        if request.use_rag_mode:
            relevant_conversations = get_relevant_conversations(request.question)
            if relevant_conversations:
                conversation_context = "\n\nRelevant previous conversations:\n"
                for conv in relevant_conversations:
                    conversation_context += f"Q: {conv['question']}\n"
                    conversation_context += f"A: {conv['response']}\n"
                    if conv.get('results'):
                        conversation_context += f"Results: {json.dumps(conv['results'][:5])}\n"
                    conversation_context += "---\n"

        # Generate and execute query
        llm_prompt = generate_query_analysis_prompt(
            request.question, 
            schema_context_for_llm, 
            semantic_schema,
            conversation_context if request.use_rag_mode else None
        )
        input_tokens_used = len(llm_prompt) // 4

        generation_config = genai.types.GenerationConfig(candidate_count=1)
        model_response = model.generate_content(llm_prompt, generation_config=generation_config)
        
        llm_output_json = model_response.text
        output_tokens_used = len(llm_output_json) // 4  # Rough estimate

        try:
            llm_response_data = json.loads(llm_output_json)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError from LLM output: {e}. Raw: {llm_output_json}")
            # Try multiple strategies to extract valid JSON
            json_strategies = [
                # Strategy 1: Look for JSON within ```json ... ```
                lambda text: re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL),
                # Strategy 2: Look for JSON within ``` ... ```
                lambda text: re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL),
                # Strategy 3: Look for JSON between curly braces
                lambda text: re.search(r"(\{.*\})", text, re.DOTALL),
                # Strategy 4: Try to fix common JSON formatting issues
                lambda text: re.sub(r'(\w+):', r'"\1":', text)  # Add quotes to keys
            ]
            
            for strategy in json_strategies:
                try:
                    if callable(strategy):
                        # For regex strategies
                        match = strategy(llm_output_json)
                        if match:
                            json_str = match.group(1)
                            # Clean up the JSON string
                            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)  # Remove control characters
                            json_str = re.sub(r',\s*}', '}', json_str)  # Remove trailing commas
                            json_str = re.sub(r',\s*]', ']', json_str)  # Remove trailing commas in arrays
                            llm_response_data = json.loads(json_str)
                            print("Successfully parsed JSON using strategy")
                            break
                    else:
                        # For string manipulation strategies
                        modified_json = strategy(llm_output_json)
                        llm_response_data = json.loads(modified_json)
                        print("Successfully parsed JSON using string manipulation")
                        break
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            if not llm_response_data:
                # If all strategies fail, construct a minimal valid response
                print("All JSON parsing strategies failed. Constructing minimal valid response.")
                llm_response_data = {
                    "thinking_steps": "Error parsing LLM response. Please try again.",
                    "analysis_type": "error",
                    "generated_sql_query": None,
                    "generated_python_script": None,
                    "required_packages": [],
                    "reason_if_not_sql_or_python": "Failed to parse LLM response. Raw output: " + llm_output_json[:200] + "...",
                    "results_table": None,
                    "plot_base64": None,
                    "simple_summary_inference": "Error in processing the request. Please try again.",
                    "chart_type": "table",
                    "input_tokens_used": input_tokens_used,
                    "output_tokens_used": output_tokens_used
                }

        # Validate required fields in the response
        required_fields = ["thinking_steps", "analysis_type"]
        missing_fields = [field for field in required_fields if field not in llm_response_data]
        if missing_fields:
            print(f"Warning: Missing required fields in LLM response: {missing_fields}")
            # Add missing fields with default values
            for field in missing_fields:
                llm_response_data[field] = "Not provided by LLM"

        # Validate analysis_type
        valid_analysis_types = ["sql", "python", "complex", "error"]
        if llm_response_data.get("analysis_type") not in valid_analysis_types:
            print(f"Warning: Invalid analysis_type: {llm_response_data.get('analysis_type')}")
            llm_response_data["analysis_type"] = "error"
            llm_response_data["reason_if_not_sql_or_python"] = f"Invalid analysis type returned by LLM: {llm_response_data.get('analysis_type')}"

        # Ensure consistent null values
        for key in ["generated_sql_query", "generated_python_script", "reason_if_not_sql_or_python"]:
            if llm_response_data.get(key) == "null" or llm_response_data.get(key) == "":
                llm_response_data[key] = None

        # Validate and clean up chart-related fields
        valid_chart_types = ["bar", "line", "scatter", "pie", "hist", "table", "box"]
        chart_type = llm_response_data.get("chart_type", "table")
        if chart_type not in valid_chart_types:
            print(f"Warning: Invalid chart_type: {chart_type}")
            llm_response_data["chart_type"] = "table"

        # Process the analysis
        analysis_type = llm_response_data.get("analysis_type")
        sql_query = llm_response_data.get("generated_sql_query")
        python_script = llm_response_data.get("generated_python_script")
        
        results_df = None
        results_data = None
        plot_base64 = None
        analysis_summary = llm_response_data.get("thinking_steps", "No thinking steps provided.")
        follow_up_questions = []

        if analysis_type == "sql" and sql_query:
            try:
                results_df = execute_sql_query(sql_query)
                results_data = results_df.to_dict(orient="records") if results_df is not None else []
                
                if not results_data:
                    analysis_summary += "\nSQL query executed successfully but returned no data."
                else:
                    analysis_summary += f"\nSuccessfully executed SQL. Found {len(results_data)} records."
                    chart_type = llm_response_data.get("chart_type", "table")
                    if chart_type != "table" and not results_df.empty:
                        plot_base64 = generate_plot(
                            results_df,
                            chart_type,
                            x_column=llm_response_data.get("chart_x_column"),
                            y_column=llm_response_data.get("chart_y_column"),
                            title=llm_response_data.get("chart_title", "Query Results")
                        )
                        if plot_base64:
                            analysis_summary += " Chart generated."
                        else:
                            analysis_summary += " Could not generate chart from SQL results."

                # Generate follow-up questions using LLM
                follow_up_prompt = f"""
                Based on the following:
                1. Original question: {request.question}
                2. Database schema: {schema_context_for_llm}
                3. Query results: {json.dumps(results_data[:5])}  # First 5 rows for context
                
                Generate 3 relevant follow-up questions that would help explore the data further.
                Questions should be:
                - Specific and actionable
                - Build upon insights from the current query
                - Consider the data relationships and patterns
                - Help uncover deeper insights
                - Be natural and conversational
                - Focus on the most interesting aspects of the data
                - Avoid redundant or obvious questions
                
                Format as a JSON array of strings.
                Example format:
                [
                    "What is the trend of [metric] over time?",
                    "How does [category] compare to other categories?",
                    "What factors influence [outcome] the most?"
                ]
                
                Respond ONLY with the JSON array, no additional text.
                """
                
                follow_up_response = model.generate_content(follow_up_prompt)
                try:
                    # Clean up the response text
                    response_text = follow_up_response.text.strip()
                    # Remove any markdown code block markers
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                    # Remove any leading/trailing quotes
                    response_text = response_text.strip('"\'')
                    
                    follow_up_questions = json.loads(response_text)
                    if not isinstance(follow_up_questions, list):
                        follow_up_questions = []
                    # Ensure we have at most 3 questions
                    follow_up_questions = follow_up_questions[:3]
                except Exception as e:
                    print(f"Error parsing follow-up questions: {e}")
                    print(f"Raw response: {follow_up_response.text}")
                    follow_up_questions = []

            except Exception as e:
                analysis_summary += f"\nError executing SQL: {str(e)}"
                llm_response_data["analysis_type"] = "error_sql_execution"
                llm_response_data["reason_if_not_sql_or_python"] = f"SQL Execution Failed: {str(e)}"

        elif analysis_type == "python":
            analysis_summary += "\nPython script generated. Ready for execution in the frontend editor."
        
        elif analysis_type == "complex":
            analysis_summary = llm_response_data.get("reason_if_not_sql_or_python", "The question is too complex for standard SQL/Python analysis with current setup.")

        # Store conversation in vector store if RAG mode is enabled
        if request.use_rag_mode:
            conversation_data = {
                "question": request.question,
                "response": llm_output_json,
                "results": results_data if 'results_data' in locals() else None,
                "analysis_type": analysis_type,
                "sql_query": sql_query if 'sql_query' in locals() else None,
                "python_script": python_script if 'python_script' in locals() else None
            }
            store_conversation(conversation_data)

        return {
            "thinking_steps": llm_response_data.get("thinking_steps"),
            "analysis_type": llm_response_data.get("analysis_type"),
            "generated_sql_query": sql_query,
            "generated_python_script": python_script,
            "required_packages": llm_response_data.get("required_packages", []),
            "reason_if_not_sql_or_python": llm_response_data.get("reason_if_not_sql_or_python"),
            "results_table": results_data,
            "plot_base64": plot_base64,
            "simple_summary_inference": analysis_summary,
            "chart_type": llm_response_data.get("chart_type", "table"),
            "chart_x_column": llm_response_data.get("chart_x_column"),
            "chart_y_column": llm_response_data.get("chart_y_column"),
            "chart_title": llm_response_data.get("chart_title"),
            "input_tokens_used": input_tokens_used,
            "output_tokens_used": output_tokens_used,
            "executed_python_output": None,
            "python_plot_base64": None,
            "follow_up_questions": follow_up_questions
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /query-analyzer: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

@router.post("/clear-conversation-history/")
async def clear_conversation_history_endpoint():
    """Clear the conversation history from the vector store."""
    try:
        clear_conversation_history()
        return {"message": "Conversation history cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing conversation history: {str(e)}") 