from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
from core.llm import configure_gemini, generate_semantic_schema_prompt
from core.database import get_db_schema
try:
    import google.generativeai as genai
except ImportError:
    print("WARNING: google.generativeai not installed. LLM features will not work.")
    genai = None

router = APIRouter()

class SemanticSchemaRequest(BaseModel):
    model_name: str = "gemini-1.5-flash"
    user_api_key: Optional[str] = None

@router.post("/semantic-schema/")
async def generate_semantic_schema_endpoint(request: SemanticSchemaRequest):
    """
    Generates a semantic layer for the database schema.
    """
    try:
        # Configure Gemini
        model = configure_gemini(request.user_api_key, request.model_name)
        db_schema = get_db_schema()

        if not db_schema:  # Check if DB is empty or not initialized
            raise HTTPException(
                status_code=400,
                detail="Database is empty or not found. Please load data first."
            )

        # Generate semantic schema
        llm_prompt = generate_semantic_schema_prompt(db_schema)
        input_tokens_used = len(llm_prompt) // 4  # Rough estimate

        generation_config = genai.types.GenerationConfig(candidate_count=1)
        model_response = model.generate_content(llm_prompt, generation_config=generation_config)
        
        llm_output_json = model_response.text.strip().replace("```json", "").replace("```", "").strip()
        output_tokens_used = len(llm_output_json) // 4  # Rough estimate
        print(f"LLM output: {llm_output_json}")

        try:
            semantic_schema = json.loads(llm_output_json)
            return semantic_schema
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError from LLM output: {e}. Raw: {llm_output_json}")
            raise HTTPException(
                status_code=500,
                detail="Failed to parse semantic schema from LLM response. Please try again."
            )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /semantic-schema: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}") 