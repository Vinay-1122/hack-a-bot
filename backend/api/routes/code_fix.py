from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import re
from core.llm import configure_gemini, generate_code_fix_prompt
from utils.security import check_dangerous_operations, sanitize_python_script
try:
    import google.generativeai as genai
except ImportError:
    print("WARNING: google.generativeai not installed. LLM features will not work.")
    genai = None


router = APIRouter()

class CodeFixRequest(BaseModel):
    code: str
    error_message: str
    model_name: str = "gemini-1.5-flash"
    user_api_key: Optional[str] = None

@router.post("/code-fix/")
async def fix_code_endpoint(request: CodeFixRequest):
    """
    Fixes Python code with error handling and visualization.
    """
    try:
        # Configure Gemini
        model = configure_gemini(request.user_api_key, request.model_name)

        # Check for dangerous operations
        dangerous_ops = check_dangerous_operations(request.code)
        if dangerous_ops:
            raise HTTPException(
                status_code=400,
                detail=f"Dangerous operations detected: {', '.join(dangerous_ops)}"
            )

        # Generate code fix
        llm_prompt = generate_code_fix_prompt(request.code, request.error_message)
        input_tokens_used = len(llm_prompt) // 4  # Rough estimate

        model_response = model.generate_content(llm_prompt)
        
        llm_output_json = model_response.text.strip()
        output_tokens_used = len(llm_output_json) // 4  # Rough estimate

        
        # Clean up the response to ensure it's just the code
        llm_output_json = re.sub(r'```python\s*', '', llm_output_json)
        llm_output_json = re.sub(r'\s*```', '', llm_output_json)

        return {
            "fixed_code": llm_output_json,
            "input_tokens_used": input_tokens_used,
            "output_tokens_used": output_tokens_used
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /code-fix: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}") 