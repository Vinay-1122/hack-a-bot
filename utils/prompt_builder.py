import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class DataFrameSchema:
    """Schema information for a DataFrame."""
    name: str
    columns: List[str]
    dtypes: Dict[str, str]
    shape: tuple[int, int]
    sample_data: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


@dataclass
class DatabaseSchema:
    """Schema information for a database."""
    tables: List[str]
    table_schemas: Dict[str, List[str]]
    relationships: Optional[Dict[str, Any]] = None


class PromptBuilder:
    """Utility class for building LLM prompts for code generation and validation."""
    
    def __init__(self):
        self.base_instructions = {
            "plotting_libraries": "pandas, plotly.express, plotly.graph_objects, and dash",
            "output_path": "/app/data/output.html",
            "security_constraints": [
                "Do not use os, subprocess, requests, urllib, or any network libraries",
                "Do not access files outside of /app/data/ directory", 
                "Do not use eval() or exec() functions",
                "Only use the provided data sources"
            ]
        }
    
    def build_code_generation_prompt(
        self,
        user_question: str,
        dataframes: List[DataFrameSchema],
        db_schema: Optional[DatabaseSchema] = None,
        provider: str = "gemini",
        additional_context: Optional[str] = None
    ) -> str:
        """Build a prompt for code generation."""
        if provider.lower() == "bedrock":
            return self._build_bedrock_generation_prompt(
                user_question, dataframes, db_schema, additional_context
            )
        else:
            return self._build_gemini_generation_prompt(
                user_question, dataframes, db_schema, additional_context
            )
    
    def _build_gemini_generation_prompt(
        self,
        user_question: str,
        dataframes: List[DataFrameSchema],
        db_schema: Optional[DatabaseSchema],
        additional_context: Optional[str]
    ) -> str:
        """Build Gemini-optimized generation prompt."""
        prompt = f"""# Data Visualization Code Generation Task

## User Request
{user_question}

## Available Data Sources

### DataFrames
"""
        # Add DataFrame information
        for i, df in enumerate(dataframes):
            prompt += f"""
**DataFrame {i+1}: {df.name}**
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Columns: {', '.join(df.columns)}
- Data Types: {json.dumps(df.dtypes, indent=2)}
"""
            if df.sample_data:
                prompt += f"- Sample Data: {json.dumps(df.sample_data, indent=2)}\n"
            if df.description:
                prompt += f"- Description: {df.description}\n"
        
        # Add database schema if available
        if db_schema:
            prompt += f"""
### Database Schema
- Tables: {', '.join(db_schema.tables)}
"""
            for table, columns in db_schema.table_schemas.items():
                prompt += f"- {table}: {', '.join(columns)}\n"
        
        # Add requirements and constraints
        prompt += f"""
## Requirements
1. **Output**: Generate Python code that creates a visualization and saves it as HTML
2. **Libraries**: Use {self.base_instructions['plotting_libraries']}
3. **Output Path**: Save the final plot to `{self.base_instructions['output_path']}`
4. **Data Loading**: DataFrames are already loaded and available as variables with their respective names

## Security Constraints
"""
        for constraint in self.base_instructions['security_constraints']:
            prompt += f"- {constraint}\n"
        
        if additional_context:
            prompt += f"\n## Additional Context\n{additional_context}\n"
        
        prompt += """
## Code Generation Instructions
1. Start with necessary imports
2. Load and explore the data if needed
3. Create appropriate visualizations based on the user's request
4. Use plotly for interactive plots when possible
5. For multiple DataFrames, consider creating a dashboard with Dash
6. Save the final visualization as HTML to the specified output path
7. Include proper error handling
8. Add comments explaining key steps

## Output Format
Please provide complete, executable Python code wrapped in ```python code blocks.
"""
        return prompt
    
    def _build_bedrock_generation_prompt(
        self,
        user_question: str,
        dataframes: List[DataFrameSchema],
        db_schema: Optional[DatabaseSchema],
        additional_context: Optional[str]
    ) -> str:
        """Build Bedrock-optimized generation prompt."""
        prompt = f"""Human: I need you to generate Python code for data visualization.

User Request: {user_question}

Available Data Sources:
"""
        # Add DataFrame information (more concise for Bedrock)
        for i, df in enumerate(dataframes):
            prompt += f"DataFrame '{df.name}': {df.shape[0]}×{df.shape[1]}, columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}\n"
        
        if db_schema:
            prompt += f"Database tables: {', '.join(db_schema.tables)}\n"
        
        prompt += f"""
Requirements:
- Use pandas, plotly.express, plotly.graph_objects, or dash
- Save output to {self.base_instructions['output_path']}
- DataFrames are pre-loaded with their names
- No file system access outside /app/data/
- No network requests or subprocess calls

Please provide complete Python code wrapped in ```python code blocks.
"""
        return prompt

    def validate_code(self, code: str) -> bool:
        """Validate the generated code."""
        # Implementation of validation logic
        return True  # Placeholder, actual implementation needed

    def validate_output(self, output: str) -> bool:
        """Validate the generated output."""
        # Implementation of validation logic
        return True  # Placeholder, actual implementation needed

    def validate_prompt(self, prompt: str) -> bool:
        """Validate the generated prompt."""
        # Implementation of validation logic
        return True  # Placeholder, actual implementation needed
    
    def build_code_validation_prompt(
        self,
        code: str,
        context: str,
        provider: str = "gemini"
    ) -> str:
        """
        Build a prompt for code validation.
        
        Args:
            code: The generated code to validate
            context: Context about the original request
            provider: LLM provider name for prompt optimization
            
        Returns:
            str: Complete prompt for code validation
        """
        if provider.lower() == "bedrock":
            return self._build_bedrock_validation_prompt(code, context)
        else:
            return self._build_gemini_validation_prompt(code, context)
    
    def _build_gemini_validation_prompt(self, code: str, context: str) -> str:
        """Build Gemini-optimized validation prompt."""
        return f"""# Code Safety Validation Task

## Context
{context}

## Code to Validate
```python
{code}
```

## Validation Criteria
Please analyze the code above for security and safety issues. Check for:

1. **Forbidden Imports**: 
   - os, subprocess, requests, urllib, socket, http
   - sys (except basic usage), shutil, glob
   - eval, exec, compile functions

2. **File System Access**:
   - Writing files outside /app/data/ directory
   - Reading files outside allowed paths
   - Creating, deleting, or modifying system files

3. **Network Access**:
   - HTTP requests or API calls
   - Socket connections
   - External service access

4. **Code Execution**:
   - Dynamic code execution (eval, exec)
   - Shell command execution
   - Process spawning

5. **Resource Usage**:
   - Infinite loops or excessive recursion
   - Memory or CPU intensive operations without limits

## Response Format
Please respond with a JSON object:
```json
{{
    "is_safe": true/false,
    "reason": "Detailed explanation of safety assessment",
    "violations": ["list", "of", "specific", "violations"],
    "recommendations": ["suggested", "fixes", "if", "unsafe"]
}}
```

Only return the JSON object, no additional text.
"""
    
    def _build_bedrock_validation_prompt(self, code: str, context: str) -> str:
        """Build Bedrock-optimized validation prompt."""
        return f"""Human: Please validate this Python code for security and safety.

Context: {context}

Code:
```python
{code}
```

Check for:
- Forbidden imports (os, subprocess, requests, etc.)
- File access outside /app/data/
- Network requests
- Code execution (eval, exec)
- Security violations

Respond with JSON:
{"is_safe": true/false, "reason": "explanation", "violations": ["list"]}"""
        return prompt
