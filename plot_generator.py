import google.generativeai as genai
import pandas as pd
import json
from typing import List, Dict, Any
from code_executor import CodeExecutor
from code_validator import CodeValidator

class PlotGenerator:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.code_executor = CodeExecutor()
        self.code_validator = CodeValidator()
    
    async def generate_plot(self, dataframes: List[Dict], schema: Dict, user_question: str) -> str:
        """
        Generate interactive Plotly plot/dashboard from dataframes
        
        Args:
            dataframes: List of dataframe info with data
            schema: Database schema
            user_question: Original question
            
        Returns:
            HTML string of the plot
        """
        
        # Prepare context for LLM
        context = self._prepare_context(dataframes, schema, user_question)
        
        # Generate code using LLM
        raw_code = await self._generate_code(context)
        
        # Sanitize the code
        sanitized_code = self.code_validator.sanitize_code(raw_code)
        
        # Validate the code for security
        is_valid, validation_errors = self.code_validator.validate_code(sanitized_code)
        
        if not is_valid:
            error_msg = "Code validation failed:\n" + "\n".join(validation_errors)
            raise Exception(error_msg)
        
        # Execute code in isolated environment
        html_result = await self.code_executor.execute_code(sanitized_code, dataframes)
        
        return html_result
    
    def _prepare_context(self, dataframes: List[Dict], schema: Dict, user_question: str) -> str:
        """Prepare context string for LLM"""
        
        context = f"""
I have {len(dataframes)} dataframe(s) that resulted from a SQL query based on this question: "{user_question}"

Database Schema:
{json.dumps(schema, indent=2)}

Dataframes Information:
"""
        
        for i, df_info in enumerate(dataframes):
            context += f"""
Dataframe {i+1} ({df_info['name']}):
- Shape: {df_info['shape']}
- Columns: {df_info['columns']}
- Data Types: {json.dumps(df_info['dtypes'], indent=2)}
- First few rows:
{df_info['data'].head().to_string()}

"""
        
        return context
    
    async def _generate_code(self, context: str) -> str:
        """Generate Python code using Gemini"""
        
        is_dashboard = len([df for df in context.split('Dataframe')]) > 2
        
        prompt = f"""
{context}

Generate Python code to create {'an interactive Plotly dashboard' if is_dashboard else 'an interactive Plotly plot'} that:

1. Analyzes the provided dataframe(s) and creates highly relevant, detailed, and interactive visualizations
2. Uses appropriate plot types based on the data characteristics and user question
3. {'Creates a dashboard with multiple subplots if multiple dataframes are provided' if is_dashboard else 'Creates a single comprehensive plot'}
4. Includes proper titles, axis labels, legends, and formatting
5. Adds interactive features like hover information, zoom, pan, etc.
6. Uses color schemes and styling that enhance readability
7. Exports the final plot/dashboard as HTML

STRICT Requirements:
- Use ONLY these imports: pandas, plotly.express, plotly.graph_objects, plotly.subplots, plotly.io
- The dataframes are available as variables: df1, df2, etc. (corresponding to the order provided)
- End with: fig.write_html('/tmp/output.html', include_plotlyjs='cdn')
- Do NOT use any file I/O operations except for the final HTML export
- Do NOT import os, sys, subprocess, or any system modules
- Do NOT use eval, exec, or dynamic code execution
- Make the visualization publication-ready with professional styling
- Ensure the code is complete and runnable

Generate ONLY the Python code, no explanations or markdown formatting:
"""
        
        try:
            response = await self.model.generate_content_async(prompt)
            code = response.text.strip()
            
            # Clean up the code (remove markdown formatting if present)
            if code.startswith('```python'):
                code = code[9:]
            if code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
            
            return code.strip()
        
        except Exception as e:
            raise Exception(f"Failed to generate code: {str(e)}") 