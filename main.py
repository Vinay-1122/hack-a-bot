from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import pandas as pd
import json
import os
import tempfile
import shutil
import uuid
from typing import List, Optional
import google.generativeai as genai
from code_executor import CodeExecutor
from plot_generator import PlotGenerator

app = FastAPI(title="LLM Plot Generator Service", version="1.0.0")

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

class PlotRequest(BaseModel):
    dataframes: List[dict]
    schema: dict
    user_question: str

@app.post("/generate-plot")
async def generate_plot(
    schema: str = Form(...),
    user_question: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Generate interactive Plotly plots from dataframes using LLM-generated code
    
    Args:
        schema: JSON string of database schema
        user_question: Original question that generated the SQL query
        files: CSV files containing the dataframes
    """
    
    try:
        # Parse schema
        schema_data = json.loads(schema)
        
        # Process uploaded dataframes
        dataframes = []
        temp_files = []
        
        for file in files:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv')
            temp_files.append(temp_file.name)
            
            # Read and save file content
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            
            # Load as DataFrame
            df = pd.read_csv(temp_file.name)
            dataframes.append({
                'name': file.filename.replace('.csv', ''),
                'data': df,
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.to_dict().items()}
            })
        
        # Generate plot using the service
        plot_generator = PlotGenerator()
        html_result = await plot_generator.generate_plot(
            dataframes=dataframes,
            schema=schema_data,
            user_question=user_question
        )
        
        # Cleanup temporary files
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
                
        return {"html": html_result, "status": "success"}
        
    except Exception as e:
        # Cleanup on error
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 