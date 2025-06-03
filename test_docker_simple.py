#!/usr/bin/env python3
"""
Simple test for Docker execution without requiring Gemini API
"""
import asyncio
import pandas as pd
from code_executor import CodeExecutor

async def test_docker_execution():
    """Test Docker execution with a simple plot"""
    
    # Create test data
    df1 = pd.DataFrame({
        'x': [1, 2, 3, 4, 5],
        'y': [2, 4, 6, 8, 10]
    })
    
    # Prepare test dataframes list
    dataframes = [{
        'name': 'test_data',
        'data': df1,
        'shape': df1.shape,
        'columns': df1.columns.tolist(),
        'dtypes': {col: str(dtype) for col, dtype in df1.dtypes.to_dict().items()}
    }]
    
    # Simple test code that should work
    test_code = """
import plotly.express as px

# Create a simple scatter plot
fig = px.scatter(df1, x='x', y='y', title='Test Plot')
fig.update_layout(
    title='Simple Test Plot',
    xaxis_title='X Values',
    yaxis_title='Y Values'
)

# Save the plot
fig.write_html('/tmp/output.html', include_plotlyjs='cdn')
"""
    
    try:
        print("Testing Docker execution...")
        executor = CodeExecutor()
        print(f"Docker command found: {executor.docker_cmd}")
        
        print("Executing test code...")
        html_result = await executor.execute_code(test_code, dataframes)
        
        print("✅ Docker execution successful!")
        print(f"HTML output length: {len(html_result)} characters")
        
        # Save test output
        with open('test_output.html', 'w') as f:
            f.write(html_result)
        print("Test plot saved as test_output.html")
        
        return True
        
    except Exception as e:
        print(f"❌ Docker execution failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_docker_execution()) 