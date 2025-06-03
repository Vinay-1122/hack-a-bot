import requests
import pandas as pd
import json

def test_service():
    """Test the plot generation service"""
    
    # Create sample data
    df1 = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=100),
        'sales': [100 + i*2 + (i%7)*10 for i in range(100)],
        'region': ['North', 'South', 'East', 'West'] * 25
    })
    
    df2 = pd.DataFrame({
        'product': ['A', 'B', 'C'] * 20,
        'revenue': [1000 + i*50 for i in range(60)],
        'category': ['Electronics', 'Clothing', 'Food'] * 20
    })
    
    # Save as CSV files
    df1.to_csv('test_sales.csv', index=False)
    df2.to_csv('test_revenue.csv', index=False)
    
    # Prepare schema
    schema = {
        "tables": {
            "sales": {
                "columns": {
                    "date": "DATE",
                    "sales": "INTEGER",
                    "region": "VARCHAR"
                }
            },
            "revenue": {
                "columns": {
                    "product": "VARCHAR",
                    "revenue": "INTEGER", 
                    "category": "VARCHAR"
                }
            }
        }
    }
    
    # Prepare request
    files = [
        ('files', ('test_sales.csv', open('test_sales.csv', 'rb'), 'text/csv')),
        ('files', ('test_revenue.csv', open('test_revenue.csv', 'rb'), 'text/csv'))
    ]
    
    data = {
        'schema': json.dumps(schema),
        'user_question': 'Show me sales trends by region and revenue by product category'
    }
    
    # Send request
    response = requests.post('http://localhost:8000/generate-plot', files=files, data=data)
    
    # Close files
    for _, file_tuple in files:
        file_tuple[1].close()
    
    if response.status_code == 200:
        result = response.json()
        print("Success!")
        
        # Save HTML output
        with open('output_plot.html', 'w') as f:
            f.write(result['html'])
        print("Plot saved as output_plot.html")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_service() 