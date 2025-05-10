# HackaBot - Natural Language Data Analysis

HackaBot is an AI-powered data analysis tool that allows users to analyze data using natural language queries. It combines the power of Google's Gemini LLM with advanced data processing capabilities to provide insights, visualizations, and code generation.

## Features

- Natural language querying of data
- Automatic SQL query generation
- Python code generation for complex analysis
- Interactive data visualization
- Semantic schema understanding
- Code fixing and optimization
- RAG-based schema context retrieval

## Project Structure

```
hackabot/
├── frontend/                 # Streamlit web interface
│   ├── app.py               # Main Streamlit application
│   ├── components/          # UI components
│   │   └── sidebar.py       # Sidebar configuration
│   ├── utils/              # Frontend utilities
│   │   └── session_state.py # Streamlit session management
│   └── requirements.txt    # Frontend dependencies
├── backend/                 # FastAPI backend service
│   ├── api/                # API endpoints
│   │   ├── app.py          # FastAPI application
│   │   └── routes/         # API route handlers
│   ├── core/              # Core functionality
│   │   ├── config.py      # Configuration settings
│   │   ├── database.py    # Database operations
│   │   ├── llm.py         # LLM integration
│   │   └── vector_store.py # Vector store for RAG
│   ├── utils/             # Backend utilities
│   │   ├── plotting.py    # Data visualization
│   │   └── security.py    # Code execution security
│   ├── main.py           # Backend entry point
│   └── requirements.txt   # Backend dependencies
└── data/                  # Data storage directory
    └── analytics_db.sqlite # SQLite database
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hackabot.git
cd hackabot
```

2. Set up the backend:
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up the frontend:
```bash
cd ../frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

4. Create a `.env` file in the backend directory:
```
GEMINI_API_KEY=your_api_key_here
```

## Running the Application

1. Start the backend server:
```bash
cd backend
python main.py
```

2. Start the frontend application:
```bash
cd frontend
streamlit run app.py
```

The application will be available at:
- Frontend: http://localhost:8501
- Backend API: http://localhost:8001
- API Documentation: http://localhost:8001/docs

## Usage

1. Configure your Gemini API key in the frontend sidebar
2. Load your data (CSV, Excel, or JSON)
3. Ask questions about your data in natural language
4. View generated SQL queries, Python code, and visualizations
5. Execute and modify generated code as needed

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 