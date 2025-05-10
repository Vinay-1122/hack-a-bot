# HackaBot Backend

This is the backend service for HackaBot, providing natural language querying and analysis capabilities for data.

## Features

- Natural language query analysis and execution
- Semantic schema generation for databases
- Code fixing and optimization
- Secure code execution
- Data visualization

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
- Create a `.env` file in the root directory
- Add your Gemini API key:
```
GEMINI_API_KEY=your_api_key_here
```

## Running the Server

Start the development server:
```bash
python main.py
```

The server will be available at `http://localhost:8000` by default.

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

- `POST /api/query-analyzer/`: Analyze natural language queries
- `POST /api/semantic-schema/`: Generate semantic schema for database
- `POST /api/code-fix/`: Fix and optimize Python code

## Project Structure

```
backend/
├── api/
│   ├── routes/
│   │   ├── query.py
│   │   ├── semantic_schema.py
│   │   └── code_fix.py
│   └── app.py
├── core/
│   ├── config.py
│   ├── database.py
│   ├── llm.py
│   └── vector_store.py
├── utils/
│   ├── plotting.py
│   └── security.py
├── main.py
├── requirements.txt
└── README.md
```

## Security

- All code execution is sandboxed
- Dangerous operations are blocked
- API keys are required for LLM operations
- CORS is configured for security

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 