import uvicorn
from api.app import app
from core.config import API_HOST, API_PORT

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=True  # Enable auto-reload during development
    ) 