from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from api.routes import query, semantic_schema, code_fix, data, data_management

app = FastAPI(
    title="HackaBot API",
    description="API for natural language querying and analysis of data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Add error handling middleware
@app.middleware("http")
async def error_handling_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"An unexpected error occurred: {str(e)}"}
        )

# Include routers
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(semantic_schema.router, prefix="/api", tags=["semantic-schema"])
app.include_router(code_fix.router, prefix="/api", tags=["code-fix"])
app.include_router(data.router, prefix="/api", tags=["data"])
app.include_router(data_management.router, prefix="/api", tags=["data-management"])

@app.get("/")
async def root():
    return {
        "message": "Welcome to HackaBot API",
        "version": "1.0.0",
        "endpoints": {
            "query": "/api/query-analyzer/",
            "semantic_schema": "/api/semantic-schema/",
            "code_fix": "/api/code-fix/",
            "data": "/api/load-data/",
            "execute_python": "/api/execute-python/"
        }
    } 