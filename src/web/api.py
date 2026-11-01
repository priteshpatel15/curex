"""
FastAPI Web Server for the Natural Language Data Analysis System.
Provides REST API endpoints for data loading, model configuration, and analysis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uvicorn

from ..app.main import DataAnalysisApp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the main application
app_instance = DataAnalysisApp()

# Create FastAPI instance
app = FastAPI(
    title="Natural Language Data Analysis API",
    description="API for natural language data analysis using configurable LLM backends",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API requests
class AnalysisRequest(BaseModel):
    query: str


class ModelConfigRequest(BaseModel):
    provider: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.1


class DataUploadResponse(BaseModel):
    success: bool
    message: str
    dataset_name: Optional[str] = None


class AnalysisResponse(BaseModel):
    success: bool
    session_id: Optional[str] = None
    user_request: Optional[str] = None
    generated_code: Optional[str] = None
    execution_output: Optional[str] = None
    error: Optional[str] = None
    model_used: Optional[str] = None
    execution_time: Optional[float] = None
    cost: Optional[float] = None


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    logger.info("Starting Natural Language Data Analysis API")
    
    # Initialize default models (if available)
    try:
        await app_instance.initialize_default_models()
        logger.info("Default models initialized")
    except Exception as e:
        logger.warning(f"Could not initialize default models: {str(e)}")
    
    # Create sample data for demonstration
    try:
        app_instance.create_sample_data("demo_sales")
        logger.info("Sample data created")
    except Exception as e:
        logger.warning(f"Could not create sample data: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Natural Language Data Analysis API is running"}


# Data management endpoints
@app.post("/data/upload", response_model=DataUploadResponse)
async def upload_data(file: UploadFile = File(...), dataset_name: Optional[str] = None):
    """Upload and load a data file"""
    try:
        # Save uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Load the data
        success = await app_instance.load_data(temp_file_path, dataset_name)
        
        # Clean up temp file
        Path(temp_file_path).unlink(missing_ok=True)
        
        if success:
            return DataUploadResponse(
                success=True,
                message="Data uploaded and loaded successfully",
                dataset_name=dataset_name or Path(file.filename).stem
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to load data file")
            
    except Exception as e:
        logger.error(f"Data upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Data upload failed: {str(e)}")


@app.post("/data/sample")
async def create_sample_data(dataset_name: str = "sample"):
    """Create sample data for demonstration"""
    try:
        success = app_instance.create_sample_data(dataset_name)
        
        if success:
            return {"success": True, "message": f"Sample dataset '{dataset_name}' created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create sample data")
            
    except Exception as e:
        logger.error(f"Sample data creation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sample data creation failed: {str(e)}")


@app.get("/data/info")
async def get_data_info():
    """Get information about loaded datasets"""
    try:
        return app_instance.get_data_info()
    except Exception as e:
        logger.error(f"Failed to get data info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get data info: {str(e)}")


# Model management endpoints
@app.post("/models/configure")
async def configure_model(config: ModelConfigRequest):
    """Configure a new LLM model"""
    try:
        success = await app_instance.configure_model(
            provider=config.provider,
            model_name=config.model_name,
            api_key=config.api_key,
            base_url=config.base_url,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        
        if success:
            return {"success": True, "message": f"Model '{config.model_name}' configured successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to configure model")
            
    except Exception as e:
        logger.error(f"Model configuration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Model configuration failed: {str(e)}")


@app.get("/models/info")
async def get_model_info():
    """Get information about configured models"""
    try:
        return app_instance.get_model_info()
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


# Analysis endpoints
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    """Analyze data using natural language query"""
    try:
        result = await app_instance.analyze(request.query)
        
        if result['success']:
            return AnalysisResponse(
                success=True,
                session_id=result.get('session_id'),
                user_request=result.get('user_request'),
                generated_code=result.get('generated_code'),
                execution_output=result.get('execution', {}).get('output'),
                error=result.get('execution', {}).get('error'),
                model_used=result.get('model_info', {}).get('model_used'),
                execution_time=result.get('execution', {}).get('execution_time'),
                cost=result.get('model_info', {}).get('cost')
            )
        else:
            return AnalysisResponse(
                success=False,
                error=result.get('error'),
                session_id=result.get('session_id')
            )
            
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/analyze/history")
async def get_analysis_history(limit: int = 10):
    """Get recent analysis session history"""
    try:
        return {"sessions": app_instance.get_session_history(limit)}
    except Exception as e:
        logger.error(f"Failed to get analysis history: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get analysis history: {str(e)}")


# File serving endpoints
@app.get("/plots/{filename}")
async def get_plot(filename: str):
    """Serve generated plot files"""
    try:
        file_path = Path(filename)
        if file_path.exists() and filename.startswith("plot_"):
            return FileResponse(file_path)
        else:
            raise HTTPException(status_code=404, detail="Plot file not found")
    except Exception as e:
        logger.error(f"Failed to serve plot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to serve plot: {str(e)}")


# System endpoints
@app.get("/system/status")
async def get_system_status():
    """Get system status and statistics"""
    try:
        data_info = app_instance.get_data_info()
        model_info = app_instance.get_model_info()
        
        return {
            "status": "running",
            "initialized": app_instance.is_initialized,
            "data_loaded": "datasets" in data_info and len(data_info["datasets"]) > 0,
            "models_configured": len(model_info.get("available_models", [])),
            "total_sessions": len(app_instance.analysis_sessions),
            "system_info": {
                "data": data_info,
                "models": model_info
            }
        }
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@app.post("/system/shutdown")
async def shutdown_system(background_tasks: BackgroundTasks):
    """Gracefully shutdown the system"""
    def shutdown():
        app_instance.shutdown()
    
    background_tasks.add_task(shutdown)
    return {"message": "System shutdown initiated"}


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )


# WebSocket endpoint for real-time analysis (optional enhancement)
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


manager = ConnectionManager()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time analysis updates"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            # Process analysis request
            try:
                result = await app_instance.analyze(data)
                await manager.send_personal_message(str(result), websocket)
            except Exception as e:
                await manager.send_personal_message(f"Error: {str(e)}", websocket)
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """Run the FastAPI server"""
    uvicorn.run(
        "src.web.api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )


if __name__ == "__main__":
    run_server(debug=True) 