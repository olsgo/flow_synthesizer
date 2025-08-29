#!/usr/bin/env python3
"""
PolyMAX Parameter Prediction Web API

A FastAPI web service that provides remote parameter prediction for PolyMAX synthesizer.
Supports audio file upload, parameter prediction, and optional Ableton Live integration.

Features:
- Audio file upload and processing
- Real-time parameter prediction
- Batch processing capabilities
- Ableton Live integration
- RESTful API with automatic documentation
- File format validation and conversion
- Prediction confidence scoring

Usage:
    python polymax_web_api.py --model path/to/model.pth --port 8000
    
API Endpoints:
    POST /predict - Upload audio file and get parameter predictions
    POST /predict/batch - Upload multiple files for batch processing
    POST /predict/apply - Predict and apply to Ableton Live
    GET /status - Get API and model status
    GET /health - Health check endpoint
    GET /docs - Interactive API documentation
"""

import argparse
import asyncio
import io
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import librosa
import numpy as np

# Import our prediction modules
try:
    from predict_polymax_params import PolyMAXPredictor
    from ableton_polymax_remote import PolyMAXClient
except ImportError as e:
    print(f"Warning: Could not import prediction modules: {e}")
    PolyMAXPredictor = None
    PolyMAXClient = None

# Pydantic models for API requests/responses
class PredictionResponse(BaseModel):
    """Response model for parameter predictions"""
    success: bool
    parameters: Optional[Dict[str, float]] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    file_info: Optional[Dict[str, Any]] = None

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions"""
    success: bool
    results: List[PredictionResponse]
    total_files: int
    successful_predictions: int
    total_processing_time: float
    error: Optional[str] = None

class AbletonApplyResponse(BaseModel):
    """Response model for Ableton Live parameter application"""
    success: bool
    prediction: Optional[PredictionResponse] = None
    ableton_status: Optional[str] = None
    applied_parameters: Optional[int] = None
    error: Optional[str] = None

class StatusResponse(BaseModel):
    """Response model for API status"""
    api_status: str
    model_loaded: bool
    model_path: Optional[str] = None
    ableton_available: bool
    supported_formats: List[str]
    version: str = "1.0.0"
    uptime: float

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str

# Global variables
predictor: Optional[PolyMAXPredictor] = None
ableton_client: Optional[PolyMAXClient] = None
api_start_time = time.time()

# Supported audio formats
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.aiff', '.m4a', '.ogg'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Create FastAPI app
app = FastAPI(
    title="PolyMAX Parameter Prediction API",
    description="AI-powered parameter prediction for PolyMAX synthesizer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def validate_audio_file(file: UploadFile) -> Dict[str, Any]:
    """
    Validate uploaded audio file
    
    Args:
        file: Uploaded file object
        
    Returns:
        Dictionary with validation results
        
    Raises:
        HTTPException: If file is invalid
    """
    # Check file size
    if hasattr(file, 'size') and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
        )
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    return {
        'filename': file.filename,
        'extension': file_ext,
        'size': getattr(file, 'size', 0)
    }

async def save_uploaded_file(file: UploadFile) -> str:
    """
    Save uploaded file to temporary location
    
    Args:
        file: Uploaded file object
        
    Returns:
        Path to saved temporary file
    """
    # Create temporary file
    suffix = Path(file.filename).suffix
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    
    try:
        # Read and save file content
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()
        return temp_file.name
    finally:
        temp_file.close()

def cleanup_temp_file(file_path: str):
    """
    Clean up temporary file
    
    Args:
        file_path: Path to temporary file
    """
    try:
        if os.path.exists(file_path):
            os.unlink(file_path)
    except Exception as e:
        print(f"Warning: Could not delete temporary file {file_path}: {e}")

@app.on_event("startup")
async def startup_event():
    """
    Initialize the API on startup
    """
    global api_start_time
    api_start_time = time.time()
    print("PolyMAX Parameter Prediction API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up resources on shutdown
    """
    print("PolyMAX Parameter Prediction API shutting down...")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    return HealthResponse(
        status="healthy",
        timestamp=time.strftime('%Y-%m-%d %H:%M:%S')
    )

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Get API and model status
    """
    global predictor, ableton_client
    
    # Check Ableton availability
    ableton_available = False
    if ableton_client:
        try:
            status = ableton_client.get_status()
            ableton_available = status.get('status') == 'success'
        except:
            pass
    
    return StatusResponse(
        api_status="running",
        model_loaded=predictor is not None,
        model_path=getattr(predictor, 'model_path', None) if predictor else None,
        ableton_available=ableton_available,
        supported_formats=list(SUPPORTED_FORMATS),
        uptime=time.time() - api_start_time
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_parameters(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file for parameter prediction")
):
    """
    Predict PolyMAX parameters from uploaded audio file
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    temp_file_path = None
    
    try:
        # Validate file
        file_info = validate_audio_file(file)
        
        # Save uploaded file
        temp_file_path = await save_uploaded_file(file)
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        # Predict parameters
        result = predictor.predict_from_file(temp_file_path)
        
        processing_time = time.time() - start_time
        
        if result['success']:
            return PredictionResponse(
                success=True,
                parameters=result['parameters'],
                confidence=result['confidence'],
                processing_time=processing_time,
                file_info=file_info
            )
        else:
            return PredictionResponse(
                success=False,
                error=result.get('error', 'Prediction failed'),
                processing_time=processing_time,
                file_info=file_info
            )
    
    except HTTPException:
        raise
    except Exception as e:
        if temp_file_path:
            background_tasks.add_task(cleanup_temp_file, temp_file_path)
        
        return PredictionResponse(
            success=False,
            error=str(e),
            processing_time=time.time() - start_time
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple audio files for batch prediction")
):
    """
    Predict PolyMAX parameters for multiple audio files
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 20:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files. Maximum: 20")
    
    start_time = time.time()
    results = []
    temp_files = []
    
    try:
        # Process each file
        for file in files:
            file_start_time = time.time()
            temp_file_path = None
            
            try:
                # Validate and save file
                file_info = validate_audio_file(file)
                temp_file_path = await save_uploaded_file(file)
                temp_files.append(temp_file_path)
                
                # Predict parameters
                result = predictor.predict_from_file(temp_file_path)
                
                file_processing_time = time.time() - file_start_time
                
                if result['success']:
                    results.append(PredictionResponse(
                        success=True,
                        parameters=result['parameters'],
                        confidence=result['confidence'],
                        processing_time=file_processing_time,
                        file_info=file_info
                    ))
                else:
                    results.append(PredictionResponse(
                        success=False,
                        error=result.get('error', 'Prediction failed'),
                        processing_time=file_processing_time,
                        file_info=file_info
                    ))
            
            except Exception as e:
                results.append(PredictionResponse(
                    success=False,
                    error=str(e),
                    processing_time=time.time() - file_start_time,
                    file_info={'filename': file.filename}
                ))
        
        # Schedule cleanup for all temp files
        for temp_file in temp_files:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        total_processing_time = time.time() - start_time
        successful_predictions = sum(1 for r in results if r.success)
        
        return BatchPredictionResponse(
            success=True,
            results=results,
            total_files=len(files),
            successful_predictions=successful_predictions,
            total_processing_time=total_processing_time
        )
    
    except Exception as e:
        # Cleanup on error
        for temp_file in temp_files:
            background_tasks.add_task(cleanup_temp_file, temp_file)
        
        return BatchPredictionResponse(
            success=False,
            results=results,
            total_files=len(files),
            successful_predictions=0,
            total_processing_time=time.time() - start_time,
            error=str(e)
        )

@app.post("/predict/apply", response_model=AbletonApplyResponse)
async def predict_and_apply(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio file for parameter prediction and Live application")
):
    """
    Predict PolyMAX parameters and apply them to Ableton Live
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not ableton_client:
        raise HTTPException(status_code=503, detail="Ableton Live integration not available")
    
    temp_file_path = None
    
    try:
        # First, get prediction
        prediction_response = await predict_parameters(background_tasks, file)
        
        if not prediction_response.success:
            return AbletonApplyResponse(
                success=False,
                prediction=prediction_response,
                error="Prediction failed"
            )
        
        # Apply to Ableton Live
        ableton_result = ableton_client.send_parameters(prediction_response.parameters)
        
        if ableton_result.get('status') == 'success':
            return AbletonApplyResponse(
                success=True,
                prediction=prediction_response,
                ableton_status="success",
                applied_parameters=ableton_result.get('applied_count', 0)
            )
        else:
            return AbletonApplyResponse(
                success=False,
                prediction=prediction_response,
                ableton_status="failed",
                error=ableton_result.get('message', 'Unknown Ableton error')
            )
    
    except Exception as e:
        return AbletonApplyResponse(
            success=False,
            error=str(e)
        )

def initialize_services(model_path: str, enable_ableton: bool = True):
    """
    Initialize prediction and Ableton services
    
    Args:
        model_path: Path to trained model
        enable_ableton: Whether to enable Ableton Live integration
    """
    global predictor, ableton_client
    
    # Initialize predictor
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        predictor = PolyMAXPredictor(model_path)
        print(f"✓ Model loaded: {model_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Initialize Ableton client if requested
    if enable_ableton:
        try:
            ableton_client = PolyMAXClient()
            print("✓ Ableton Live client initialized")
        except Exception as e:
            print(f"⚠ Ableton Live client failed to initialize: {e}")
            ableton_client = None

def main():
    parser = argparse.ArgumentParser(description='PolyMAX Parameter Prediction Web API')
    parser.add_argument('--model', required=True,
                       help='Path to trained PolyMAX model')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to run the server on')
    parser.add_argument('--no-ableton', action='store_true',
                       help='Disable Ableton Live integration')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload for development')
    parser.add_argument('--log-level', default='info',
                       choices=['debug', 'info', 'warning', 'error'],
                       help='Log level')
    
    args = parser.parse_args()
    
    # Initialize services
    try:
        initialize_services(args.model, enable_ableton=not args.no_ableton)
    except Exception as e:
        print(f"Error initializing services: {e}")
        return 1
    
    # Run server
    print(f"\nStarting PolyMAX Parameter Prediction API...")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Documentation: http://{args.host}:{args.port}/docs")
    print(f"Health check: http://{args.host}:{args.port}/health")
    
    uvicorn.run(
        "polymax_web_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())