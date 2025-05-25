import sys
from pathlib import Path

# Add the project root to Python path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from typing import Dict, Any

from backend.services.inference import predict_sentiment
from backend.config.settings import API_HOST, API_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MCP Sentiment Analysis API",
    description="A Machine Learning API for sentiment analysis using MCP server architecture",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input validation model
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Text to analyze for sentiment")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "healthy", "service": "sentiment-analysis"}

@app.post("/predict", response_model=Dict[str, Any])
async def get_sentiment(data: TextRequest):
    """
    Predict sentiment for the given text.
    
    Args:
        data (TextRequest): Input text to analyze
        
    Returns:
        Dict[str, Any]: Prediction result with sentiment and confidence
        
    Raises:
        HTTPException: If prediction fails
    """
    try:
        result = predict_sentiment(data.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT) 