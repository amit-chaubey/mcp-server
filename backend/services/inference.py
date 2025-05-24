import logging
from typing import Dict, Any
from backend.services.model_loader import load_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model at module level
try:
    model = load_model()
    logger.info("Model loaded successfully in inference service")
except Exception as e:
    logger.error(f"Failed to load model in inference service: {str(e)}")
    raise

def predict_sentiment(text: str) -> Dict[str, Any]:
    """
    Predict sentiment for the given text.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, Any]: Prediction result with sentiment and confidence
        
    Raises:
        ValueError: If input text is empty or invalid
        Exception: For prediction errors
    """
    try:
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string")
            
        # Clean and preprocess text
        text = text.strip()
        
        # Make prediction
        prediction = model.predict([text])[0]
        confidence = model.predict_proba([text]).max()
        
        result = {
            "sentiment": "positive" if prediction == 1 else "negative",
            "confidence": float(confidence),
            "text": text
        }
        
        logger.info(f"Prediction successful for text: {text[:50]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise 