import joblib
import logging
from pathlib import Path
from backend.config.settings import MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """
    Load the sentiment analysis model from disk.
    
    Returns:
        The loaded model pipeline
        
    Raises:
        FileNotFoundError: If the model file doesn't exist
        Exception: For other loading errors
    """
    try:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
            
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise 