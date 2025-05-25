import logging
from typing import Dict, Any, List
import numpy as np
from pathlib import Path
from backend.services.model_loader import load_model
from backend.config.settings import MODEL_PATH
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and metadata at module level
try:
    model = load_model()
    metadata_path = Path(MODEL_PATH).with_suffix('.metadata.pkl')
    if metadata_path.exists():
        metadata = joblib.load(str(metadata_path))
    else:
        metadata = None
    logger.info("Model and metadata loaded successfully in inference service")
except Exception as e:
    logger.error(f"Failed to load model in inference service: {str(e)}")
    raise

def preprocess_text(text: str) -> str:
    """
    Preprocess input text
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = ''.join(c for c in text if c.isalnum() or c.isspace())
    return text

def get_top_features(text: str, n: int = 5) -> List[Dict[str, float]]:
    """
    Get top contributing features for the prediction
    
    Args:
        text (str): Input text
        n (int): Number of top features to return
        
    Returns:
        List[Dict[str, float]]: Top features with their coefficients
    """
    if metadata is None or 'feature_names' not in metadata:
        return []
        
    # Get feature names and coefficients
    feature_names = metadata['feature_names']
    coefficients = model.named_steps['classifier'].coef_[0]
    
    # Transform text
    vectorizer = model.named_steps['vectorizer']
    features = vectorizer.transform([text])
    
    # Get feature scores
    feature_scores = []
    for i, feature in enumerate(feature_names):
        if feature in text.lower():
            idx = vectorizer.vocabulary_.get(feature)
            if idx is not None:
                score = coefficients[idx] * features[0, idx]
                feature_scores.append({
                    'feature': feature,
                    'score': float(score)
                })
    
    # Sort by absolute score
    feature_scores.sort(key=lambda x: abs(x['score']), reverse=True)
    return feature_scores[:n]

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
        text = preprocess_text(text)
        
        # Make prediction
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        confidence = probabilities.max()
        
        # Get top contributing features
        top_features = get_top_features(text)
        
        result = {
            "sentiment": "positive" if prediction == 1 else "negative",
            "confidence": float(confidence),
            "text": text,
            "probabilities": {
                "positive": float(probabilities[1]),
                "negative": float(probabilities[0])
            },
            "top_features": top_features
        }
        
        logger.info(f"Prediction successful for text: {text[:50]}...")
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise 