import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import logging
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.config.settings import MODEL_PATH

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample training data"""
    texts = [
        "I love this product, it's amazing!",
        "This is the best thing I've ever bought",
        "Absolutely fantastic experience",
        "Great service and fast delivery",
        "Highly recommended",
        "This is terrible, I want a refund",
        "Worst experience ever",
        "Complete waste of money",
        "I'm very disappointed",
        "Would not recommend to anyone"
    ]
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]  # 1 for positive, 0 for negative
    return texts, labels

def train_model():
    """Train and save the sentiment analysis model"""
    try:
        # Create sample data
        texts, labels = create_sample_data()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Create and train pipeline
        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(max_features=1000)),
            ('classifier', LogisticRegression(random_state=42))
        ])
        
        logger.info("Training model...")
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        
        logger.info(f"Training accuracy: {train_score:.2f}")
        logger.info(f"Testing accuracy: {test_score:.2f}")
        
        # Create models directory if it doesn't exist
        Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(pipeline, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 