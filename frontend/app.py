import streamlit as st
import requests
from typing import Dict, Any
import json
import sys
from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.config.settings import API_HOST, API_PORT

# Configure page
st.set_page_config(
    page_title="MCP Sentiment Analysis",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .stTextInput>div>div>input {
        font-size: 18px;
    }
    .stButton>button {
        width: 100%;
        font-size: 18px;
    }
    </style>
    """, unsafe_allow_html=True)

def create_session():
    """Create a requests session with retry mechanism"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def call_api(text: str) -> Dict[str, Any]:
    """
    Call the sentiment analysis API.
    
    Args:
        text (str): Input text to analyze
        
    Returns:
        Dict[str, Any]: API response
        
    Raises:
        Exception: If API call fails
    """
    try:
        session = create_session()
        response = session.post(
            f"http://{API_HOST}:{API_PORT}/predict",
            json={"text": text},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API server. Please make sure the backend server is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("Request timed out. Please try again.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def main():
    st.title("🤖 MCP Sentiment Analysis")
    st.markdown("---")
    
    # Input section
    st.subheader("Enter Text")
    user_input = st.text_area(
        "Type or paste your text here:",
        height=150,
        placeholder="Enter text to analyze sentiment..."
    )
    
    # Analyze button
    if st.button("Analyze Sentiment", type="primary"):
        if not user_input:
            st.warning("Please enter some text to analyze.")
            return
            
        with st.spinner("Analyzing sentiment..."):
            result = call_api(user_input)
            
            if result:
                # Display results
                st.markdown("---")
                st.subheader("Results")
                
                # Create columns for results
                col1, col2 = st.columns(2)
                
                with col1:
                    sentiment = result["sentiment"]
                    sentiment_color = "green" if sentiment == "positive" else "red"
                    st.markdown(f"**Sentiment:** :{sentiment_color}[{sentiment.upper()}]")
                
                with col2:
                    confidence = result["confidence"] * 100
                    st.markdown(f"**Confidence:** {confidence:.1f}%")
                
                # Display confidence bar
                st.progress(confidence / 100)
                
                # Display original text
                st.markdown("---")
                st.markdown("**Original Text:**")
                st.info(user_input)

if __name__ == "__main__":
    main() 