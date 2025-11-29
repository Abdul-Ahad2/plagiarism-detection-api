"""
AI detection using HuggingFace fakespot-ai/roberta-base-ai-text-detection-v1
Fast, accurate ML model. Returns AI similarity score (0-1).
"""
import logging
import os
from huggingface_hub import InferenceClient
from app.config import HF_TOKEN
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai_detector")


# Initialize HF client
try:
    hf_client = InferenceClient(api_key=HF_TOKEN)
    logger.info("✓ HuggingFace client initialized")
except Exception as e:
    logger.error(f"❌ HF_TOKEN not available: {e}")
    hf_client = None


def detect_ai_similarity(text: str) -> float:
    """
    Detect AI-generated text using HuggingFace model.
    Fast, accurate ML-based detection.
    
    Returns:
        float: 0.0-1.0 where 1.0 = definitely AI, 0.0 = definitely human
    """
    try:
        logger.info("Starting AI detection (HuggingFace model)...")
        
        if not text or len(text) < 20:
            logger.warning("Text too short for detection")
            return 0.0
        
        if hf_client is None:
            logger.error("HuggingFace client not initialized")
            return 0.0
        
        # Truncate to avoid token limits
        text = text[:2000]
        
        logger.info("Sending request to HuggingFace...")
        result = hf_client.text_classification(
            text,
            model="fakespot-ai/roberta-base-ai-text-detection-v1",
        )
        
        logger.info(f"API response: {result}")
        
        # Extract top result
        if not result or len(result) == 0:
            logger.error("No result from API")
            return 0.0
        
        top_result = result[0]
        label = top_result.get('label', '').upper()
        confidence = top_result.get('score', 0.5)
        
        logger.info(f"Classification: {label} (confidence: {confidence:.4f})")
        
        # Map to 0-1 score
        if label == 'AI':
            ai_score = confidence
        elif label == 'HUMAN':
            ai_score = 1.0 - confidence
        else:
            ai_score = 0.5
        
        ai_score = min(max(ai_score, 0.0), 1.0)  # Clamp to 0-1
        logger.info(f"✅ AI detection complete: {ai_score:.3f}")
        
        return round(ai_score, 3)
    
    except Exception as e:
        logger.error(f"❌ AI detection error: {str(e)}", exc_info=True)
        return 0.0


