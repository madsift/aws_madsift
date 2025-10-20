#!/usr/bin/env python3
"""
Centralized LLM Model Configuration
Provides consistent model access across the application
"""

import os
from typing import Optional

try:
    from strands.models import BedrockModel
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    print("⚠️ Strands models not available")

def get_ollama_model() -> Optional[object]:
    """Get configured Ollama model"""
    
    if not STRANDS_AVAILABLE:
        return None
    from strands.models.ollama import OllamaModel
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

    print(f"🔗 Attempting to connect to Ollama: {ollama_url} ({model_name})")

    try:
        return OllamaModel(
            host=ollama_url,
            model_id=model_name
        )
    except Exception as e:
        print(f"❌ Failed to create Ollama model: {e}")
        return None

def get_bedrock_model() -> Optional[object]:
    """Get configured Bedrock model"""
    if not STRANDS_AVAILABLE:
        return None

    try:
        return BedrockModel(
            # Ensure this is CORRECT
            model_id='anthropic.claude-3-haiku-20240307-v1:0',
            #model_id="meta.llama3-1-8b-instruct-v1:0", # <--- Alternative option
            temperature=0.3,
            streaming=True,
        )
    except Exception as e:
        print(f"❌ Failed to create Bedrock model: {e}")
        return None
        
def get_titan_model() -> Optional[object]:
    """Get configured Bedrock model"""
    if not STRANDS_AVAILABLE:
        return None

    try:
        return BedrockModel(
            # Ensure this is CORRECT
            model_id='amazon.titan-text-express-v1',
            #model_id="meta.llama3-1-8b-instruct-v1:0", # <--- Alternative option
            temperature=0.0,
            max_tokens=800,
            streaming=False,
        )
    except Exception as e:
        print(f"❌ Failed to create Bedrock model: {e}")
        return None

# Initialize models
#ollama_model = get_ollama_model()
bedrock_model = get_bedrock_model()

# Default model (can be switched as needed)
default_model = bedrock_model

def get_default_model():
    """Get the default model for the application"""
    return default_model

def switch_to_bedrock():
    """Switch default model to Bedrock"""
    global default_model
    default_model = bedrock_model
    print("🔄 Switched to Bedrock model")

def switch_to_ollama():
    """Switch default model to Ollama"""
    global default_model
    default_model = ollama_model
    print("🔄 Switched to Ollama model")

def get_current_model_info() -> dict:
    """Get information about the current default model"""
    if default_model == ollama_model:
        return {
            "type": "ollama",
            "url": os.getenv("OLLAMA_BASE_URL", "https://2128d83ee318.ngrok-free.app"),
            "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            "available": ollama_model is not None
        }
    elif default_model == bedrock_model:
        return {
            "type": "bedrock",
            "model": "anthropic.claude-3-haiku-20240307-v1:0",
            "available": bedrock_model is not None
        }
    else:
        return {
            "type": "none",
            "available": False
        }
