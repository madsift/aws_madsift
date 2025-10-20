#!/usr/bin/env python3
"""
Session Management - Handles session initialization and management

This module handles:
- S3 session manager setup
- Session information tracking
- Conversation history management
"""

from datetime import datetime
from typing import Dict, Any, Optional, List

# Strands imports with better error handling
STRANDS_AVAILABLE = False
STRANDS_ERROR = None

try:
    from strands.session.s3_session_manager import S3SessionManager
    STRANDS_AVAILABLE = True
    print("✅ Strands session imports successful")
except ImportError as e:
    STRANDS_ERROR = str(e)
    print(f"⚠️ Strands session not available: {e}")
except Exception as e:
    STRANDS_ERROR = str(e)
    print(f"⚠️ Strands session error: {e}")


class SessionManager:
    """Handles session management and conversation history"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"chat_{int(datetime.now().timestamp())}"
        self.session_manager = None
        self.conversation_history = []
        self.bucket_name = None
        self.region = None

    def initialize_session_manager(self, bucket_name: str, username: str, region: Optional[str] = None):
        """Initialize S3 session manager"""
        self.bucket_name = bucket_name
        self.region = region
        safe_username = "".join(c if c.isalnum() or c in "-_" else "_" for c in username)
        prefix = f"strand_session/{safe_username}/{self.session_id}/"

        print(f"[DEBUG] Using S3 session path: s3://{bucket_name}/{prefix}")

        if STRANDS_AVAILABLE:
            self.session_manager = S3SessionManager(
                session_id=self.session_id,
                bucket=self.bucket_name,
                region=region,
                prefix=prefix,
            )
        else:
            print(f"⚠️ S3SessionManager not available: {STRANDS_ERROR}")
            self.session_manager = None

    def add_to_conversation_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session"""
        return {
            "session_id": self.session_id,
            "strands_available": STRANDS_AVAILABLE,
            "session_manager": self.session_manager is not None,
            "session_manager_type": "S3SessionManager",
            "session_storage": self.bucket_name,
            "conversation_length": len(self.conversation_history),
            "strands_error": STRANDS_ERROR if not STRANDS_AVAILABLE else None
        }

    def _check_conversation_context(self, user_message: str) -> Optional[Dict[str, Any]]:
        """Check if the question relates to previous conversation history and respond contextually."""
        if len(self.conversation_history) < 2:
            return None

        user_lower = user_message.lower()
        followups = [
            "tell me more",
            "what about",
            "and that",
            "more details",
            "explain that",
            "what do you mean",
            "can you elaborate",
            "what else",
            "anything else",
            "more information",
        ]

        if any(p in user_lower for p in followups):
            # Use last assistant message for continuity
            last_assistant_msgs = [m for m in self.conversation_history if m["role"] == "assistant"]
            if last_assistant_msgs:
                last_response = last_assistant_msgs[-1]["content"]
                content = (
                    f"Building on what I mentioned earlier:\n\n{last_response}\n\n"
                    f"If you want more details, you can ask about specific entities or relationships from the Knowledge Graph."
                )

                return {
                    "content": content,
                    "tools_used": ["Session Memory", "Conversation Context"],
                    "citations": [],
                    "confidence": 80,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "source": "conversation_context",
                    "session_id": self.session_id,
                }

        return None