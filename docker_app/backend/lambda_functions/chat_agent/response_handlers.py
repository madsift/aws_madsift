#!/usr/bin/env python3
"""
Response Handlers - Unified, Agent-First Response Generation

This module handles:
- Intelligent pre-routing to distinguish chit-chat from substantive queries.
- Defaulting to the powerful Strands agent for all real questions.
- Generating simple contextual responses for greetings and fallbacks.
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from common.llm_models import get_default_model


class ResponseHandlers:
    """Handles the primary response generation logic with an Agent-First approach."""
    
    def __init__(self, session_manager, kg_operations):
        self.session_manager = session_manager
        self.kg_operations = kg_operations

    # --- NEW: PRIMARY ENTRY POINT ---
    def handle_chat_request(self, agent, user_message: str, chat_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for handling a user's chat message.
        It uses an LLM router to separate simple chit-chat from complex queries,
        ensuring the powerful Strands agent is used for all substantive questions.
        """
        print("🤖 Handling chat request with new Agent-First logic...")

        # 1. First, check if the message is simple chit-chat.
        if self._is_chitchat(user_message):
            print("✅ Intent classified as chit-chat. Generating simple contextual response.")
            content = self._generate_contextual_response(user_message)
            source = "contextual_chitchat"
            tools_used = ["Intent Classifier", "Session Manager"]
        
        # 2. If it's not chit-chat, ALWAYS use the powerful Strands agent.
        else:
            print("✅ Intent classified as KG Query. Invoking main Strands agent...")
            try:
                # The get_strands_response method already handles history and context.
                # We are just wrapping it to catch potential failures.
                agent_response_dict = self.get_strands_response(agent, user_message, chat_options)
                
                # Check if the agent call was successful
                if agent_response_dict and "error" not in agent_response_dict.get("content", "").lower():
                    return agent_response_dict
                else:
                    print("⚠️ Strands agent returned an error or empty response. Falling back.")
                    content = self._generate_contextual_response(user_message)
                    source = "agent_failure_fallback"
                    tools_used = ["Session Manager"]

            except Exception as e:
                print(f"❌ Catastrophic failure during agent invocation: {e}. Falling back.")
                content = "I encountered an unexpected issue while processing your request. Please try again."
                source = "exception_fallback"
                tools_used = ["Error Handler"]

        # 3. Format and return the fallback/chitchat response.
        self.session_manager.add_to_conversation_history("assistant", content)
        return {
            "content": content,
            "tools_used": tools_used,
            "citations": [],
            "confidence": 70,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "session_id": self.session_manager.session_id
        }

    # --- NEW: LLM-based Intent Classifier ---
    def _is_chitchat(self, user_message: str) -> bool:
        """
        Uses a fast LLM to classify user intent: is this a query for the
        Knowledge Graph or just general chit-chat?
        """
        # Short messages are often chit-chat, but we let the LLM decide.
        if len(user_message) > 150:
            return False

        print("[Router] Classifying user intent...")
        
        try:
            # Use a fast and cheap model for classification if available
            router_model = get_default_model(model_id="anthropic.claude-3-haiku-20240307-v1:0") 
        except Exception:
            print("[Router] Haiku not available, using default model for routing.")
            router_model = get_default_model()

        router_prompt = f"""
            Your task is to classify the user's intent into one of two categories: "KG_QUERY" or "CHITCHAT".

            - "KG_QUERY" is for any question that seeks information, asks for data, or could be answered by a knowledge base. This includes searching, listing, summarizing, or asking about topics, claims, posts, people, or events.
            - "CHITCHAT" is for simple greetings (hi, hello), thank yous, or conversational filler that does not seek specific information.

            User message: "{user_message}"

            Based on the message, what is the user's intent? Respond with only the single word "KG_QUERY" or "CHITCHAT".
        """
        
        try:
            # For a simple classification, a direct, non-streaming call is best.
            # We use the internal _invoke_agent_sync to handle the async complexity.
            classification = self._invoke_agent_sync(router_model, router_prompt)
            
            if classification:
                classification = classification.strip().replace('"', '')
                print(f"[Router] Intent classified as: {classification}")
                return classification == "CHITCHAT"
            
            return False # Default to not chit-chat if classification returns nothing
            
        except Exception as e:
            print(f"[Router] Intent classification failed: {e}. Defaulting to not chit-chat.")
            # When in doubt, assume it's a real query to be safe.
            return False

    def get_strands_response(self, agent, user_message: str, chat_options: Dict[str, Any]) -> Dict[str, Any]:
        """Get response from Strands agent (uses invoke_async safely)."""
        enhanced_message = self.kg_operations._add_kg_context(user_message)
        print(f"🤖 Invoking Strands agent with message: {enhanced_message[:100]}...")

        text = self._invoke_agent_sync(agent, enhanced_message)
        if text is None:
            print("[ResponseHandlers] Agent returned no text.")
            content = "Sorry, I couldn't get a response from the agent. There might be an issue with the underlying model."
        else:
            content = text

        print(f"✅ Strands agent response received: {len(content)} chars")
        self.session_manager.add_to_conversation_history("assistant", content)

        return {
            "content": content,
            "tools_used": ["Strands Agent", "Session Memory", "GraphRAGTool"],
            "citations": [],
            "confidence": 90,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "source": "strands_agent",
            "session_id": self.session_manager.session_id
        }

    def _invoke_agent_sync(self, agent, prompt: str, timeout: int = 60) -> Optional[str]:
        """
        Safely invoke an async agent method from a synchronous context.
        This handles cases where an event loop may or may not be running.
        """
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                task = asyncio.create_task(agent.invoke_async(prompt))
                response = asyncio.run_coroutine_threadsafe(task, loop).result(timeout)
            else:
                response = asyncio.run(agent.invoke_async(prompt))
        except RuntimeError:
            response = asyncio.run(agent.invoke_async(prompt))
        except Exception as e:
            print(f"[ResponseHandlers] Error invoking agent async: {e}")
            return None

        try:
            if hasattr(response, "content"): return response.content
            elif isinstance(response, dict): return response.get("response") or response.get("content") or json.dumps(response)
            else: return str(response)
        except Exception as e:
            print(f"[ResponseHandlers] Failed to extract text from agent response: {e}")
            return None

    def _generate_contextual_response(self, user_message: str) -> str:
        """
        Generate a conversational fallback answer using history.
        Used for simple greetings or as a final safety net.
        """
        msg_lower = user_message.lower()
        if any(greet in msg_lower for greet in ["hi", "hello", "hey"]):
            if len(self.session_manager.conversation_history) > 1:
                return "Hello again! How can I help you explore the Knowledge Graph?"
            return "Hello! I'm your assistant for the Rumour Verification Framework. What would you like to know?"
        
        return "I can help you query the Knowledge Graph and vector store. Please ask me about claims, posts, or any topic you'd like to explore."
