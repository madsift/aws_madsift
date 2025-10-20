#!/usr/bin/env python3
"""
Response Handlers - Different response generation strategies

This module handles:
- Strands agent response generation
- Intelligent fallback responses
- Conversation context responses
- KG-related response routing
"""

import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from common.llm_models import get_default_model


class ResponseHandlers:
    """Handles different types of response generation"""
    
    def __init__(self, session_manager, kg_operations):
        self.session_manager = session_manager
        self.kg_operations = kg_operations

    def _invoke_agent_sync(self, agent, prompt: str, timeout: int = 60) -> Optional[str]:
        """
        Invoke agent.invoke_async(prompt) and return the textual content.
        Works both when an event loop is running or not.
        Returns None on failure.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # create a task and wrap it to run thread-safely
                task = asyncio.create_task(agent.invoke_async(prompt))
                response = asyncio.run_coroutine_threadsafe(task, loop).result(timeout)
            else:
                # no running loop in this thread: run directly
                response = asyncio.run(agent.invoke_async(prompt))
        except RuntimeError:
            # fallback if get_event_loop() fails for some reason
            try:
                response = asyncio.run(agent.invoke_async(prompt))
            except Exception as e:
                print(f"[ResponseHandlers] _invoke_agent_sync runtime error: {e}")
                return None
        except Exception as e:
            print(f"[ResponseHandlers] Error invoking agent async: {e}")
            return None

        # Normalize response -> text
        try:
            if isinstance(response, dict):
                # common keys used by different runtimes
                return response.get("response") or response.get("content") or response.get("output") or json.dumps(response)
            elif hasattr(response, "content"):
                return response.content
            else:
                return str(response)
        except Exception as e:
            print(f"[ResponseHandlers] Failed to extract text from agent response: {e}")
            return None

    def get_strands_response(self, agent, user_message: str, chat_options: Dict[str, Any]) -> Dict[str, Any]:
        """Get response from Strands agent (uses invoke_async safely)."""

        # Add KG context to the message
        enhanced_message = self.kg_operations._add_kg_context(user_message)

        print(f"🤖 Invoking Strands agent with message: {enhanced_message[:100]}...")

        # Use the same async-safe invocation pattern (delegated to helper)
        text = self._invoke_agent_sync(agent, enhanced_message)
        if text is None:
            # fallback: return an error-like assistant response
            print("[ResponseHandlers] Agent returned no text.")
            content = "Sorry — I couldn't get a response from the agent."
        else:
            content = text

        print(f"✅ Strands agent response received: {len(content)} chars")

        # Add to conversation history
        self.session_manager.add_to_conversation_history("assistant", content)

        return {
            "content": content,
            "tools_used": ["Strands Agent", "Session Memory"],
            "citations": [],
            "confidence": 90,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "source": "strands_agent",
            "session_id": self.session_manager.session_id
        }

    def _generate_contextual_response(self, user_message: str) -> str:
        """
        Generate a conversational fallback answer using history.
        Used when no KG or vector results are found.
        """
        if len(self.session_manager.conversation_history) > 1:
            context_intro = "Based on our earlier conversation, "
        else:
            context_intro = ""

        msg = user_message.lower()
        if any(greet in msg for greet in ["hi", "hello", "hey"]):
            if len(self.session_manager.conversation_history) > 1:
                return (
                    "Hello again! I remember our earlier discussion. "
                    "Would you like to keep exploring the Knowledge Graph or vector store?"
                )
            return (
                "Hello! I'm your assistant for the Rumour Verification Framework. "
                "I can query Knowledge Graphs, search the LanceDB vector store, "
                "and maintain context across sessions. What would you like to do?"
            )

        return (
            f"{context_intro}I'm here to help you navigate the Knowledge Graph and vector store. "
            "Ask me about specific claims, posts, or datasets—or anything else you'd like to explore."
        )

    def _should_query_kg(self, user_message: str, agent_response: str = "") -> bool:
        """
        Heuristic to decide if a user query should trigger KG/vector retrieval.
        Checks for schema-related keywords or LLM hints.
        """
        msg = user_message.lower()
        kg_terms = [
            "claim", "claims", "post", "posts", "user", "users",
            "dataset", "data", "graph", "node", "edge",
            "find", "search", "list", "show", "retrieve",
            "knowledge graph", "triple", "sparql", "rdf", "sioc", "schema"
        ]

        if any(t in msg for t in kg_terms):
            return True

        # fall back to inspecting agent suggestion
        resp = agent_response.lower()
        if any(p in resp for p in ["query the", "from the graph", "in the kg", "knowledge graph"]):
            return True

        return False

    def get_intelligent_fallback_response(self, user_message: str, chat_options: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent fallback with session management and smart routing"""

        print(f"🧠 Using intelligent fallback with session management")

        # Check conversation history for context
        context_response = self.session_manager._check_conversation_context(user_message)
        if context_response:
            print("✅ Answered from conversation history")
            return context_response

        # Check if this is a KG-related question - ALWAYS use real KG data
        if self._should_query_kg(user_message, ""):
            print(f"🔍 Detected KG question, querying real Knowledge Graph...")
            try:
                from .real_kg_chat import kg_chat_system

                # Check if KG is loaded
                if not kg_chat_system.kg_loaded:
                    content = "No Knowledge Graph is currently loaded. Please load a Knowledge Graph first to query data about claims, posts, or users."
                else:
                    # Get KG statistics for context
                    stats = kg_chat_system.get_kg_statistics()
                    platform = stats.get('platform', 'Unknown')
                    nodes = stats.get('nodes', {})

                    print(f"🔍 Querying {platform} KG with: {user_message[:50]}...")

                    # Execute the query using real KG system
                    kg_response = kg_chat_system.answer_question(user_message)

                    if kg_response.get('results_count', 0) > 0:
                        content = kg_response.get('content', '')
                        print(f"✅ Found {kg_response.get('results_count', 0)} results from KG")

                        # Add conversational context if we have history
                        if len(self.session_manager.conversation_history) > 1:
                            content = f"Based on our conversation and the {platform} Knowledge Graph:\n\n{content}"

                        # Add to conversation history
                        self.session_manager.add_to_conversation_history("assistant", content)

                        return {
                            "content": content,
                            "tools_used": ["Session Manager", "Knowledge Graph", f"{platform} KG Query"],
                            "citations": [],
                            "confidence": kg_response.get('confidence', 85),
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "source": "enhanced_fallback_kg",
                            "session_id": self.session_manager.session_id
                        }
                    else:
                        print(f"⚠️ No results found in KG")
                        content = f"No specific results found for '{user_message}' in the {platform} Knowledge Graph. The {platform} KG contains {nodes.get('posts', 0)} posts and {nodes.get('claims', 0)} claims. Try asking about available claims, posts, or users."

            except Exception as kg_error:
                print(f"❌ KG query failed: {kg_error}")
                content = "I'm having trouble accessing the Knowledge Graph right now. Please make sure it's loaded and try again."
        else:
            # General conversation response with history context
            content = self._generate_contextual_response(user_message)

        # Add to conversation history
        self.session_manager.add_to_conversation_history("assistant", content)

        return {
            "content": content,
            "tools_used": ["Session Manager", "Conversation Context"],
            "citations": [],
            "confidence": 75,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "source": "enhanced_fallback",
            "session_id": self.session_manager.session_id
        }