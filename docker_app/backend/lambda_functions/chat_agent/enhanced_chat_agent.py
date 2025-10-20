#!/usr/bin/env python3
"""
Enhanced Chat Agent - Refactored modular version

This is the main orchestrator for the Enhanced Chat Agent with:
- Modular architecture with separated concerns
- Session management
- Knowledge Graph operations
- Hybrid Graph-RAG functionality
- Response handling strategies
"""

from datetime import datetime
from typing import Dict, Any, Optional
from strands.tools.decorator import tool

from common.llm_models import get_default_model
from config import SYSTEM_PROMPT
from session_manager import SessionManager, STRANDS_AVAILABLE, STRANDS_ERROR
from kg_operations import KGOperations
from graph_rag_tool import GraphRAGTool
from response_handlers import ResponseHandlers

# Strands imports with better error handling
try:
    from strands import Agent
    STRANDS_AGENT_AVAILABLE = True
    print("✅ Strands Agent imports successful")
except ImportError as e:
    STRANDS_AGENT_AVAILABLE = False
    print(f"⚠️ Strands Agent not available: {e}")
except Exception as e:
    STRANDS_AGENT_AVAILABLE = False
    print(f"⚠️ Strands Agent error: {e}")


class EnhancedChatAgent:
    """Enhanced chat agent with modular architecture and intelligent tool routing"""

    def __init__(self, session_id: Optional[str] = None):
        # Initialize modular components
        self.session_manager = SessionManager(session_id)
        self.kg_operations = KGOperations()
        self.response_handlers = ResponseHandlers(self.session_manager, self.kg_operations)
        
        # Agent state
        self.agent = None
        self.initialized = False
        self.model = None
        self.graph_rag_tool = None


    def _initialize_agent(self, bucket_name: str, username: str, kg_path: str, ldb_path: Optional[str] = None, region: Optional[str] = None, ldb_table_name: Optional[str] = None):
        """Initialize Strands agent with session management, KG tool, and hybrid Graph-RAG tool

        ldb_path: optional path to LanceDB vector store (e.g. s3://bucket/prefix or local path)
        ldb_table_name: optional specific table name within the LanceDB
        """
        print('Initializing agent .......')
        
        # Initialize session manager
        self.session_manager.initialize_session_manager(bucket_name, username, region)

        # Load knowledge graph and vector store
        self.kg_operations.load_knowledge_graph(kg_path, region)
        
        try:
            self.kg_operations.load_vector_store(ldb_path, username=username, table_name=ldb_table_name)
        except Exception as e:
            print(f"⚠️ Vector store not loaded: {e}")

        # Initialize Graph RAG tool
        self.graph_rag_tool = GraphRAGTool(
            self.kg_operations.kg_store,
            self.kg_operations.vector_db,
            self.kg_operations.vector_table
        )

        # Define tools as closures to maintain existing behavior
        @tool
        def query_knowledge_graph_oxigraph(sparql_query: str) -> str:
            """
            Executes a SPARQL query against the loaded in-memory Oxigraph knowledge graph.
            """
            return self.kg_operations.query_knowledge_graph_oxigraph(sparql_query)

        @tool
        def query_knowledge_graph_fuseki(query: str) -> str:
            """Query external Fuseki endpoint"""
            return self.kg_operations.query_knowledge_graph_fuseki(query)

        @tool
        def query_combined_graph_rag(
            user_query: str,
            top_k_graph: int = 5,
            top_k_vector: int = 10,
            re_rank_with_cross_encoder: bool = False
        ) -> str:
            """
            Full-power hybrid Graph-RAG tool
            """
            return self.graph_rag_tool.query_combined_graph_rag(
                user_query, top_k_graph, top_k_vector, re_rank_with_cross_encoder
            )

        # Create agent with tools
        if STRANDS_AGENT_AVAILABLE and self.session_manager.session_manager:
            self.model = get_default_model()
            if not self.model:
                raise Exception("No model available from centralized configuration")
            
            self.agent = Agent(
                model=self.model,
                session_manager=self.session_manager.session_manager,
                tools=[query_combined_graph_rag, query_knowledge_graph_oxigraph],
                system_prompt=SYSTEM_PROMPT
            )

            print(f"✅ Added KG + Hybrid Graph-RAG tool to Strands agent")
            print(f"📊 Agent tools: {self.agent.tool_names}")
            print(f"✅ Enhanced agent with default model initialized successfully with session: {self.session_manager.session_id}")
        else:
            print(f"⚠️ Strands Agent not available, using fallback mode")
            
        self.initialized = True

    @property
    def session_id(self):
        """Get session ID from session manager"""
        return self.session_manager.session_id

    @session_id.setter
    def session_id(self, value):
        """Set session ID in session manager"""
        self.session_manager.session_id = value

    def get_response(self, user_message: str, chat_options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get response from the enhanced agent with intelligent routing"""

        # Add to conversation history
        self.session_manager.add_to_conversation_history("user", user_message)

        # If we have a working Strands agent, use it
        if self.initialized and self.agent:
            try:
                return self.response_handlers.get_strands_response(self.agent, user_message, chat_options or {})
            except Exception as e:
                print(f"⚠️ Strands agent failed: {e}")

        # Intelligent fallback with session management
        return self.response_handlers.get_intelligent_fallback_response(user_message, chat_options or {})

    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the current session"""
        session_info = self.session_manager.get_session_info()
        session_info.update({
            "agent_available": self.agent is not None,
            "initialized": self.initialized,
        })
        return session_info
