#!/usr/bin/env python3
"""
Real KG Chat System - Queries actual Knowledge Graph with SPARQL
"""

import streamlit as st
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# RDF and SPARQL imports
try:
    from rdflib import Graph, Namespace
    from rdflib.plugins.sparql import prepareQuery
    RDF_AVAILABLE = True
except ImportError:
    RDF_AVAILABLE = False
    st.error("RDFLib not available - install with: pip install rdflib")

# Strands imports for enhanced responses
try:
    from strands import Agent
    #from strands.models.ollama import OllamaModel
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False

# Namespaces (matching the KG builder)
EX = Namespace("http://example.org/")
SIOC = Namespace("http://rdfs.org/sioc/ns#")
SCHEMA = Namespace("http://schema.org/")
PROV = Namespace("http://www.w3.org/ns/prov#")

class RealKGChatSystem:
    """Real Knowledge Graph Chat System with SPARQL queries"""
    
    def __init__(self):
        self.kg = None
        self.agent = None
        self.kg_loaded = False
        self.current_kg_file = None
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize Strands agent for enhanced responses"""
        try:
            if STRANDS_AVAILABLE:
                # Use centralized model configuration
                from models.llm_models import get_default_model
                ollama_model = get_default_model()
                
                if ollama_model:
                    self.agent = Agent(model=ollama_model)
                else:
                    print("⚠️ No model available from centralized configuration")
        except Exception as e:
            st.warning(f"Could not initialize Strands agent: {e}")
    
    def load_kg_from_file(self, kg_file: str) -> bool:
        """Load Knowledge Graph from TTL file"""
        if not RDF_AVAILABLE:
            return False
        
        try:
            if not os.path.exists(kg_file):
                return False
            
            self.kg = Graph()
            self.kg.parse(kg_file, format="turtle")
            self.kg_loaded = True
            self.current_kg_file = kg_file  # Fix: Set current file
            return True
            
        except Exception as e:
            st.error(f"Failed to load KG: {e}")
            return False
    
    def get_kg_statistics(self) -> Dict[str, Any]:
        """Get basic KG statistics"""
        if not self.kg_loaded or not self.kg:
            return {}
        
        try:
            # Count different node types - Fix: Use proper RDF syntax
            from rdflib import RDF
            post_count = len(list(self.kg.subjects(RDF.type, SIOC.Post)))
            user_count = len(list(self.kg.subjects(RDF.type, SIOC.UserAccount)))
            claim_count = len(list(self.kg.subjects(RDF.type, EX.Claim)))
            platform_count = len(list(self.kg.subjects(RDF.type, EX.Platform)))
            
            # Get platform name
            platform = "Unknown"
            try:
                platform_query = """
                SELECT ?platform WHERE {
                    ?p a ex:Platform ;
                       schema:name ?platform .
                }
                """
                pq = prepareQuery(platform_query, initNs={
                    "ex": EX, 
                    "schema": SCHEMA
                })
                platform_results = list(self.kg.query(pq))
                if platform_results:
                    platform = str(platform_results[0][0])
            except Exception as e:
                pass  # Keep default "Unknown"
            
            # Count relationships - Fix: Use proper predicate query
            reply_count = len(list(self.kg.triples((None, SIOC.reply_of, None))))
            
            return {
                "total_triples": len(self.kg),
                "nodes": {
                    "posts": post_count,
                    "users": user_count,
                    "claims": claim_count,
                    "platforms": platform_count
                },
                "relationships": {
                    "replies": reply_count
                },
                "platform": platform,
                "loaded_from_file": self.current_kg_file,
                "load_timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {"error": str(e), "loaded": False}
    
    def query_kg(self, sparql_query: str) -> List[Dict[str, str]]:
        """Execute SPARQL query on the KG"""
        if not self.kg_loaded or not self.kg:
            return []
        
        try:
            # Prepare query with namespaces
            pq = prepareQuery(sparql_query, initNs={
                "ex": EX, 
                "sioc": SIOC, 
                "schema": SCHEMA, 
                "prov": PROV
            })
            
            qres = self.kg.query(pq)
            
            results = []
            if qres.vars:
                headers = [str(var) for var in qres.vars]
                for row in qres:
                    row_dict = {header: str(value) for header, value in zip(headers, row)}
                    results.append(row_dict)
            
            return results
            
        except Exception as e:
            st.error(f"SPARQL query failed: {e}")
            return []
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer question using KG queries and optional agent enhancement"""
        
        if not self.kg_loaded:
            return {
                "content": "❌ No Knowledge Graph loaded. Please build a KG first using the KG Management page.",
                "tools_used": ["Error Handler"],
                "citations": [],
                "confidence": 0,
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "source": "error"
            }
        
        # Check if this is an enhanced query with chat history context
        original_question = question
        has_context = "Context from recent conversation:" in question
        
        if has_context:
            # Extract the actual question from the enhanced query
            parts = question.split("Current question: ")
            if len(parts) > 1:
                original_question = parts[1].strip()
            else:
                original_question = question
        
        # Get KG statistics
        stats = self.get_kg_statistics()
        
        # Determine query type and execute appropriate SPARQL using original question
        query_results = self._execute_question_query(original_question.lower())
        
        # Format response with enhanced display
        if query_results["results"]:
            result_count = len(query_results["results"])
            query_desc = query_results.get("description", "search results")
            
            content_parts = [f"🧠 **{query_desc.title()}** ({result_count} found):"]
            
            # Enhanced result formatting
            for i, result in enumerate(query_results["results"][:12]):  # Show more results
                if "claimText" in result:
                    claim_text = result['claimText']
                    # Truncate very long claims
                    if len(claim_text) > 300:
                        claim_text = claim_text[:300] + "..."
                    content_parts.append(f"\n**{i+1}.** {claim_text}")
                    
                elif "postText" in result:
                    post_text = result['postText']
                    if len(post_text) > 250:
                        post_text = post_text[:250] + "..."
                    
                    # Add user info if available
                    if "userName" in result:
                        content_parts.append(f"\n**{i+1}.** *by {result['userName']}:* {post_text}")
                    else:
                        content_parts.append(f"\n**{i+1}.** {post_text}")
                        
                elif "userName" in result:
                    user_name = result['userName']
                    if "postText" in result:
                        post_preview = result['postText'][:150] + "..." if len(result['postText']) > 150 else result['postText']
                        content_parts.append(f"\n**{i+1}.** **{user_name}:** {post_preview}")
                    else:
                        content_parts.append(f"\n**{i+1}.** **User:** {user_name}")
                        
                elif "platformName" in result:
                    content_parts.append(f"\n**{i+1}.** **Platform:** {result['platformName']}")
                    
                else:
                    # Generic result display with better formatting
                    result_items = []
                    for k, v in result.items():
                        if len(str(v)) > 100:
                            v = str(v)[:100] + "..."
                        result_items.append(f"**{k}:** {v}")
                    content_parts.append(f"\n**{i+1}.** {' | '.join(result_items)}")
            
            if result_count > 12:
                content_parts.append(f"\n*... and {result_count - 12} more results*")
            
            # Add contextual KG information
            nodes = stats.get('nodes', {})
            relationships = stats.get('relationships', {})
            platform = stats.get('platform', 'Unknown')
            
            content_parts.append(f"\n📊 **{platform} KG:** {nodes.get('posts', 0)} posts, {nodes.get('claims', 0)} claims, {nodes.get('users', 0)} users")
            
            # Dynamic confidence based on result quality
            confidence = min(75 + (result_count * 3) + (15 if result_count >= 5 else 0), 95)
            
        else:
            nodes = stats.get('nodes', {})
            relationships = stats.get('relationships', {})
            platform = stats.get('platform', 'Unknown')
            
            # More helpful "no results" message with specific suggestions
            content_parts = [
                f"🔍 **No specific results found** for '{question}' in the {platform} Knowledge Graph."
            ]
            
            # Add platform-specific suggestions
            if platform.lower() == "reddit":
                content_parts.extend([
                    f"\n📊 **Available Reddit Data:** {nodes.get('posts', 0)} posts, {nodes.get('claims', 0)} claims, {nodes.get('users', 0)} users",
                    f"\n💡 **Try asking:**",
                    f"- 'What claims are available?'",
                    f"- 'Show me all posts'",
                    f"- 'Who are the users?'",
                    f"- 'What posts mention Nepal?'",
                    f"- 'Tell me about earthquake claims'"
                ])
            elif platform.lower() == "twitter":
                content_parts.extend([
                    f"\n📊 **Available Twitter Data:** {nodes.get('posts', 0)} posts, {nodes.get('claims', 0)} claims, {nodes.get('users', 0)} users",
                    f"\n💡 **Try asking:**",
                    f"- 'What claims mention Putin?'",
                    f"- 'Show me all claims'",
                    f"- 'Who posted about this?'",
                    f"- 'What are the rumours?'",
                    f"- 'Show me verified posts'"
                ])
            else:
                content_parts.extend([
                    f"\n📊 **Available Data:** {nodes.get('posts', 0)} posts, {nodes.get('claims', 0)} claims, {nodes.get('users', 0)} users",
                    f"\n💡 **Try asking:**",
                    f"- 'What claims are available?'",
                    f"- 'Show me all posts'",
                    f"- 'Who are the users?'",
                    f"- 'What data do you have?'"
                ])
            
            confidence = 40  # Slightly higher since we're providing helpful guidance
        
        # Enhance with agent if available, using full context if provided
        if self.agent and STRANDS_AVAILABLE and query_results["results"]:
            try:
                # Use full question with context for agent enhancement
                enhancement_question = question if has_context else original_question
                enhancement = self._enhance_response_with_agent(enhancement_question, query_results["results"][:3])
                if enhancement:
                    context_note = " (with conversation context)" if has_context else ""
                    content_parts.append(f"\n🤖 **AI Analysis{context_note}:** {enhancement}")
                    confidence += 15 if has_context else 10
            except Exception as e:
                pass  # Silently fail agent enhancement
        
        return {
            "content": "\n".join(content_parts),
            "tools_used": ["SPARQL Query", "Knowledge Graph"],
            "citations": self._create_citations(query_results["results"]),
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "source": "knowledge_graph",
            "query_type": query_results["query_type"],
            "results_count": len(query_results["results"])
        }
    
    def _execute_question_query(self, question: str) -> Dict[str, Any]:
        """Execute appropriate SPARQL query based on question content"""
        
        # Enhanced query templates with better matching
        queries = {
            "all_claims": {
                "keywords": ["claim", "claims", "what claims", "show claims", "list claims", "tell me claims"],
                "query": """
                    SELECT ?claimText WHERE {
                        ?claim a ex:Claim ;
                               ex:canonicalForm ?claimText .
                    } LIMIT 30
                """,
                "description": "All extracted claims"
            },
            "all_posts": {
                "keywords": ["post", "posts", "what posts", "show posts", "content", "data", "available", "what data"],
                "query": """
                    SELECT ?postText WHERE {
                        ?post a sioc:Post ;
                              schema:headline ?postText .
                    } LIMIT 20
                """,
                "description": "All posts in the knowledge graph"
            },
            "all_users": {
                "keywords": ["user", "users", "who", "people", "authors", "show users"],
                "query": """
                    SELECT DISTINCT ?userName WHERE {
                        ?user a sioc:UserAccount ;
                              schema:name ?userName .
                    } LIMIT 20
                """,
                "description": "All users in the knowledge graph"
            },
            "posts_and_users": {
                "keywords": ["posted", "author", "created", "wrote"],
                "query": """
                    SELECT ?userName ?postText WHERE {
                        ?post sioc:has_creator ?user ;
                              schema:headline ?postText .
                        ?user schema:name ?userName .
                    } LIMIT 15
                """,
                "description": "Posts with their authors"
            },
            "claims_about_topic": {
                "keywords": ["about", "mention", "topic", "related to"],
                "query": """
                    SELECT ?claimText WHERE {
                        ?claim a ex:Claim ;
                               ex:canonicalForm ?claimText .
                        FILTER(CONTAINS(LCASE(?claimText), "{topic}"))
                    } LIMIT 20
                """,
                "description": "Claims about specific topics"
            },
            "posts_by_label": {
                "keywords": ["rumour", "rumor", "non-rumour", "non-rumor", "labeled", "label", "verified"],
                "query": """
                    SELECT ?postText ?label WHERE {
                        ?post ex:label ?label ;
                              schema:headline ?postText .
                        FILTER(CONTAINS(LCASE(?label), "{label}"))
                    } LIMIT 20
                """,
                "description": "Posts by verification label"
            },
            "platform_info": {
                "keywords": ["platform", "source", "from where", "reddit", "twitter", "pheme"],
                "query": """
                    SELECT ?platformName WHERE {
                        ?platform a ex:Platform ;
                                  schema:name ?platformName .
                    }
                """,
                "description": "Platform information"
            },
            "general_search": {
                "keywords": ["search", "find", "look for", "tell me about"],
                "query": """
                    SELECT ?claimText WHERE {
                        ?claim a ex:Claim ;
                               ex:canonicalForm ?claimText .
                    } LIMIT 10
                    UNION
                    SELECT ?postText WHERE {
                        ?post a sioc:Post ;
                              schema:headline ?postText .
                    } LIMIT 10
                """,
                "description": "General search across claims and posts"
            }
        }
        
        # Enhanced matching with fuzzy search
        question_lower = question.lower()
        best_match = None
        best_score = 0
        
        # Find best matching query using fuzzy matching
        for query_type, query_info in queries.items():
            score = 0
            for keyword in query_info["keywords"]:
                if keyword.lower() in question_lower:
                    score += len(keyword)  # Longer matches get higher scores
            
            if score > best_score:
                best_score = score
                best_match = (query_type, query_info)
        
        # Execute best match if found
        if best_match and best_score > 0:
            query_type, query_info = best_match
            sparql_query = query_info["query"]
            
            # Replace placeholders
            if "{topic}" in sparql_query:
                topic = self._extract_topic_from_question(question_lower, query_info["keywords"])
                sparql_query = sparql_query.replace("{topic}", topic)
            
            if "{label}" in sparql_query:
                label = "rumour" if "rumour" in question_lower or "rumor" in question_lower else "non-rumour"
                sparql_query = sparql_query.replace("{label}", label)
            
            results = self.query_kg(sparql_query)
            return {
                "results": results,
                "query_type": query_type,
                "description": query_info["description"]
            }
        
        # Fallback 1: Try broad keyword search
        keywords = [word for word in question.split() if len(word) > 3 and word.lower() not in ['what', 'show', 'tell', 'about', 'the']]
        if keywords:
            keyword = keywords[0].lower()
            
            # Try claims first
            sparql_query = f"""
                SELECT ?claimText WHERE {{
                    ?claim a ex:Claim ;
                           ex:canonicalForm ?claimText .
                    FILTER(CONTAINS(LCASE(?claimText), "{keyword}"))
                }} LIMIT 15
            """
            results = self.query_kg(sparql_query)
            
            if results:
                return {
                    "results": results,
                    "query_type": "keyword_search_claims",
                    "description": f"Claims containing '{keyword}'"
                }
            
            # Try posts if no claims found
            sparql_query = f"""
                SELECT ?postText WHERE {{
                    ?post a sioc:Post ;
                          schema:headline ?postText .
                    FILTER(CONTAINS(LCASE(?postText), "{keyword}"))
                }} LIMIT 15
            """
            results = self.query_kg(sparql_query)
            
            if results:
                return {
                    "results": results,
                    "query_type": "keyword_search_posts",
                    "description": f"Posts containing '{keyword}'"
                }
        
        # Fallback 2: Return sample data to show what's available
        sample_query = """
            SELECT ?claimText WHERE {
                ?claim a ex:Claim ;
                       ex:canonicalForm ?claimText .
            } LIMIT 5
        """
        results = self.query_kg(sample_query)
        
        return {
            "results": results,
            "query_type": "sample_data",
            "description": "Sample claims from the knowledge graph"
        }
    
    def _extract_topic_from_question(self, question: str, keywords: List[str]) -> str:
        """Extract topic from question for targeted queries"""
        # Remove query keywords to find the actual topic
        words = question.split()
        topic_words = [word for word in words if word not in keywords and len(word) > 2]
        
        # Extract topics from question
        common_topics = ["earthquake", "nepal", "disaster", "news", "breaking", "update"]
        
        for topic in common_topics:
            if topic in question:
                return topic
        
        # Return first significant word if no common topic found
        return topic_words[0] if topic_words else "topic"
    
    def _enhance_response_with_agent(self, question: str, results: List[Dict]) -> Optional[str]:
        """Enhance response using Strands agent"""
        if not self.agent or not results:
            return None
        
        try:
            # Create context from results
            context = "\n".join([
                f"- {result.get('claimText', result.get('postText', str(result)))[:100]}"
                for result in results[:3]
            ])
            
            prompt = f"""Based on this data from a rumour verification knowledge graph, provide a brief analysis of the question: "{question}"

Data found:
{context}

Provide a 1-2 sentence analysis focusing on patterns, credibility, or insights."""
            
            response = self.agent.invoke(prompt)
            if isinstance(response, dict) and "response" in response:
                return response["response"][:200]  # Limit length
            elif isinstance(response, str):
                return response[:200]
            
        except Exception:
            pass  # Silently fail
        
        return None
    
    def _create_citations(self, results: List[Dict]) -> List[Dict[str, str]]:
        """Create citations from query results"""
        citations = []
        for i, result in enumerate(results[:5]):  # Limit citations
            if "claimText" in result:
                citations.append({
                    "source": f"Knowledge Graph Claim {i+1}",
                    "content": result["claimText"][:100] + "...",
                    "type": "claim"
                })
            elif "postText" in result:
                citations.append({
                    "source": f"Knowledge Graph Post {i+1}",
                    "content": result["postText"][:100] + "...",
                    "type": "post"
                })
        return citations

# Global instance
kg_chat_system = RealKGChatSystem()

def get_real_kg_response(user_query: str, chat_options: Dict[str, Any]) -> Dict[str, Any]:
    """Get response from real KG chat system"""
    
    # Check if user has manually selected a KG in session state
    import streamlit as st
    
    # Priority 1: Use manually selected KG (highest priority)
    if hasattr(st, 'session_state') and "selected_kg_file" in st.session_state:
        selected_kg_file = st.session_state.selected_kg_file
        
        # Force reload if different from current
        if kg_chat_system.current_kg_file != selected_kg_file:
            if kg_chat_system.load_kg_from_file(selected_kg_file):
                print(f"Using manually selected KG: {selected_kg_file}")
            else:
                print(f"Failed to load manually selected KG: {selected_kg_file}")
    
    # Priority 2: If no KG loaded at all, try auto-selection as fallback
    elif not kg_chat_system.kg_loaded or not kg_chat_system.current_kg_file:
        selected_kg = select_appropriate_kg(user_query)
        
        if selected_kg:
            if kg_chat_system.load_kg_from_file(str(selected_kg)):
                print(f"Auto-selected KG: {selected_kg}")
                kg_chat_system.current_kg_file = str(selected_kg)
    
    # Priority 3: Use whatever KG is currently loaded (don't change)
    # This prevents auto-switching when a KG is already loaded
    
    # Enhance response with chat history if available
    enhanced_query = user_query
    if chat_options.get("has_history") and chat_options.get("chat_history"):
        # Add context prefix for better responses
        history_context = "\n".join(chat_options["chat_history"][-3:])  # Last 3 exchanges
        enhanced_query = f"Context from recent conversation:\n{history_context}\n\nCurrent question: {user_query}"
    
    return kg_chat_system.answer_question(enhanced_query)

def select_appropriate_kg(user_query: str) -> Optional[Path]:
    """Select the most appropriate KG file based on query content"""
    
    # Get all available KG files
    kg_files = list(Path(".").glob("*_kg.ttl"))
    
    if not kg_files:
        return None
    
    query_lower = user_query.lower()
    
    # Reddit-specific keywords
    reddit_keywords = ["reddit", "subreddit", "r/", "nepal", "worldnews", "technology"]
    
    # PHEME-specific keywords  
    pheme_keywords = ["pheme", "missing", "coup", "political"]
    
    # Check for Reddit KG preference
    if any(keyword in query_lower for keyword in reddit_keywords):
        # Look for Reddit KG files
        reddit_kgs = [f for f in kg_files if "reddit" in f.name.lower()]
        if reddit_kgs:
            # Return the most recent Reddit KG
            return max(reddit_kgs, key=lambda x: x.stat().st_mtime)
    
    # Check for PHEME KG preference
    if any(keyword in query_lower for keyword in pheme_keywords):
        # Look for PHEME KG files
        pheme_kgs = [f for f in kg_files if "pheme" in f.name.lower()]
        if pheme_kgs:
            return max(pheme_kgs, key=lambda x: x.stat().st_mtime)
    
    # Default: return the most recent KG
    return max(kg_files, key=lambda x: x.stat().st_mtime)

def show_kg_chat_interface():
    """Show KG chat interface for testing"""
    st.title("🧠 Real Knowledge Graph Chat")
    
    # KG status
    stats = kg_chat_system.get_kg_statistics()
    if stats.get("total_triples", 0) > 0:
        nodes = stats.get('nodes', {})
        st.success(f"✅ KG Loaded: {nodes.get('posts', 0)} posts, {nodes.get('claims', 0)} claims")
    else:
        st.warning("⚠️ No KG loaded. Build a KG first using KG Management.")
        
        # Try to find and load KG files
        kg_files = list(Path(".").glob("*_kg.ttl"))
        if kg_files:
            selected_kg = st.selectbox("Available KG files:", kg_files)
            if st.button("Load KG"):
                if kg_chat_system.load_kg_from_file(str(selected_kg)):
                    st.success("KG loaded successfully!")
                    st.rerun()
    
    # Chat interface
    if stats.get("total_triples", 0) > 0:
        st.subheader("Ask questions about the Knowledge Graph:")
        
        # Sample questions
        sample_questions = [
            "What claims are in the data?",
            "Show me posts labeled as rumours",
            "Who are the most active users?",
            "What are all the claims in the dataset?",
            "Show me reply conversations"
        ]
        
        selected_sample = st.selectbox("Sample questions:", [""] + sample_questions)
        
        user_input = st.text_input("Your question:", value=selected_sample)
        
        if st.button("Ask") and user_input:
            with st.spinner("Querying Knowledge Graph..."):
                response = kg_chat_system.answer_question(user_input)
            
            st.write("**Response:**")
            st.write(response["content"])
            
            # Show metadata
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{response['confidence']}%")
            with col2:
                st.metric("Results", response.get("results_count", 0))
            with col3:
                st.metric("Query Type", response.get("query_type", "unknown"))

if __name__ == "__main__":
    show_kg_chat_interface()
