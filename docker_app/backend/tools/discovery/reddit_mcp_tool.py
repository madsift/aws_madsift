"""
Reddit MCP Tool - Fetch Reddit posts using MCP + Strands Agent
Uses Smithery MCP server for Reddit data retrieval
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urlencode
from pydantic import BaseModel, Field
from common.aws_clients import get_secret
from common.llm_models import get_default_model

from tools.base import BaseRumourTool, ToolMetadata, ToolCategory

region =  os.getenv("AWS_REGION", "us-east-1")
api_secret_name = os.getenv("SECRET_NAME", "api_secret")

# Pydantic models for Reddit posts
class RedditPost(BaseModel):
    """Schema for a single Reddit post fetched by the agent."""
    id: str = Field(description="The unique ID of the post.")
    title: str = Field(description="The title of the post.")
    selftext: Optional[str] = Field(default=None, description="The body text of the post, if any.")
    score: int = Field(description="The score or number of upvotes.")
    permalink: str = Field(description="The permanent link to the post.")
    author: str = Field(description="The username of the post's author.")
    created_utc: float = Field(description="The UTC timestamp of when the post was created.")

class PostExtractionResult(BaseModel):
    """A container for the list of Reddit posts returned by the agent."""
    posts: List[RedditPost]

class RedditMCPTool(BaseRumourTool):
    """Reddit MCP Tool for fetching posts via Smithery MCP server"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        self._logger = logging.getLogger("tools.reddit_mcp_tool")
        
        # Configuration from environment
        self.smithery_key = get_secret(api_secret_name, region, 'smithery')
        self.base_url = os.getenv("REDDIT_MCP_BASE_URL", "https://server.smithery.ai/@ruradium/mcp-reddit/mcp")
        
        if not self.smithery_key:
            self._logger.warning("SMITHERY_KEY not found in environment variables")
    
    @classmethod
    def create_instance(cls) -> 'RedditMCPTool':
        """Create an instance of RedditMCPTool"""
        metadata = ToolMetadata(
            name="reddit_mcp_tool",
            description="Fetch Reddit posts from subreddits using MCP + Strands Agent",
            category=ToolCategory.DISCOVERY
        )
        return cls(metadata)
    
    def fetch_latest_posts_from_mcp(self, subreddit: str = "r/Nepal", limit: int = 5) -> List[Dict[str, Any]]:
        """Fetch the latest N Reddit posts from a subreddit using MCP + Strands Agent."""
        
        if not self.smithery_key:
            self._logger.error("SMITHERY_KEY not configured")
            return []
        
        try:
            # Import MCP and Strands dependencies
            import json
            from urllib.parse import urlencode
            from strands.tools.mcp import MCPClient
            from mcp.client.streamable_http import streamablehttp_client
            from strands import Agent
            #from strands.models.ollama import OllamaModel
            
            # Build MCP URL with API key
            params = {"api_key": self.smithery_key}
            url = f"{self.base_url}?{urlencode(params)}"
            
            self._logger.info(f"Connecting to Reddit MCP server: {self.base_url}")
            
            # Create MCP client
            reddit_client = MCPClient(lambda: streamablehttp_client(url))
            
            # Use centralized model configuration
            model = get_default_model()
            
            # Fetch posts using MCP + Strands
            with reddit_client:
                tools = reddit_client.list_tools_sync()
                agent = Agent(model=model, tools=tools)
                
                prompt = f"""Get me the latest {limit} posts from https://www.reddit.com/{subreddit}.
                Include: id, title, selftext, score, permalink, author, and created_utc."""
                
                self._logger.info(f"Fetching {limit} posts from {subreddit}")
                agent_response = agent.structured_output(PostExtractionResult, prompt)
            
               
                if isinstance(agent_response, str):
                    try:
                        # Parse the JSON string into a Python dictionary
                        parsed_data = json.loads(agent_response)
                        # Create the Pydantic model from the parsed dictionary
                        structured_result = PostExtractionResult(**parsed_data)
                    except (json.JSONDecodeError, TypeError) as e:
                        self._logger.error(f"Failed to parse string response from agent: {e}")
                        return []
                else:
                    # If it's already a Pydantic object, use it directly
                    structured_result = agent_response
                
                # Convert to dictionaries
                posts_as_dicts = [post.model_dump() for post in structured_result.posts]
                
                self._logger.info(f"Successfully fetched {len(posts_as_dicts)} posts from {subreddit}")
                
                return posts_as_dicts[:limit]
                
        except ImportError as e:
            self._logger.error(f"Missing dependencies for MCP/Strands: {e}")
            return []
        except Exception as e:
            self._logger.error(f"Error fetching Reddit posts: {e}")
            return []
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute Reddit MCP tool to fetch posts
        
        Args:
            subreddit: Subreddit to fetch from (e.g., 'r/Nepal')
            limit: Number of posts to fetch (default: 5)
            
        Returns:
            Dict containing posts and metadata
        """
        # Validate parameters
        if not self.validate_input(**kwargs):
            return {
                "success": False,
                "error": "Invalid parameters",
                "posts": [],
                "metadata": {}
            }
        
        subreddit = kwargs.get("subreddit", "r/Nepal")
        limit = kwargs.get("limit", 5)
        
        # Ensure subreddit format
        if not subreddit.startswith("r/"):
            subreddit = f"r/{subreddit}"
        
        try:
            # Fetch posts using MCP
            posts = self.fetch_latest_posts_from_mcp(subreddit, limit)
            
            if not posts:
                return {
                    "success": False,
                    "error": f"No posts found for {subreddit}",
                    "posts": [],
                    "metadata": {
                        "subreddit": subreddit,
                        "requested_limit": limit,
                        "actual_count": 0
                    }
                }
            
            # Add platform metadata for KG building
            for post in posts:
                post["platform"] = "Reddit"
                post["subreddit"] = subreddit
                post["fetched_at"] = datetime.now().isoformat()
            
            return {
                "success": True,
                "posts": posts,
                "metadata": {
                    "subreddit": subreddit,
                    "requested_limit": limit,
                    "actual_count": len(posts),
                    "platform": "Reddit",
                    "fetched_at": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            self._logger.error(f"Error in Reddit MCP tool execution: {e}")
            return {
                "success": False,
                "error": str(e),
                "posts": [],
                "metadata": {
                    "subreddit": subreddit,
                    "requested_limit": limit,
                    "actual_count": 0
                }
            }
    
    def validate_input(self, **kwargs) -> bool:
        """Validate input parameters (required by BaseRumourTool)"""
        subreddit = kwargs.get("subreddit")
        limit = kwargs.get("limit", 5)
        
        if not subreddit:
            self._logger.error("Parameter 'subreddit' is required")
            return False
        
        if not isinstance(limit, int) or limit < 1 or limit > 50:
            self._logger.error("Parameter 'limit' must be an integer between 1 and 50")
            return False
        
        return True
    
    def _validate_parameters(self, **kwargs) -> bool:
        """Validate input parameters (legacy method)"""
        return self.validate_input(**kwargs)
    
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """Get parameters schema for the tool"""
        return {
            "type": "object",
            "properties": {
                "subreddit": {
                    "type": "string",
                    "description": "Subreddit to fetch posts from (e.g., 'r/Nepal', 'r/worldnews')",
                    "examples": ["r/Nepal", "r/worldnews", "r/technology"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of posts to fetch",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 5
                }
            },
            "required": ["subreddit"]
        }

# Export for tool system
def create_reddit_mcp_tool() -> RedditMCPTool:
    """Factory function to create RedditMCPTool instance"""
    return RedditMCPTool.create_instance()
