"""
Reddit Public API Tool - Fetch Reddit posts using public Reddit JSON API
No authentication required, more reliable than MCP server
"""

import os
import logging
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from tools.base import BaseRumourTool, ToolMetadata, ToolCategory

class RedditPost(BaseModel):
    """Schema for a single Reddit post."""
    id: str = Field(description="The unique ID of the post.")
    title: str = Field(description="The title of the post.")
    selftext: Optional[str] = Field(default=None, description="The body text of the post, if any.")
    score: int = Field(description="The score or number of upvotes.")
    permalink: str = Field(description="The permanent link to the post.")
    author: str = Field(description="The username of the post's author.")
    created_utc: float = Field(description="The UTC timestamp of when the post was created.")

class RedditPublicTool(BaseRumourTool):
    """Reddit Public API Tool for fetching posts via Reddit's JSON API"""
    
    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        self._logger = logging.getLogger("tools.reddit_public_tool")
        
        # User agent for Reddit API (required)
        self.user_agent = "MadSift KG Builder/1.0"
    
    @classmethod
    def create_instance(cls) -> 'RedditPublicTool':
        """Create an instance of RedditPublicTool"""
        metadata = ToolMetadata(
            name="reddit_public_tool",
            description="Fetch Reddit posts from subreddits using public Reddit JSON API",
            category=ToolCategory.DISCOVERY
        )
        return cls(metadata)
    
    def fetch_posts_from_public_api(self, subreddit: str = "r/Nepal", limit: int = 25) -> List[Dict[str, Any]]:
        """
        Fetch the latest N Reddit posts from a subreddit using public JSON API.
        
        Args:
            subreddit: Subreddit name (e.g., 'r/Nepal' or 'Nepal')
            limit: Number of posts to fetch (max 100)
        
        Returns:
            List of post dictionaries
        """
        # Ensure subreddit format (remove r/ if present for API call)
        clean_subreddit = subreddit.replace('r/', '')
        
        # Reddit JSON API endpoint
        url = f"https://www.reddit.com/r/{clean_subreddit}/new.json"
        
        # Request parameters
        params = {
            'limit': min(limit, 100)  # Reddit API max is 100
        }
        
        headers = {
            'User-Agent': self.user_agent
        }
        
        try:
            self._logger.info(f"FINAL REQUEST: URL='{url}', Headers='{headers}'")
            self._logger.info(f"Fetching {limit} posts from r/{clean_subreddit} via public API")
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract posts from Reddit JSON structure
            posts = []
            if 'data' in data and 'children' in data['data']:
                for child in data['data']['children']:
                    post_data = child.get('data', {})
                    
                    # Extract relevant fields
                    post = {
                        'id': post_data.get('id', ''),
                        'title': post_data.get('title', ''),
                        'selftext': post_data.get('selftext', ''),
                        'score': post_data.get('score', 0),
                        'permalink': f"https://www.reddit.com{post_data.get('permalink', '')}",
                        'author': post_data.get('author', '[deleted]'),
                        'created_utc': post_data.get('created_utc', 0)
                    }
                    
                    posts.append(post)
                
                self._logger.info(f"Successfully fetched {len(posts)} posts from r/{clean_subreddit}")
            else:
                self._logger.warning(f"Unexpected JSON structure from Reddit API for r/{clean_subreddit}")
            
            return posts[:limit]
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                self._logger.error(f"Subreddit r/{clean_subreddit} not found (404)")
            elif e.response.status_code == 403:
                self._logger.error(f"Access forbidden to r/{clean_subreddit} (403) - may be private")
            else:
                self._logger.error(f"HTTP error fetching from r/{clean_subreddit}: {e}", exc_info=True)
            return []
        except requests.exceptions.Timeout:
            self._logger.error(f"Timeout fetching posts from r/{clean_subreddit}")
            return []
        except requests.exceptions.RequestException as e:
            self._logger.error(f"Request error fetching from r/{clean_subreddit}: {e}", exc_info=True)
            return []
        except Exception as e:
            self._logger.error(f"Unexpected error fetching from r/{clean_subreddit}: {e}", exc_info=True)
            return []
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute Reddit Public API tool to fetch posts
        
        Args:
            subreddit: Subreddit to fetch from (e.g., 'r/Nepal' or 'Nepal')
            limit: Number of posts to fetch (default: 25, max: 100)
            
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
        limit = kwargs.get("limit", 25)
        
        # Ensure subreddit format
        if not subreddit.startswith("r/"):
            subreddit = f"r/{subreddit}"
        
        try:
            # Fetch posts using public API
            posts = self.fetch_posts_from_public_api(subreddit, limit)
            
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
            self._logger.error(f"Error in Reddit Public API tool execution: {e}", exc_info=True)
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
        """Validate input parameters"""
        subreddit = kwargs.get("subreddit")
        limit = kwargs.get("limit", 25)
        
        if not subreddit:
            self._logger.error("Parameter 'subreddit' is required")
            return False
        
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            self._logger.error("Parameter 'limit' must be an integer between 1 and 100")
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
                    "description": "Subreddit to fetch posts from (e.g., 'r/Nepal', 'worldnews')",
                    "examples": ["r/Nepal", "worldnews", "r/technology"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of posts to fetch",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 25
                }
            },
            "required": ["subreddit"]
        }

# Export for tool system
def create_reddit_public_tool() -> RedditPublicTool:
    """Factory function to create RedditPublicTool instance"""
    return RedditPublicTool.create_instance()

if __name__ == "__main__":
    # Simple manual test for RedditPublicTool
    import json
    logging.basicConfig(level=logging.INFO)

    # Create an instance of the tool
    tool = create_reddit_public_tool()

    # Example 1: Fetch latest 5 posts from r/Nepal
    print("\n--- Fetching from r/Nepal ---")
    result = tool.execute(subreddit="r/worldnews", limit=5)
    print (len(result.get("posts", [])))
    #print(json.dumps(result, indent=2))

    # Example 2: Fetch latest 3 posts from r/worldnews
    #print("\n--- Fetching from r/worldnews ---")
    #result = tool.execute(subreddit="r/worldnews", limit=3)
    #print(json.dumps(result, indent=2))

    # Example 3: Handle invalid subreddit gracefully
    #print("\n--- Fetching from non-existent subreddit ---")
    #result = tool.execute(subreddit="r/thissubredditdoesnotexistlol", limit=3)
    #print(json.dumps(result, indent=2))
