"""
Reddit PRAW Tool - Fetch Reddit posts using Reddit API via PRAW
Authenticated version of reddit_mcp_public_tool.py
"""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

import praw
from tools.base import BaseRumourTool, ToolMetadata, ToolCategory
from common.aws_clients import get_secret

api_secret_name = os.getenv("SECRET_NAME", "api_secret")
region =  os.getenv("AWS_REGION", "us-east-1")
client_id = get_secret(api_secret_name, region, 'reddit_id')
client_secret = get_secret(api_secret_name, region, 'reddit_secret')

class RedditPost(BaseModel):
    """Schema for a single Reddit post."""
    id: str = Field(description="The unique ID of the post.")
    title: str = Field(description="The title of the post.")
    selftext: Optional[str] = Field(default=None, description="The body text of the post, if any.")
    score: int = Field(description="The score or number of upvotes.")
    permalink: str = Field(description="The permanent link to the post.")
    author: str = Field(description="The username of the post's author.")
    created_utc: float = Field(description="The UTC timestamp of when the post was created.")


class RedditPRAWTool(BaseRumourTool):
    """Reddit Tool using PRAW (Python Reddit API Wrapper)"""

    def __init__(self, metadata: ToolMetadata):
        super().__init__(metadata)
        self._logger = logging.getLogger("tools.reddit_praw_tool")

        # Environment variables for Reddit API credentials
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = "MadSiftKG/1.0 by u/madsift" #os.getenv("REDDIT_USER_AGENT", "MadSiftKG/1.0 by u/example_user")

        if not self.client_id or not self.client_secret:
            raise ValueError("Reddit API credentials not set in environment variables.")

        # Initialize PRAW Reddit instance
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )

    @classmethod
    def create_instance(cls) -> 'RedditPRAWTool':
        """Create an instance of RedditPRAWTool"""
        metadata = ToolMetadata(
            name="reddit_praw_tool",
            description="Fetch Reddit posts using authenticated PRAW API",
            category=ToolCategory.DISCOVERY
        )
        return cls(metadata)

    def fetch_posts_with_praw(self, subreddit: str = "r/Nepal", limit: int = 25) -> List[Dict[str, Any]]:
        """Fetch latest posts from subreddit using PRAW"""
        clean_subreddit = subreddit.replace('r/', '')
        self._logger.info(f"Fetching {limit} posts from r/{clean_subreddit} via PRAW")

        try:
            subreddit_obj = self.reddit.subreddit(clean_subreddit)
            posts = []

            for submission in subreddit_obj.new(limit=limit):
                post = {
                    "id": submission.id,
                    "title": submission.title,
                    "selftext": submission.selftext or "",
                    "score": submission.score,
                    "permalink": f"https://www.reddit.com{submission.permalink}",
                    "author": str(submission.author) if submission.author else "[deleted]",
                    "created_utc": submission.created_utc,
                }
                posts.append(post)

            self._logger.info(f"Successfully fetched {len(posts)} posts from r/{clean_subreddit}")
            return posts

        except Exception as e:
            self._logger.error(f"Error fetching posts from r/{clean_subreddit}: {e}", exc_info=True)
            return []

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute Reddit PRAW tool to fetch posts"""
        if not self.validate_input(**kwargs):
            return {"success": False, "error": "Invalid parameters", "posts": [], "metadata": {}}

        subreddit = kwargs.get("subreddit", "r/Nepal")
        limit = kwargs.get("limit", 25)

        if not subreddit.startswith("r/"):
            subreddit = f"r/{subreddit}"

        try:
            posts = self.fetch_posts_with_praw(subreddit, limit)

            if not posts:
                return {
                    "success": False,
                    "error": f"No posts found for {subreddit}",
                    "posts": [],
                    "metadata": {"subreddit": subreddit, "requested_limit": limit, "actual_count": 0}
                }

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
            self._logger.error(f"Error executing Reddit PRAW tool: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "posts": [],
                "metadata": {"subreddit": subreddit, "requested_limit": limit, "actual_count": 0}
            }

    def validate_input(self, **kwargs) -> bool:
        subreddit = kwargs.get("subreddit")
        limit = kwargs.get("limit", 25)
        if not subreddit:
            self._logger.error("Parameter 'subreddit' is required")
            return False
        if not isinstance(limit, int) or limit < 1 or limit > 100:
            self._logger.error("Parameter 'limit' must be integer between 1 and 100")
            return False
        return True

    def _get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "subreddit": {
                    "type": "string",
                    "description": "Subreddit to fetch posts from",
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


def create_reddit_praw_tool() -> RedditPRAWTool:
    """Factory function to create RedditPRAWTool instance"""
    return RedditPRAWTool.create_instance()


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    # Example run
    tool = create_reddit_praw_tool()
    result = tool.execute(subreddit="r/nepalsocial", limit=5)
    print(json.dumps(result, indent=2))
