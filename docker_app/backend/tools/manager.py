"""
Tool Manager for Rumour Verification Framework
Provides high-level interface for tool management and agent integration
"""

import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Try to import Strands components, fall back to mocks if not available
try:
    from strands import Agent
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    # Mock Agent for development
    class Agent:
        def __init__(self, tools=None, **kwargs):
            self.tools = tools or []
            self.kwargs = kwargs

try:
    from strands_tools import (
        file_read, file_write, http_request, tavily_search, 
        use_aws, memory, environment
    )
    STRANDS_TOOLS_AVAILABLE = True
except ImportError:
    STRANDS_TOOLS_AVAILABLE = False
    # Mock tools for development
    def file_read(**kwargs):
        return {"mock": "file_read", "args": kwargs}
    
    def file_write(**kwargs):
        return {"mock": "file_write", "args": kwargs}
    
    def http_request(**kwargs):
        return {"mock": "http_request", "args": kwargs}
    
    def tavily_search(**kwargs):
        return {"mock": "tavily_search", "args": kwargs}
    
    def use_aws(**kwargs):
        return {"mock": "use_aws", "args": kwargs}
    
    def memory(**kwargs):
        return {"mock": "memory", "args": kwargs}
    
    def environment(**kwargs):
        return {"mock": "environment", "args": kwargs}

from .base import (
    BaseRumourTool, ToolRegistry, ToolLoader, ToolCategory,
    tool_registry, tool_loader, create_agent_with_tools,
    initialize_tool_system
)

logger = logging.getLogger(__name__)


class ToolManager:
    """
    High-level tool manager for the Rumour Verification Framework
    Provides simplified interface for tool management and agent creation
    """
    
    def __init__(self, tools_directory: str = "./tools"):
        self.tools_directory = Path(tools_directory)
        self.registry = tool_registry
        self.loader = tool_loader
        self._logger = logging.getLogger("tools.manager")
        self._initialized = False
        
    def initialize(self, reload: bool = False) -> Dict[str, Any]:
        """
        Initialize the tool system
        
        Args:
            reload: Whether to reload existing tools
            
        Returns:
            Initialization results
        """
        try:
            if reload or not self._initialized:
                tools_loaded = initialize_tool_system(str(self.tools_directory))
                self._initialized = True
                
                result = {
                    "success": True,
                    "tools_loaded": tools_loaded,
                    "tools_directory": str(self.tools_directory),
                    "categories": {
                        category.value: len(self.registry.get_tools_by_category(category))
                        for category in ToolCategory
                    }
                }
                
                self._logger.info(f"Tool system initialized: {tools_loaded} tools loaded")
                return result
            else:
                return {
                    "success": True,
                    "message": "Tool system already initialized",
                    "tools_loaded": len(self.registry._tools)
                }
                
        except Exception as e:
            error_msg = f"Failed to initialize tool system: {e}"
            self._logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    def create_research_agent(self, **kwargs) -> Agent:
        """
        Create an agent optimized for research tasks
        
        Args:
            **kwargs: Additional agent configuration
            
        Returns:
            Configured research agent
        """
        research_categories = [
            ToolCategory.DISCOVERY,
            ToolCategory.ENRICHMENT,
            ToolCategory.REASONING
        ]
        
        additional_tools = [
            tavily_search,
            use_aws,
            memory,
            environment
        ]
        
        return create_agent_with_tools(
            categories=research_categories,
            additional_tools=additional_tools,
            **kwargs
        )
    
    def create_verification_agent(self, **kwargs) -> Agent:
        """
        Create an agent optimized for content verification
        
        Args:
            **kwargs: Additional agent configuration
            
        Returns:
            Configured verification agent
        """
        verification_categories = [
            ToolCategory.DISCOVERY,
            ToolCategory.REASONING,
            ToolCategory.ACTIONS
        ]
        
        additional_tools = [
            tavily_search,
            http_request,
            use_aws,
            memory
        ]
        
        return create_agent_with_tools(
            categories=verification_categories,
            additional_tools=additional_tools,
            **kwargs
        )
    
    def create_interactive_agent(self, **kwargs) -> Agent:
        """
        Create an agent optimized for interactive chat
        
        Args:
            **kwargs: Additional agent configuration
            
        Returns:
            Configured interactive agent
        """
        interactive_categories = [
            ToolCategory.INTERACTIVE,
            ToolCategory.DISCOVERY,
            ToolCategory.REASONING
        ]
        
        additional_tools = [
            tavily_search,
            http_request,
            memory,
            environment
        ]
        
        return create_agent_with_tools(
            categories=interactive_categories,
            additional_tools=additional_tools,
            **kwargs
        )
    
    def create_monitoring_agent(self, **kwargs) -> Agent:
        """
        Create an agent optimized for monitoring and alerting
        
        Args:
            **kwargs: Additional agent configuration
            
        Returns:
            Configured monitoring agent
        """
        monitoring_categories = [
            ToolCategory.DISCOVERY,
            ToolCategory.ACTIONS,
            ToolCategory.REASONING
        ]
        
        additional_tools = [
            tavily_search,
            use_aws,
            memory,
            environment
        ]
        
        return create_agent_with_tools(
            categories=monitoring_categories,
            additional_tools=additional_tools,
            **kwargs
        )
    
    def create_custom_agent(
        self, 
        categories: Optional[List[Union[ToolCategory, str]]] = None,
        tool_names: Optional[List[str]] = None,
        additional_strands_tools: Optional[List] = None,
        **kwargs
    ) -> Agent:
        """
        Create a custom agent with specific tools
        
        Args:
            categories: Tool categories to include
            tool_names: Specific tool names to include
            additional_strands_tools: Additional Strands tools
            **kwargs: Additional agent configuration
            
        Returns:
            Configured custom agent
        """
        tools = []
        
        # Add tools by category
        if categories:
            category_objs = []
            for cat in categories:
                if isinstance(cat, str):
                    category_objs.append(ToolCategory(cat))
                else:
                    category_objs.append(cat)
            
            for category in category_objs:
                category_tools = self.registry.get_tools_by_category(category)
                for tool in category_tools:
                    strands_tool = self.registry.get_strands_tool(tool.metadata.name)
                    if strands_tool:
                        tools.append(strands_tool)
        
        # Add specific tools by name
        if tool_names:
            for name in tool_names:
                strands_tool = self.registry.get_strands_tool(name)
                if strands_tool:
                    tools.append(strands_tool)
                else:
                    self._logger.warning(f"Tool not found: {name}")
        
        # Add additional Strands tools
        if additional_strands_tools:
            tools.extend(additional_strands_tools)
        
        # Add core tools
        tools.extend([file_read, file_write, http_request])
        
        return Agent(tools=tools, **kwargs)
    
    def list_available_tools(self) -> Dict[str, Any]:
        """
        List all available tools with their metadata
        
        Returns:
            Dictionary of tools and their information
        """
        return {
            "tools": self.registry.list_tools(),
            "categories": {
                category.value: [
                    tool.metadata.name 
                    for tool in self.registry.get_tools_by_category(category)
                ]
                for category in ToolCategory
            },
            "total_tools": len(self.registry._tools)
        }
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific tool
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information or None if not found
        """
        tool = self.registry.get_tool(tool_name)
        if tool:
            return tool.get_schema()
        return None
    
    def reload_tools(self) -> Dict[str, Any]:
        """
        Reload all tools from directory
        
        Returns:
            Reload results
        """
        try:
            tools_loaded = self.loader.reload_tools()
            
            result = {
                "success": True,
                "tools_reloaded": tools_loaded,
                "message": f"Successfully reloaded {tools_loaded} tools"
            }
            
            self._logger.info(f"Tools reloaded: {tools_loaded} tools")
            return result
            
        except Exception as e:
            error_msg = f"Failed to reload tools: {e}"
            self._logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    def validate_tool_directory(self) -> Dict[str, Any]:
        """
        Validate the tools directory structure
        
        Returns:
            Validation results
        """
        results = {
            "success": True,
            "directory_exists": self.tools_directory.exists(),
            "categories": {},
            "issues": []
        }
        
        if not results["directory_exists"]:
            results["success"] = False
            results["issues"].append(f"Tools directory does not exist: {self.tools_directory}")
            return results
        
        # Check category directories
        for category in ToolCategory:
            category_dir = self.tools_directory / category.value
            category_info = {
                "exists": category_dir.exists(),
                "python_files": [],
                "init_file": False
            }
            
            if category_info["exists"]:
                # Check for Python files
                python_files = list(category_dir.glob("*.py"))
                category_info["python_files"] = [f.name for f in python_files]
                
                # Check for __init__.py
                init_file = category_dir / "__init__.py"
                category_info["init_file"] = init_file.exists()
                
                if not category_info["init_file"]:
                    results["issues"].append(f"Missing __init__.py in {category_dir}")
            
            results["categories"][category.value] = category_info
        
        if results["issues"]:
            results["success"] = False
        
        return results


# Global tool manager instance
tool_manager = ToolManager()