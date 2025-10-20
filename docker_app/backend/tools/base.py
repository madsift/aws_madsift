"""
Base tool interface and loading mechanism for Rumour Verification Framework
Built on Strands Agents SDK for dynamic tool loading and management
"""

import os
import sys
import importlib
import importlib.util
import inspect
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Type, Callable
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

# Try to import Strands components, fall back to mocks if not available
try:
    from strands import Agent, tool
    STRANDS_AVAILABLE = True
except ImportError:
    STRANDS_AVAILABLE = False
    # Mock implementations for development without Strands
    class Agent:
        def __init__(self, tools=None, **kwargs):
            self.tools = tools or []
            self.kwargs = kwargs
    
    def tool(name=None, description=None):
        """Mock tool decorator"""
        def decorator(func):
            func._tool_name = name
            func._tool_description = description
            return func
        return decorator

try:
    from strands_tools import file_read, file_write, http_request
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

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Tool categories for organization and filtering"""
    DISCOVERY = "discovery"
    ENRICHMENT = "enrichment"
    INTERACTIVE = "interactive"
    REASONING = "reasoning"
    ACTIONS = "actions"


@dataclass
class ToolMetadata:
    """Metadata for tool registration and discovery"""
    name: str
    description: str
    category: ToolCategory
    version: str = "1.0.0"
    author: str = "Rumour Verification Framework"
    dependencies: List[str] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class BaseRumourTool(ABC):
    """
    Abstract base class for all Rumour Verification Framework tools
    Provides standard interface and integration with Strands Agents
    """
    
    def __init__(self, metadata: ToolMetadata):
        self.metadata = metadata
        self._logger = logging.getLogger(f"tools.{metadata.name}")
        
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            Dict containing tool execution results
        """
        pass
    
    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters before execution
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get tool schema for Strands Agent integration
        
        Returns:
            Tool schema dictionary
        """
        return {
            "name": self.metadata.name,
            "description": self.metadata.description,
            "category": self.metadata.category.value,
            "version": self.metadata.version,
            "parameters": self._get_parameters_schema()
        }
    
    @abstractmethod
    def _get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get parameters schema for the tool
        
        Returns:
            Parameters schema dictionary
        """
        pass
    
    def log_execution(self, action: str, **kwargs):
        """Log tool execution for monitoring and debugging"""
        self._logger.info(f"Tool {self.metadata.name} - {action}", extra=kwargs)


class ToolRegistry:
    """
    Registry for managing and discovering tools
    Integrates with Strands Agents for dynamic tool loading
    """
    
    def __init__(self):
        self._tools: Dict[str, BaseRumourTool] = {}
        self._strands_tools: Dict[str, Callable] = {}
        self._categories: Dict[ToolCategory, List[str]] = {
            category: [] for category in ToolCategory
        }
        self._logger = logging.getLogger("tools.registry")
        
    def register_tool(self, tool: BaseRumourTool) -> bool:
        """
        Register a tool in the registry
        
        Args:
            tool: Tool instance to register
            
        Returns:
            True if registered successfully, False otherwise
        """
        try:
            if not tool.metadata.enabled:
                self._logger.info(f"Tool {tool.metadata.name} is disabled, skipping registration")
                return False
                
            # Register in internal registry
            self._tools[tool.metadata.name] = tool
            self._categories[tool.metadata.category].append(tool.metadata.name)
            
            # Create Strands-compatible tool function
            strands_tool_func = self._create_strands_tool(tool)
            self._strands_tools[tool.metadata.name] = strands_tool_func
            
            self._logger.info(f"Successfully registered tool: {tool.metadata.name}")
            return True
            
        except Exception as e:
            self._logger.error(f"Failed to register tool {tool.metadata.name}: {e}")
            return False
    
    def _create_strands_tool(self, rumour_tool: BaseRumourTool) -> Callable:
        """
        Create a Strands-compatible tool function from a RumourTool
        
        Args:
            rumour_tool: The RumourTool to wrap
            
        Returns:
            Strands-compatible tool function
        """
        if STRANDS_AVAILABLE:
            @tool(
                name=rumour_tool.metadata.name,
                description=rumour_tool.metadata.description
            )
            def strands_tool_wrapper(**kwargs) -> Dict[str, Any]:
                return self._execute_tool_wrapper(rumour_tool, **kwargs)
        else:
            # Mock implementation when Strands is not available
            def strands_tool_wrapper(**kwargs) -> Dict[str, Any]:
                return self._execute_tool_wrapper(rumour_tool, **kwargs)
            
            strands_tool_wrapper._tool_name = rumour_tool.metadata.name
            strands_tool_wrapper._tool_description = rumour_tool.metadata.description
        
        return strands_tool_wrapper
    
    def _execute_tool_wrapper(self, rumour_tool: BaseRumourTool, **kwargs) -> Dict[str, Any]:
        """Strands tool wrapper for RumourTool"""
        try:
            # Validate input
            if not rumour_tool.validate_input(**kwargs):
                return {
                    "success": False,
                    "error": "Invalid input parameters",
                    "tool": rumour_tool.metadata.name
                }
            
            # Log execution
            rumour_tool.log_execution("execute", parameters=kwargs)
            
            # Execute tool
            result = rumour_tool.execute(**kwargs)
            
            # Ensure result has standard format
            if not isinstance(result, dict):
                result = {"result": result}
            
            result.update({
                "success": True,
                "tool": rumour_tool.metadata.name,
                "category": rumour_tool.metadata.category.value
            })
            
            return result
            
        except Exception as e:
            error_msg = f"Tool execution failed: {str(e)}"
            rumour_tool._logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg,
                "tool": rumour_tool.metadata.name
            }
    
    def get_tool(self, name: str) -> Optional[BaseRumourTool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def get_strands_tool(self, name: str) -> Optional[Callable]:
        """Get a Strands-compatible tool function by name"""
        return self._strands_tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[BaseRumourTool]:
        """Get all tools in a specific category"""
        tool_names = self._categories.get(category, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_all_strands_tools(self) -> List[Callable]:
        """Get all Strands-compatible tool functions"""
        return list(self._strands_tools.values())
    
    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all registered tools with their metadata"""
        return {
            name: tool.get_schema() 
            for name, tool in self._tools.items()
        }
    
    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool"""
        if name in self._tools:
            tool = self._tools[name]
            category = tool.metadata.category
            
            # Remove from registries
            del self._tools[name]
            if name in self._strands_tools:
                del self._strands_tools[name]
            if name in self._categories[category]:
                self._categories[category].remove(name)
                
            self._logger.info(f"Unregistered tool: {name}")
            return True
        return False


class ToolLoader:
    """
    Dynamic tool loader for discovering and loading tools from directories
    Supports hot reloading and automatic tool discovery
    """
    
    def __init__(self, registry: ToolRegistry, tools_directory: str = "./tools"):
        self.registry = registry
        self.tools_directory = Path(tools_directory)
        self._logger = logging.getLogger("tools.loader")
        self._loaded_modules: Dict[str, Any] = {}
        
    def load_tools_from_directory(self, reload: bool = False) -> int:
        """
        Load all tools from the tools directory
        
        Args:
            reload: Whether to reload already loaded modules
            
        Returns:
            Number of tools loaded
        """
        loaded_count = 0
        
        if not self.tools_directory.exists():
            self._logger.warning(f"Tools directory does not exist: {self.tools_directory}")
            return loaded_count
        
        # Load tools from each category directory
        for category in ToolCategory:
            category_dir = self.tools_directory / category.value
            if category_dir.exists():
                loaded_count += self._load_category_tools(category_dir, category, reload)
        
        # Also load from root tools directory
        loaded_count += self._load_category_tools(self.tools_directory, None, reload)
        
        self._logger.info(f"Loaded {loaded_count} tools from directory")
        return loaded_count
    
    def _load_category_tools(self, directory: Path, category: Optional[ToolCategory], reload: bool) -> int:
        """Load tools from a specific category directory"""
        loaded_count = 0
        
        for py_file in directory.glob("*.py"):
            if py_file.name.startswith("__") or py_file.name == "base.py":
                continue
                
            try:
                loaded_count += self._load_tool_from_file(py_file, category, reload)
            except Exception as e:
                self._logger.error(f"Failed to load tool from {py_file}: {e}")
        
        return loaded_count
    
    def _load_tool_from_file(self, file_path: Path, category: Optional[ToolCategory], reload: bool) -> int:
        """Load a tool from a specific Python file"""
        module_name = f"tools.{file_path.stem}"
        if category:
            module_name = f"tools.{category.value}.{file_path.stem}"
        
        try:
            # Import or reload module
            if module_name in sys.modules and reload:
                module = importlib.reload(sys.modules[module_name])
            else:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[module_name] = module
            
            self._loaded_modules[module_name] = module
            
            # Find and register tool classes
            return self._register_tools_from_module(module, category)
            
        except Exception as e:
            self._logger.error(f"Failed to load module {module_name}: {e}")
            return 0
    
    def _register_tools_from_module(self, module: Any, category: Optional[ToolCategory]) -> int:
        """Register all tool classes found in a module"""
        registered_count = 0
        
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseRumourTool) and 
                obj != BaseRumourTool):
                
                try:
                    # Instantiate tool (assuming default constructor)
                    if hasattr(obj, 'create_instance'):
                        tool_instance = obj.create_instance()
                    else:
                        # Try to create with default metadata if no create_instance method
                        continue
                    
                    # Override category if specified
                    if category and tool_instance.metadata.category != category:
                        tool_instance.metadata.category = category
                    
                    if self.registry.register_tool(tool_instance):
                        registered_count += 1
                        
                except Exception as e:
                    self._logger.error(f"Failed to instantiate tool {name}: {e}")
        
        return registered_count
    
    def reload_tools(self) -> int:
        """Reload all tools from directory"""
        # Clear existing tools
        for tool_name in list(self.registry._tools.keys()):
            self.registry.unregister_tool(tool_name)
        
        # Reload from directory
        return self.load_tools_from_directory(reload=True)


# Global tool registry instance
tool_registry = ToolRegistry()
tool_loader = ToolLoader(tool_registry)


def create_agent_with_tools(
    categories: Optional[List[ToolCategory]] = None,
    additional_tools: Optional[List[Callable]] = None,
    **agent_kwargs
) -> Agent:
    """
    Create a Strands Agent with loaded tools
    
    Args:
        categories: Tool categories to include (None for all)
        additional_tools: Additional Strands tools to include
        **agent_kwargs: Additional arguments for Agent constructor
        
    Returns:
        Configured Strands Agent
    """
    # Get tools by category
    tools = []
    
    if categories:
        for category in categories:
            category_tools = tool_registry.get_tools_by_category(category)
            for tool in category_tools:
                strands_tool = tool_registry.get_strands_tool(tool.metadata.name)
                if strands_tool:
                    tools.append(strands_tool)
    else:
        # Include all tools
        tools = tool_registry.get_all_strands_tools()
    
    # Add additional tools
    if additional_tools:
        tools.extend(additional_tools)
    
    # Add core Strands tools
    tools.extend([file_read, file_write, http_request])
    
    # Create agent
    return Agent(tools=tools, **agent_kwargs)


def initialize_tool_system(tools_directory: str = "./tools") -> int:
    """
    Initialize the tool system by loading all tools from directory
    
    Args:
        tools_directory: Directory to load tools from
        
    Returns:
        Number of tools loaded
    """
    global tool_loader
    tool_loader = ToolLoader(tool_registry, tools_directory)
    return tool_loader.load_tools_from_directory()