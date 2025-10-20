"""
Tools package for Rumour Verification Framework
Provides base tool interface and loading mechanism using Strands Agents SDK
"""

from .base import (
    BaseRumourTool,
    ToolMetadata,
    ToolCategory,
    ToolRegistry,
    ToolLoader,
    tool_registry,
    tool_loader,
    create_agent_with_tools,
    initialize_tool_system
)

from .manager import ToolManager, tool_manager

# Version information
__version__ = "1.0.0"
__author__ = "Rumour Verification Framework Team"

# Export main classes and functions
__all__ = [
    # Base classes
    "BaseRumourTool",
    "ToolMetadata", 
    "ToolCategory",
    "ToolRegistry",
    "ToolLoader",
    
    # Global instances
    "tool_registry",
    "tool_loader",
    "tool_manager",
    
    # Factory functions
    "create_agent_with_tools",
    "initialize_tool_system",
    
    # High-level interface
    "ToolManager"
]