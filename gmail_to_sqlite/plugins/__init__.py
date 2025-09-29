"""
Advanced Plugin System for Gmail to SQLite.

Provides a flexible plugin architecture for extending functionality
with hooks, filters, and custom processing capabilities.
"""

import os
import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    name: str
    version: str
    description: str
    author: str
    email: Optional[str] = None
    url: Optional[str] = None
    dependencies: List[str] = None
    min_app_version: Optional[str] = None
    max_app_version: Optional[str] = None


class PluginError(Exception):
    """Raised when plugin operations fail."""
    pass


class Hook:
    """Represents a hook point in the application."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable, priority: int = 0) -> None:
        """Add a callback to this hook."""
        self.callbacks.append((priority, callback))
        # Sort by priority (higher priority first)
        self.callbacks.sort(key=lambda x: x[0], reverse=True)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove a callback from this hook."""
        self.callbacks = [(p, c) for p, c in self.callbacks if c != callback]
    
    def call(self, *args, **kwargs) -> List[Any]:
        """Call all callbacks registered to this hook."""
        results = []
        for priority, callback in self.callbacks:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in hook {self.name} callback {callback.__name__}: {e}")
        return results


class Filter:
    """Represents a filter point in the application."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.callbacks: List[Callable] = []
    
    def add_callback(self, callback: Callable, priority: int = 0) -> None:
        """Add a callback to this filter."""
        self.callbacks.append((priority, callback))
        # Sort by priority (higher priority first)
        self.callbacks.sort(key=lambda x: x[0], reverse=True)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove a callback from this filter."""
        self.callbacks = [(p, c) for p, c in self.callbacks if c != callback]
    
    def apply(self, value: Any, *args, **kwargs) -> Any:
        """Apply all filters to a value."""
        for priority, callback in self.callbacks:
            try:
                value = callback(value, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error in filter {self.name} callback {callback.__name__}: {e}")
        return value


class BasePlugin(ABC):
    """Abstract base class for all plugins."""
    
    def __init__(self):
        self.metadata: Optional[PluginMetadata] = None
        self.enabled: bool = True
        self.config: Dict[str, Any] = {}
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    @abstractmethod
    def initialize(self, plugin_manager: 'PluginManager') -> None:
        """Initialize the plugin."""
        pass
    
    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the plugin with settings."""
        self.config = config
    
    def enable(self) -> None:
        """Enable the plugin."""
        self.enabled = True
    
    def disable(self) -> None:
        """Disable the plugin."""
        self.enabled = False
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        pass


class MessageProcessorPlugin(BasePlugin):
    """Base class for message processing plugins."""
    
    @abstractmethod
    def process_message(self, message: Any) -> Any:
        """Process a message and return modified message."""
        pass
    
    def should_process(self, message: Any) -> bool:
        """Determine if this plugin should process the message."""
        return True


class ExporterPlugin(BasePlugin):
    """Base class for data export plugins."""
    
    @abstractmethod
    def export_data(self, data: Any, output_path: str, **options) -> bool:
        """Export data to specified format and path."""
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported export formats."""
        pass


class AnalyticsPlugin(BasePlugin):
    """Base class for analytics plugins."""
    
    @abstractmethod
    def generate_report(self, data: Any, report_type: str) -> Dict[str, Any]:
        """Generate analytics report."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        pass


class PluginManager:
    """Manages plugin loading, registration, and lifecycle."""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_paths: List[str] = []
        self.hooks: Dict[str, Hook] = {}
        self.filters: Dict[str, Filter] = {}
        self._initialize_core_hooks()
    
    def _initialize_core_hooks(self) -> None:
        """Initialize core application hooks."""
        self.register_hook("before_message_sync", "Called before syncing messages")
        self.register_hook("after_message_sync", "Called after syncing messages")
        self.register_hook("before_message_process", "Called before processing a message")
        self.register_hook("after_message_process", "Called after processing a message")
        self.register_hook("before_database_save", "Called before saving to database")
        self.register_hook("after_database_save", "Called after saving to database")
        self.register_hook("sync_complete", "Called when sync operation completes")
        self.register_hook("sync_error", "Called when sync encounters an error")
        
        self.register_filter("message_content", "Filter message content before processing")
        self.register_filter("message_metadata", "Filter message metadata")
        self.register_filter("search_query", "Filter search queries")
        self.register_filter("export_data", "Filter data before export")
    
    def add_plugin_path(self, path: str) -> None:
        """Add a directory to search for plugins."""
        if path not in self.plugin_paths:
            self.plugin_paths.append(path)
    
    def register_hook(self, name: str, description: str = "") -> Hook:
        """Register a new hook."""
        if name not in self.hooks:
            self.hooks[name] = Hook(name, description)
        return self.hooks[name]
    
    def register_filter(self, name: str, description: str = "") -> Filter:
        """Register a new filter."""
        if name not in self.filters:
            self.filters[name] = Filter(name, description)
        return self.filters[name]
    
    def get_hook(self, name: str) -> Optional[Hook]:
        """Get a hook by name."""
        return self.hooks.get(name)
    
    def get_filter(self, name: str) -> Optional[Filter]:
        """Get a filter by name."""
        return self.filters.get(name)
    
    def call_hook(self, name: str, *args, **kwargs) -> List[Any]:
        """Call all callbacks for a hook."""
        hook = self.get_hook(name)
        if hook:
            return hook.call(*args, **kwargs)
        return []
    
    def apply_filter(self, name: str, value: Any, *args, **kwargs) -> Any:
        """Apply all filters to a value."""
        filter_obj = self.get_filter(name)
        if filter_obj:
            return filter_obj.apply(value, *args, **kwargs)
        return value
    
    def discover_plugins(self) -> List[str]:
        """Discover available plugins in plugin paths."""
        discovered = []
        
        for path in self.plugin_paths:
            plugin_dir = Path(path)
            if not plugin_dir.exists():
                continue
            
            for file_path in plugin_dir.rglob("*.py"):
                if file_path.name.startswith("_"):
                    continue
                
                # Convert path to module name
                relative_path = file_path.relative_to(plugin_dir)
                module_name = str(relative_path).replace("/", ".").replace("\\", ".")[:-3]
                discovered.append(f"{path}.{module_name}")
        
        return discovered
    
    def load_plugin(self, plugin_path: str) -> bool:
        """Load a single plugin from path or module name."""
        try:
            # Try to import as module
            module = importlib.import_module(plugin_path)
            
            # Find plugin classes in module
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, BasePlugin) and 
                    obj != BasePlugin):
                    plugin_classes.append(obj)
            
            if not plugin_classes:
                logger.warning(f"No plugin classes found in {plugin_path}")
                return False
            
            # Instantiate and register plugins
            for plugin_class in plugin_classes:
                try:
                    plugin_instance = plugin_class()
                    metadata = plugin_instance.get_metadata()
                    
                    if metadata.name in self.plugins:
                        logger.warning(f"Plugin {metadata.name} already loaded, skipping")
                        continue
                    
                    # Initialize plugin
                    plugin_instance.metadata = metadata
                    plugin_instance.initialize(self)
                    
                    self.plugins[metadata.name] = plugin_instance
                    logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")
                    
                except Exception as e:
                    logger.error(f"Failed to initialize plugin {plugin_class.__name__}: {e}")
            
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import plugin {plugin_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error loading plugin {plugin_path}: {e}")
            return False
    
    def load_all_plugins(self) -> int:
        """Load all discoverable plugins."""
        discovered = self.discover_plugins()
        loaded_count = 0
        
        for plugin_path in discovered:
            if self.load_plugin(plugin_path):
                loaded_count += 1
        
        logger.info(f"Loaded {loaded_count} plugins")
        return loaded_count
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin by name."""
        if plugin_name not in self.plugins:
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            plugin.cleanup()
            del self.plugins[plugin_name]
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """Enable a plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].enable()
            return True
        return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """Disable a plugin."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].disable()
            return True
        return False
    
    def get_plugin(self, plugin_name: str) -> Optional[BasePlugin]:
        """Get a plugin by name."""
        return self.plugins.get(plugin_name)
    
    def get_plugins_by_type(self, plugin_type: Type[BasePlugin]) -> List[BasePlugin]:
        """Get all plugins of a specific type."""
        return [p for p in self.plugins.values() if isinstance(p, plugin_type)]
    
    def get_enabled_plugins(self) -> List[BasePlugin]:
        """Get all enabled plugins."""
        return [p for p in self.plugins.values() if p.enabled]
    
    def list_plugins(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded plugins with their metadata."""
        result = {}
        for name, plugin in self.plugins.items():
            metadata = plugin.metadata
            result[name] = {
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "enabled": plugin.enabled
            }
        return result
    
    def configure_plugin(self, plugin_name: str, config: Dict[str, Any]) -> bool:
        """Configure a plugin with settings."""
        if plugin_name in self.plugins:
            self.plugins[plugin_name].configure(config)
            return True
        return False
    
    def cleanup_all_plugins(self) -> None:
        """Clean up all loaded plugins."""
        for plugin in self.plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin {plugin.metadata.name}: {e}")
        
        self.plugins.clear()


# Global plugin manager instance
plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    return plugin_manager


def register_hook(name: str, description: str = "") -> Hook:
    """Register a new hook in the global plugin manager."""
    return plugin_manager.register_hook(name, description)


def register_filter(name: str, description: str = "") -> Filter:
    """Register a new filter in the global plugin manager."""
    return plugin_manager.register_filter(name, description)


def call_hook(name: str, *args, **kwargs) -> List[Any]:
    """Call a hook in the global plugin manager."""
    return plugin_manager.call_hook(name, *args, **kwargs)


def apply_filter(name: str, value: Any, *args, **kwargs) -> Any:
    """Apply a filter in the global plugin manager."""
    return plugin_manager.apply_filter(name, value, *args, **kwargs)