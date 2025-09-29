"""
Example plugin for Gmail to SQLite.

This demonstrates how to create a plugin that processes messages
and extends the application functionality.
"""

import logging
from gmail_to_sqlite.plugins import (
    BasePlugin, MessageProcessorPlugin, PluginMetadata
)

logger = logging.getLogger(__name__)


class ExampleMessageProcessor(MessageProcessorPlugin):
    """Example message processor plugin."""
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="example-processor",
            version="1.0.0",
            description="Example message processing plugin",
            author="Gmail to SQLite Team",
            email="support@example.com"
        )
    
    def initialize(self, plugin_manager) -> None:
        """Initialize the plugin."""
        logger.info("Example message processor plugin initialized")
        
        # Register hooks
        hook = plugin_manager.get_hook("before_message_process")
        if hook:
            hook.add_callback(self.before_process_hook, priority=10)
        
        filter_obj = plugin_manager.get_filter("message_content")
        if filter_obj:
            filter_obj.add_callback(self.filter_message_content, priority=5)
    
    def process_message(self, message) -> any:
        """Process a message and return modified message."""
        # Example: Add custom metadata
        if hasattr(message, 'custom_data'):
            message.custom_data = message.custom_data or {}
        else:
            message.custom_data = {}
        
        # Example processing: detect urgent messages
        if message.subject and any(word in message.subject.lower() 
                                 for word in ['urgent', 'asap', 'emergency']):
            message.custom_data['priority'] = 'urgent'
        else:
            message.custom_data['priority'] = 'normal'
        
        # Example: extract phone numbers
        import re
        if message.body:
            phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b'
            phones = re.findall(phone_pattern, message.body)
            if phones:
                message.custom_data['phone_numbers'] = phones
        
        return message
    
    def should_process(self, message) -> bool:
        """Determine if this plugin should process the message."""
        # Example: only process unread messages
        return not message.is_read
    
    def before_process_hook(self, message):
        """Hook called before message processing."""
        logger.debug(f"Processing message: {message.id}")
    
    def filter_message_content(self, content, *args, **kwargs):
        """Filter to modify message content."""
        # Example: remove sensitive information
        if isinstance(content, str):
            # Remove credit card numbers
            import re
            content = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 
                           '[CREDIT_CARD_REDACTED]', content)
        
        return content


class ExampleAnalyticsPlugin(BasePlugin):
    """Example analytics plugin."""
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="example-analytics",
            version="1.0.0", 
            description="Example analytics plugin",
            author="Gmail to SQLite Team"
        )
    
    def initialize(self, plugin_manager) -> None:
        """Initialize the plugin."""
        logger.info("Example analytics plugin initialized")
        
        # Register custom hook
        plugin_manager.register_hook("custom_analytics", "Custom analytics processing")
        
        # Add callback to sync complete hook
        hook = plugin_manager.get_hook("sync_complete")
        if hook:
            hook.add_callback(self.on_sync_complete)
    
    def on_sync_complete(self, sync_stats):
        """Called when sync operation completes."""
        logger.info(f"Sync completed with stats: {sync_stats}")
        
        # Example: generate custom analytics report
        self.generate_custom_report()
    
    def generate_custom_report(self):
        """Generate a custom analytics report."""
        # This would contain custom analytics logic
        logger.info("Generating custom analytics report")
        
        # Example analytics could include:
        # - Sender reputation scoring
        # - Email pattern detection
        # - Custom metrics calculation
        pass


# Plugin registration (required)
# The plugin system will automatically discover and load these classes
plugins = [ExampleMessageProcessor, ExampleAnalyticsPlugin]