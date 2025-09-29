"""
Advanced configuration system for Gmail to SQLite.

Supports YAML, JSON, and environment variable configuration with
validation, type conversion, and hierarchical configuration merging.
"""

import os
import json
import yaml
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    path: str = "data/messages.db"
    enable_logging: bool = False
    connection_pool_size: int = 20
    timeout: int = 30
    pragma_settings: Dict[str, Any] = field(default_factory=lambda: {
        "journal_mode": "WAL",
        "synchronous": "NORMAL",
        "cache_size": -64000,  # 64MB cache
        "temp_store": "MEMORY",
        "mmap_size": 268435456  # 256MB mmap
    })


@dataclass
class SyncConfig:
    """Synchronization configuration settings."""
    workers: int = 4
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5
    batch_size: int = 100
    rate_limit_requests_per_minute: int = 250
    enable_incremental_sync: bool = True
    full_sync_interval_hours: int = 24
    deleted_message_check_interval_hours: int = 6


@dataclass
class CacheConfig:
    """Caching configuration settings."""
    enabled: bool = True
    type: str = "memory"  # memory, redis, file
    ttl_seconds: int = 3600
    max_size: int = 1000
    redis_url: Optional[str] = None
    file_cache_dir: str = "data/cache"


@dataclass
class AttachmentConfig:
    """Attachment handling configuration."""
    enabled: bool = False
    download_path: str = "data/attachments"
    max_size_mb: int = 100
    allowed_types: List[str] = field(default_factory=lambda: [
        "pdf", "doc", "docx", "txt", "jpg", "jpeg", "png", "gif"
    ])
    extract_text: bool = True
    virus_scan_enabled: bool = False


@dataclass
class AnalyticsConfig:
    """Analytics and reporting configuration."""
    enabled: bool = True
    metrics_retention_days: int = 90
    generate_daily_reports: bool = True
    generate_weekly_reports: bool = True
    generate_monthly_reports: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])


@dataclass
class WebConfig:
    """Web interface configuration."""
    enabled: bool = False
    host: str = "localhost"
    port: int = 8080
    debug: bool = False
    secret_key: Optional[str] = None
    cors_enabled: bool = True
    auth_enabled: bool = True
    session_timeout_minutes: int = 60


@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    enabled: bool = False
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/gmail-to-sqlite.log"
    log_rotation: bool = True
    log_max_size: str = "10MB"
    log_backup_count: int = 5
    health_check_interval: int = 300


@dataclass
class BackupConfig:
    """Backup and restore configuration."""
    enabled: bool = True
    backup_path: str = "data/backups"
    schedule_cron: str = "0 2 * * *"  # Daily at 2 AM
    retention_days: int = 30
    compression_enabled: bool = True
    encryption_enabled: bool = False
    encryption_key_file: Optional[str] = None


@dataclass
class PluginConfig:
    """Plugin system configuration."""
    enabled: bool = True
    plugin_paths: List[str] = field(default_factory=lambda: ["plugins"])
    auto_load: bool = True
    enabled_plugins: List[str] = field(default_factory=list)
    disabled_plugins: List[str] = field(default_factory=list)


@dataclass
class Config:
    """Main configuration container."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    attachments: AttachmentConfig = field(default_factory=AttachmentConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)
    web: WebConfig = field(default_factory=WebConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    backup: BackupConfig = field(default_factory=BackupConfig)
    plugins: PluginConfig = field(default_factory=PluginConfig)
    
    # Gmail API settings
    credentials_file: str = "credentials.json"
    token_file: str = "token.json"
    scopes: List[str] = field(default_factory=lambda: [
        "https://www.googleapis.com/auth/gmail.readonly"
    ])


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


class ConfigManager:
    """Advanced configuration manager with multiple source support."""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[Config] = None
        self._config_files = [
            "default.yaml",
            "local.yaml",
            f"{os.getenv('ENVIRONMENT', 'development')}.yaml"
        ]
    
    def load_config(self, config_file: Optional[str] = None) -> Config:
        """
        Load configuration from multiple sources with priority:
        1. Environment variables
        2. Command line config file
        3. Local config files
        4. Default config
        """
        config_data = {}
        
        # Load from multiple config files (lowest to highest priority)
        config_files = self._config_files if config_file is None else [config_file]
        
        for file in config_files:
            file_path = self.config_dir / file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        if file.endswith('.yaml') or file.endswith('.yml'):
                            file_data = yaml.safe_load(f)
                        elif file.endswith('.json'):
                            file_data = json.load(f)
                        else:
                            continue
                        
                        if file_data:
                            config_data = self._deep_merge(config_data, file_data)
                            logger.info(f"Loaded configuration from {file_path}")
                            
                except Exception as e:
                    logger.warning(f"Failed to load config from {file_path}: {e}")
        
        # Override with environment variables
        env_config = self._load_from_env()
        if env_config:
            config_data = self._deep_merge(config_data, env_config)
        
        # Convert to Config object
        try:
            self._config = self._dict_to_config(config_data)
            self._validate_config(self._config)
            return self._config
        except Exception as e:
            raise ConfigurationError(f"Invalid configuration: {e}")
    
    def get_config(self) -> Config:
        """Get the current configuration, loading if necessary."""
        if self._config is None:
            self.load_config()
        return self._config
    
    def save_config(self, config: Config, filename: str = "local.yaml") -> None:
        """Save configuration to file."""
        config_path = self.config_dir / filename
        config_dict = self._config_to_dict(config)
        
        try:
            with open(config_path, 'w') as f:
                if filename.endswith('.yaml') or filename.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif filename.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError("Unsupported config file format")
            
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")
    
    def create_default_config(self) -> None:
        """Create default configuration files."""
        default_config = Config()
        
        # Create default.yaml
        self.save_config(default_config, "default.yaml")
        
        # Create example local.yaml with comments
        example_config = {
            "# Gmail to SQLite Advanced Configuration": None,
            "# Copy this file to local.yaml and modify as needed": None,
            "database": {
                "path": "data/messages.db",
                "enable_logging": False
            },
            "sync": {
                "workers": 4,
                "batch_size": 100
            },
            "web": {
                "enabled": False,
                "port": 8080
            },
            "monitoring": {
                "enabled": False,
                "log_level": "INFO"
            }
        }
        
        example_path = self.config_dir / "local.yaml.example"
        with open(example_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False, indent=2)
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def _load_from_env(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        prefix = "GMAIL_TO_SQLITE_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                # Convert SECTION_KEY to nested dict
                if '_' in config_key:
                    parts = config_key.split('_', 1)
                    section, sub_key = parts
                    if section not in env_config:
                        env_config[section] = {}
                    env_config[section][sub_key] = self._convert_env_value(value)
                else:
                    env_config[config_key] = self._convert_env_value(value)
        
        return env_config
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool, List]:
        """Convert environment variable string to appropriate type."""
        # Boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # List values (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        return value
    
    def _dict_to_config(self, data: Dict[str, Any]) -> Config:
        """Convert dictionary to Config object."""
        config = Config()
        
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        if 'sync' in data:
            config.sync = SyncConfig(**data['sync'])
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        if 'attachments' in data:
            config.attachments = AttachmentConfig(**data['attachments'])
        if 'analytics' in data:
            config.analytics = AnalyticsConfig(**data['analytics'])
        if 'web' in data:
            config.web = WebConfig(**data['web'])
        if 'monitoring' in data:
            config.monitoring = MonitoringConfig(**data['monitoring'])
        if 'backup' in data:
            config.backup = BackupConfig(**data['backup'])
        if 'plugins' in data:
            config.plugins = PluginConfig(**data['plugins'])
        
        # Direct config fields
        for field in ['credentials_file', 'token_file', 'scopes']:
            if field in data:
                setattr(config, field, data[field])
        
        return config
    
    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        """Convert Config object to dictionary."""
        return {
            'database': {
                'path': config.database.path,
                'enable_logging': config.database.enable_logging,
                'connection_pool_size': config.database.connection_pool_size,
                'timeout': config.database.timeout,
                'pragma_settings': config.database.pragma_settings
            },
            'sync': {
                'workers': config.sync.workers,
                'max_retry_attempts': config.sync.max_retry_attempts,
                'retry_delay_seconds': config.sync.retry_delay_seconds,
                'batch_size': config.sync.batch_size,
                'rate_limit_requests_per_minute': config.sync.rate_limit_requests_per_minute,
                'enable_incremental_sync': config.sync.enable_incremental_sync,
                'full_sync_interval_hours': config.sync.full_sync_interval_hours,
                'deleted_message_check_interval_hours': config.sync.deleted_message_check_interval_hours
            },
            'cache': {
                'enabled': config.cache.enabled,
                'type': config.cache.type,
                'ttl_seconds': config.cache.ttl_seconds,
                'max_size': config.cache.max_size,
                'redis_url': config.cache.redis_url,
                'file_cache_dir': config.cache.file_cache_dir
            },
            'attachments': {
                'enabled': config.attachments.enabled,
                'download_path': config.attachments.download_path,
                'max_size_mb': config.attachments.max_size_mb,
                'allowed_types': config.attachments.allowed_types,
                'extract_text': config.attachments.extract_text,
                'virus_scan_enabled': config.attachments.virus_scan_enabled
            },
            'analytics': {
                'enabled': config.analytics.enabled,
                'metrics_retention_days': config.analytics.metrics_retention_days,
                'generate_daily_reports': config.analytics.generate_daily_reports,
                'generate_weekly_reports': config.analytics.generate_weekly_reports,
                'generate_monthly_reports': config.analytics.generate_monthly_reports,
                'export_formats': config.analytics.export_formats
            },
            'web': {
                'enabled': config.web.enabled,
                'host': config.web.host,
                'port': config.web.port,
                'debug': config.web.debug,
                'secret_key': config.web.secret_key,
                'cors_enabled': config.web.cors_enabled,
                'auth_enabled': config.web.auth_enabled,
                'session_timeout_minutes': config.web.session_timeout_minutes
            },
            'monitoring': {
                'enabled': config.monitoring.enabled,
                'prometheus_enabled': config.monitoring.prometheus_enabled,
                'prometheus_port': config.monitoring.prometheus_port,
                'log_level': config.monitoring.log_level,
                'log_file': config.monitoring.log_file,
                'log_rotation': config.monitoring.log_rotation,
                'log_max_size': config.monitoring.log_max_size,
                'log_backup_count': config.monitoring.log_backup_count,
                'health_check_interval': config.monitoring.health_check_interval
            },
            'backup': {
                'enabled': config.backup.enabled,
                'backup_path': config.backup.backup_path,
                'schedule_cron': config.backup.schedule_cron,
                'retention_days': config.backup.retention_days,
                'compression_enabled': config.backup.compression_enabled,
                'encryption_enabled': config.backup.encryption_enabled,
                'encryption_key_file': config.backup.encryption_key_file
            },
            'plugins': {
                'enabled': config.plugins.enabled,
                'plugin_paths': config.plugins.plugin_paths,
                'auto_load': config.plugins.auto_load,
                'enabled_plugins': config.plugins.enabled_plugins,
                'disabled_plugins': config.plugins.disabled_plugins
            },
            'credentials_file': config.credentials_file,
            'token_file': config.token_file,
            'scopes': config.scopes
        }
    
    def _validate_config(self, config: Config) -> None:
        """Validate configuration values."""
        # Database validation
        if not config.database.path:
            raise ValueError("Database path cannot be empty")
        
        # Sync validation
        if config.sync.workers < 1:
            raise ValueError("Number of workers must be positive")
        if config.sync.batch_size < 1:
            raise ValueError("Batch size must be positive")
        
        # Web validation
        if config.web.enabled:
            if not (1 <= config.web.port <= 65535):
                raise ValueError("Web port must be between 1 and 65535")
        
        # Cache validation
        if config.cache.enabled and config.cache.type == "redis":
            if not config.cache.redis_url:
                raise ValueError("Redis URL required when using Redis cache")
        
        # Monitoring validation
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.monitoring.log_level not in valid_log_levels:
            raise ValueError(f"Log level must be one of {valid_log_levels}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config_manager.get_config()


def load_config(config_file: Optional[str] = None) -> Config:
    """Load configuration from file or default sources."""
    return config_manager.load_config(config_file)


def save_config(config: Config, filename: str = "local.yaml") -> None:
    """Save configuration to file."""
    config_manager.save_config(config, filename)


def create_default_config() -> None:
    """Create default configuration files."""
    config_manager.create_default_config()