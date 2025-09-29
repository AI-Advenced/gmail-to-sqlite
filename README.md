# Gmail to SQLite Advanced

🚀 **Advanced Gmail to SQLite synchronization tool** with comprehensive analytics, plugin system, multi-format export capabilities, and enterprise-grade features.

## ✨ Advanced Features

### 🔧 **Core Enhancements**
- **Advanced Configuration System** - YAML/JSON configuration with environment variable support
- **Plugin Architecture** - Extensible plugin system with hooks and filters
- **Multiple Cache Backends** - Memory, Redis, and file-based caching
- **Comprehensive Analytics** - Email insights, contact analysis, and trend reporting
- **Advanced Search** - Full-text search with SQLite FTS and Whoosh backends
- **Multi-Format Export** - CSV, JSON, XML, XLSX, MBOX, and HTML export formats

### 📊 **Analytics & Reporting**
- **Email Metrics** - Volume analysis, size distribution, thread patterns
- **Contact Insights** - Communication frequency, response time analysis
- **Time Pattern Analysis** - Peak hours, seasonal trends, activity patterns  
- **Label Distribution** - Gmail label usage statistics
- **Automated Reports** - Daily, weekly, and monthly report generation
- **Custom Visualizations** - Charts and graphs with Plotly/Matplotlib

### 🔌 **Plugin System**
- **Extensible Architecture** - Hook and filter system for customization
- **Message Processing Plugins** - Custom message transformation and enrichment
- **Analytics Plugins** - Custom metrics and reporting extensions
- **Export Plugins** - Additional export format support
- **Auto-Discovery** - Automatic plugin loading and registration

### 🗂️ **Attachment Handling**
- **Intelligent Download** - Size limits, type filtering, virus scanning
- **Text Extraction** - PDF, DOCX, image OCR support
- **Metadata Management** - Hash verification, duplicate detection
- **Secure Storage** - Organized file storage with cleanup policies

### 🔍 **Advanced Search**
- **Full-Text Search** - SQLite FTS5 and Whoosh backend support
- **Complex Filtering** - Multiple field filters with operators
- **Search Suggestions** - Auto-complete based on content
- **Saved Searches** - Query templates and bookmarks
- **Performance Optimization** - Indexed search with caching

### 📤 **Multi-Format Export**
- **CSV Export** - Configurable delimiters and formatting
- **JSON Export** - Structured data with nested objects
- **XML Export** - Hierarchical data representation
- **XLSX Export** - Excel-compatible spreadsheet format
- **MBOX Export** - Standard email archive format
- **HTML Export** - Formatted web pages with styling

### ⚡ **Performance & Scalability**
- **Intelligent Caching** - Multi-level caching with TTL support
- **Connection Pooling** - Optimized database connections
- **Batch Processing** - Efficient bulk operations
- **Memory Optimization** - Streaming processing for large datasets
- **Rate Limiting** - Gmail API quota management

## 📋 **Prerequisites**

- **Python 3.8+**
- **Google Cloud Project** with Gmail API enabled
- **OAuth 2.0 credentials** (`credentials.json`)

### Optional Dependencies
- **Redis** - For Redis caching backend
- **ClamAV** - For attachment virus scanning  
- **Tesseract** - For image text extraction (OCR)
- **LibMagic** - For advanced MIME type detection

## 🚀 **Installation**

### Basic Installation
```bash
git clone https://github.com/marcboeker/gmail-to-sqlite-advanced.git
cd gmail-to-sqlite-advanced
pip install -e .
```

### Full Installation (All Features)
```bash
pip install -e ".[all]"
```

### Feature-Specific Installation
```bash
# Analytics features
pip install -e ".[analytics]"

# Export capabilities  
pip install -e ".[export]"

# Search enhancements
pip install -e ".[search]"

# Caching backends
pip install -e ".[cache]"

# Attachment processing
pip install -e ".[attachments]"

# Web interface
pip install -e ".[web]"
```

## ⚙️ **Configuration**

### Quick Start Configuration
```bash
# Create default configuration
gmail-to-sqlite config init

# Edit configuration
vim config/local.yaml
```

### Configuration Example
```yaml
# config/local.yaml
database:
  path: "data/messages.db"
  
sync:
  workers: 8
  batch_size: 200
  
cache:
  enabled: true
  type: "redis"
  redis_url: "redis://localhost:6379/0"
  
attachments:
  enabled: true
  download_path: "data/attachments"
  max_size_mb: 50
  extract_text: true
  
analytics:
  enabled: true
  generate_daily_reports: true
  
web:
  enabled: true
  port: 8080
```

## 🔧 **Usage**

### Basic Synchronization
```bash
# Incremental sync
gmail-to-sqlite sync --config config/local.yaml

# Full sync with analytics
gmail-to-sqlite sync --full-sync --generate-reports

# Sync with attachment download
gmail-to-sqlite sync --enable-attachments
```

### Advanced Analytics
```bash
# Generate comprehensive report
gmail-to-sqlite analytics generate-report --format html

# Contact analysis
gmail-to-sqlite analytics contacts --top 50

# Time pattern analysis  
gmail-to-sqlite analytics time-patterns --months 12

# Custom metrics
gmail-to-sqlite analytics custom --plugin custom-analytics
```

### Search Operations
```bash
# Full-text search
gmail-to-sqlite search "project proposal" --backend whoosh

# Advanced filtering
gmail-to-sqlite search --sender "boss@company.com" --date-range "2024-01-01,2024-12-31"

# Export search results
gmail-to-sqlite search "meeting notes" --export csv --output search_results.csv
```

### Export Operations
```bash
# Export all messages to Excel
gmail-to-sqlite export xlsx messages.xlsx

# Filtered export
gmail-to-sqlite export csv --sender-contains "@important-client.com" --output client_emails.csv

# Custom export with template
gmail-to-sqlite export mbox --query "SELECT * FROM messages WHERE size > 1048576" --output large_emails.mbox

# Contact summary report
gmail-to-sqlite export json --report-type contacts --output contact_summary.json
```

### Plugin Management
```bash
# List available plugins
gmail-to-sqlite plugins list

# Enable plugin
gmail-to-sqlite plugins enable custom-analytics

# Install plugin from file
gmail-to-sqlite plugins install /path/to/plugin.py

# Plugin development mode
gmail-to-sqlite plugins dev-mode --watch plugins/
```

### Web Interface
```bash
# Start web interface
gmail-to-sqlite web start --host 0.0.0.0 --port 8080

# Web interface with authentication
gmail-to-sqlite web start --auth-required --secret-key "your-secret-key"
```

## 🔌 **Plugin Development**

### Simple Message Processor Plugin
```python
from gmail_to_sqlite.plugins import MessageProcessorPlugin, PluginMetadata

class CustomProcessor(MessageProcessorPlugin):
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="custom-processor",
            version="1.0.0",
            description="Custom message processing",
            author="Your Name"
        )
    
    def initialize(self, plugin_manager) -> None:
        # Register hooks
        hook = plugin_manager.get_hook("before_message_process")
        hook.add_callback(self.process_message)
    
    def process_message(self, message):
        # Custom processing logic
        message.custom_field = "processed"
        return message
```

### Analytics Plugin
```python  
from gmail_to_sqlite.plugins import AnalyticsPlugin

class CustomAnalytics(AnalyticsPlugin):
    def generate_report(self, data, report_type):
        # Generate custom analytics
        return {"custom_metric": 42}
    
    def get_metrics(self):
        # Return current metrics
        return {"active_threads": 150}
```

## 📊 **Database Schema**

### Enhanced Schema
The advanced version includes additional tables and fields:

```sql
-- Core messages table (enhanced)
CREATE TABLE messages (
    message_id TEXT UNIQUE,
    thread_id TEXT,
    sender JSON,
    recipients JSON, 
    labels JSON,
    subject TEXT,
    body TEXT,
    size INTEGER,
    timestamp DATETIME,
    is_read BOOLEAN,
    is_outgoing BOOLEAN,
    is_deleted BOOLEAN,
    last_indexed DATETIME,
    custom_data JSON,        -- For plugin data
    attachment_count INTEGER, -- Attachment metadata
    extracted_text TEXT      -- OCR/attachment text
);

-- Analytics metrics
CREATE TABLE email_metrics (
    date TEXT,
    metric_name TEXT,
    metric_value REAL,
    metadata JSON
);

-- Attachment metadata  
CREATE TABLE attachments (
    message_id TEXT,
    filename TEXT,
    size INTEGER,
    mime_type TEXT,
    md5_hash TEXT,
    file_path TEXT,
    extracted_text TEXT
);

-- Full-text search index
CREATE VIRTUAL TABLE messages_fts USING fts5(
    message_id, subject, body, sender, recipients
);
```

## 🔍 **Advanced Queries**

### Analytics Queries
```sql
-- Top senders by volume
SELECT 
    json_extract(sender, '$.email') as email,
    COUNT(*) as message_count,
    SUM(size)/1024/1024 as total_mb
FROM messages 
GROUP BY email 
ORDER BY message_count DESC;

-- Response time analysis
SELECT 
    strftime('%H', timestamp) as hour,
    AVG(size) as avg_size,
    COUNT(*) as count
FROM messages 
GROUP BY hour 
ORDER BY hour;

-- Thread complexity analysis
SELECT 
    thread_id,
    COUNT(*) as message_count,
    COUNT(DISTINCT json_extract(sender, '$.email')) as participants,
    MIN(timestamp) as thread_start,
    MAX(timestamp) as thread_end
FROM messages 
GROUP BY thread_id 
HAVING message_count > 5
ORDER BY message_count DESC;
```

### Search Queries
```sql
-- Find messages with attachments
SELECT * FROM messages 
WHERE attachment_count > 0
AND json_extract(custom_data, '$.has_large_attachment') = 1;

-- Sentiment analysis results (from plugin)
SELECT 
    subject,
    json_extract(custom_data, '$.sentiment') as sentiment,
    timestamp
FROM messages 
WHERE json_extract(custom_data, '$.sentiment') IS NOT NULL
ORDER BY timestamp DESC;
```

## 🎯 **Performance Optimization**

### Configuration Tuning
```yaml
# High-performance configuration
database:
  pragma_settings:
    cache_size: -128000  # 128MB cache
    mmap_size: 536870912 # 512MB mmap
    journal_mode: "WAL"
    synchronous: "NORMAL"

sync:
  workers: 12           # Scale based on CPU cores
  batch_size: 500       # Larger batches for speed
  
cache:
  type: "redis"         # Redis for better performance
  max_size: 10000       # Larger cache
  
attachments:
  enabled: false        # Disable if not needed
```

### Monitoring
```bash
# Performance monitoring
gmail-to-sqlite monitor --metrics-port 9090

# Database statistics
gmail-to-sqlite db stats --analyze

# Cache performance
gmail-to-sqlite cache stats --detailed
```

## 🔒 **Security Features**

- **Credential Encryption** - Secure token storage
- **Virus Scanning** - ClamAV integration for attachments
- **Content Filtering** - Sensitive data redaction plugins
- **Access Control** - Web interface authentication
- **Audit Logging** - Comprehensive operation logs

## 🌐 **Web Interface Features**

- **Dashboard** - Overview of email statistics
- **Search Interface** - Advanced search with filters
- **Analytics Viewer** - Interactive charts and reports  
- **Export Manager** - GUI export operations
- **Plugin Management** - Web-based plugin control
- **Configuration Editor** - Online configuration management

## 🤝 **Contributing**

### Development Setup
```bash
# Development installation
git clone https://github.com/marcboeker/gmail-to-sqlite-advanced.git
cd gmail-to-sqlite-advanced
pip install -e ".[dev,all]"

# Run tests
pytest tests/

# Code formatting
black gmail_to_sqlite tests
flake8 gmail_to_sqlite tests

# Type checking
mypy gmail_to_sqlite
```

### Plugin Development
```bash
# Create plugin template
gmail-to-sqlite plugins create-template MyPlugin

# Test plugin
gmail-to-sqlite plugins test plugins/my_plugin.py

# Package plugin
gmail-to-sqlite plugins package plugins/my_plugin.py
```

## 📄 **License**

MIT License - see [LICENSE](LICENSE) file for details.

## 🆘 **Support**

- **Documentation**: [GitHub Wiki](https://github.com/marcboeker/gmail-to-sqlite-advanced/wiki)
- **Issues**: [GitHub Issues](https://github.com/marcboeker/gmail-to-sqlite-advanced/issues)
- **Discussions**: [GitHub Discussions](https://github.com/marcboeker/gmail-to-sqlite-advanced/discussions)

## 🎉 **Acknowledgments**

- Original Gmail to SQLite project by Marc Boeker
- Contributors and community feedback
- Open source libraries and dependencies

---

⭐ **Star this repository if you find it useful!**