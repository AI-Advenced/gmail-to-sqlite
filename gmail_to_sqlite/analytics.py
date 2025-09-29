"""
Advanced analytics and reporting system for Gmail to SQLite.

Provides comprehensive email analytics, trend analysis, contact insights,
and automated report generation capabilities.
"""

import sqlite3
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta, date
from dataclasses import dataclass
from collections import defaultdict, Counter
import json
import logging

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.dates import DateFormatter
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EmailMetrics:
    """Container for email metrics."""
    total_messages: int
    total_sent: int
    total_received: int
    total_unread: int
    total_size_mb: float
    avg_message_size_kb: float
    date_range: Tuple[datetime, datetime]


@dataclass
class ContactInsights:
    """Container for contact analysis."""
    email: str
    name: str
    total_messages: int
    sent_count: int
    received_count: int
    avg_response_time_hours: Optional[float]
    last_contact: datetime
    contact_frequency_score: float


@dataclass
class TimeAnalysis:
    """Container for time-based analysis."""
    peak_hours: List[int]
    peak_days: List[str]
    monthly_trends: Dict[str, int]
    yearly_trends: Dict[int, int]
    response_time_stats: Dict[str, float]


class EmailAnalyzer:
    """Advanced email analytics engine."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection = None
    
    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with proper configuration."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection
    
    def close_connection(self) -> None:
        """Close database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def get_basic_metrics(self, start_date: Optional[datetime] = None, 
                         end_date: Optional[datetime] = None) -> EmailMetrics:
        """Get basic email metrics for date range."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Build date filter
        date_filter = ""
        params = []
        if start_date:
            date_filter += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            date_filter += " AND timestamp <= ?"
            params.append(end_date)
        
        try:
            # Total messages
            cursor.execute(f"""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN is_outgoing = 1 THEN 1 ELSE 0 END) as sent,
                       SUM(CASE WHEN is_outgoing = 0 THEN 1 ELSE 0 END) as received,
                       SUM(CASE WHEN is_read = 0 THEN 1 ELSE 0 END) as unread,
                       SUM(size) as total_size,
                       AVG(size) as avg_size,
                       MIN(timestamp) as min_date,
                       MAX(timestamp) as max_date
                FROM messages 
                WHERE is_deleted = 0 {date_filter}
            """, params)
            
            row = cursor.fetchone()
            
            total_size_mb = (row['total_size'] or 0) / (1024 * 1024)
            avg_size_kb = (row['avg_size'] or 0) / 1024
            
            min_date = datetime.fromisoformat(row['min_date']) if row['min_date'] else datetime.now()
            max_date = datetime.fromisoformat(row['max_date']) if row['max_date'] else datetime.now()
            
            return EmailMetrics(
                total_messages=row['total'] or 0,
                total_sent=row['sent'] or 0,
                total_received=row['received'] or 0,
                total_unread=row['unread'] or 0,
                total_size_mb=round(total_size_mb, 2),
                avg_message_size_kb=round(avg_size_kb, 2),
                date_range=(min_date, max_date)
            )
            
        except Exception as e:
            logger.error(f"Failed to get basic metrics: {e}")
            return EmailMetrics(0, 0, 0, 0, 0.0, 0.0, (datetime.now(), datetime.now()))
    
    def analyze_contacts(self, limit: int = 50) -> List[ContactInsights]:
        """Analyze contact interactions and patterns."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Get contact statistics
            cursor.execute("""
                WITH contact_stats AS (
                    SELECT 
                        json_extract(sender, '$.email') as email,
                        json_extract(sender, '$.name') as name,
                        COUNT(*) as total_messages,
                        SUM(CASE WHEN is_outgoing = 1 THEN 1 ELSE 0 END) as sent_count,
                        SUM(CASE WHEN is_outgoing = 0 THEN 1 ELSE 0 END) as received_count,
                        MAX(timestamp) as last_contact,
                        AVG(size) as avg_size
                    FROM messages 
                    WHERE is_deleted = 0 
                        AND json_extract(sender, '$.email') IS NOT NULL
                        AND json_extract(sender, '$.email') != ''
                    GROUP BY json_extract(sender, '$.email')
                    HAVING total_messages >= 2
                    ORDER BY total_messages DESC
                    LIMIT ?
                )
                SELECT * FROM contact_stats
            """, (limit,))
            
            contacts = []
            for row in cursor.fetchall():
                # Calculate contact frequency score (messages per day since first contact)
                last_contact = datetime.fromisoformat(row['last_contact'])
                days_since_contact = (datetime.now() - last_contact).days + 1
                frequency_score = row['total_messages'] / days_since_contact
                
                # Calculate average response time (simplified)
                avg_response_time = self._calculate_response_time(row['email'])
                
                contact = ContactInsights(
                    email=row['email'] or '',
                    name=row['name'] or row['email'] or '',
                    total_messages=row['total_messages'],
                    sent_count=row['sent_count'],
                    received_count=row['received_count'],
                    avg_response_time_hours=avg_response_time,
                    last_contact=last_contact,
                    contact_frequency_score=round(frequency_score, 4)
                )
                contacts.append(contact)
            
            return contacts
            
        except Exception as e:
            logger.error(f"Failed to analyze contacts: {e}")
            return []
    
    def analyze_time_patterns(self) -> TimeAnalysis:
        """Analyze temporal patterns in email activity."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            # Peak hours analysis
            cursor.execute("""
                SELECT 
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as count
                FROM messages 
                WHERE is_deleted = 0
                GROUP BY hour
                ORDER BY count DESC
            """)
            
            hour_data = cursor.fetchall()
            peak_hours = [int(row['hour']) for row in hour_data[:3]]
            
            # Peak days analysis
            cursor.execute("""
                SELECT 
                    CASE strftime('%w', timestamp)
                        WHEN '0' THEN 'Sunday'
                        WHEN '1' THEN 'Monday'
                        WHEN '2' THEN 'Tuesday'
                        WHEN '3' THEN 'Wednesday'
                        WHEN '4' THEN 'Thursday'
                        WHEN '5' THEN 'Friday'
                        WHEN '6' THEN 'Saturday'
                    END as day_name,
                    COUNT(*) as count
                FROM messages 
                WHERE is_deleted = 0
                GROUP BY strftime('%w', timestamp)
                ORDER BY count DESC
            """)
            
            day_data = cursor.fetchall()
            peak_days = [row['day_name'] for row in day_data[:3]]
            
            # Monthly trends
            cursor.execute("""
                SELECT 
                    strftime('%Y-%m', timestamp) as month,
                    COUNT(*) as count
                FROM messages 
                WHERE is_deleted = 0
                    AND timestamp >= date('now', '-24 months')
                GROUP BY month
                ORDER BY month
            """)
            
            monthly_trends = {row['month']: row['count'] for row in cursor.fetchall()}
            
            # Yearly trends
            cursor.execute("""
                SELECT 
                    strftime('%Y', timestamp) as year,
                    COUNT(*) as count
                FROM messages 
                WHERE is_deleted = 0
                GROUP BY year
                ORDER BY year
            """)
            
            yearly_trends = {int(row['year']): row['count'] for row in cursor.fetchall()}
            
            # Response time statistics
            response_stats = self._calculate_global_response_stats()
            
            return TimeAnalysis(
                peak_hours=peak_hours,
                peak_days=peak_days,
                monthly_trends=monthly_trends,
                yearly_trends=yearly_trends,
                response_time_stats=response_stats
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze time patterns: {e}")
            return TimeAnalysis([], [], {}, {}, {})
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of email labels."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT labels FROM messages 
                WHERE is_deleted = 0 AND labels IS NOT NULL
            """)
            
            label_counts = Counter()
            for row in cursor.fetchall():
                try:
                    labels = json.loads(row['labels'])
                    if isinstance(labels, list):
                        for label in labels:
                            label_counts[label] += 1
                except (json.JSONDecodeError, TypeError):
                    continue
            
            return dict(label_counts.most_common(20))
            
        except Exception as e:
            logger.error(f"Failed to get label distribution: {e}")
            return {}
    
    def get_size_analysis(self) -> Dict[str, Any]:
        """Analyze message size patterns."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    AVG(size) as avg_size,
                    MIN(size) as min_size,
                    MAX(size) as max_size,
                    COUNT(CASE WHEN size < 1024 THEN 1 END) as very_small,
                    COUNT(CASE WHEN size BETWEEN 1024 AND 10240 THEN 1 END) as small,
                    COUNT(CASE WHEN size BETWEEN 10240 AND 102400 THEN 1 END) as medium,
                    COUNT(CASE WHEN size BETWEEN 102400 AND 1048576 THEN 1 END) as large,
                    COUNT(CASE WHEN size > 1048576 THEN 1 END) as very_large
                FROM messages 
                WHERE is_deleted = 0
            """)
            
            row = cursor.fetchone()
            
            return {
                'avg_size_kb': round((row['avg_size'] or 0) / 1024, 2),
                'min_size_bytes': row['min_size'] or 0,
                'max_size_mb': round((row['max_size'] or 0) / (1024 * 1024), 2),
                'size_distribution': {
                    'very_small_<1KB': row['very_small'],
                    'small_1-10KB': row['small'],
                    'medium_10-100KB': row['medium'],
                    'large_100KB-1MB': row['large'],
                    'very_large_>1MB': row['very_large']
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze message sizes: {e}")
            return {}
    
    def get_thread_analysis(self) -> Dict[str, Any]:
        """Analyze email thread patterns."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                WITH thread_stats AS (
                    SELECT 
                        thread_id,
                        COUNT(*) as message_count,
                        COUNT(DISTINCT json_extract(sender, '$.email')) as participant_count,
                        MIN(timestamp) as thread_start,
                        MAX(timestamp) as thread_end
                    FROM messages 
                    WHERE is_deleted = 0
                    GROUP BY thread_id
                )
                SELECT 
                    AVG(message_count) as avg_messages_per_thread,
                    MAX(message_count) as max_messages_in_thread,
                    AVG(participant_count) as avg_participants_per_thread,
                    COUNT(CASE WHEN message_count = 1 THEN 1 END) as single_message_threads,
                    COUNT(CASE WHEN message_count > 10 THEN 1 END) as long_threads,
                    COUNT(*) as total_threads
                FROM thread_stats
            """)
            
            row = cursor.fetchone()
            
            return {
                'total_threads': row['total_threads'],
                'avg_messages_per_thread': round(row['avg_messages_per_thread'] or 0, 2),
                'max_messages_in_thread': row['max_messages_in_thread'] or 0,
                'avg_participants_per_thread': round(row['avg_participants_per_thread'] or 0, 2),
                'single_message_threads': row['single_message_threads'] or 0,
                'long_threads_10plus': row['long_threads'] or 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze threads: {e}")
            return {}
    
    def _calculate_response_time(self, email: str) -> Optional[float]:
        """Calculate average response time for a specific contact."""
        # Simplified implementation - would need more sophisticated logic
        # to properly match sent/received message pairs
        return None  # Placeholder
    
    def _calculate_global_response_stats(self) -> Dict[str, float]:
        """Calculate global response time statistics."""
        # Placeholder for complex response time analysis
        return {
            'avg_response_hours': 0.0,
            'median_response_hours': 0.0,
            'response_rate_percentage': 0.0
        }


class ReportGenerator:
    """Generate various types of analytical reports."""
    
    def __init__(self, analyzer: EmailAnalyzer, output_dir: str = "data/reports"):
        self.analyzer = analyzer
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def generate_summary_report(self, format: str = "json") -> str:
        """Generate comprehensive summary report."""
        try:
            # Gather all analytics data
            basic_metrics = self.analyzer.get_basic_metrics()
            top_contacts = self.analyzer.analyze_contacts(limit=10)
            time_analysis = self.analyzer.analyze_time_patterns()
            label_dist = self.analyzer.get_label_distribution()
            size_analysis = self.analyzer.get_size_analysis()
            thread_analysis = self.analyzer.get_thread_analysis()
            
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'basic_metrics': {
                    'total_messages': basic_metrics.total_messages,
                    'total_sent': basic_metrics.total_sent,
                    'total_received': basic_metrics.total_received,
                    'total_unread': basic_metrics.total_unread,
                    'total_size_mb': basic_metrics.total_size_mb,
                    'avg_message_size_kb': basic_metrics.avg_message_size_kb,
                    'date_range': [
                        basic_metrics.date_range[0].isoformat(),
                        basic_metrics.date_range[1].isoformat()
                    ]
                },
                'top_contacts': [
                    {
                        'email': c.email,
                        'name': c.name,
                        'total_messages': c.total_messages,
                        'sent_count': c.sent_count,
                        'received_count': c.received_count,
                        'last_contact': c.last_contact.isoformat(),
                        'frequency_score': c.contact_frequency_score
                    } for c in top_contacts
                ],
                'time_patterns': {
                    'peak_hours': time_analysis.peak_hours,
                    'peak_days': time_analysis.peak_days,
                    'monthly_trends': time_analysis.monthly_trends,
                    'yearly_trends': time_analysis.yearly_trends
                },
                'label_distribution': label_dist,
                'size_analysis': size_analysis,
                'thread_analysis': thread_analysis
            }
            
            # Save report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if format.lower() == "json":
                filename = f"email_summary_{timestamp}.json"
                filepath = Path(self.output_dir) / filename
                
                with open(filepath, 'w') as f:
                    json.dump(report_data, f, indent=2)
                
                logger.info(f"Summary report generated: {filepath}")
                return str(filepath)
            
            elif format.lower() == "html":
                filename = f"email_summary_{timestamp}.html"
                filepath = Path(self.output_dir) / filename
                
                html_content = self._generate_html_report(report_data)
                
                with open(filepath, 'w') as f:
                    f.write(html_content)
                
                logger.info(f"HTML report generated: {filepath}")
                return str(filepath)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
        except Exception as e:
            logger.error(f"Failed to generate summary report: {e}")
            return ""
    
    def _generate_html_report(self, data: Dict[str, Any]) -> str:
        """Generate HTML version of the report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gmail Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                         background: #e8f4f8; border-radius: 5px; }}
                .contact {{ margin: 5px 0; padding: 5px; background: #f9f9f9; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Gmail Analytics Report</h1>
                <p>Generated: {data['generated_at']}</p>
            </div>
            
            <div class="section">
                <h2>Basic Metrics</h2>
                <div class="metric">Total Messages: {data['basic_metrics']['total_messages']}</div>
                <div class="metric">Sent: {data['basic_metrics']['total_sent']}</div>
                <div class="metric">Received: {data['basic_metrics']['total_received']}</div>
                <div class="metric">Unread: {data['basic_metrics']['total_unread']}</div>
                <div class="metric">Total Size: {data['basic_metrics']['total_size_mb']} MB</div>
            </div>
            
            <div class="section">
                <h2>Top Contacts</h2>
                <table>
                    <tr><th>Email</th><th>Name</th><th>Total</th><th>Sent</th><th>Received</th></tr>
        """
        
        for contact in data['top_contacts'][:10]:
            html += f"""
                    <tr>
                        <td>{contact['email']}</td>
                        <td>{contact['name']}</td>
                        <td>{contact['total_messages']}</td>
                        <td>{contact['sent_count']}</td>
                        <td>{contact['received_count']}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Time Patterns</h2>
                <p><strong>Peak Hours:</strong> """ + ", ".join(map(str, data['time_patterns']['peak_hours'])) + """</p>
                <p><strong>Peak Days:</strong> """ + ", ".join(data['time_patterns']['peak_days']) + """</p>
            </div>
            
        </body>
        </html>
        """
        
        return html
    
    def generate_visualization_report(self) -> Optional[str]:
        """Generate report with charts and visualizations."""
        if not PANDAS_AVAILABLE or not PLOTLY_AVAILABLE:
            logger.warning("Pandas or Plotly not available for visualization report")
            return None
        
        try:
            # This would contain complex visualization generation
            # using plotly or matplotlib
            logger.info("Visualization report generation not fully implemented")
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate visualization report: {e}")
            return None


class MetricsCollector:
    """Collect and store metrics over time."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._setup_metrics_table()
    
    def _setup_metrics_table(self) -> None:
        """Set up metrics storage table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS email_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL,
                        metric_name TEXT NOT NULL,
                        metric_value REAL NOT NULL,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(date, metric_name)
                    )
                """)
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to setup metrics table: {e}")
    
    def collect_daily_metrics(self, target_date: Optional[date] = None) -> None:
        """Collect and store daily metrics."""
        if target_date is None:
            target_date = date.today()
        
        try:
            analyzer = EmailAnalyzer(self.db_path)
            
            # Get metrics for the day
            start_time = datetime.combine(target_date, datetime.min.time())
            end_time = datetime.combine(target_date, datetime.max.time())
            
            daily_metrics = analyzer.get_basic_metrics(start_time, end_time)
            
            # Store metrics
            metrics_to_store = [
                ('total_messages', daily_metrics.total_messages),
                ('total_sent', daily_metrics.total_sent),
                ('total_received', daily_metrics.total_received),
                ('total_unread', daily_metrics.total_unread),
                ('total_size_mb', daily_metrics.total_size_mb),
                ('avg_message_size_kb', daily_metrics.avg_message_size_kb)
            ]
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for metric_name, value in metrics_to_store:
                    cursor.execute("""
                        INSERT OR REPLACE INTO email_metrics 
                        (date, metric_name, metric_value) 
                        VALUES (?, ?, ?)
                    """, (target_date.isoformat(), metric_name, value))
                
                conn.commit()
                
            analyzer.close_connection()
            logger.info(f"Daily metrics collected for {target_date}")
            
        except Exception as e:
            logger.error(f"Failed to collect daily metrics: {e}")
    
    def get_metric_history(self, metric_name: str, days: int = 30) -> List[Tuple[str, float]]:
        """Get historical values for a metric."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT date, metric_value 
                    FROM email_metrics 
                    WHERE metric_name = ? 
                        AND date >= date('now', '-{} days')
                    ORDER BY date
                """.format(days), (metric_name,))
                
                return [(row[0], row[1]) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get metric history: {e}")
            return []