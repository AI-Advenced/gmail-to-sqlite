"""
Advanced search and filtering capabilities for Gmail to SQLite.

Provides full-text search, complex filtering, and query building functionality
with support for multiple search backends and indexing strategies.
"""

import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

try:
    import whoosh
    from whoosh.index import create_index, open_dir, exists_in
    from whoosh.fields import Schema, TEXT, ID, DATETIME, NUMERIC, BOOLEAN
    from whoosh.qparser import QueryParser, MultifieldParser
    from whoosh.query import Query
    WHOOSH_AVAILABLE = True
except ImportError:
    WHOOSH_AVAILABLE = False

logger = logging.getLogger(__name__)


class SearchBackend(Enum):
    """Available search backends."""
    SQLITE_FTS = "sqlite_fts"
    WHOOSH = "whoosh"
    BASIC = "basic"


class SortOrder(Enum):
    """Sort order options."""
    ASC = "ASC"
    DESC = "DESC"


@dataclass
class SearchFilter:
    """Represents a search filter condition."""
    field: str
    operator: str  # eq, ne, lt, le, gt, ge, like, in, not_in, between
    value: Any
    case_sensitive: bool = False


@dataclass
class SearchQuery:
    """Complete search query specification."""
    text: Optional[str] = None
    filters: List[SearchFilter] = None
    sort_field: Optional[str] = None
    sort_order: SortOrder = SortOrder.DESC
    limit: Optional[int] = None
    offset: int = 0
    include_deleted: bool = False
    date_range: Optional[Tuple[datetime, datetime]] = None


@dataclass
class SearchResult:
    """Search result container."""
    total_count: int
    results: List[Dict[str, Any]]
    query_time_ms: float
    backend_used: SearchBackend


class QueryBuilder:
    """Builds SQL queries from search specifications."""
    
    OPERATOR_MAP = {
        'eq': '= ?',
        'ne': '!= ?',
        'lt': '< ?',
        'le': '<= ?',
        'gt': '> ?',
        'ge': '>= ?',
        'like': 'LIKE ?',
        'not_like': 'NOT LIKE ?',
        'in': 'IN ({})',
        'not_in': 'NOT IN ({})',
        'between': 'BETWEEN ? AND ?',
        'is_null': 'IS NULL',
        'is_not_null': 'IS NOT NULL'
    }
    
    def build_search_query(self, query: SearchQuery, use_fts: bool = False) -> Tuple[str, List[Any]]:
        """Build SQL query from search specification."""
        params = []
        where_clauses = []
        
        # Base table selection
        if use_fts:
            base_query = "SELECT messages.* FROM messages JOIN messages_fts ON messages.rowid = messages_fts.rowid"
        else:
            base_query = "SELECT * FROM messages"
        
        # Full-text search
        if query.text and use_fts:
            where_clauses.append("messages_fts MATCH ?")
            params.append(query.text)
        elif query.text:
            # Fallback to LIKE search on multiple fields
            text_conditions = []
            for field in ['subject', 'body']:
                text_conditions.append(f"{field} LIKE ?")
                params.append(f"%{query.text}%")
            where_clauses.append(f"({' OR '.join(text_conditions)})")
        
        # Apply filters
        if query.filters:
            for filter_obj in query.filters:
                clause, filter_params = self._build_filter_clause(filter_obj)
                if clause:
                    where_clauses.append(clause)
                    params.extend(filter_params)
        
        # Date range filter
        if query.date_range:
            where_clauses.append("timestamp BETWEEN ? AND ?")
            params.extend(query.date_range)
        
        # Include/exclude deleted messages
        if not query.include_deleted:
            where_clauses.append("is_deleted = 0")
        
        # Build complete query
        sql_query = base_query
        if where_clauses:
            sql_query += " WHERE " + " AND ".join(where_clauses)
        
        # Add sorting
        if query.sort_field:
            sql_query += f" ORDER BY {query.sort_field} {query.sort_order.value}"
        else:
            sql_query += " ORDER BY timestamp DESC"
        
        # Add pagination
        if query.limit:
            sql_query += f" LIMIT {query.limit}"
        if query.offset:
            sql_query += f" OFFSET {query.offset}"
        
        return sql_query, params
    
    def _build_filter_clause(self, filter_obj: SearchFilter) -> Tuple[str, List[Any]]:
        """Build WHERE clause for a single filter."""
        field = filter_obj.field
        operator = filter_obj.operator
        value = filter_obj.value
        
        if operator not in self.OPERATOR_MAP:
            logger.warning(f"Unknown operator: {operator}")
            return "", []
        
        # Handle case sensitivity for text fields
        if not filter_obj.case_sensitive and isinstance(value, str):
            field = f"LOWER({field})"
            if isinstance(value, str):
                value = value.lower()
            elif isinstance(value, list):
                value = [v.lower() if isinstance(v, str) else v for v in value]
        
        # Handle special operators
        if operator in ['is_null', 'is_not_null']:
            return f"{field} {self.OPERATOR_MAP[operator]}", []
        elif operator in ['in', 'not_in']:
            if not isinstance(value, (list, tuple)):
                value = [value]
            placeholders = ','.join(['?'] * len(value))
            clause = f"{field} {self.OPERATOR_MAP[operator].format(placeholders)}"
            return clause, list(value)
        elif operator == 'between':
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                logger.error(f"Between operator requires exactly 2 values, got: {value}")
                return "", []
            return f"{field} {self.OPERATOR_MAP[operator]}", list(value)
        else:
            # Standard operators
            clause = f"{field} {self.OPERATOR_MAP[operator]}"
            return clause, [value]


class SQLiteSearchEngine:
    """SQLite-based search engine with FTS support."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.query_builder = QueryBuilder()
        self._setup_fts()
    
    def _setup_fts(self) -> None:
        """Set up SQLite FTS (Full-Text Search) virtual table."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if FTS table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='messages_fts'
                """)
                
                if not cursor.fetchone():
                    # Create FTS virtual table
                    cursor.execute("""
                        CREATE VIRTUAL TABLE messages_fts USING fts5(
                            message_id,
                            subject,
                            body,
                            sender,
                            recipients,
                            content='messages',
                            content_rowid='rowid'
                        )
                    """)
                    
                    # Create triggers to keep FTS table in sync
                    cursor.execute("""
                        CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
                            INSERT INTO messages_fts(rowid, message_id, subject, body, sender, recipients)
                            VALUES (NEW.rowid, NEW.message_id, NEW.subject, NEW.body, NEW.sender, NEW.recipients);
                        END
                    """)
                    
                    cursor.execute("""
                        CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
                            DELETE FROM messages_fts WHERE rowid = OLD.rowid;
                        END
                    """)
                    
                    cursor.execute("""
                        CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
                            DELETE FROM messages_fts WHERE rowid = OLD.rowid;
                            INSERT INTO messages_fts(rowid, message_id, subject, body, sender, recipients)
                            VALUES (NEW.rowid, NEW.message_id, NEW.subject, NEW.body, NEW.sender, NEW.recipients);
                        END
                    """)
                    
                    # Populate FTS table with existing data
                    cursor.execute("""
                        INSERT INTO messages_fts(rowid, message_id, subject, body, sender, recipients)
                        SELECT rowid, message_id, subject, body, sender, recipients FROM messages
                    """)
                    
                    conn.commit()
                    logger.info("SQLite FTS setup completed")
                
        except Exception as e:
            logger.error(f"Failed to setup SQLite FTS: {e}")
    
    def search(self, query: SearchQuery) -> SearchResult:
        """Execute search query using SQLite."""
        start_time = datetime.now()
        
        try:
            # Try FTS first if text search is involved
            use_fts = bool(query.text)
            sql_query, params = self.query_builder.build_search_query(query, use_fts)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Execute main query
                cursor.execute(sql_query, params)
                results = [dict(row) for row in cursor.fetchall()]
                
                # Get total count (without LIMIT/OFFSET)
                count_query = sql_query.split(" ORDER BY")[0]
                if " LIMIT " in count_query:
                    count_query = count_query.split(" LIMIT ")[0]
                count_query = f"SELECT COUNT(*) FROM ({count_query})"
                
                cursor.execute(count_query, params[:-2] if query.limit else params)
                total_count = cursor.fetchone()[0]
                
                query_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return SearchResult(
                    total_count=total_count,
                    results=results,
                    query_time_ms=query_time,
                    backend_used=SearchBackend.SQLITE_FTS if use_fts else SearchBackend.BASIC
                )
                
        except Exception as e:
            logger.error(f"SQLite search failed: {e}")
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            return SearchResult(
                total_count=0,
                results=[],
                query_time_ms=query_time,
                backend_used=SearchBackend.BASIC
            )


class WhooshSearchEngine:
    """Whoosh-based full-text search engine."""
    
    def __init__(self, index_dir: str = "data/search_index"):
        if not WHOOSH_AVAILABLE:
            raise ImportError("Whoosh is not available")
        
        self.index_dir = index_dir
        self.schema = Schema(
            message_id=ID(stored=True, unique=True),
            subject=TEXT(stored=True),
            body=TEXT(stored=True),
            sender_email=TEXT(stored=True),
            sender_name=TEXT(stored=True),
            recipients=TEXT(stored=True),
            labels=TEXT(stored=True),
            timestamp=DATETIME(stored=True),
            size=NUMERIC(stored=True),
            is_read=BOOLEAN(stored=True),
            is_outgoing=BOOLEAN(stored=True),
            is_deleted=BOOLEAN(stored=True)
        )
        
        self._setup_index()
    
    def _setup_index(self) -> None:
        """Set up Whoosh search index."""
        try:
            import os
            os.makedirs(self.index_dir, exist_ok=True)
            
            if not exists_in(self.index_dir):
                self.index = create_index(self.schema, self.index_dir)
                logger.info("Created new Whoosh index")
            else:
                self.index = open_dir(self.index_dir)
                logger.info("Opened existing Whoosh index")
                
        except Exception as e:
            logger.error(f"Failed to setup Whoosh index: {e}")
            self.index = None
    
    def index_message(self, message: Dict[str, Any]) -> None:
        """Index a single message."""
        if not self.index:
            return
        
        try:
            writer = self.index.writer()
            
            # Extract sender information
            sender = message.get('sender', {})
            sender_email = sender.get('email', '') if isinstance(sender, dict) else str(sender)
            sender_name = sender.get('name', '') if isinstance(sender, dict) else ''
            
            # Extract recipients
            recipients = message.get('recipients', {})
            recipient_text = ''
            if isinstance(recipients, dict):
                for key, value in recipients.items():
                    if isinstance(value, list):
                        for recipient in value:
                            if isinstance(recipient, dict):
                                recipient_text += f"{recipient.get('email', '')} {recipient.get('name', '')} "
            
            # Extract labels
            labels = message.get('labels', [])
            labels_text = ' '.join(labels) if isinstance(labels, list) else str(labels)
            
            writer.add_document(
                message_id=message.get('message_id', ''),
                subject=message.get('subject', '') or '',
                body=message.get('body', '') or '',
                sender_email=sender_email,
                sender_name=sender_name,
                recipients=recipient_text,
                labels=labels_text,
                timestamp=message.get('timestamp'),
                size=message.get('size', 0),
                is_read=message.get('is_read', False),
                is_outgoing=message.get('is_outgoing', False),
                is_deleted=message.get('is_deleted', False)
            )
            
            writer.commit()
            
        except Exception as e:
            logger.error(f"Failed to index message {message.get('message_id')}: {e}")
    
    def search(self, query: SearchQuery) -> SearchResult:
        """Execute search query using Whoosh."""
        if not self.index:
            return SearchResult(0, [], 0, SearchBackend.WHOOSH)
        
        start_time = datetime.now()
        
        try:
            with self.index.searcher() as searcher:
                whoosh_query = self._build_whoosh_query(query)
                
                # Execute search
                results = searcher.search(whoosh_query, limit=query.limit or 100)
                
                # Convert results
                search_results = []
                for hit in results:
                    result_dict = dict(hit)
                    search_results.append(result_dict)
                
                query_time = (datetime.now() - start_time).total_seconds() * 1000
                
                return SearchResult(
                    total_count=len(results),
                    results=search_results,
                    query_time_ms=query_time,
                    backend_used=SearchBackend.WHOOSH
                )
                
        except Exception as e:
            logger.error(f"Whoosh search failed: {e}")
            query_time = (datetime.now() - start_time).total_seconds() * 1000
            return SearchResult(0, [], query_time, SearchBackend.WHOOSH)
    
    def _build_whoosh_query(self, query: SearchQuery) -> Query:
        """Build Whoosh query from search specification."""
        from whoosh.query import And, Or, Term, Phrase, DateRange
        
        conditions = []
        
        # Text search
        if query.text:
            parser = MultifieldParser(["subject", "body", "sender_email", "sender_name"], 
                                    schema=self.schema)
            text_query = parser.parse(query.text)
            conditions.append(text_query)
        
        # Filters
        if query.filters:
            for filter_obj in query.filters:
                condition = self._build_whoosh_filter(filter_obj)
                if condition:
                    conditions.append(condition)
        
        # Date range
        if query.date_range:
            date_condition = DateRange("timestamp", query.date_range[0], query.date_range[1])
            conditions.append(date_condition)
        
        # Combine conditions
        if conditions:
            return And(conditions)
        else:
            from whoosh.query import Every
            return Every()
    
    def _build_whoosh_filter(self, filter_obj: SearchFilter) -> Optional[Query]:
        """Build Whoosh filter condition."""
        from whoosh.query import Term, NumericRange, Or, And, Not
        
        field = filter_obj.field
        operator = filter_obj.operator
        value = filter_obj.value
        
        try:
            if operator == 'eq':
                return Term(field, value)
            elif operator == 'ne':
                return Not(Term(field, value))
            elif operator in ['lt', 'le', 'gt', 'ge']:
                # Convert to numeric range
                if operator == 'lt':
                    return NumericRange(field, None, value, startexcl=True, endexcl=True)
                elif operator == 'le':
                    return NumericRange(field, None, value, startexcl=True, endexcl=False)
                elif operator == 'gt':
                    return NumericRange(field, value, None, startexcl=True, endexcl=True)
                elif operator == 'ge':
                    return NumericRange(field, value, None, startexcl=False, endexcl=True)
            elif operator == 'in':
                if isinstance(value, (list, tuple)):
                    return Or([Term(field, v) for v in value])
                else:
                    return Term(field, value)
            elif operator == 'between':
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    return NumericRange(field, value[0], value[1])
            
        except Exception as e:
            logger.error(f"Failed to build Whoosh filter: {e}")
        
        return None


class AdvancedSearchManager:
    """Main search manager with multiple backend support."""
    
    def __init__(self, db_path: str, config: Any):
        self.config = config
        self.sqlite_engine = SQLiteSearchEngine(db_path)
        
        # Initialize Whoosh if available and enabled
        self.whoosh_engine = None
        if WHOOSH_AVAILABLE and config.get('search', {}).get('whoosh_enabled', False):
            try:
                self.whoosh_engine = WhooshSearchEngine()
                logger.info("Whoosh search engine initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Whoosh engine: {e}")
    
    def search(self, query: SearchQuery, backend: Optional[SearchBackend] = None) -> SearchResult:
        """Execute search with automatic backend selection."""
        if backend == SearchBackend.WHOOSH and self.whoosh_engine:
            return self.whoosh_engine.search(query)
        elif backend in [SearchBackend.SQLITE_FTS, SearchBackend.BASIC, None]:
            return self.sqlite_engine.search(query)
        else:
            # Fallback to SQLite
            return self.sqlite_engine.search(query)
    
    def build_query(self, **kwargs) -> SearchQuery:
        """Build search query from keyword arguments."""
        return SearchQuery(
            text=kwargs.get('text'),
            filters=[SearchFilter(**f) for f in kwargs.get('filters', [])],
            sort_field=kwargs.get('sort_field'),
            sort_order=SortOrder(kwargs.get('sort_order', 'DESC')),
            limit=kwargs.get('limit'),
            offset=kwargs.get('offset', 0),
            include_deleted=kwargs.get('include_deleted', False),
            date_range=kwargs.get('date_range')
        )
    
    def search_by_sender(self, email: str, limit: int = 100) -> SearchResult:
        """Search messages by sender email."""
        query = SearchQuery(
            filters=[SearchFilter('sender', 'like', f'%{email}%')],
            limit=limit
        )
        return self.search(query)
    
    def search_by_date_range(self, start_date: datetime, end_date: datetime) -> SearchResult:
        """Search messages within date range."""
        query = SearchQuery(date_range=(start_date, end_date))
        return self.search(query)
    
    def search_unread_messages(self, limit: int = 100) -> SearchResult:
        """Search unread messages."""
        query = SearchQuery(
            filters=[SearchFilter('is_read', 'eq', False)],
            limit=limit
        )
        return self.search(query)
    
    def search_large_messages(self, min_size_mb: float = 1.0) -> SearchResult:
        """Search messages larger than specified size."""
        min_size_bytes = int(min_size_mb * 1024 * 1024)
        query = SearchQuery(
            filters=[SearchFilter('size', 'ge', min_size_bytes)],
            sort_field='size'
        )
        return self.search(query)
    
    def get_search_suggestions(self, partial_text: str, max_suggestions: int = 10) -> List[str]:
        """Get search suggestions based on partial text input."""
        suggestions = set()
        
        # Add common search terms based on database content
        try:
            with sqlite3.connect(self.sqlite_engine.db_path) as conn:
                cursor = conn.cursor()
                
                # Subject line suggestions
                cursor.execute("""
                    SELECT DISTINCT subject FROM messages 
                    WHERE subject LIKE ? AND subject IS NOT NULL 
                    LIMIT ?
                """, (f"%{partial_text}%", max_suggestions))
                
                for row in cursor.fetchall():
                    if row[0]:
                        suggestions.add(row[0])
                
                # Sender suggestions
                cursor.execute("""
                    SELECT DISTINCT json_extract(sender, '$.email') as email FROM messages 
                    WHERE email LIKE ? AND email IS NOT NULL 
                    LIMIT ?
                """, (f"%{partial_text}%", max_suggestions // 2))
                
                for row in cursor.fetchall():
                    if row[0]:
                        suggestions.add(row[0])
        
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
        
        return sorted(list(suggestions))[:max_suggestions]