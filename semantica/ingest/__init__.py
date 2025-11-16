"""
Data Ingestion Module

This module provides comprehensive data ingestion capabilities from various sources
including files, web content, feeds, streams, repositories, emails, and databases.

Algorithms Used:

File Ingestion:
    - File Type Detection: Multi-method detection (extension-based, MIME type, magic number analysis)
    - Directory Scanning: Recursive directory traversal with filtering
    - Cloud Storage Integration: AWS S3, Google Cloud Storage, Azure Blob Storage API integration
    - File Validation: Size limits, format validation, content verification
    - Batch Processing: Parallel file processing with progress tracking
    - Magic Number Analysis: Binary file signature detection for accurate type identification

Web Ingestion:
    - HTTP Request Handling: GET/POST requests with retry logic and error handling
    - Rate Limiting: Time-based delay enforcement between requests
    - Robots.txt Compliance: RobotFileParser-based robots.txt checking and URL filtering
    - Content Extraction: BeautifulSoup-based HTML parsing and text extraction
    - Sitemap Crawling: XML sitemap parsing and URL discovery
    - Link Discovery: HTML link extraction and domain filtering
    - JavaScript Rendering: Optional Selenium/Playwright integration for dynamic content
    - URL Normalization: URL parsing, joining, and canonicalization

Feed Ingestion:
    - RSS/Atom Parsing: XML parsing with format auto-detection
    - Feed Discovery: HTML link tag parsing for feed discovery
    - Date Parsing: Multiple date format parsing (RFC 822, ISO 8601, etc.)
    - Feed Validation: XML schema validation and format verification
    - Update Monitoring: Polling-based feed update detection with callbacks
    - Content Extraction: Feed item content and metadata extraction

Stream Ingestion:
    - Kafka Processing: Kafka consumer group management and message processing
    - RabbitMQ Processing: AMQP protocol handling and queue management
    - AWS Kinesis Processing: Kinesis stream reading and shard management
    - Apache Pulsar Processing: Pulsar consumer and subscription management
    - Message Transformation: Custom transformation pipeline for stream messages
    - Error Handling: Retry logic, dead letter queue handling, error recovery
    - Stream Monitoring: Health checks, metrics collection, performance monitoring
    - Partition Management: Partition assignment and load balancing

Repository Ingestion:
    - Git Operations: Repository cloning, branch checking, commit traversal
    - Code Extraction: File content extraction with language detection
    - Commit Analysis: Git log parsing, diff analysis, statistics calculation
    - Language Detection: File extension and content-based programming language identification
    - Code Structure Analysis: AST parsing for classes, functions, imports extraction
    - Dependency Analysis: Package manager file parsing (requirements.txt, package.json, etc.)
    - Documentation Extraction: README, docstring, and comment extraction

Email Ingestion:
    - IMAP Protocol: IMAP connection, mailbox selection, message retrieval
    - POP3 Protocol: POP3 connection and message downloading
    - Email Parsing: RFC 822 email message parsing with header and body extraction
    - Attachment Processing: MIME attachment extraction and file saving
    - Content Extraction: Plain text and HTML body extraction with BeautifulSoup
    - Thread Analysis: Message-ID and In-Reply-To header-based conversation threading
    - Link Extraction: URL extraction from email HTML content

Database Ingestion:
    - Database Connection: SQLAlchemy-based connection management with connection pooling
    - SQL Query Execution: Parameterized query execution with result set processing
    - Schema Introspection: Database schema analysis and table/column discovery
    - Data Type Conversion: Database-specific type to Python type conversion
    - Pagination: Large dataset processing with LIMIT/OFFSET or cursor-based pagination
    - Data Export: Result set to dictionary/JSON conversion
    - Multi-database Support: PostgreSQL, MySQL, SQLite, Oracle, SQL Server abstraction

Key Features:
    - Multiple ingestion source types (file, web, feed, stream, repo, email, db)
    - Unified ingestion function with source type dispatch
    - Method registry for extensibility
    - Configuration management with environment variables and config files
    - Batch processing and progress tracking
    - Error handling and retry logic
    - Content extraction and transformation

Main Classes:
    - FileIngestor: Local and cloud file processing
    - WebIngestor: Web scraping and crawling
    - FeedIngestor: RSS/Atom feed processing
    - StreamIngestor: Real-time stream processing
    - RepoIngestor: Git repository processing
    - EmailIngestor: Email protocol handling
    - DBIngestor: Database export handling
    - MethodRegistry: Registry for custom ingestion methods
    - IngestConfig: Configuration manager for ingest module

Convenience Functions:
    - ingest: Unified ingestion function with source type dispatch
    - ingest_file: File ingestion wrapper
    - ingest_web: Web ingestion wrapper
    - ingest_feed: Feed ingestion wrapper
    - ingest_stream: Stream ingestion wrapper
    - ingest_repository: Repository ingestion wrapper
    - ingest_email: Email ingestion wrapper
    - ingest_database: Database ingestion wrapper
    - build: Module-level build function for data ingestion

Example Usage:
    >>> from semantica.ingest import ingest, ingest_file, ingest_web
    >>> # Unified ingestion
    >>> result = ingest("document.pdf", source_type="file")
    >>> # File ingestion
    >>> files = ingest_file("./documents", method="directory")
    >>> # Web ingestion
    >>> content = ingest_web("https://example.com", method="url")
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .file_ingestor import FileIngestor, FileObject, FileTypeDetector, CloudStorageIngestor
from .web_ingestor import WebIngestor, WebContent, RateLimiter, RobotsChecker, ContentExtractor, SitemapCrawler
from .feed_ingestor import FeedIngestor, FeedItem, FeedData, FeedParser, FeedMonitor
from .stream_ingestor import (
    StreamIngestor,
    StreamMessage,
    StreamProcessor,
    KafkaProcessor,
    RabbitMQProcessor,
    KinesisProcessor,
    PulsarProcessor,
    StreamMonitor,
)
from .repo_ingestor import RepoIngestor, CodeFile, CommitInfo, CodeExtractor, GitAnalyzer
from .email_ingestor import EmailIngestor, EmailData, AttachmentProcessor, EmailParser as EmailIngestorParser
from .db_ingestor import DBIngestor, TableData, DatabaseConnector, DataExporter
from .registry import MethodRegistry, method_registry
from .methods import (
    ingest,
    ingest_file,
    ingest_web,
    ingest_feed,
    ingest_stream,
    ingest_repository,
    ingest_email,
    ingest_database,
    get_ingest_method,
    list_available_methods,
)
from .config import IngestConfig, ingest_config

__all__ = [
    # File ingestion
    "FileIngestor",
    "FileObject",
    "FileTypeDetector",
    "CloudStorageIngestor",
    # Web ingestion
    "WebIngestor",
    "WebContent",
    "RateLimiter",
    "RobotsChecker",
    "ContentExtractor",
    "SitemapCrawler",
    # Feed ingestion
    "FeedIngestor",
    "FeedItem",
    "FeedData",
    "FeedParser",
    "FeedMonitor",
    # Stream ingestion
    "StreamIngestor",
    "StreamMessage",
    "StreamProcessor",
    "KafkaProcessor",
    "RabbitMQProcessor",
    "KinesisProcessor",
    "PulsarProcessor",
    "StreamMonitor",
    # Repository ingestion
    "RepoIngestor",
    "CodeFile",
    "CommitInfo",
    "CodeExtractor",
    "GitAnalyzer",
    # Email ingestion
    "EmailIngestor",
    "EmailData",
    "AttachmentProcessor",
    "EmailIngestorParser",
    # Database ingestion
    "DBIngestor",
    "TableData",
    "DatabaseConnector",
    "DataExporter",
    # Registry and Methods
    "MethodRegistry",
    "method_registry",
    "ingest",
    "ingest_file",
    "ingest_web",
    "ingest_feed",
    "ingest_stream",
    "ingest_repository",
    "ingest_email",
    "ingest_database",
    "get_ingest_method",
    "list_available_methods",
    # Configuration
    "IngestConfig",
    "ingest_config",
    # Legacy
    "build",
]


def build(
    sources: Union[List[Union[str, Path]], str, Path],
    source_type: str = "file",
    recursive: bool = True,
    read_content: bool = True,
    **options
) -> Dict[str, Any]:
    """
    Ingest data from sources (module-level convenience function).
    
    This is a user-friendly wrapper that automatically selects the appropriate
    ingestor based on source type and ingests the data.
    
    Args:
        sources: Data source(s) - can be file paths, URLs, directories, etc.
        source_type: Type of source - "file", "web", "feed", "stream", "repo", "email", "db" (default: "file")
        recursive: For directories, whether to ingest recursively (default: True)
        read_content: Whether to read file content (default: True)
        **options: Additional ingestion options
        
    Returns:
        Dictionary containing:
            - files: List of ingested file objects (for file ingestion)
            - content: Ingested content (for web/feed ingestion)
            - metadata: Ingestion metadata
            - statistics: Ingestion statistics
            
    Examples:
        >>> import semantica
        >>> result = semantica.ingest.build(
        ...     sources=["doc1.pdf", "doc2.docx"],
        ...     source_type="file",
        ...     read_content=True
        ... )
        >>> print(f"Ingested {len(result['files'])} files")
    """
    # Normalize sources to list
    if isinstance(sources, (str, Path)):
        sources = [sources]
    
    results = {
        "files": [],
        "content": [],
        "metadata": {},
        "statistics": {}
    }
    
    if source_type == "file":
        # Use FileIngestor
        ingestor = FileIngestor(config=options.get("config", {}), **options)
        
        file_objects = []
        for source in sources:
            source_path = Path(source)
            if source_path.is_dir():
                # Ingest directory
                files = ingestor.ingest_directory(source_path, recursive=recursive, **options)
                file_objects.extend(files)
            elif source_path.is_file():
                # Ingest single file
                file_obj = ingestor.ingest_file(source_path, read_content=read_content, **options)
                file_objects.append(file_obj)
            else:
                # Try as file path string
                try:
                    file_obj = ingestor.ingest_file(source, read_content=read_content, **options)
                    file_objects.append(file_obj)
                except Exception as e:
                    results["statistics"].setdefault("errors", []).append({
                        "source": str(source),
                        "error": str(e)
                    })
        
        results["files"] = file_objects
        results["statistics"] = {
            "total_sources": len(sources),
            "ingested_files": len(file_objects),
            "errors": len(results["statistics"].get("errors", []))
        }
        
    elif source_type == "web":
        # Use WebIngestor
        from .web_ingestor import WebIngestor
        ingestor = WebIngestor(config=options.get("config", {}), **options)
        
        web_contents = []
        for source in sources:
            if isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
                content = ingestor.ingest_url(source, **options)
                web_contents.append(content)
        
        results["content"] = web_contents
        results["statistics"] = {
            "total_urls": len(sources),
            "ingested_pages": len(web_contents)
        }
        
    else:
        # For other types, return placeholder
        results["statistics"] = {
            "message": f"Source type '{source_type}' ingestion not yet implemented in build() function",
            "suggestion": f"Use {source_type.capitalize()}Ingestor class directly"
        }
    
    return results
