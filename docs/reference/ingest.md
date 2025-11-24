# Ingest Module

The `ingest` module provides comprehensive data ingestion capabilities for loading data from various sources including files, web pages, feeds, databases, and real-time streams.

## Overview

The ingest module supports **50+ file formats** and multiple data sources:

- **File Ingestion**: PDF, DOCX, XLSX, TXT, MD, JSON, CSV, and more
- **Web Scraping**: HTML pages, sitemaps, with JavaScript rendering
- **Feed Processing**: RSS, Atom feeds with automatic updates
- **Database Connectivity**: SQL and NoSQL databases
- **Stream Processing**: Real-time data streams
- **Email Processing**: EML, MSG, MBOX, PST archives
- **Repository Analysis**: Git repositories and code analysis
- **MCP Server Ingestion**: Connect to your own Python/FastMCP MCP servers via URL for resource and tool-based data ingestion

---

## Algorithms Used

### File Discovery
- **Recursive Traversal**: Depth-first search for file discovery
- **Pattern Matching**: Glob patterns with regex support
- **Filtering**: Extension-based and size-based filtering

### Web Scraping
- **HTML Parsing**: BeautifulSoup/lxml DOM parsing
- **JavaScript Rendering**: Headless browser (Selenium/Playwright)
- **Rate Limiting**: Token bucket algorithm
- **Robots.txt**: Compliance checking

### Stream Processing
- **Batch Buffering**: Sliding window with configurable size
- **Backpressure Handling**: Flow control mechanisms

---

## Quick Start

```python
from semantica.ingest import FileIngestor, WebIngestor, FeedIngestor

# Ingest local files
file_ingestor = FileIngestor(recursive=True)
documents = file_ingestor.ingest("documents/", formats=["pdf", "docx"])

# Ingest web content
web_ingestor = WebIngestor(max_depth=2)
web_docs = web_ingestor.ingest("https://example.com/articles")

# Ingest RSS feeds
feed_ingestor = FeedIngestor(max_items=100)
feed_docs = feed_ingestor.ingest("https://example.com/rss")

print(f"Total documents: {len(documents) + len(web_docs) + len(feed_docs)}")
```

---

## Main Classes

### FileIngestor


**Supported Formats:**

| Category | Formats |
|----------|---------|
| **Documents** | PDF, DOCX, XLSX, PPTX, TXT, RTF, ODT, EPUB, LaTeX, Markdown |
| **Structured** | JSON, YAML, TOML, CSV, TSV, Parquet, Avro, ORC |
| **Web** | HTML, XHTML, XML, JSON-LD, RDFa |
| **Archives** | ZIP, TAR, RAR, 7Z, GZ, BZ2 |
| **Scientific** | BibTeX, EndNote, RIS, JATS XML |
| **Email** | EML, MSG, MBOX, PST |

**Example Usage:**

```python
from semantica.ingest import FileIngestor

# Basic usage
ingestor = FileIngestor()
docs = ingestor.ingest("documents/")

# Advanced configuration
ingestor = FileIngestor(
    recursive=True,
    max_file_size=100 * 1024 * 1024,  # 100MB
    supported_formats=["pdf", "docx", "xlsx"],
    extract_archives=True,
    ocr_enabled=True,
    ocr_language="eng"
)

# Ingest with filters
docs = ingestor.ingest(
    "documents/",
    formats=["pdf", "docx"],
    exclude_patterns=["*draft*", "*temp*"],
    metadata={"source": "company_docs", "version": "1.0"}
)

# Process results
for doc in docs:
    print(f"File: {doc.filename}")
    print(f"Format: {doc.format}")
    print(f"Size: {doc.size} bytes")
    print(f"Pages: {doc.metadata.get('pages', 'N/A')}")
```

---

### WebIngestor


**Example Usage:**

```python
from semantica.ingest import WebIngestor

# Basic web scraping
ingestor = WebIngestor()
docs = ingestor.ingest("https://example.com")

# Advanced configuration
ingestor = WebIngestor(
    max_depth=3,
    respect_robots_txt=True,
    delay_between_requests=1.0,
    user_agent="Semantica/1.0",
    render_javascript=True,
    timeout=30
)

# Scrape with patterns
docs = ingestor.ingest(
    "https://blog.example.com",
    patterns=["*.html", "*/articles/*"],
    exclude_patterns=["*/admin/*", "*/login/*"],
    follow_links=True,
    max_pages=100
)

# Extract metadata
for doc in docs:
    print(f"URL: {doc.url}")
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Author: {doc.metadata.get('author')}")
    print(f"Published: {doc.metadata.get('published_date')}")
```

---

### FeedIngestor


**Example Usage:**

```python
from semantica.ingest import FeedIngestor

# Basic feed ingestion
ingestor = FeedIngestor()
docs = ingestor.ingest("https://example.com/rss")

# Advanced configuration
ingestor = FeedIngestor(
    max_items=1000,
    update_interval=3600,  # 1 hour
    include_content=True,
    fetch_full_content=True
)

# Ingest multiple feeds
feeds = [
    "https://news.ycombinator.com/rss",
    "https://example.com/atom",
    "https://blog.example.com/feed"
]

all_docs = []
for feed_url in feeds:
    docs = ingestor.ingest(feed_url)
    all_docs.extend(docs)

print(f"Total feed items: {len(all_docs)}")
```

---

### DBIngestor


**Example Usage:**

```python
from semantica.ingest import DBIngestor

# SQL database ingestion
ingestor = DBIngestor(
    connection_string="postgresql://user:pass@localhost/db"
)

docs = ingestor.ingest(
    query="SELECT title, content, author, created_at FROM articles WHERE published = true",
    metadata={"source": "articles_db", "version": "1.0"}
)

# NoSQL database ingestion
mongo_ingestor = DBIngestor(
    connection_string="mongodb://localhost:27017/mydb"
)

docs = mongo_ingestor.ingest(
    collection="articles",
    query={"status": "published"},
    projection={"title": 1, "content": 1, "author": 1}
)
```

---

### StreamIngestor


**Example Usage:**

```python
from semantica.ingest import StreamIngestor

# Kafka stream ingestion
ingestor = StreamIngestor(
    stream_type="kafka",
    bootstrap_servers=["localhost:9092"],
    topic="documents",
    group_id="semantica-consumer"
)

# Process stream
for doc in ingestor.stream():
    print(f"Received: {doc.id}")
    # Process document
    
# RabbitMQ stream ingestion
rabbitmq_ingestor = StreamIngestor(
    stream_type="rabbitmq",
    host="localhost",
    queue="documents"
)
```

---

### EmailIngestor


**Example Usage:**

```python
from semantica.ingest import EmailIngestor

# Ingest email files
ingestor = EmailIngestor()
docs = ingestor.ingest("emails/", formats=["eml", "msg"])

# Extract attachments
ingestor = EmailIngestor(
    extract_attachments=True,
    attachment_dir="attachments/"
)

docs = ingestor.ingest("archive.mbox")

# Process emails
for doc in docs:
    print(f"From: {doc.metadata['from']}")
    print(f"Subject: {doc.metadata['subject']}")
    print(f"Date: {doc.metadata['date']}")
    print(f"Attachments: {len(doc.metadata.get('attachments', []))}")
```

---

### RepoIngestor


**Example Usage:**

```python
from semantica.ingest import RepoIngestor

# Ingest Git repository
ingestor = RepoIngestor()
docs = ingestor.ingest(
    "https://github.com/user/repo.git",
    branch="main",
    include_history=True
)

# Analyze code
ingestor = RepoIngestor(
    analyze_code=True,
    extract_functions=True,
    extract_classes=True,
    languages=["python", "javascript"]
)

docs = ingestor.ingest("path/to/local/repo")
```

---

### MCPIngestor

**Bring Your Own MCP Server**: The MCP (Model Context Protocol) ingestion allows you to connect to your own Python-based or FastMCP MCP servers via URL. You can bring any MCP server you've built or have access to, and Semantica will dynamically discover and ingest data from its resources and tools.

**IMPORTANT**: This implementation supports **ONLY Python-based MCP servers and FastMCP servers**. Users can bring their own Python or FastMCP MCP servers via URL connections. JavaScript, TypeScript, C#, Java, and other language implementations are **NOT supported**.

**Key Features:**
- **URL-based connection**: Connect to any Python/FastMCP MCP server via HTTP/HTTPS URL
- **Generic implementation**: Works with any Python/FastMCP MCP server across diverse domains
- **Dynamic discovery**: Automatically discovers available resources and tools
- **Multiple servers**: Connect to multiple MCP servers simultaneously
- **Resource ingestion**: Ingest data from MCP server resources
- **Tool execution**: Execute MCP server tools and ingest their outputs
- **Authentication support**: API keys, OAuth, and custom headers

**Example Usage:**

```python
from semantica.ingest import MCPIngestor

# Create ingestor
ingestor = MCPIngestor()

# Connect to your MCP server via URL (primary method)
ingestor.connect(
    "my_server",  # Server identifier
    url="http://localhost:8000/mcp"  # Your MCP server URL
)

# List available resources from your server
resources = ingestor.list_available_resources("my_server")
print(f"Available resources: {[r.uri for r in resources]}")

# List available tools from your server
tools = ingestor.list_available_tools("my_server")
print(f"Available tools: {[t.name for t in tools]}")
```

**Resource-Based Ingestion:**

```python
from semantica.ingest import MCPIngestor

ingestor = MCPIngestor()
ingestor.connect("file_server", url="http://localhost:8000/mcp")

# Ingest specific resources
data = ingestor.ingest_resources(
    "file_server",
    resource_uris=["resource://file/documents", "resource://file/reports"]
)

for item in data:
    print(f"Resource: {item.resource_uri}")
    print(f"Content type: {item.data_type}")
    print(f"Metadata: {item.metadata}")

# Ingest all available resources
all_data = ingestor.ingest_all_resources("file_server")
print(f"Ingested {len(all_data)} resources")
```

**Tool-Based Ingestion:**

```python
from semantica.ingest import MCPIngestor

ingestor = MCPIngestor()
ingestor.connect("db_server", url="http://localhost:8000/mcp")

# Execute a tool and ingest its output
result = ingestor.ingest_tool_output(
    "db_server",
    tool_name="query_database",
    arguments={"query": "SELECT * FROM users LIMIT 10"}
)

print(f"Tool result: {result.content}")
print(f"Metadata: {result.metadata}")
```

**Multiple MCP Servers:**

```python
from semantica.ingest import MCPIngestor

ingestor = MCPIngestor()

# Connect to multiple MCP servers simultaneously
ingestor.connect("db_server", url="http://localhost:8001/mcp")
ingestor.connect("file_server", url="http://localhost:8002/mcp")
ingestor.connect(
    "api_server",
    url="https://api.example.com/mcp",
    headers={"Authorization": "Bearer your_token"}
)

# Ingest from different servers
db_data = ingestor.ingest_resources("db_server", resource_uris=["resource://database/tables"])
file_data = ingestor.ingest_all_resources("file_server")
api_result = ingestor.ingest_tool_output("api_server", tool_name="fetch_data", arguments={})

# List all connected servers
servers = ingestor.get_connected_servers()
print(f"Connected servers: {servers}")
```

**Authentication and HTTPS:**

```python
from semantica.ingest import MCPIngestor

ingestor = MCPIngestor()

# Connect via HTTPS with authentication (transport auto-detected from URL)
ingestor.connect(
    "secure_server",
    url="https://api.example.com/mcp",
    headers={
        "Authorization": "Bearer your_api_token",
        "X-API-Key": "your_api_key"
    }
)

# The client automatically handles HTTPS and authentication headers
resources = ingestor.list_available_resources("secure_server")
```

**Use Cases:**

- **Database Integration**: Connect to a database MCP server to ingest table schemas, query results, or metadata
- **File System Access**: Connect to a file system MCP server to ingest documents, logs, or configuration files
- **API Integration**: Connect to an API MCP server to ingest data from external services
- **Custom Data Sources**: Bring your own MCP server to expose any data source through the MCP protocol

**Best Practices:**

1. **Server Naming**: Use descriptive names for your MCP servers to easily identify them
2. **Connection Management**: Reuse the same MCPIngestor instance for multiple servers
3. **Error Handling**: Always handle connection errors and check server availability
4. **Resource Discovery**: List available resources and tools before ingestion to understand server capabilities
5. **Authentication**: Store credentials securely and use environment variables for sensitive data

---

## Common Patterns

### Pattern 1: Multi-Source Ingestion

```python
from semantica.ingest import FileIngestor, WebIngestor, FeedIngestor

sources = []

# Ingest files
file_ingestor = FileIngestor(recursive=True)
sources.extend(file_ingestor.ingest("documents/"))

# Ingest web
web_ingestor = WebIngestor()
sources.extend(web_ingestor.ingest("https://example.com"))

# Ingest feeds
feed_ingestor = FeedIngestor()
sources.extend(feed_ingestor.ingest("https://example.com/rss"))

print(f"Total sources: {len(sources)}")
```

### Pattern 2: Batch Processing with Progress

```python
from semantica.ingest import FileIngestor
from tqdm import tqdm

ingestor = FileIngestor()
files = ingestor.list_files("documents/", recursive=True)

docs = []
for file_path in tqdm(files, desc="Ingesting"):
    doc = ingestor.ingest_file(file_path)
    docs.append(doc)
```

### Pattern 3: Error Handling

```python
from semantica.ingest import FileIngestor, IngestionError

ingestor = FileIngestor()

successful = []
failed = []

for file_path in file_paths:
    try:
        doc = ingestor.ingest_file(file_path)
        successful.append(doc)
    except IngestionError as e:
        print(f"Failed to ingest {file_path}: {e}")
        failed.append((file_path, str(e)))

print(f"Successful: {len(successful)}, Failed: {len(failed)}")
```

---

## Configuration

```yaml
# config.yaml - Ingest Configuration

ingest:
  file:
    recursive: true
    max_file_size: 104857600  # 100MB
    supported_formats: [pdf, docx, xlsx, txt, md, json, csv]
    extract_archives: true
    ocr_enabled: true
    ocr_language: eng
    
  web:
    max_depth: 3
    respect_robots_txt: true
    delay_between_requests: 1.0
    render_javascript: true
    timeout: 30
    max_pages: 1000
    
  feed:
    max_items: 1000
    update_interval: 3600
    fetch_full_content: true
    
  stream:
    batch_size: 100
    max_wait_time: 5
```

---

## See Also

- [Parse Module](parse.md) - Document parsing and content extraction
- [Normalize Module](normalize.md) - Data cleaning and normalization
- [Core Module](core.md) - Framework orchestration
