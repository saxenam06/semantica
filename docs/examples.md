# Examples

Real-world examples and use cases for Semantica.

!!! tip "Interactive Learning"
    For hands-on interactive tutorials, check out our [Cookbook](cookbook.md) with Jupyter notebooks covering everything from basics to advanced use cases.

---

## Table of Contents

- [Getting Started (5 min examples)](#getting-started-5-min-examples)
- [Core Workflows (15 min examples)](#core-workflows-15-min-examples)
- [Advanced Patterns (30+ min examples)](#advanced-patterns-30-min-examples)
- [Production Patterns](#production-patterns)

---

!!! note "Prerequisites"
    All examples assume you have Semantica installed. See the [Installation Guide](installation.md) if you need to set it up first.

---

## Getting Started (5 min examples)

Quick examples to get you up and running with Semantica.

### Example 1: Basic Knowledge Graph

**Difficulty**: Beginner  
**Time**: 5 minutes  
**Prerequisites**: Semantica installed, sample PDF

Build a knowledge graph from a single document.

**Code**:

```python
from semantica import Semantica

semantica = Semantica()

# Build KG from PDF
result = semantica.build_knowledge_base(
    sources=["research_paper.pdf"],
    embeddings=True,
    graph=True
)

kg = result["knowledge_graph"]
print(f"Entities: {len(kg['entities'])}")
print(f"Relationships: {len(kg['relationships'])}")
```

**Expected Output**:

```
Entities: 45
Relationships: 32
```

**Common Errors**:

- `FileNotFoundError`: Ensure the PDF file exists in the current directory
- `ImportError`: Make sure Semantica is installed: `pip install semantica`

**Next Steps**:

- [Example 2: Entity Extraction](#example-2-entity-extraction)
- [Core Workflows](#core-workflows-15-min-examples)

---

### Example 2: Entity Extraction

**Difficulty**: Beginner  
**Time**: 5 minutes  
**Prerequisites**: Semantica installed

Extract entities from text using Named Entity Recognition.

**Code**:

```python
from semantica import Semantica

semantica = Semantica()

text = """
Apple Inc. is a technology company founded by Steve Jobs.
The company is headquartered in Cupertino, California.
Tim Cook is the current CEO of Apple.
"""

entities = semantica.semantic_extract.extract_entities(text)
for entity in entities["entities"]:
    print(f"{entity['text']}: {entity['type']}")
```

**Expected Output**:

```
Apple Inc.: ORGANIZATION
Steve Jobs: PERSON
Cupertino: LOCATION
California: LOCATION
Tim Cook: PERSON
```

**Common Errors**:

- `AttributeError`: Ensure you're using the correct API: `semantica.semantic_extract.extract_entities()`
- Empty results: Check that your text contains named entities

**Next Steps**:

- [Example 3: Multi-Source Integration](#example-3-multi-source-integration)
- [Core Workflows](#core-workflows-15-min-examples)

---

### Example 3: Multi-Source Integration

**Difficulty**: Beginner  
**Time**: 5 minutes  
**Prerequisites**: Multiple data sources

Combine data from multiple sources into a unified knowledge graph.

**Code**:

```python
from semantica import Semantica

semantica = Semantica()

sources = [
    "documents/finance_report.pdf",
    "documents/market_analysis.docx",
    "https://example.com/news-article"
]

result = semantica.build_knowledge_base(sources)
kg = result["knowledge_graph"]

print(f"Unified knowledge graph with {len(kg['entities'])} entities")
```

**Expected Output**:

```
Unified knowledge graph with 120 entities
```

**Common Errors**:

- Network errors for URLs: Check internet connection and URL accessibility
- File format errors: Ensure all file formats are supported

**Next Steps**:

- [Core Workflows](#core-workflows-15-min-examples)

---

### Example 4: Export Formats

**Difficulty**: Beginner  
**Time**: 5 minutes  
**Prerequisites**: Knowledge graph built

Export knowledge graph to multiple formats.

**Code**:

```python
from semantica import Semantica

semantica = Semantica()
kg = semantica.kg.build_graph(["data.pdf"])

# Export to different formats
semantica.export.to_rdf(kg, "output.rdf")
semantica.export.to_json(kg, "output.json")
semantica.export.to_csv(kg, "output.csv")
semantica.export.to_owl(kg, "output.owl")
```

---

## Core Workflows (15 min examples)

Common workflows for building production-ready knowledge graphs.

### Example 5: Conflict Resolution

**Difficulty**: Intermediate  
**Time**: 15 minutes  
**Prerequisites**: Multiple data sources with potential conflicts

Resolve conflicts in data from multiple sources.

**Code**:

```python
from semantica import Semantica
from semantica.conflicts import ConflictResolver

semantica = Semantica()

# Build graph from multiple sources
result = semantica.build_knowledge_base([
    "source1.pdf",
    "source2.pdf",
    "source3.pdf"
])

# Detect conflicts
conflicts = semantica.kg.detect_conflicts(result["knowledge_graph"])

# Resolve conflicts
resolver = ConflictResolver(default_strategy="voting")
resolved = resolver.resolve_conflicts(conflicts)

print(f"Resolved {len(resolved)} conflicts")
```

### Example 6: Custom Configuration

**Difficulty**: Intermediate  
**Time**: 15 minutes  
**Prerequisites**: Understanding of configuration options

Use custom configuration for specific use cases.

**Code**:

```python
from semantica import Semantica, Config

# Custom configuration
config = Config(
    embeddings=True,
    graph=True,
    normalize=True,
    conflict_resolution="highest_confidence"
)

semantica = Semantica(config=config)
result = semantica.build_knowledge_base(["document.pdf"])
```

### Example 7: Incremental Graph Building

**Difficulty**: Intermediate  
**Time**: 15 minutes  
**Prerequisites**: Multiple data sources to process incrementally

Build knowledge graph incrementally.

**Code**:

```python
from semantica import Semantica

semantica = Semantica()

# Build graphs separately
kg1 = semantica.kg.build_graph(["source1.pdf"])
kg2 = semantica.kg.build_graph(["source2.pdf"])
kg3 = semantica.kg.build_graph(["source3.pdf"])

# Merge into unified graph
merged_kg = semantica.kg.merge([kg1, kg2, kg3])

print(f"Merged graph: {len(merged_kg['entities'])} entities")
```

### Example 8: Visualization

**Difficulty**: Beginner  
**Time**: 10 minutes  
**Prerequisites**: Knowledge graph built

Create interactive visualizations.

**Code**:

```python
from semantica import Semantica

semantica = Semantica()

# Build graph
result = semantica.build_knowledge_base(["document.pdf"])
kg = result["knowledge_graph"]

# Visualize
semantica.kg.visualize(kg, output_path="graph.html")

# Also analyze
analysis = semantica.kg.analyze(kg)
print(f"Graph density: {analysis['density']}")
print(f"Connected components: {analysis['components']}")
```

---

## Advanced Patterns (30+ min examples)

Advanced patterns for complex use cases and production deployments.

### Example 9: Graph Store (Persistent Storage)

**Difficulty**: Intermediate  
**Time**: 30 minutes  
**Prerequisites**: Neo4j or KuzuDB installed

Store and query knowledge graphs in a persistent graph database.

**Code**:

```python
from semantica.graph_store import GraphStore

# Initialize with Neo4j
store = GraphStore(
    backend="neo4j",
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
store.connect()

# Create nodes
apple = store.create_node(
    labels=["Company"],
    properties={"name": "Apple Inc.", "founded": 1976}
)

tim_cook = store.create_node(
    labels=["Person"],
    properties={"name": "Tim Cook", "title": "CEO"}
)

# Create relationship
store.create_relationship(
    start_node_id=tim_cook["id"],
    end_node_id=apple["id"],
    rel_type="CEO_OF",
    properties={"since": 2011}
)

# Query with Cypher
results = store.execute_query("""
    MATCH (p:Person)-[:CEO_OF]->(c:Company)
    RETURN p.name, c.name
""")

print(f"Query results: {results}")
store.close()
```

### Example 10: Using KuzuDB (Embedded)

**Difficulty**: Intermediate  
**Time**: 20 minutes  
**Prerequisites**: KuzuDB installed

For embedded graph storage without external dependencies.

**Code**:

```python
from semantica.graph_store import GraphStore

# KuzuDB - no server required
store = GraphStore(backend="kuzu", database_path="./my_graph_db")
store.connect()

# Store your knowledge graph
node = store.create_node(["Entity"], {"name": "Test"})
neighbors = store.get_neighbors(node["id"], depth=2)
stats = store.get_stats()

print(f"Graph stats: {stats}")
store.close()
```

### Example 11: FalkorDB for Real-Time Applications

**Difficulty**: Intermediate  
**Time**: 25 minutes  
**Prerequisites**: Redis and FalkorDB installed

Ultra-fast graph queries for LLM applications.

**Code**:

```python
from semantica.graph_store import GraphStore

# FalkorDB - Redis-based, ultra-fast
store = GraphStore(
    backend="falkordb",
    host="localhost",
    port=6379,
    graph_name="knowledge_graph"
)
store.connect()

# Fast queries for RAG applications
results = store.execute_query("""
    MATCH (n)-[r]->(m)
    WHERE n.name CONTAINS $query
    RETURN n, r, m LIMIT 10
""", parameters={"query": "AI"})

store.close()
```

---

## Production Patterns

Production-ready patterns for scalable deployments.

### Example 12: Streaming Data Processing

**Difficulty**: Advanced  
**Time**: 45 minutes  
**Prerequisites**: Understanding of streaming architectures

Process data streams in real-time.

**Code**:

```python
from semantica.ingest import StreamIngestor
from semantica import Semantica

semantica = Semantica()

# Set up stream ingestor
stream_ingestor = StreamIngestor(stream_uri="kafka://localhost:9092/topic")

# Process stream
for batch in stream_ingestor.stream(batch_size=100):
    result = semantica.build_knowledge_base(
        sources=batch,
        embeddings=True,
        graph=True
    )
    # Process and store results
    process_results(result)
```

**Expected Output**:
```
Processing batch 1: 100 documents
Processing batch 2: 100 documents
...
```

---

### Example 13: Custom Extractors

**Difficulty**: Advanced  
**Time**: 60 minutes  
**Prerequisites**: Understanding of extraction patterns

Create custom entity extractors for domain-specific entities.

**Code**:

```python
from semantica.semantic_extract import BaseExtractor

class CustomProductExtractor(BaseExtractor):
    def extract(self, text):
        # Your custom extraction logic
        products = []
        # Pattern matching or ML model
        return products

# Use custom extractor
extractor = CustomProductExtractor()
entities = extractor.extract(text)
```

---

### Example 14: Batch Processing Large Datasets

**Difficulty**: Intermediate  
**Time**: 30 minutes  
**Prerequisites**: Large dataset to process

Process large datasets efficiently with batching.

**Code**:

```python
from semantica import Semantica
import os

semantica = Semantica()
sources = [f"data/doc_{i}.pdf" for i in range(1000)]
batch_size = 50

for i in range(0, len(sources), batch_size):
    batch = sources[i:i+batch_size]
    result = semantica.build_knowledge_base(
        sources=batch,
        embeddings=True,
        graph=True
    )
    # Save intermediate results
    save_batch_result(result, batch_num=i//batch_size)
```

---

## Use Case Examples

### Research Paper Analysis

Extract knowledge from research papers:

```python
from semantica import Semantica

semantica = Semantica()

# Process research paper
result = semantica.build_knowledge_base([
    "papers/ai_research.pdf",
    "papers/ml_survey.pdf"
])

kg = result["knowledge_graph"]

# Find key concepts
concepts = [e for e in kg["entities"] if e["type"] == "CONCEPT"]
print(f"Found {len(concepts)} key concepts")
```

### Company Intelligence

Build knowledge graph from company documents:

```python
from semantica import Semantica

semantica = Semantica()

# Company documents
sources = [
    "company/annual_report.pdf",
    "company/press_releases/",
    "company/website_content.html"
]

result = semantica.build_knowledge_base(sources)
kg = result["knowledge_graph"]

# Export for analysis
semantica.export.to_json(kg, "company_intelligence.json")
```

### News Article Processing

Process and analyze news articles:

```python
from semantica import Semantica

semantica = Semantica()

# News articles
articles = [
    "https://example.com/article1",
    "https://example.com/article2",
    "https://example.com/article3"
]

result = semantica.build_knowledge_base(articles)
kg = result["knowledge_graph"]

# Extract key entities
people = [e for e in kg["entities"] if e["type"] == "PERSON"]
organizations = [e for e in kg["entities"] if e["type"] == "ORGANIZATION"]

print(f"People mentioned: {len(people)}")
print(f"Organizations: {len(organizations)}")
```

## Interactive Examples

For more interactive examples and tutorials, check out our [Cookbook](cookbook.md) with Jupyter notebooks covering:

- **Introduction**: Getting started tutorials
- **Advanced**: Advanced techniques and patterns
- **Use Cases**: Real-world applications in various domains

---

## Example Gallery

Visual previews of what you can build with Semantica:

- **Knowledge Graph Visualization**: Interactive network graphs
- 📈 **Analytics Dashboards**: Quality metrics and insights
- 🔍 **Semantic Search**: Vector-based document retrieval
- 🧠 **GraphRAG Applications**: Enhanced LLM responses

---

## More Resources

- **[Quick Start Guide](quickstart.md)** - Step-by-step tutorial
- **[API Reference](reference/core.md)** - Complete API documentation
- **[Cookbook](cookbook.md)** - Interactive Jupyter notebooks
- **[Use Cases](use-cases.md)** - Real-world applications
- **[Modules Guide](modules.md)** - Module documentation

---

!!! info "Contribute"
    Have an example to share? [Contribute on GitHub](https://github.com/Hawksight-AI/semantica)

**Last Updated**: 2024
