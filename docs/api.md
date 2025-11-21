# API Reference

Complete API documentation for Semantica.

## Core Classes

### Semantica

Main framework class for building semantic layers and knowledge graphs.

```python
from semantica import Semantica

semantica = Semantica(config=None)
```

#### Methods

##### `build_knowledge_base(sources, **kwargs)`

Build a knowledge base from one or more data sources.

**Parameters:**
- `sources` (List[str] | str): Data source(s) - file paths or URLs
- `embeddings` (bool): Generate embeddings (default: True)
- `graph` (bool): Build knowledge graph (default: True)
- `normalize` (bool): Normalize data (default: True)

**Returns:**
- `Dict[str, Any]`: Dictionary containing:
  - `knowledge_graph`: Knowledge graph data
  - `embeddings`: Embedding vectors
  - `metadata`: Processing metadata
  - `statistics`: Processing statistics

**Example:**
```python
result = semantica.build_knowledge_base(
    sources=["document.pdf"],
    embeddings=True,
    graph=True
)
kg = result["knowledge_graph"]
```

##### `process_document(source)`

Process a single document.

**Parameters:**
- `source` (str): File path or URL

**Returns:**
- `Dict[str, Any]`: Processed document data

##### `extract_entities(text)`

Extract entities from text.

**Parameters:**
- `text` (str): Input text

**Returns:**
- `Dict[str, List]`: Dictionary with `entities` list

##### `extract_relationships(text)`

Extract relationships from text.

**Parameters:**
- `text` (str): Input text

**Returns:**
- `Dict[str, List]`: Dictionary with `relationships` list

---

## Knowledge Graph Module

### `semantica.kg`

Knowledge graph construction and analysis.

#### Methods

##### `build_graph(sources)`

Build a knowledge graph from sources.

```python
kg = semantica.kg.build_graph(["document.pdf"])
```

##### `analyze(graph)`

Analyze a knowledge graph.

```python
analysis = semantica.kg.analyze(kg)
print(analysis["statistics"])
```

##### `visualize(graph, output_path=None)`

Visualize a knowledge graph.

```python
semantica.kg.visualize(kg, output_path="graph.html")
```

##### `merge(graphs)`

Merge multiple knowledge graphs.

```python
merged = semantica.kg.merge([kg1, kg2, kg3])
```

---

## Semantic Extraction Module

### `semantica.semantic_extract`

Entity and relationship extraction.

#### Methods

##### `extract_entities(text)`

Extract named entities from text.

```python
result = semantica.semantic_extract.extract_entities(text)
entities = result["entities"]
```

##### `extract_relationships(text)`

Extract relationships from text.

```python
result = semantica.semantic_extract.extract_relationships(text)
relationships = result["relationships"]
```

##### `extract_triples(text)`

Extract subject-predicate-object triples.

```python
result = semantica.semantic_extract.extract_triples(text)
triples = result["triples"]
```

---

## Embeddings Module

### `semantica.embeddings`

Embedding generation and management.

#### Methods

##### `generate(text)`

Generate embedding for a single text.

```python
embedding = semantica.embeddings.generate("Your text here")
```

##### `generate_batch(texts)`

Generate embeddings for multiple texts.

```python
texts = ["text1", "text2", "text3"]
embeddings = semantica.embeddings.generate_batch(texts)
```

---

## Export Module

### `semantica.export`

Export knowledge graphs to various formats.

#### Methods

##### `to_rdf(kg, path)`

Export to RDF format.

```python
semantica.export.to_rdf(kg, "output.rdf")
```

##### `to_json(kg, path)`

Export to JSON format.

```python
semantica.export.to_json(kg, "output.json")
```

##### `to_csv(kg, path)`

Export to CSV format.

```python
semantica.export.to_csv(kg, "output.csv")
```

##### `to_owl(kg, path)`

Export to OWL ontology format.

```python
semantica.export.to_owl(kg, "output.owl")
```

##### `to_yaml(kg, path)`

Export to YAML format.

```python
semantica.export.to_yaml(kg, "output.yaml")
```

---

## Conflict Resolution Module

### `semantica.conflicts`

Conflict detection and resolution.

#### Classes

##### `ConflictResolver`

```python
from semantica.conflicts import ConflictResolver

resolver = ConflictResolver(default_strategy="voting")
```

**Methods:**
- `resolve_conflicts(conflicts)`: Resolve multiple conflicts
- `resolve_conflict(conflict, strategy=None)`: Resolve a single conflict
- `set_resolution_rule(property, strategy)`: Set custom resolution rules

**Example:**
```python
resolver = ConflictResolver(default_strategy="voting")
resolved = resolver.resolve_conflicts(conflicts)
```

---

## Configuration

### `Config`

Configuration class for Semantica.

```python
from semantica import Config

config = Config(
    embeddings=True,
    graph=True,
    normalize=True,
    conflict_resolution="voting"
)

semantica = Semantica(config=config)
```

**Parameters:**
- `embeddings` (bool): Enable embedding generation
- `graph` (bool): Enable knowledge graph construction
- `normalize` (bool): Enable data normalization
- `conflict_resolution` (str): Default conflict resolution strategy

---

## Full Documentation

For complete module documentation, see:
- [MODULES_DOCUMENTATION.md](../MODULES_DOCUMENTATION.md) - Detailed module documentation
- [GitHub Repository](https://github.com/Hawksight-AI/semantica) - Source code
