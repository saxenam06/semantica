# Knowledge Graph

> **High-level KG construction, management, and analysis system.**

---

## 🎯 Overview

The **Knowledge Graph (KG) Module** is the core module for building, managing, and analyzing knowledge graphs. It transforms extracted entities and relationships into structured, queryable knowledge graphs.

### What is a Knowledge Graph?

A **knowledge graph** is a structured representation of information where:
- **Nodes** represent entities (people, organizations, concepts, etc.)
- **Edges** represent relationships between entities
- **Properties** store additional information about nodes and edges

Knowledge graphs enable semantic queries, relationship traversal, and complex reasoning that traditional databases cannot handle.

### Why Use the KG Module?

- **Structured Knowledge**: Transform unstructured data into structured, queryable graphs
- **Entity Resolution**: Automatically merge duplicate entities using fuzzy matching
- **Temporal Support**: Track how knowledge changes over time
- **Graph Analytics**: Analyze graph structure, importance, and communities
- **Provenance Tracking**: Know where every piece of information came from

### How It Works

1. **Input**: Entities and relationships from semantic extraction
2. **Entity Resolution**: Merge similar entities to avoid duplicates
3. **Graph Construction**: Build nodes and edges from entities and relationships
4. **Enrichment**: Add temporal information, provenance, and metadata
5. **Analysis**: Perform graph analytics (centrality, communities, etc.)

<div class="grid cards" markdown>

-   :material-graph-outline:{ .lg .middle } **KG Construction**

    ---

    Build graphs from entities and relationships with automatic merging

-   :material-clock-time-four-outline:{ .lg .middle } **Temporal Graphs**

    ---

    Time-aware edges (`valid_from`, `valid_until`) and temporal queries

-   :material-account-multiple-check:{ .lg .middle } **Entity Resolution**

    ---

    Resolve entities using fuzzy matching and semantic similarity

-   :material-chart-network:{ .lg .middle } **Graph Analytics**

    ---

    Centrality, Community Detection, and Connectivity analysis

-   :material-history:{ .lg .middle } **Provenance**

    ---

    Track the source and lineage of every node and edge

</div>

!!! tip "When to Use"
    - **KG Building**: The primary module for assembling a KG from extracted data
    - **Entity Resolution**: Resolving and merging similar entities
    - **Analysis**: Understanding the structure and importance of nodes
    - **Time-Series**: Modeling how the graph evolves over time

!!! note "Related Modules"
    - **Conflict Detection**: Use `semantica.conflicts` module for conflict detection and resolution
    - **Deduplication**: Use `semantica.deduplication` module for advanced deduplication

---

## ⚙️ Algorithms Used

### Entity Resolution
- **Fuzzy Matching**: Levenshtein/Jaro-Winkler distance for string similarity.
- **Semantic Matching**: Cosine similarity of embeddings.
- **Transitive Merging**: If A=B and B=C, then A=B=C.

### Graph Analytics
- **Centrality**: Degree, Betweenness, Closeness, Eigenvector.
- **Communities**: Louvain, Leiden, K-Clique.
- **Connectivity**: Connected Components, Bridge Detection.

### Temporal Analysis
- **Time-Slicing**: Viewing the graph at a specific point in time.
- **Interval Algebra**: Allen's interval algebra for temporal reasoning (overlaps, during, before).

---

## Main Classes

### GraphBuilder

Constructs the KG from raw data.

**Methods:**

| Method | Description |
|--------|-------------|
| `` `build(sources)` `` | Build graph from inputs |
| `` `merge_entities()` `` | Merge duplicate entities during building |

**Example:**

```python
from semantica.kg import GraphBuilder

builder = GraphBuilder(merge_entities=True)
kg = builder.build([source1, source2])
```

### GraphAnalyzer

Runs analytical algorithms.

**Methods:**

| Method | Description |
|--------|-------------|
| `` `centrality(method)` `` | Calculate importance |
| `` `communities(method)` `` | Find clusters |

### TemporalGraphQuery

Queries time-aware graphs.

**Methods:**

| Method | Description |
|--------|-------------|
| `` `at_time(timestamp)` `` | Graph state at T |
| `` `during(start, end)` `` | Graph state in interval |

---

## Using Classes

```python
from semantica.kg import GraphBuilder, GraphAnalyzer

# Build using GraphBuilder
builder = GraphBuilder(merge_entities=True)
kg = builder.build(sources)

# Analyze
analyzer = GraphAnalyzer()
stats = analyzer.analyze_graph(kg)
print(f"Communities: {stats.get('communities', [])}")
```

---

## Configuration

### Environment Variables

```bash
export KG_MERGE_STRATEGY=fuzzy
export KG_TEMPORAL_GRANULARITY=day
export KG_RESOLUTION_STRATEGY=fuzzy
```

### YAML Configuration

```yaml
kg:
  resolution:
    threshold: 0.9
    strategy: semantic
    
  temporal:
    enabled: true
    default_validity: infinite
```

---

## Integration Examples

### Temporal Analysis Pipeline

```python
from semantica.kg import GraphBuilder, TemporalGraphQuery

# 1. Build Temporal Graph
builder = GraphBuilder(enable_temporal=True)
kg = builder.build(temporal_data)

# 2. Query Evolution
query = TemporalGraphQuery(kg)
snapshot_2020 = query.at_time("2020-01-01")
snapshot_2023 = query.at_time("2023-01-01")

# 3. Compare
diff = snapshot_2023.minus(snapshot_2020)
print(f"New nodes since 2020: {len(diff.nodes)}")
```

---

## Best Practices

1.  **Clean Data First**: Use `EntityResolver` to resolve similar entities and prevent "entity explosion" (too many duplicate nodes).
2.  **Use Provenance**: Always track sources (`track_history=True`) to debug where bad data came from.
3.  **Temporal Granularity**: Choose the right granularity (Day vs Second) to balance performance and precision.
4.  **Deduplication**: Use `semantica.deduplication` module for advanced deduplication needs.
5.  **Conflict Resolution**: Use `semantica.conflicts` module for conflict detection and resolution.

---

## See Also

- [Graph Store Module](graph_store.md) - Persistence layer
- [Semantic Extract Module](semantic_extract.md) - Data source
- [Visualization Module](visualization.md) - Visualizing the KG
- [Conflicts Module](conflicts.md) - Conflict detection and resolution

## Cookbook

Interactive tutorials to learn knowledge graph construction and analysis:

- **[Building Knowledge Graphs](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/07_Building_Knowledge_Graphs.ipynb)**: Learn the fundamentals of building knowledge graphs
  - **Topics**: Graph construction, entity resolution, relationship mapping
  - **Difficulty**: Beginner
  - **Use Cases**: Understanding graph construction basics

- **[Your First Knowledge Graph](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/08_Your_First_Knowledge_Graph.ipynb)**: Build your first knowledge graph from scratch
  - **Topics**: Entity extraction, relationship extraction, graph construction, visualization
  - **Difficulty**: Beginner
  - **Use Cases**: First-time users, quick start

- **[Graph Analytics](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/10_Graph_Analytics.ipynb)**: Analyze knowledge graphs with centrality and community detection
  - **Topics**: Centrality measures, community detection, graph metrics
  - **Difficulty**: Intermediate
  - **Use Cases**: Understanding graph structure, finding important nodes

- **[Advanced Graph Analytics](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/02_Advanced_Graph_Analytics.ipynb)**: Advanced graph analysis techniques
  - **Topics**: PageRank, Louvain algorithm, shortest path, graph mining
  - **Difficulty**: Advanced
  - **Use Cases**: Complex graph analysis, research applications

- **[Temporal Knowledge Graphs](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/advanced/10_Temporal_Knowledge_Graphs.ipynb)**: Model and query data that changes over time
  - **Topics**: Time series, temporal logic, temporal queries, graph evolution
  - **Difficulty**: Advanced
  - **Use Cases**: Tracking changes over time, temporal reasoning

- **[Deduplication Module](deduplication.md)**: Advanced deduplication techniques for entity resolution
