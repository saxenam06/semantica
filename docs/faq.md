# Frequently Asked Questions

Common questions and answers about Semantica.

## General Questions

### What is Semantica?

Semantica is an open-source framework for building semantic layers and knowledge graphs from unstructured data. It transforms raw data into structured, queryable knowledge that powers AI applications.

### What can I use Semantica for?

- Building knowledge graphs from documents
- Creating semantic layers for AI applications
- Extracting entities and relationships from text
- Powering GraphRAG systems
- Integrating data from multiple sources
- Building AI agent memory systems

### Is Semantica free?

Yes! Semantica is 100% open source and free to use under the MIT License.

## Installation

### How do I install Semantica?

```bash
pip install semantica
```

See the [Installation Guide](installation.md) for detailed instructions.

### What Python version do I need?

Python 3.8 or higher. Python 3.11+ is recommended.

### Do I need GPU?

No, GPU is optional. Semantica works on CPU, but GPU acceleration is available for faster processing.

## Usage

### How do I get started?

1. Install Semantica: `pip install semantica`
2. Follow the [Quick Start Guide](quickstart.md)
3. Try the [Examples](examples.md)

### Can I process PDF files?

Yes! Semantica supports PDF, DOCX, HTML, JSON, CSV, and many other formats.

### How do I extract entities from text?

```python
from semantica import Semantica

semantica = Semantica()
result = semantica.semantic_extract.extract_entities("Your text here")
entities = result["entities"]
```

### Can I use my own models?

Yes, Semantica is extensible. You can plug in custom models for entity extraction, embeddings, etc.

## Knowledge Graphs

### What is a knowledge graph?

A knowledge graph is a structured representation where entities (nodes) are connected by relationships (edges). It captures semantic meaning and relationships in data.

### How do I build a knowledge graph?

```python
from semantica import Semantica

semantica = Semantica()
result = semantica.build_knowledge_base(["document.pdf"])
kg = result["knowledge_graph"]
```

### Can I merge multiple knowledge graphs?

Yes! Use the `merge` method:

```python
merged = semantica.kg.merge([kg1, kg2, kg3])
```

### How do I visualize a knowledge graph?

```python
semantica.kg.visualize(kg, output_path="graph.html")
```

## Conflict Resolution

### What is conflict resolution?

When the same entity appears in multiple sources with different information, conflict resolution determines which information to use.

### What strategies are available?

- **Voting**: Majority wins
- **Credibility Weighted**: Weight by source credibility
- **Most Recent**: Use latest information
- **Highest Confidence**: Use highest confidence score
- **First Seen**: Use first encountered value

### How do I set a resolution strategy?

```python
from semantica.conflicts import ConflictResolver

resolver = ConflictResolver(default_strategy="voting")
```

## Export & Integration

### What formats can I export to?

- RDF/XML
- OWL (Ontology)
- JSON
- CSV
- YAML
- And more

### Can I use Semantica with other tools?

Yes! Semantica exports to standard formats that work with:
- Neo4j
- Graph databases
- RDF stores
- Vector databases
- Any tool that accepts RDF/JSON/CSV

## Performance

### How fast is Semantica?

Performance depends on:
- Document size
- Number of documents
- Hardware (CPU/GPU)
- Configuration options

For typical documents, processing takes seconds to minutes.

### Can I process large datasets?

Yes, but consider:
- Processing in batches
- Using GPU acceleration
- Incremental building
- Optimizing configuration

## Troubleshooting

### Installation fails

- Upgrade pip: `pip install --upgrade pip`
- Use virtual environment
- Check Python version: `python --version`

### No entities extracted

- Verify document contains text (not just images)
- Check document format is supported
- Review extraction configuration

### Memory errors

- Process documents one at a time
- Reduce batch sizes
- Use smaller models
- Increase available RAM

### Slow processing

- Enable GPU if available
- Process in smaller batches
- Optimize configuration
- Use faster models

## Getting Help

### Where can I get help?

- **Documentation**: This site
- **GitHub Issues**: [Report bugs](https://github.com/Hawksight-AI/semantica/issues)
- **Discussions**: [Ask questions](https://github.com/Hawksight-AI/semantica/discussions)

### How do I report a bug?

Open an issue on [GitHub](https://github.com/Hawksight-AI/semantica/issues) with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details

### Can I contribute?

Yes! We welcome contributions. See our [Contributing Guide](https://github.com/Hawksight-AI/semantica/blob/main/CONTRIBUTING.md).

---

Still have questions? Check the [API Reference](api.md) or [open an issue](https://github.com/Hawksight-AI/semantica/issues).

