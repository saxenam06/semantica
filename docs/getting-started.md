# Getting Started

## Welcome to Semantica

**Semantica** is a comprehensive knowledge graph and semantic processing framework designed for building production-ready semantic AI applications.

### 🎯 What You'll Learn
- What Semantica is and why it's useful
- How to install and configure the framework
- Understanding the framework architecture
- Key concepts and terminology
- Next steps for getting started

---

## 🚀 What is Semantica?

Semantica is a powerful, production-ready framework for:

- **Building Knowledge Graphs**: Transform unstructured data into structured knowledge graphs.
- **Semantic Processing**: Extract entities, relationships, and meaning from text, images, and audio.
- **GraphRAG**: Next-generation retrieval augmented generation using knowledge graphs.
- **Temporal Analysis**: Time-aware knowledge graphs for tracking changes over time.
- **Multi-Modal Processing**: Handle text, images, audio, and structured data.
- **Enterprise Features**: Quality assurance, conflict resolution, ontology generation, and more.

---

## 💡 Use Cases

| Domain | Application |
| :--- | :--- |
| **Cybersecurity** | Threat intelligence and analysis |
| **Healthcare** | Medical research and patient data analysis |
| **Finance** | Fraud detection and financial analysis |
| **Supply Chain** | Optimization and risk management |
| **Research** | Knowledge management and literature review |
| **AI Systems** | Multi-agent memory and reasoning |

---

## 📦 Installation & Setup

### Prerequisites
Before installing Semantica, ensure you have:
- **Python 3.8** or higher
- **pip** package manager
- (Optional) Virtual environment for isolation

### Installation Methods

=== "PyPI (Stable)"
    ```bash
    pip install semantica
    ```

=== "Source (Dev)"
    ```bash
    git clone https://github.com/Hawksight-AI/semantica.git
    cd semantica
    pip install -e .
    ```

=== "Extras"
    ```bash
    pip install semantica[all]           # Install all optional dependencies
    pip install semantica[gpu]           # Install GPU support
    pip install semantica[visualization] # Install visualization tools
    ```

### Verify Installation

```python
import semantica
print(semantica.__version__)
```

---

## 🏗️ Understanding Semantica's Architecture

Semantica uses a **modular architecture** where each module handles a specific aspect of semantic processing. This design gives you flexibility and control over your pipeline.

### Primary Approach: Individual Modules

The recommended approach is to use individual modules directly. Each module can be imported and used independently:

- **`semantica.ingest`**: Data ingestion from files, web, databases
- **`semantica.parse`**: Document parsing and text extraction
- **`semantica.semantic_extract`**: Entity and relationship extraction
- **`semantica.kg`**: Knowledge graph construction
- **`semantica.embeddings`**: Vector embedding generation
- **`semantica.vector_store`**: Vector database operations

**Benefits of the modular approach:**
- **Full control**: Customize each step of your pipeline
- **Flexibility**: Mix and match modules as needed
- **Transparency**: Clear understanding of what each step does
- **Easy debugging**: Isolate issues to specific modules

**Quick Example:**
```python
from semantica.ingest import FileIngestor
from semantica.parse import DocumentParser
from semantica.semantic_extract import NERExtractor, RelationExtractor
from semantica.kg import GraphBuilder

# Each module is used independently
ingestor = FileIngestor()
parser = DocumentParser()
ner = NERExtractor()
builder = GraphBuilder()
```

**For detailed examples, see:**
- **[Welcome to Semantica Cookbook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/01_Welcome_to_Semantica.ipynb)**: Comprehensive introduction to all modules and architecture
  - **Topics**: Framework overview, all modules, architecture, configuration
  - **Difficulty**: Beginner
  - **Time**: 30-45 minutes
  - **Use Cases**: First-time users, understanding the framework structure

### Alternative Approach: Orchestration Class

For complex workflows, you can use the `` `Semantica` `` class for orchestration. This class coordinates multiple modules and provides lifecycle management.

**When to use orchestration:**
- Complex multi-step workflows spanning multiple modules
- Need lifecycle management (initialization, shutdown)
- Want centralized configuration
- Building applications with multiple components

!!! tip "Getting Started"
    For beginners, start with individual modules to understand how each component works. As you build more complex applications, consider using the orchestration class for workflow management. See the [Core Module Reference](reference/core.md) for orchestration details.

## ⚙️ Configuration

Semantica modules can be configured individually or through environment variables. Configuration options vary by module, allowing you to customize behavior for your specific needs.

### Environment Variables

Common configuration via environment variables:

```bash
export OPENAI_API_KEY=your_openai_key
export EMBEDDING_MODEL=all-MiniLM-L6-v2
export EMBEDDING_DEVICE=cuda
```

### Module-Specific Configuration

Each module accepts configuration parameters when instantiated. For example, the NER extractor can be configured with different methods, providers, and thresholds.

### Config File (`config.yaml`)

For centralized configuration, you can use a YAML config file to manage settings across multiple modules:

```yaml
api_keys:
  openai: your_key_here

embedding:
  provider: openai
  model: text-embedding-3-large

knowledge_graph:
  backend: networkx
  temporal: true
```

**For detailed configuration examples, see:**
- **[Welcome to Semantica Cookbook](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/01_Welcome_to_Semantica.ipynb)**: Configuration examples for all modules
- **[Core Module Reference](reference/core.md)**: Complete configuration documentation

---

## ⏭️ Next Steps

Now that you understand the basics, here are recommended next steps:

### 🍳 Interactive Tutorials (Cookbook)

Get hands-on experience with these interactive Jupyter notebooks:

1. **[Welcome to Semantica](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/01_Welcome_to_Semantica.ipynb)**: Comprehensive introduction to all Semantica modules
   - **Topics**: Framework overview, all modules, architecture, configuration
   - **Difficulty**: Beginner
   - **Time**: 30-45 minutes
   - **Use Cases**: First-time users, understanding the framework structure

2. **[Your First Knowledge Graph](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/08_Your_First_Knowledge_Graph.ipynb)**: Build your first knowledge graph from a document
   - **Topics**: Entity extraction, relationship extraction, graph construction, visualization
   - **Difficulty**: Beginner
   - **Time**: 20-30 minutes
   - **Use Cases**: Learning the basics, quick start

3. **[Data Ingestion](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/02_Data_Ingestion.ipynb)**: Learn to ingest from multiple sources
   - **Topics**: File, web, feed, stream, database ingestion
   - **Difficulty**: Beginner
   - **Time**: 15-20 minutes
   - **Use Cases**: Loading data from various sources

4. **[Document Parsing](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/03_Document_Parsing.ipynb)**: Parse various document formats
   - **Topics**: PDF, DOCX, HTML, JSON parsing
   - **Difficulty**: Beginner
   - **Time**: 15-20 minutes
   - **Use Cases**: Extracting text from different file formats

### 📚 Documentation

- **[Quick Start Guide](quickstart.md)**: Step-by-step tutorial to build your first knowledge graph
- **[Core Concepts](concepts.md)**: Deep dive into knowledge graphs, ontologies, and semantic reasoning
- **[API Reference](reference/core.md)**: Complete technical documentation for all modules
- **[Examples](examples.md)**: Real-world examples and use cases
- **[Cookbook](cookbook.md)**: Full list of interactive Jupyter notebooks
