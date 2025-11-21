# Installation

Get Semantica up and running in minutes.

## Prerequisites

Before installing Semantica, ensure you have:

- **Python 3.8 or higher** - Check your version:
  ```bash
  python --version
  ```
- **pip** - Python package installer (usually comes with Python)

## Basic Installation

Install Semantica from PyPI:

```bash
pip install semantica
```

This installs Semantica with all core dependencies.

## Verify Installation

Verify that Semantica is installed correctly:

```bash
python -c "import semantica; print(semantica.__version__)"
```

Expected output:
```
0.0.1
```

You can also check the installation:

```bash
pip show semantica
```

## Development Installation

To install Semantica in development mode (for contributing):

```bash
# Clone the repository
git clone https://github.com/Hawksight-AI/semantica.git
cd semantica

# Install in editable mode
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

## Optional Dependencies

Semantica supports optional features that can be installed separately:

### GPU Support

For GPU-accelerated operations:

```bash
pip install semantica[gpu]
```

This includes:
- PyTorch with CUDA support
- FAISS GPU
- CuPy

### Visualization

For enhanced visualization capabilities:

```bash
pip install semantica[viz]
```

Includes:
- PyVis for interactive graphs
- Graphviz for static diagrams
- UMAP for dimensionality reduction

### LLM Providers

Install all LLM provider integrations:

```bash
pip install semantica[llm-all]
```

Or install specific providers:

```bash
# OpenAI
pip install semantica[llm-openai]

# Anthropic
pip install semantica[llm-anthropic]

# Google Gemini
pip install semantica[llm-gemini]

# Groq
pip install semantica[llm-groq]

# Ollama
pip install semantica[llm-ollama]
```

### Cloud Integrations

For cloud storage and deployment:

```bash
pip install semantica[cloud]
```

Includes:
- AWS S3 (boto3)
- Azure Blob Storage
- Google Cloud Storage
- Kubernetes support

### All Optional Features

Install everything:

```bash
pip install semantica[all]
```

## Virtual Environment (Recommended)

It's recommended to use a virtual environment:

=== "venv"

    ```bash
    # Create virtual environment
    python -m venv venv

    # Activate (Windows)
    venv\Scripts\activate

    # Activate (Linux/Mac)
    source venv/bin/activate

    # Install Semantica
    pip install semantica
    ```

=== "conda"

    ```bash
    # Create conda environment
    conda create -n semantica python=3.11
    conda activate semantica

    # Install Semantica
    pip install semantica
    ```

## Troubleshooting

### Common Issues

#### ModuleNotFoundError

**Error**: `ModuleNotFoundError: No module named 'semantica'`

**Solutions**:
- Make sure you've activated the correct Python environment
- Verify installation: `pip list | grep semantica`
- Reinstall: `pip install --upgrade semantica`

#### Installation Fails

**Error**: Installation fails with dependency errors

**Solutions**:
- Upgrade pip: `pip install --upgrade pip`
- Install build tools: `pip install build wheel`
- Try installing without optional dependencies first: `pip install semantica --no-deps`

#### GPU Dependencies Fail

**Error**: GPU dependencies fail to install

**Solutions**:
- Install CPU-only version first: `pip install semantica`
- Then add GPU support: `pip install semantica[gpu]`
- Check CUDA compatibility for your system

#### Permission Errors

**Error**: Permission denied during installation

**Solutions**:
- Use `--user` flag: `pip install --user semantica`
- Use virtual environment (recommended)
- On Linux/Mac, avoid using `sudo` with pip

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.8 | 3.11+ |
| RAM | 4 GB | 8 GB+ |
| Disk Space | 2 GB | 5 GB+ |
| OS | Windows/Linux/Mac | Linux/Mac |

## Next Steps

Now that Semantica is installed:

1. **[Quick Start Guide](quickstart.md)** - Build your first knowledge graph
2. **[Examples](examples.md)** - See real-world use cases
3. **[API Reference](api.md)** - Explore the full API
4. **[Cookbook](cookbook.md)** - Interactive tutorials

## Getting Help

If you encounter issues:

- Check the [troubleshooting section](#troubleshooting) above
- Review [GitHub Issues](https://github.com/Hawksight-AI/semantica/issues)
- Ask questions in discussions
