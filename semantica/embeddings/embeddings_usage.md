# Embeddings Module Usage Guide

This guide demonstrates how to use the embeddings module for generating, optimizing, and managing embeddings for text, image, audio, and multimodal content.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Text Embedding](#text-embedding)
3. [Image Embedding](#image-embedding)
4. [Audio Embedding](#audio-embedding)
5. [Multimodal Embedding](#multimodal-embedding)
6. [Embedding Optimization](#embedding-optimization)
7. [Pooling Strategies](#pooling-strategies)
8. [Similarity Calculation](#similarity-calculation)
9. [Context Management](#context-management)
10. [Provider Adapters](#provider-adapters)
11. [Using Methods](#using-methods)
12. [Using Registry](#using-registry)
13. [Configuration](#configuration)
14. [Advanced Examples](#advanced-examples)

## Basic Usage

### Using the Convenience Function

```python
from semantica.embeddings import build

# Generate embeddings from text data
result = build(
    data=["Hello world", "How are you?", "Python programming"],
    data_type="text",
    model="sentence-transformers"
)

print(f"Generated {len(result['embeddings'])} embeddings")
print(f"Shape: {result['embeddings'].shape}")
print(f"Statistics: {result['statistics']}")
```

### Using Main Classes

```python
from semantica.embeddings import EmbeddingGenerator

# Create embedding generator
generator = EmbeddingGenerator()

# Generate embeddings
embeddings = generator.generate_embeddings(
    "Hello world",
    data_type="text"
)

print(f"Embedding shape: {embeddings.shape}")
print(f"Embedding dimension: {len(embeddings)}")
```

## Text Embedding

### Sentence-Transformers Embedding

```python
from semantica.embeddings import TextEmbedder

# Create text embedder with specific model
embedder = TextEmbedder(
    model_name="all-MiniLM-L6-v2",
    device="cpu",
    normalize=True
)

# Single text embedding
embedding = embedder.embed_text("The quick brown fox jumps over the lazy dog")
print(f"Embedding shape: {embedding.shape}")
print(f"Embedding dimension: {embedder.get_embedding_dimension()}")

# Batch text embedding
texts = [
    "Python is a programming language",
    "Machine learning is fascinating",
    "Natural language processing"
]
embeddings = embedder.embed_batch(texts)
print(f"Batch embeddings shape: {embeddings.shape}")  # (3, embedding_dim)
```

### Sentence-Level Embedding

```python
from semantica.embeddings import TextEmbedder

embedder = TextEmbedder()

# Extract embeddings for each sentence
text = "First sentence. Second sentence! Third sentence?"
sentence_embeddings = embedder.embed_sentences(text)

print(f"Found {len(sentence_embeddings)} sentence embeddings")
for i, emb in enumerate(sentence_embeddings):
    print(f"Sentence {i+1}: shape {emb.shape}")
```

### Using Text Embedding Methods

```python
from semantica.embeddings.methods import embed_text

# Using sentence-transformers
emb = embed_text("Hello world", method="sentence_transformers")

# Using fallback (hash-based)
emb = embed_text("Hello world", method="fallback")

# Batch processing
texts = ["text1", "text2", "text3"]
embs = embed_text(texts, method="sentence_transformers")
```

## Image Embedding

### CLIP Image Embedding

```python
from semantica.embeddings import ImageEmbedder

# Create image embedder
embedder = ImageEmbedder(
    model_name="ViT-B/32",
    device="cpu",
    normalize=True
)

# Single image embedding
embedding = embedder.embed_image("photo.jpg")
print(f"Image embedding shape: {embedding.shape}")

# Batch image embedding
image_paths = ["img1.jpg", "img2.png", "img3.jpeg"]
embeddings = embedder.embed_batch(image_paths)
print(f"Batch embeddings shape: {embeddings.shape}")  # (3, embedding_dim)
```

### Using Image Embedding Methods

```python
from semantica.embeddings.methods import embed_image

# Using CLIP
emb = embed_image("image.jpg", method="clip")

# Using fallback
emb = embed_image("image.jpg", method="fallback")

# Batch processing
images = ["img1.jpg", "img2.png"]
embs = embed_image(images, method="clip")
```

## Audio Embedding

### Librosa Feature Extraction

```python
from semantica.embeddings import AudioEmbedder

# Create audio embedder
embedder = AudioEmbedder(
    sample_rate=16000,
    normalize=True
)

# Single audio embedding
embedding = embedder.embed_audio("speech.wav")
print(f"Audio embedding shape: {embedding.shape}")

# Extract detailed features
features = embedder.extract_features("speech.wav")
print(f"MFCC shape: {features['mfcc'].shape}")
print(f"Chroma shape: {features['chroma'].shape}")
print(f"Tempo: {features['tempo']} BPM")
print(f"Duration: {features['duration']:.2f}s")
```

### Using Audio Embedding Methods

```python
from semantica.embeddings.methods import embed_audio

# Using librosa (MFCC + chroma + spectral contrast + tonnetz)
emb = embed_audio("audio.wav", method="librosa")

# Using fallback
emb = embed_audio("audio.wav", method="fallback")

# Batch processing
audio_files = ["audio1.wav", "audio2.mp3"]
embs = embed_audio(audio_files, method="librosa")
```

## Multimodal Embedding

### Concatenation-Based Multimodal Embedding

```python
from semantica.embeddings import MultimodalEmbedder

# Create multimodal embedder
embedder = MultimodalEmbedder(
    align_modalities=True,
    normalize=True
)

# Text + Image
embedding = embedder.embed_multimodal(
    text="A cat sitting on a mat",
    image_path="cat.jpg",
    combine_method="concat"
)
print(f"Multimodal embedding shape: {embedding.shape}")

# All three modalities
embedding = embedder.embed_multimodal(
    text="Speech about cats",
    image_path="cat.jpg",
    audio_path="speech.wav",
    combine_method="concat"
)
```

### Averaging-Based Multimodal Embedding

```python
from semantica.embeddings import MultimodalEmbedder

embedder = MultimodalEmbedder()

# Mean pooling for compact representation
embedding = embedder.embed_multimodal(
    text="A cat",
    image_path="cat.jpg",
    combine_method="mean"
)
print(f"Mean-pooled embedding shape: {embedding.shape}")
```

### Cross-Modal Similarity

```python
from semantica.embeddings import MultimodalEmbedder

embedder = MultimodalEmbedder()

# Calculate text-image similarity
similarity = embedder.compute_cross_modal_similarity(
    text="A cat sitting on a mat",
    image_path="cat.jpg"
)
print(f"Text-Image similarity: {similarity:.3f}")

# Calculate similarity across all modalities
similarity = embedder.compute_cross_modal_similarity(
    text="Speech about cats",
    image_path="cat.jpg",
    audio_path="speech.wav"
)
print(f"Cross-modal similarity: {similarity:.3f}")
```

### Using Multimodal Embedding Methods

```python
from semantica.embeddings.methods import embed_multimodal

# Concatenation
emb = embed_multimodal(
    text="A cat",
    image_path="cat.jpg",
    method="concat"
)

# Averaging
emb = embed_multimodal(
    text="A cat",
    image_path="cat.jpg",
    method="mean"
)
```

## Embedding Optimization

### PCA Dimension Reduction

```python
from semantica.embeddings import EmbeddingOptimizer

# Create optimizer
optimizer = EmbeddingOptimizer(
    compression_method="pca",
    target_dimension=128
)

# Compress embeddings using PCA
embeddings = ...  # Your embeddings array (n_samples, original_dim)
compressed = optimizer.compress(
    embeddings,
    target_dim=64,
    method="pca"
)
print(f"Original shape: {embeddings.shape}")
print(f"Compressed shape: {compressed.shape}")
```

### Quantization

```python
from semantica.embeddings import EmbeddingOptimizer

optimizer = EmbeddingOptimizer()

# Quantize to 8-bit
quantized = optimizer.compress(
    embeddings,
    method="quantization",
    bits=8
)
print(f"Memory reduction: {embeddings.nbytes / quantized.nbytes:.2f}x")
```

### Dimension Truncation

```python
from semantica.embeddings import EmbeddingOptimizer

optimizer = EmbeddingOptimizer()

# Simple truncation
truncated = optimizer.reduce_dimensions(
    embeddings,
    target_dim=64,
    method="truncate"
)
```

### Performance Profiling

```python
from semantica.embeddings import EmbeddingOptimizer

optimizer = EmbeddingOptimizer()

# Profile embedding characteristics
metrics = optimizer.profile_performance(embeddings)
print(f"Shape: {metrics['shape']}")
print(f"Memory: {metrics['memory_mb']:.2f} MB")
print(f"Mean: {metrics['mean']:.4f}")
print(f"Std: {metrics['std']:.4f}")
print(f"Sparsity: {metrics['sparsity']:.3f}")
```

### Using Optimization Methods

```python
from semantica.embeddings.methods import optimize_embeddings

# PCA compression
compressed = optimize_embeddings(
    embeddings,
    method="pca",
    target_dim=64
)

# Quantization
quantized = optimize_embeddings(
    embeddings,
    method="quantization",
    bits=8
)

# Truncation
truncated = optimize_embeddings(
    embeddings,
    method="truncate",
    target_dim=64
)
```

## Pooling Strategies

### Mean Pooling

```python
from semantica.embeddings import MeanPooling

pooling = MeanPooling()
embeddings = ...  # (n_embeddings, dim)
pooled = pooling.pool(embeddings)
print(f"Pooled shape: {pooled.shape}")  # (dim,)
```

### Max Pooling

```python
from semantica.embeddings import MaxPooling

pooling = MaxPooling()
pooled = pooling.pool(embeddings)
```

### CLS Token Pooling

```python
from semantica.embeddings import CLSPooling

pooling = CLSPooling()
pooled = pooling.pool(embeddings)  # Returns first embedding
```

### Attention-Based Pooling

```python
from semantica.embeddings import AttentionPooling

pooling = AttentionPooling()
pooled = pooling.pool(embeddings)  # Softmax-weighted sum
```

### Hierarchical Pooling

```python
from semantica.embeddings import HierarchicalPooling

pooling = HierarchicalPooling()
pooled = pooling.pool(embeddings, chunk_size=10)
```

### Using Pooling Methods

```python
from semantica.embeddings.methods import pool_embeddings

# Mean pooling
pooled = pool_embeddings(embeddings, method="mean")

# Max pooling
pooled = pool_embeddings(embeddings, method="max")

# Attention pooling
pooled = pool_embeddings(embeddings, method="attention")

# Hierarchical pooling
pooled = pool_embeddings(embeddings, method="hierarchical", chunk_size=10)
```

## Similarity Calculation

### Cosine Similarity

```python
from semantica.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator()

emb1 = generator.generate_embeddings("Hello world", data_type="text")
emb2 = generator.generate_embeddings("Hi there", data_type="text")

similarity = generator.compare_embeddings(emb1, emb2, method="cosine")
print(f"Cosine similarity: {similarity:.3f}")
```

### Euclidean Similarity

```python
from semantica.embeddings import EmbeddingGenerator

generator = EmbeddingGenerator()

similarity = generator.compare_embeddings(emb1, emb2, method="euclidean")
print(f"Euclidean similarity: {similarity:.3f}")
```

### Using Similarity Methods

```python
from semantica.embeddings.methods import calculate_similarity

# Cosine similarity
sim = calculate_similarity(emb1, emb2, method="cosine")

# Euclidean similarity
sim = calculate_similarity(emb1, emb2, method="euclidean")
```

## Context Management

### Text Splitting into Windows

```python
from semantica.embeddings import ContextManager

# Create context manager
manager = ContextManager(
    window_size=512,
    overlap=50,
    max_contexts=100
)

# Split long text into windows
long_text = "..."  # Your long text
windows = manager.split_into_windows(
    long_text,
    preserve_sentences=True
)

print(f"Created {len(windows)} context windows")
for window in windows:
    print(f"Window {window.context_id}: {window.start_index}-{window.end_index}")
```

### Context Merging

```python
from semantica.embeddings import ContextManager

manager = ContextManager()

# Merge multiple context windows
merged = manager.merge_contexts(
    [window1.context_id, window2.context_id, window3.context_id]
)
print(f"Merged text length: {len(merged.text)}")
```

### Context Retrieval

```python
from semantica.embeddings import ContextManager

manager = ContextManager()

# Get context by ID
window = manager.get_context(context_id)
if window:
    print(f"Context text: {window.text}")
    print(f"Metadata: {window.metadata}")
```

## Provider Adapters

### OpenAI Embeddings

```python
from semantica.embeddings import OpenAIAdapter

# Create OpenAI adapter
adapter = OpenAIAdapter(
    api_key="your-api-key",
    model="text-embedding-3-small"
)

# Generate embedding
embedding = adapter.embed("Hello world")
print(f"OpenAI embedding shape: {embedding.shape}")
```

### BGE Embeddings

```python
from semantica.embeddings import BGEAdapter

# Create BGE adapter
adapter = BGEAdapter(
    model_name="BAAI/bge-small-en-v1.5"
)

# Generate embedding
embedding = adapter.embed("Hello world")
```

### Provider Factory

```python
from semantica.embeddings import ProviderAdapterFactory

# Create provider using factory
adapter = ProviderAdapterFactory.create(
    "openai",
    api_key="your-api-key"
)

embedding = adapter.embed("Hello world")
```

## Using Methods

### Generate Embeddings

```python
from semantica.embeddings.methods import generate_embeddings

# Auto-detect data type
emb = generate_embeddings("Hello world", method="default")

# Explicit text embedding
emb = generate_embeddings("Hello world", method="text")

# Image embedding
emb = generate_embeddings("image.jpg", method="image")

# Audio embedding
emb = generate_embeddings("audio.wav", method="audio")
```

### Batch Processing

```python
from semantica.embeddings.methods import embed_text, embed_image

# Batch text embeddings
texts = ["text1", "text2", "text3"]
embeddings = embed_text(texts, method="sentence_transformers")

# Batch image embeddings
images = ["img1.jpg", "img2.png"]
embeddings = embed_image(images, method="clip")
```

## Using Registry

### Registering Custom Methods

```python
from semantica.embeddings.registry import method_registry

def custom_text_embedding(text: str, **kwargs):
    """Custom text embedding function."""
    # Your custom implementation
    return embedding_vector

# Register custom method
method_registry.register("text", "custom_method", custom_text_embedding)

# Use custom method
from semantica.embeddings.methods import embed_text
emb = embed_text("Hello world", method="custom_method")
```

### Listing Available Methods

```python
from semantica.embeddings.methods import list_available_methods

# List all methods
all_methods = list_available_methods()
print(all_methods)

# List methods for specific task
text_methods = list_available_methods("text")
print(text_methods)
```

### Getting Registered Methods

```python
from semantica.embeddings.methods import get_embedding_method

# Get method from registry
method = get_embedding_method("text", "custom_method")
if method:
    result = method("Hello world")
```

## Configuration

### Environment Variables

```bash
# Set embedding configuration via environment variables
export EMBEDDING_MODEL="all-MiniLM-L6-v2"
export EMBEDDING_BATCH_SIZE=32
export EMBEDDING_DIMENSION=384
export EMBEDDING_NORMALIZE="true"
export EMBEDDING_DEVICE="cpu"
export EMBEDDING_COMPRESSION_METHOD="pca"
export EMBEDDING_WINDOW_SIZE=512
export EMBEDDING_OVERLAP=50
```

### Programmatic Configuration

```python
from semantica.embeddings.config import embeddings_config

# Set configuration
embeddings_config.set("model", "sentence-transformers/all-mpnet-base-v2")
embeddings_config.set("batch_size", 64)
embeddings_config.set("normalize", True)

# Get configuration
model = embeddings_config.get("model", default="all-MiniLM-L6-v2")
batch_size = embeddings_config.get("batch_size", default=32)

# Method-specific configuration
embeddings_config.set_method_config("text", model_name="all-MiniLM-L6-v2")
text_config = embeddings_config.get_method_config("text")
```

### Config File (YAML)

```yaml
embeddings:
  model: "all-MiniLM-L6-v2"
  batch_size: 32
  dimension: 384
  normalize: true
  device: "cpu"
  compression_method: "pca"
  window_size: 512
  overlap: 50

embeddings_methods:
  text:
    model_name: "all-MiniLM-L6-v2"
    device: "cpu"
    normalize: true
  image:
    model_name: "ViT-B/32"
    device: "cpu"
    normalize: true
  audio:
    sample_rate: 16000
    normalize: true
```

```python
from semantica.embeddings.config import EmbeddingsConfig

# Load from config file
config = EmbeddingsConfig(config_file="config.yaml")
```

## Advanced Examples

### Complete Embedding Pipeline

```python
from semantica.embeddings import (
    EmbeddingGenerator,
    EmbeddingOptimizer,
    calculate_similarity
)

# Step 1: Generate embeddings
generator = EmbeddingGenerator()
texts = ["text1", "text2", "text3"]
embeddings = generator.generate_embeddings(texts, data_type="text")

# Step 2: Optimize embeddings
optimizer = EmbeddingOptimizer()
compressed = optimizer.compress(embeddings, target_dim=64, method="pca")

# Step 3: Calculate similarities
similarity = calculate_similarity(compressed[0], compressed[1], method="cosine")
print(f"Similarity: {similarity:.3f}")
```

### Multimodal Search

```python
from semantica.embeddings import MultimodalEmbedder, calculate_similarity

embedder = MultimodalEmbedder()

# Embed query (text)
query_emb = embedder.embed_text("A cat sitting on a mat")

# Embed documents (images)
doc_embs = [
    embedder.embed_image("cat1.jpg"),
    embedder.embed_image("dog1.jpg"),
    embedder.embed_image("cat2.jpg"),
]

# Find most similar
similarities = [
    calculate_similarity(query_emb, doc_emb, method="cosine")
    for doc_emb in doc_embs
]

most_similar_idx = max(range(len(similarities)), key=lambda i: similarities[i])
print(f"Most similar image: doc_{most_similar_idx}")
```

### Batch Processing with Context Windows

```python
from semantica.embeddings import ContextManager, TextEmbedder

# Create context manager
context_manager = ContextManager(window_size=512, overlap=50)

# Create text embedder
embedder = TextEmbedder()

# Process long document
long_document = "..."  # Your long document
windows = context_manager.split_into_windows(long_document)

# Generate embeddings for each window
window_embeddings = []
for window in windows:
    emb = embedder.embed_text(window.text)
    window_embeddings.append(emb)

# Pool window embeddings
from semantica.embeddings.methods import pool_embeddings
document_embedding = pool_embeddings(
    np.array(window_embeddings),
    method="mean"
)
```

### Custom Embedding Method

```python
from semantica.embeddings.registry import method_registry
from semantica.embeddings.methods import embed_text
import numpy as np

def tfidf_embedding(text: str, **kwargs):
    """Custom TF-IDF based embedding."""
    # Your TF-IDF implementation
    # This is a placeholder
    return np.random.rand(128).astype(np.float32)

# Register custom method
method_registry.register("text", "tfidf", tfidf_embedding)

# Use custom method
embedding = embed_text("Hello world", method="tfidf")
```

## Best Practices

1. **Normalize Embeddings**: Always normalize embeddings for consistent similarity calculations
   ```python
   embedder = TextEmbedder(normalize=True)
   ```

2. **Batch Processing**: Use batch processing for multiple items to improve efficiency
   ```python
   embeddings = embedder.embed_batch(texts)  # Faster than loop
   ```

3. **Dimension Optimization**: Use PCA for dimension reduction when storage is a concern
   ```python
   compressed = optimizer.compress(embeddings, target_dim=64, method="pca")
   ```

4. **Context Windows**: Use context windows for long texts to maintain semantic coherence
   ```python
   windows = manager.split_into_windows(long_text, preserve_sentences=True)
   ```

5. **Cross-Modal Similarity**: Use cross-modal similarity for multimodal search and retrieval
   ```python
   similarity = embedder.compute_cross_modal_similarity(text="A cat", image_path="cat.jpg")
   ```

6. **Configuration Management**: Use environment variables or config files for consistent settings
   ```python
   embeddings_config.set("model", "all-MiniLM-L6-v2")
   ```

7. **Error Handling**: Always handle fallback methods when dependencies are unavailable
   ```python
   try:
       emb = embed_text(text, method="sentence_transformers")
   except:
       emb = embed_text(text, method="fallback")
   ```

## Performance Tips

1. **GPU Acceleration**: Use GPU for faster processing when available
   ```python
   embedder = TextEmbedder(device="cuda")
   ```

2. **Batch Size**: Adjust batch size based on available memory
   ```python
   embeddings_config.set("batch_size", 64)  # Larger for more memory
   ```

3. **Quantization**: Use quantization for memory-constrained environments
   ```python
   quantized = optimize_embeddings(embeddings, method="quantization", bits=8)
   ```

4. **Caching**: Cache embeddings for repeated queries
   ```python
   # Store embeddings in cache/database for reuse
   ```

5. **Parallel Processing**: Process multiple files in parallel when possible
   ```python
   from concurrent.futures import ThreadPoolExecutor
   with ThreadPoolExecutor() as executor:
       embeddings = list(executor.map(embed_image, image_paths))
   ```

