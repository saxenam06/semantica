"""
Embeddings Generation Module

This module provides comprehensive embedding generation and management capabilities
for the Semantica framework, supporting text, image, audio, and multimodal content
with multiple embedding models, optimization strategies, and similarity calculations.

Algorithms Used:

Embedding Generation:
    - Sentence-transformers Encoding: Transformer-based sentence embedding generation using pre-trained models
    - CLIP Image Encoding: Vision transformer-based image embedding generation for cross-modal understanding
    - Librosa Audio Feature Extraction: Multi-feature audio analysis including:
      * MFCC (Mel-frequency Cepstral Coefficients): 13 coefficients for spectral representation
      * Chroma Features: 12-dimensional pitch class representation
      * Spectral Contrast: 7-dimensional spectral contrast features
      * Tonnetz: 6-dimensional tonal centroid features
      * Temporal Aggregation: Mean pooling across time frames for fixed-size vectors
    - Hash-based Fallback Embeddings: SHA-256 hash-based deterministic embeddings (128-dimensional)
    - Batch Processing: Vectorized processing for multiple items simultaneously
    - Data Type Auto-detection: File extension-based and content-based data type detection

Similarity Calculation:
    - Cosine Similarity: Dot product divided by vector norms for normalized similarity (0-1 range)
    - Euclidean Similarity: L2 norm distance converted to similarity score (0-1 range)
    - Cross-modal Similarity: Pairwise cosine similarity between different modalities (text-image, image-audio, etc.)

Embedding Optimization:
    - PCA (Principal Component Analysis): Variance-preserving dimension reduction using eigenvalue decomposition
    - Quantization: Bit-depth reduction (8-bit, 16-bit) for memory efficiency with dequantization
    - Dimension Truncation: Simple truncation to first N dimensions
    - Batch Normalization: L2 normalization across batch dimension for unit vectors

Pooling Strategies:
    - Mean Pooling: Arithmetic mean across embedding dimension
    - Max Pooling: Element-wise maximum across embedding dimension
    - CLS Token Pooling: First token/embedding extraction (for transformer models)
    - Attention-based Pooling: Softmax-weighted sum using dot product attention scores
    - Hierarchical Pooling: Two-level pooling (chunk-level then global-level mean pooling)

Multimodal Processing:
    - Dimension Alignment: Truncation/padding to align embedding dimensions across modalities
    - Concatenation: Vector concatenation for combined multimodal embeddings
    - Averaging: Mean aggregation for compact multimodal representation
    - Cross-modal Similarity: Pairwise similarity computation between different input modalities

Context Management:
    - Text Splitting: Sliding window text chunking with configurable window size
    - Sentence Boundary Preservation: Regex-based sentence boundary detection (., !, ?, newline) with minimum window size constraints
    - Overlapping Windows: Sliding window with configurable overlap for context continuity
    - Context Merging: Text concatenation and metadata aggregation from multiple context windows
    - LRU-style Context Cleanup: Oldest context removal when maximum context limit reached

Provider Adapters:
    - OpenAI API Integration: REST API-based embedding generation
    - BGE Model Integration: Sentence-transformers wrapper for BAAI General Embedding models
    - Sentence-transformers Integration: Hugging Face transformers-based embedding models

Key Features:
    - Text, image, audio, and multimodal embedding generation
    - Multiple embedding models and providers
    - Embedding optimization and compression
    - Similarity calculation and comparison
    - Pooling strategies for aggregation
    - Context window management for long texts
    - Method registry for extensibility
    - Configuration management with environment variables and config files

Main Classes:
    - EmbeddingGenerator: Main embedding generation handler
    - TextEmbedder: Text embedding generation
    - ImageEmbedder: Image embedding generation
    - AudioEmbedder: Audio embedding generation
    - MultimodalEmbedder: Multi-modal embedding support
    - EmbeddingOptimizer: Embedding optimization and fine-tuning
    - ContextManager: Embedding context management
    - MethodRegistry: Registry for custom embedding methods
    - EmbeddingsConfig: Configuration manager for embeddings module

Convenience Functions:
    - build: Generate embeddings from data in one call
    - generate_embeddings: Generate embeddings with method dispatch
    - embed_text: Text embedding wrapper
    - embed_image: Image embedding wrapper
    - embed_audio: Audio embedding wrapper
    - embed_multimodal: Multimodal embedding wrapper
    - optimize_embeddings: Embedding optimization wrapper
    - calculate_similarity: Similarity calculation wrapper
    - pool_embeddings: Pooling strategy wrapper

Example Usage:
    >>> from semantica.embeddings import build, embed_text, embed_image, calculate_similarity
    >>> # Using convenience function
    >>> result = build(data=["text1", "text2"], data_type="text")
    >>> # Using method functions
    >>> text_emb = embed_text("Hello world", method="sentence_transformers")
    >>> img_emb = embed_image("image.jpg", method="clip")
    >>> similarity = calculate_similarity(text_emb, img_emb, method="cosine")
    >>> # Using classes directly
    >>> from semantica.embeddings import EmbeddingGenerator
    >>> generator = EmbeddingGenerator()
    >>> embeddings = generator.generate_embeddings("Hello world", data_type="text")

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import numpy as np

from .embedding_generator import EmbeddingGenerator
from .text_embedder import TextEmbedder
from .image_embedder import ImageEmbedder
from .audio_embedder import AudioEmbedder
from .multimodal_embedder import MultimodalEmbedder
from .embedding_optimizer import EmbeddingOptimizer
from .context_manager import ContextManager, ContextWindow
from .provider_adapters import (
    ProviderAdapter,
    OpenAIAdapter,
    BGEAdapter,
    LlamaAdapter,
    ProviderAdapterFactory,
)
from .pooling_strategies import (
    PoolingStrategy,
    MeanPooling,
    MaxPooling,
    CLSPooling,
    AttentionPooling,
    HierarchicalPooling,
    PoolingStrategyFactory,
)
from .registry import MethodRegistry, method_registry
from .methods import (
    generate_embeddings,
    embed_text,
    embed_image,
    embed_audio,
    embed_multimodal,
    optimize_embeddings,
    calculate_similarity,
    pool_embeddings,
    get_embedding_method,
    list_available_methods,
)
from .config import EmbeddingsConfig, embeddings_config

__all__ = [
    # Core Classes
    "EmbeddingGenerator",
    "TextEmbedder",
    "ImageEmbedder",
    "AudioEmbedder",
    "MultimodalEmbedder",
    "EmbeddingOptimizer",
    "ContextManager",
    "ContextWindow",
    # Provider adapters
    "ProviderAdapter",
    "OpenAIAdapter",
    "BGEAdapter",
    "LlamaAdapter",
    "ProviderAdapterFactory",
    # Pooling strategies
    "PoolingStrategy",
    "MeanPooling",
    "MaxPooling",
    "CLSPooling",
    "AttentionPooling",
    "HierarchicalPooling",
    "PoolingStrategyFactory",
    # Registry and Methods
    "MethodRegistry",
    "method_registry",
    "generate_embeddings",
    "embed_text",
    "embed_image",
    "embed_audio",
    "embed_multimodal",
    "optimize_embeddings",
    "calculate_similarity",
    "pool_embeddings",
    "get_embedding_method",
    "list_available_methods",
    # Configuration
    "EmbeddingsConfig",
    "embeddings_config",
    # Convenience Functions
    "build",
]


def build(
    data: Union[str, Path, List[Union[str, Path]]],
    data_type: Optional[str] = None,
    model: Optional[str] = None,
    batch_size: int = 32,
    **options
) -> Dict[str, Any]:
    """
    Generate embeddings from data (module-level convenience function).
    
    This is a user-friendly wrapper around EmbeddingGenerator.generate_embeddings()
    that creates an EmbeddingGenerator instance and generates embeddings.
    
    Args:
        data: Input data - can be text string, file path, or list of texts/paths
        data_type: Data type - "text", "image", "audio" (auto-detected if None)
        model: Embedding model to use (default: None, uses default model)
        batch_size: Batch size for processing multiple items (default: 32)
        **options: Additional generation options
        
    Returns:
        Dictionary containing:
            - embeddings: Generated embeddings (numpy array or list of arrays)
            - metadata: Embedding metadata
            - statistics: Generation statistics
            
    Examples:
        >>> import semantica
        >>> result = semantica.embeddings.build(
        ...     data=["text1", "text2", "text3"],
        ...     data_type="text",
        ...     model="sentence-transformers"
        ... )
        >>> print(f"Generated {len(result['embeddings'])} embeddings")
    """
    # Create EmbeddingGenerator instance
    config = {}
    if model:
        config["model"] = model
    config.update(options)
    
    generator = EmbeddingGenerator(config=config, **options)
    
    # Normalize data to list if single item
    is_single = not isinstance(data, list)
    if is_single:
        data = [data]
    
    # Generate embeddings
    if len(data) == 1:
        # Single item
        embeddings = generator.generate_embeddings(data[0], data_type=data_type, **options)
        if is_single:
            # Return single embedding array
            return {
                "embeddings": embeddings,
                "metadata": {
                    "data_type": data_type or generator._detect_data_type(data[0]),
                    "model": model or "default",
                    "shape": embeddings.shape if isinstance(embeddings, np.ndarray) else None
                },
                "statistics": {
                    "total_items": 1,
                    "successful": 1,
                    "failed": 0
                }
            }
    else:
        # Batch processing
        results = generator.process_batch(data, **options)
        return {
            "embeddings": results["embeddings"],
            "metadata": {
                "data_type": data_type or "auto-detected",
                "model": model or "default",
                "batch_size": batch_size
            },
            "statistics": {
                "total_items": results["total"],
                "successful": results["success_count"],
                "failed": results["failure_count"]
            },
            "failed_items": results.get("failed", [])
        }
