"""
Embedding Methods Module

This module provides all embedding methods as simple, reusable functions for
embedding generation, optimization, similarity calculation, and pooling.
It supports multiple embedding approaches and integrates with the method registry
for extensibility.

Supported Methods:

Embedding Generation:
    - "default": Default embedding generation using EmbeddingGenerator
    - "text": Text embedding generation
    - "image": Image embedding generation
    - "audio": Audio embedding generation
    - "multimodal": Multimodal embedding generation

Text Embedding:
    - "sentence_transformers": Sentence-transformers model-based embedding
    - "fallback": Hash-based fallback embedding

Image Embedding:
    - "clip": CLIP model-based embedding
    - "fallback": Feature-based fallback embedding

Audio Embedding:
    - "librosa": Librosa feature extraction (MFCC, chroma, spectral contrast, tonnetz)
    - "fallback": Simple fallback embedding

Multimodal Embedding:
    - "concat": Concatenation-based combination
    - "mean": Averaging-based combination

Embedding Optimization:
    - "pca": Principal Component Analysis dimension reduction
    - "quantization": Bit-depth quantization
    - "truncate": Simple dimension truncation

Pooling Strategies:
    - "mean": Mean pooling
    - "max": Max pooling
    - "cls": CLS token pooling
    - "attention": Attention-based pooling
    - "hierarchical": Hierarchical pooling

Similarity Calculation:
    - "cosine": Cosine similarity
    - "euclidean": Euclidean similarity

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

Key Features:
    - Multiple embedding generation methods
    - Multiple optimization methods
    - Multiple pooling strategies
    - Multiple similarity calculation methods
    - Method dispatchers with registry support
    - Custom method registration capability
    - Consistent interface across all methods

Main Functions:
    - generate_embeddings: Embedding generation wrapper
    - embed_text: Text embedding wrapper
    - embed_image: Image embedding wrapper
    - embed_audio: Audio embedding wrapper
    - embed_multimodal: Multimodal embedding wrapper
    - optimize_embeddings: Embedding optimization wrapper
    - calculate_similarity: Similarity calculation wrapper
    - pool_embeddings: Pooling strategy wrapper
    - get_embedding_method: Get embedding method by name
    - list_available_methods: List registered methods

Example Usage:
    >>> from semantica.embeddings.methods import embed_text, embed_image, calculate_similarity
    >>> text_emb = embed_text("Hello world", method="sentence_transformers")
    >>> img_emb = embed_image("image.jpg", method="clip")
    >>> similarity = calculate_similarity(text_emb, img_emb, method="cosine")
    >>> from semantica.embeddings.methods import get_embedding_method
    >>> method = get_embedding_method("text", "custom_method")
"""

from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path

import numpy as np

from ..utils.logging import get_logger
from ..utils.exceptions import ProcessingError, ConfigurationError
from .embedding_generator import EmbeddingGenerator
from .text_embedder import TextEmbedder
from .image_embedder import ImageEmbedder
from .audio_embedder import AudioEmbedder
from .multimodal_embedder import MultimodalEmbedder
from .embedding_optimizer import EmbeddingOptimizer
from .pooling_strategies import PoolingStrategyFactory
from .registry import method_registry
from .config import embeddings_config

logger = get_logger("embeddings_methods")


def generate_embeddings(
    data: Union[str, Path, List[Union[str, Path]]],
    data_type: Optional[str] = None,
    method: str = "default",
    **kwargs
) -> np.ndarray:
    """
    Generate embeddings from data (convenience function).
    
    This is a user-friendly wrapper that generates embeddings using the specified method.
    
    Args:
        data: Input data - can be text string, file path, or list of texts/paths
        data_type: Data type - "text", "image", "audio" (auto-detected if None)
        method: Generation method (default: "default")
            - "default": Use EmbeddingGenerator with auto-detection
            - "text": Text embedding generation
            - "image": Image embedding generation
            - "audio": Audio embedding generation
        **kwargs: Additional options passed to embedder
    
    Returns:
        np.ndarray: Generated embeddings (1D for single item, 2D for batch)
        
    Examples:
        >>> from semantica.embeddings.methods import generate_embeddings
        >>> emb = generate_embeddings("Hello world", method="default")
        >>> embs = generate_embeddings(["text1", "text2"], method="text")
    """
    # Check for custom method in registry
    custom_method = method_registry.get("generation", method)
    if custom_method:
        try:
            return custom_method(data, data_type=data_type, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        if method == "default":
            generator = EmbeddingGenerator(**kwargs)
            return generator.generate_embeddings(data, data_type=data_type, **kwargs)
        elif method == "text":
            return embed_text(data, **kwargs)
        elif method == "image":
            return embed_image(data, **kwargs)
        elif method == "audio":
            return embed_audio(data, **kwargs)
        else:
            raise ProcessingError(f"Unknown generation method: {method}")
            
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise


def embed_text(
    text: Union[str, List[str]],
    method: str = "sentence_transformers",
    **kwargs
) -> np.ndarray:
    """
    Generate text embeddings (convenience function).
    
    This is a user-friendly wrapper that generates text embeddings using the specified method.
    
    Args:
        text: Input text string or list of texts
        method: Text embedding method (default: "sentence_transformers")
            - "sentence_transformers": Sentence-transformers model-based embedding
            - "fallback": Hash-based fallback embedding
        **kwargs: Additional options passed to TextEmbedder
    
    Returns:
        np.ndarray: Text embeddings (1D for single text, 2D for batch)
        
    Examples:
        >>> from semantica.embeddings.methods import embed_text
        >>> emb = embed_text("Hello world", method="sentence_transformers")
        >>> embs = embed_text(["text1", "text2"], method="sentence_transformers")
    """
    # Check for custom method in registry
    custom_method = method_registry.get("text", method)
    if custom_method:
        try:
            return custom_method(text, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        # Get config
        config = embeddings_config.get_method_config("text")
        config.update(kwargs)
        
        embedder = TextEmbedder(**config)
        
        if isinstance(text, list):
            return embedder.embed_batch(text, **kwargs)
        else:
            return embedder.embed_text(text, **kwargs)
            
    except Exception as e:
        logger.error(f"Failed to embed text: {e}")
        raise


def embed_image(
    image_path: Union[str, Path, List[Union[str, Path]]],
    method: str = "clip",
    **kwargs
) -> np.ndarray:
    """
    Generate image embeddings (convenience function).
    
    This is a user-friendly wrapper that generates image embeddings using the specified method.
    
    Args:
        image_path: Path to image file or list of paths
        method: Image embedding method (default: "clip")
            - "clip": CLIP model-based embedding
            - "fallback": Feature-based fallback embedding
        **kwargs: Additional options passed to ImageEmbedder
    
    Returns:
        np.ndarray: Image embeddings (1D for single image, 2D for batch)
        
    Examples:
        >>> from semantica.embeddings.methods import embed_image
        >>> emb = embed_image("image.jpg", method="clip")
        >>> embs = embed_image(["img1.jpg", "img2.png"], method="clip")
    """
    # Check for custom method in registry
    custom_method = method_registry.get("image", method)
    if custom_method:
        try:
            return custom_method(image_path, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        # Get config
        config = embeddings_config.get_method_config("image")
        config.update(kwargs)
        
        embedder = ImageEmbedder(**config)
        
        if isinstance(image_path, list):
            return embedder.embed_batch(image_path, **kwargs)
        else:
            return embedder.embed_image(image_path, **kwargs)
            
    except Exception as e:
        logger.error(f"Failed to embed image: {e}")
        raise


def embed_audio(
    audio_path: Union[str, Path, List[Union[str, Path]]],
    method: str = "librosa",
    **kwargs
) -> np.ndarray:
    """
    Generate audio embeddings (convenience function).
    
    This is a user-friendly wrapper that generates audio embeddings using the specified method.
    
    Args:
        audio_path: Path to audio file or list of paths
        method: Audio embedding method (default: "librosa")
            - "librosa": Librosa feature extraction (MFCC, chroma, spectral contrast, tonnetz)
            - "fallback": Simple fallback embedding
        **kwargs: Additional options passed to AudioEmbedder
    
    Returns:
        np.ndarray: Audio embeddings (1D for single audio, 2D for batch)
        
    Examples:
        >>> from semantica.embeddings.methods import embed_audio
        >>> emb = embed_audio("audio.wav", method="librosa")
        >>> embs = embed_audio(["audio1.wav", "audio2.mp3"], method="librosa")
    """
    # Check for custom method in registry
    custom_method = method_registry.get("audio", method)
    if custom_method:
        try:
            return custom_method(audio_path, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        # Get config
        config = embeddings_config.get_method_config("audio")
        config.update(kwargs)
        
        embedder = AudioEmbedder(**config)
        
        if isinstance(audio_path, list):
            return embedder.embed_batch(audio_path, **kwargs)
        else:
            return embedder.embed_audio(audio_path, **kwargs)
            
    except Exception as e:
        logger.error(f"Failed to embed audio: {e}")
        raise


def embed_multimodal(
    text: Optional[str] = None,
    image_path: Optional[Union[str, Path]] = None,
    audio_path: Optional[Union[str, Path]] = None,
    method: str = "concat",
    **kwargs
) -> np.ndarray:
    """
    Generate multimodal embeddings (convenience function).
    
    This is a user-friendly wrapper that generates multimodal embeddings using the specified method.
    
    Args:
        text: Input text string (optional)
        image_path: Path to image file (optional)
        audio_path: Path to audio file (optional)
        method: Multimodal combination method (default: "concat")
            - "concat": Concatenation-based combination
            - "mean": Averaging-based combination
        **kwargs: Additional options passed to MultimodalEmbedder
    
    Returns:
        np.ndarray: Multimodal embedding vector
        
    Examples:
        >>> from semantica.embeddings.methods import embed_multimodal
        >>> emb = embed_multimodal(
        ...     text="A cat",
        ...     image_path="cat.jpg",
        ...     method="concat"
        ... )
    """
    # Check for custom method in registry
    custom_method = method_registry.get("multimodal", method)
    if custom_method:
        try:
            return custom_method(text, image_path, audio_path, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        # Get config
        config = embeddings_config.get_method_config("multimodal")
        config.update(kwargs)
        
        embedder = MultimodalEmbedder(**config)
        
        return embedder.embed_multimodal(
            text=text,
            image_path=image_path,
            audio_path=audio_path,
            combine_method=method,
            **kwargs
        )
            
    except Exception as e:
        logger.error(f"Failed to embed multimodal: {e}")
        raise


def optimize_embeddings(
    embeddings: np.ndarray,
    method: str = "pca",
    target_dim: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Optimize embeddings (convenience function).
    
    This is a user-friendly wrapper that optimizes embeddings using the specified method.
    
    Args:
        embeddings: Input embeddings array
        method: Optimization method (default: "pca")
            - "pca": Principal Component Analysis dimension reduction
            - "quantization": Bit-depth quantization
            - "truncate": Simple dimension truncation
        target_dim: Target dimension for reduction (required for pca/truncate)
        **kwargs: Additional options passed to EmbeddingOptimizer
    
    Returns:
        np.ndarray: Optimized embeddings
        
    Examples:
        >>> from semantica.embeddings.methods import optimize_embeddings
        >>> compressed = optimize_embeddings(embeddings, method="pca", target_dim=64)
        >>> quantized = optimize_embeddings(embeddings, method="quantization", bits=8)
    """
    # Check for custom method in registry
    custom_method = method_registry.get("optimization", method)
    if custom_method:
        try:
            return custom_method(embeddings, target_dim=target_dim, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        # Get config
        config = embeddings_config.get_method_config("optimization")
        config.update(kwargs)
        
        optimizer = EmbeddingOptimizer(**config)
        
        if method == "pca":
            if target_dim is None:
                raise ProcessingError("target_dim required for PCA method")
            return optimizer.compress(embeddings, target_dim=target_dim, method="pca", **kwargs)
        elif method == "quantization":
            return optimizer.compress(embeddings, method="quantization", **kwargs)
        elif method == "truncate":
            if target_dim is None:
                raise ProcessingError("target_dim required for truncate method")
            return optimizer.reduce_dimensions(embeddings, target_dim=target_dim, method="truncate", **kwargs)
        else:
            raise ProcessingError(f"Unknown optimization method: {method}")
            
    except Exception as e:
        logger.error(f"Failed to optimize embeddings: {e}")
        raise


def calculate_similarity(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    method: str = "cosine",
    **kwargs
) -> float:
    """
    Calculate similarity between embeddings (convenience function).
    
    This is a user-friendly wrapper that calculates similarity using the specified method.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        method: Similarity method (default: "cosine")
            - "cosine": Cosine similarity (dot product / norms)
            - "euclidean": Euclidean similarity (converted to 0-1 range)
        **kwargs: Additional options (unused)
    
    Returns:
        float: Similarity score (0-1 range)
        
    Examples:
        >>> from semantica.embeddings.methods import calculate_similarity
        >>> similarity = calculate_similarity(emb1, emb2, method="cosine")
        >>> print(f"Similarity: {similarity:.3f}")
    """
    # Check for custom method in registry
    custom_method = method_registry.get("similarity", method)
    if custom_method:
        try:
            return custom_method(embedding1, embedding2, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        generator = EmbeddingGenerator(**kwargs)
        return generator.compare_embeddings(embedding1, embedding2, method=method, **kwargs)
            
    except Exception as e:
        logger.error(f"Failed to calculate similarity: {e}")
        raise


def pool_embeddings(
    embeddings: np.ndarray,
    method: str = "mean",
    **kwargs
) -> np.ndarray:
    """
    Pool embeddings (convenience function).
    
    This is a user-friendly wrapper that pools embeddings using the specified strategy.
    
    Args:
        embeddings: Input embeddings array (n_embeddings, dim)
        method: Pooling strategy (default: "mean")
            - "mean": Mean pooling
            - "max": Max pooling
            - "cls": CLS token pooling
            - "attention": Attention-based pooling
            - "hierarchical": Hierarchical pooling
        **kwargs: Additional options passed to pooling strategy
    
    Returns:
        np.ndarray: Pooled embedding vector (dim,)
        
    Examples:
        >>> from semantica.embeddings.methods import pool_embeddings
        >>> pooled = pool_embeddings(embeddings, method="mean")
        >>> attention_pooled = pool_embeddings(embeddings, method="attention")
    """
    # Check for custom method in registry
    custom_method = method_registry.get("pooling", method)
    if custom_method:
        try:
            return custom_method(embeddings, **kwargs)
        except Exception as e:
            logger.warning(f"Custom method {method} failed: {e}, falling back to default")
    
    try:
        strategy = PoolingStrategyFactory.create(method, **kwargs)
        return strategy.pool(embeddings, **kwargs)
            
    except Exception as e:
        logger.error(f"Failed to pool embeddings: {e}")
        raise


def get_embedding_method(task: str, name: str) -> Optional[Callable]:
    """
    Get a registered embedding method.
    
    Args:
        task: Task type ("generation", "text", "image", "audio", "multimodal", "optimization", "pooling", "provider", "similarity")
        name: Method name
        
    Returns:
        Registered method or None if not found
        
    Examples:
        >>> from semantica.embeddings.methods import get_embedding_method
        >>> method = get_embedding_method("text", "custom_method")
        >>> if method:
        ...     result = method("Hello world")
    """
    return method_registry.get(task, name)


def list_available_methods(task: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List all available embedding methods.
    
    Args:
        task: Optional task type filter
        
    Returns:
        Dictionary mapping task types to method names
        
    Examples:
        >>> from semantica.embeddings.methods import list_available_methods
        >>> all_methods = list_available_methods()
        >>> text_methods = list_available_methods("text")
    """
    return method_registry.list_all(task)


# Register default methods
method_registry.register("generation", "default", generate_embeddings)
method_registry.register("generation", "text", generate_embeddings)
method_registry.register("generation", "image", generate_embeddings)
method_registry.register("generation", "audio", generate_embeddings)
method_registry.register("text", "sentence_transformers", embed_text)
method_registry.register("text", "fallback", embed_text)
method_registry.register("image", "clip", embed_image)
method_registry.register("image", "fallback", embed_image)
method_registry.register("audio", "librosa", embed_audio)
method_registry.register("audio", "fallback", embed_audio)
method_registry.register("multimodal", "concat", embed_multimodal)
method_registry.register("multimodal", "mean", embed_multimodal)
method_registry.register("optimization", "pca", optimize_embeddings)
method_registry.register("optimization", "quantization", optimize_embeddings)
method_registry.register("optimization", "truncate", optimize_embeddings)
method_registry.register("similarity", "cosine", calculate_similarity)
method_registry.register("similarity", "euclidean", calculate_similarity)
method_registry.register("pooling", "mean", pool_embeddings)
method_registry.register("pooling", "max", pool_embeddings)
method_registry.register("pooling", "cls", pool_embeddings)
method_registry.register("pooling", "attention", pool_embeddings)
method_registry.register("pooling", "hierarchical", pool_embeddings)

