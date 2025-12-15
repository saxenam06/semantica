"""
Triplet Store Methods Module

This module provides all triplet store methods as simple, reusable functions for
registering stores, adding triplets, querying, and managing triplet stores. It supports
multiple approaches and integrates with the method registry for extensibility.

Supported Methods:

Store Registration:
    - "default": Default store registration using TripletManager
    - "blazegraph": Blazegraph-specific registration
    - "jena": Jena-specific registration
    - "rdf4j": RDF4J-specific registration
    - "virtuoso": Virtuoso-specific registration

Triplet Addition:
    - "default": Default triplet addition
    - "single": Single triplet addition
    - "batch": Batch triplet addition
    - "bulk": Bulk triplet addition

Triplet Retrieval:
    - "default": Default triplet retrieval
    - "pattern": Pattern-based retrieval
    - "sparql": SPARQL-based retrieval

Triplet Deletion:
    - "default": Default triplet deletion
    - "single": Single triplet deletion
    - "batch": Batch triplet deletion

Triplet Update:
    - "default": Default triplet update (delete-then-add)
    - "atomic": Atomic update (when supported)

SPARQL Query:
    - "default": Default SPARQL query execution
    - "optimized": Optimized query execution
    - "cached": Cached query execution

Query Optimization:
    - "default": Default query optimization
    - "basic": Basic optimization (whitespace, LIMIT)
    - "advanced": Advanced optimization (cost-based)

Bulk Loading:
    - "default": Default bulk loading
    - "batch": Batch-based bulk loading
    - "stream": Stream-based bulk loading

Validation:
    - "default": Default validation
    - "triplet": Triplet structure validation
    - "pre_load": Pre-load validation

Algorithms Used:

Store Registration:
    - Store Type Detection: Backend type identification, adapter factory pattern
    - Configuration Management: Store configuration storage, default store selection
    - Adapter Instantiation: Backend-specific adapter creation, connection initialization

Triplet Operations:
    - Triplet Validation: Required field checking, confidence validation, URI validation
    - Batch Processing: Chunking algorithm, batch size optimization
    - Pattern Matching: Subject/predicate/object filtering, SPARQL query construction

SPARQL Query:
    - Query Validation: Syntax checking, query type detection
    - Query Optimization: Whitespace normalization, LIMIT injection, cost estimation
    - Query Caching: MD5-based cache key generation, LRU-style eviction
    - Result Processing: Binding extraction, variable extraction

Bulk Loading:
    - Progress Tracking: Load percentage calculation, elapsed time tracking, throughput calculation
    - Retry Mechanism: Exponential backoff, max retry limit
    - Stream Processing: Iterator-based processing, incremental batch collection

Key Features:
    - Multiple triplet store operation methods
    - Store registration with method dispatch
    - Method dispatchers with registry support
    - Custom method registration capability
    - Consistent interface across all methods

Main Functions:
    - register_store: Store registration wrapper
    - add_triplet: Triplet addition wrapper
    - add_triplets: Multiple triplet addition wrapper
    - get_triplets: Triplet retrieval wrapper
    - delete_triplet: Triplet deletion wrapper
    - update_triplet: Triplet update wrapper
    - execute_query: SPARQL query execution wrapper
    - optimize_query: Query optimization wrapper
    - bulk_load: Bulk loading wrapper
    - validate_triplets: Triplet validation wrapper
    - get_triplet_store_method: Get triplet store method by task and name
    - list_available_methods: List registered methods

Example Usage:
    >>> from semantica.triplet_store.methods import register_store, add_triplet, execute_query
    >>> store = register_store("main", "blazegraph", "http://localhost:9999/blazegraph", method="default")
    >>> result = add_triplet(triplet, store_id="main", method="default")
    >>> query_result = execute_query(sparql_query, store_adapter, method="default")
"""

from typing import Any, Dict, List, Optional, Union

from ..semantic_extract.triplet_extractor import Triplet
from .bulk_loader import BulkLoader, LoadProgress
from .config import triplet_store_config
from .query_engine import QueryEngine, QueryPlan, QueryResult
from .registry import method_registry
from .triplet_manager import TripletManager, TripletStore

# Global manager instances
_global_manager: Optional[TripletManager] = None
_global_query_engine: Optional[QueryEngine] = None
_global_bulk_loader: Optional[BulkLoader] = None


def _get_manager() -> TripletManager:
    """Get or create global TripletManager instance."""
    global _global_manager
    if _global_manager is None:
        config = triplet_store_config.get_all()
        _global_manager = TripletManager(config=config)
    return _global_manager


def _get_query_engine() -> QueryEngine:
    """Get or create global QueryEngine instance."""
    global _global_query_engine
    if _global_query_engine is None:
        config = triplet_store_config.get_all()
        _global_query_engine = QueryEngine(config=config)
    return _global_query_engine


def _get_bulk_loader() -> BulkLoader:
    """Get or create global BulkLoader instance."""
    global _global_bulk_loader
    if _global_bulk_loader is None:
        config = triplet_store_config.get_all()
        _global_bulk_loader = BulkLoader(config=config)
    return _global_bulk_loader


def register_store(
    store_id: str, store_type: str, endpoint: str, method: str = "default", **options
) -> TripletStore:
    """
    Register a triplet store.

    Args:
        store_id: Store identifier
        store_type: Store type (blazegraph, jena, rdf4j, virtuoso)
        endpoint: Store endpoint URL
        method: Registration method name (default: "default")
        **options: Additional options

    Returns:
        Registered store
    """
    # Check registry for custom method
    custom_method = method_registry.get("register", method)
    if custom_method:
        return custom_method(store_id, store_type, endpoint, **options)

    # Default implementation
    manager = _get_manager()
    return manager.register_store(store_id, store_type, endpoint, **options)


def add_triplet(
    triplet: Triplet, store_id: Optional[str] = None, method: str = "default", **options
) -> Dict[str, Any]:
    """
    Add single triplet to store.

    Args:
        triplet: Triplet to add
        store_id: Store identifier (uses default if not provided)
        method: Addition method name (default: "default")
        **options: Additional options

    Returns:
        Operation result
    """
    # Check registry for custom method
    custom_method = method_registry.get("add", method)
    if custom_method:
        return custom_method(triplet, store_id=store_id, **options)

    # Default implementation
    manager = _get_manager()
    return manager.add_triplet(triplet, store_id=store_id, **options)


def add_triplets(
    triplets: List[Triplet],
    store_id: Optional[str] = None,
    method: str = "default",
    **options,
) -> Dict[str, Any]:
    """
    Add multiple triplets to store.

    Args:
        triplets: List of triplets to add
        store_id: Store identifier
        method: Addition method name (default: "default")
        **options: Additional options

    Returns:
        Operation result
    """
    # Check registry for custom method
    custom_method = method_registry.get("add", method)
    if custom_method:
        return custom_method(triplets, store_id=store_id, **options)

    # Default implementation
    manager = _get_manager()
    return manager.add_triplets(triplets, store_id=store_id, **options)


def get_triplets(
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object: Optional[str] = None,
    store_id: Optional[str] = None,
    method: str = "default",
    **options,
) -> List[Triplet]:
    """
    Get triplets matching criteria.

    Args:
        subject: Optional subject URI
        predicate: Optional predicate URI
        object: Optional object URI
        store_id: Store identifier
        method: Retrieval method name (default: "default")
        **options: Additional options

    Returns:
        List of matching triplets
    """
    # Check registry for custom method
    custom_method = method_registry.get("get", method)
    if custom_method:
        return custom_method(subject, predicate, object, store_id=store_id, **options)

    # Default implementation
    manager = _get_manager()
    return manager.get_triplets(subject, predicate, object, store_id=store_id, **options)


def delete_triplet(
    triplet: Triplet, store_id: Optional[str] = None, method: str = "default", **options
) -> Dict[str, Any]:
    """
    Delete triplet from store.

    Args:
        triplet: Triplet to delete
        store_id: Store identifier
        method: Deletion method name (default: "default")
        **options: Additional options

    Returns:
        Operation result
    """
    # Check registry for custom method
    custom_method = method_registry.get("delete", method)
    if custom_method:
        return custom_method(triplet, store_id=store_id, **options)

    # Default implementation
    manager = _get_manager()
    return manager.delete_triplet(triplet, store_id=store_id, **options)


def update_triplet(
    old_triplet: Triplet,
    new_triplet: Triplet,
    store_id: Optional[str] = None,
    method: str = "default",
    **options,
) -> Dict[str, Any]:
    """
    Update triplet in store.

    Args:
        old_triplet: Original triplet
        new_triplet: Updated triplet
        store_id: Store identifier
        method: Update method name (default: "default")
        **options: Additional options

    Returns:
        Operation result
    """
    # Check registry for custom method
    custom_method = method_registry.get("update", method)
    if custom_method:
        return custom_method(old_triplet, new_triplet, store_id=store_id, **options)

    # Default implementation
    manager = _get_manager()
    return manager.update_triplet(old_triplet, new_triplet, store_id=store_id, **options)


def execute_query(
    query: str, store_adapter: Any, method: str = "default", **options
) -> QueryResult:
    """
    Execute SPARQL query.

    Args:
        query: SPARQL query string
        store_adapter: Triplet store adapter instance
        method: Query method name (default: "default")
        **options: Additional options

    Returns:
        Query result
    """
    # Check registry for custom method
    custom_method = method_registry.get("query", method)
    if custom_method:
        return custom_method(query, store_adapter, **options)

    # Default implementation
    engine = _get_query_engine()
    return engine.execute_query(query, store_adapter, **options)


def optimize_query(query: str, method: str = "default", **options) -> str:
    """
    Optimize SPARQL query.

    Args:
        query: Original SPARQL query
        method: Optimization method name (default: "default")
        **options: Additional options

    Returns:
        Optimized query
    """
    # Check registry for custom method
    custom_method = method_registry.get("optimize", method)
    if custom_method:
        return custom_method(query, **options)

    # Default implementation
    engine = _get_query_engine()
    return engine.optimize_query(query, **options)


def plan_query(query: str, **options) -> QueryPlan:
    """
    Create query execution plan.

    Args:
        query: SPARQL query
        **options: Planning options

    Returns:
        Query execution plan
    """
    engine = _get_query_engine()
    return engine.plan_query(query, **options)


def bulk_load(
    triplets: List[Triplet], store_adapter: Any, method: str = "default", **options
) -> LoadProgress:
    """
    Load triplets in bulk.

    Args:
        triplets: List of triplets to load
        store_adapter: Triplet store adapter instance
        method: Loading method name (default: "default")
        **options: Additional options

    Returns:
        Load progress information
    """
    # Check registry for custom method
    custom_method = method_registry.get("bulk_load", method)
    if custom_method:
        return custom_method(triplets, store_adapter, **options)

    # Default implementation
    loader = _get_bulk_loader()
    return loader.load_triplets(triplets, store_adapter, **options)


def validate_triplets(
    triplets: List[Triplet], method: str = "default", **options
) -> Dict[str, Any]:
    """
    Validate triplets before loading.

    Args:
        triplets: List of triplets to validate
        method: Validation method name (default: "default")
        **options: Validation options

    Returns:
        Validation results
    """
    # Check registry for custom method
    custom_method = method_registry.get("validate", method)
    if custom_method:
        return custom_method(triplets, **options)

    # Default implementation
    loader = _get_bulk_loader()
    return loader.validate_before_load(triplets, **options)


def get_triplet_store_method(task: str, method_name: str) -> Optional[Any]:
    """
    Get triplet store method by task and name.

    Args:
        task: Task type (register, add, get, delete, update, query, optimize, bulk_load, validate)
        method_name: Method name

    Returns:
        Method function or None if not found
    """
    return method_registry.get(task, method_name)


def list_available_methods(task: Optional[str] = None) -> Dict[str, List[str]]:
    """
    List all available triplet store methods.

    Args:
        task: Optional task type to filter by

    Returns:
        Dictionary mapping task types to lists of method names
    """
    return method_registry.list_all(task)
