"""
Triplet Manager Module

This module provides comprehensive CRUD operations for RDF triplets and triplet
store management, enabling unified access to multiple triplet store backends
through a common interface.

Key Features:
    - CRUD operations for RDF triplets
    - Multi-store management and registration
    - Batch operations and bulk loading
    - Triplet validation and consistency
    - Store adapter pattern
    - Error handling and recovery

Main Classes:
    - TripletManager: Main triplet store management coordinator
    - TripletStore: Triplet store configuration dataclass

Example Usage:
    >>> from semantica.triplet_store import TripletManager
    >>> manager = TripletManager()
    >>> store = manager.register_store("main", "blazegraph", "http://localhost:9999/blazegraph")
    >>> result = manager.add_triplet(triplet, store_id="main")
    >>> triplets = manager.get_triplets(subject="http://example.org/entity1")
    >>> result = manager.add_triplets(triplet_list, store_id="main", batch_size=1000)

Author: Semantica Contributors
License: MIT
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..semantic_extract.triplet_extractor import Triplet
from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.logging import get_logger
from ..utils.progress_tracker import get_progress_tracker


@dataclass
class TripletStore:
    """Triplet store configuration."""

    store_id: str
    store_type: str  # "blazegraph", "jena", "rdf4j", "virtuoso"
    endpoint: str
    config: Dict[str, Any] = field(default_factory=dict)
    connected: bool = False


class TripletManager:
    """
    Triplet store management system.

    • CRUD operations for RDF triplets
    • Batch operations and bulk loading
    • Triplet validation and consistency
    • Transaction support
    • Performance optimization
    • Error handling and recovery
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """
        Initialize triplet manager.

        Args:
            config: Configuration dictionary
            **kwargs: Additional configuration options:
                - default_store: Default triplet store to use
        """
        self.logger = get_logger("triplet_manager")
        self.config = config or {}
        self.config.update(kwargs)
        self.progress_tracker = get_progress_tracker()

        self.stores: Dict[str, TripletStore] = {}
        self.default_store_id = self.config.get("default_store")

    def register_store(
        self, store_id: str, store_type: str, endpoint: str, **config
    ) -> TripletStore:
        """
        Register a triplet store.

        Args:
            store_id: Store identifier
            store_type: Store type
            endpoint: Store endpoint URL
            **config: Additional configuration

        Returns:
            Registered store
        """
        store = TripletStore(
            store_id=store_id, store_type=store_type, endpoint=endpoint, config=config
        )

        self.stores[store_id] = store

        if not self.default_store_id:
            self.default_store_id = store_id

        self.logger.info(f"Registered triplet store: {store_id} ({store_type})")

        return store

    def add_triplet(
        self, triplet: Triplet, store_id: Optional[str] = None, **options
    ) -> Dict[str, Any]:
        """
        Add single triplet to store.

        Args:
            triplet: Triplet to add
            store_id: Store identifier (uses default if not provided)
            **options: Additional options

        Returns:
            Operation status
        """
        store = self._get_store(store_id)

        # Validate triplet
        if not self._validate_triplet(triplet):
            raise ValidationError("Invalid triplet")

        # Add to store (delegates to adapter)
        try:
            adapter = self._get_adapter(store)
            result = adapter.add_triplet(triplet, **options)

            return {
                "success": True,
                "store_id": store.store_id,
                "triplet": triplet,
                **result,
            }
        except Exception as e:
            self.logger.error(f"Failed to add triplet: {e}")
            raise ProcessingError(f"Failed to add triplet: {e}")

    def add_triplets(
        self, triplets: List[Triplet], store_id: Optional[str] = None, **options
    ) -> Dict[str, Any]:
        """
        Add multiple triplets to store.

        Args:
            triplets: List of triplets
            store_id: Store identifier
            **options: Additional options:
                - batch_size: Batch size for bulk operations
                - validate: Validate triplets before adding

        Returns:
            Operation status
        """
        tracking_id = self.progress_tracker.start_tracking(
            module="triplet_store",
            submodule="TripletManager",
            message=f"Adding {len(triplets)} triplets to store",
        )

        try:
            store = self._get_store(store_id)

            # Validate triplets
            if options.get("validate", True):
                self.progress_tracker.update_tracking(
                    tracking_id, message="Validating triplets..."
                )
                valid_triplets = [t for t in triplets if self._validate_triplet(t)]
                invalid_count = len(triplets) - len(valid_triplets)
                if invalid_count > 0:
                    self.logger.warning(f"{invalid_count} invalid triplets filtered out")
            else:
                valid_triplets = triplets

            # Add to store
            adapter = self._get_adapter(store)
            batch_size = options.get("batch_size", 1000)
            total_batches = (len(valid_triplets) + batch_size - 1) // batch_size

            results = []
            for i in range(0, len(valid_triplets), batch_size):
                batch_num = i // batch_size + 1
                self.progress_tracker.update_tracking(
                    tracking_id,
                    message=f"Processing batch {batch_num}/{total_batches}...",
                )
                batch = valid_triplets[i : i + batch_size]
                result = adapter.add_triplets(batch, **options)
                results.append(result)

            self.progress_tracker.stop_tracking(
                tracking_id,
                status="completed",
                message=f"Added {len(valid_triplets)} triplets in {len(results)} batches",
            )
            return {
                "success": True,
                "store_id": store.store_id,
                "total_triplets": len(valid_triplets),
                "batches": len(results),
                "results": results,
            }
        except Exception as e:
            self.logger.error(f"Failed to add triplets: {e}")
            self.progress_tracker.stop_tracking(
                tracking_id, status="failed", message=str(e)
            )
            raise ProcessingError(f"Failed to add triplets: {e}")

    def get_triplets(
        self,
        subject: str,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        store_id: Optional[str] = None,
        **options,
    ) -> List[Triplet]:
        """
        Get triplets matching criteria.

        Args:
            subject: Subject URI
            predicate: Optional predicate URI
            object: Optional object URI
            store_id: Store identifier
            **options: Additional options

        Returns:
            List of matching triplets
        """
        store = self._get_store(store_id)

        try:
            adapter = self._get_adapter(store)
            return adapter.get_triplets(subject, predicate, object, **options)
        except Exception as e:
            self.logger.error(f"Failed to get triplets: {e}")
            raise ProcessingError(f"Failed to get triplets: {e}")

    def delete_triplet(
        self, triplet: Triplet, store_id: Optional[str] = None, **options
    ) -> Dict[str, Any]:
        """
        Delete triplet from store.

        Args:
            triplet: Triplet to delete
            store_id: Store identifier
            **options: Additional options

        Returns:
            Operation status
        """
        store = self._get_store(store_id)

        try:
            adapter = self._get_adapter(store)
            result = adapter.delete_triplet(triplet, **options)

            return {"success": True, "store_id": store.store_id, **result}
        except Exception as e:
            self.logger.error(f"Failed to delete triplet: {e}")
            raise ProcessingError(f"Failed to delete triplet: {e}")

    def update_triplet(
        self,
        old_triplet: Triplet,
        new_triplet: Triplet,
        store_id: Optional[str] = None,
        **options,
    ) -> Dict[str, Any]:
        """
        Update triplet in store.

        Args:
            old_triplet: Original triplet
            new_triplet: Updated triplet
            store_id: Store identifier
            **options: Additional options

        Returns:
            Operation status
        """
        # Delete old and add new
        self.delete_triplet(old_triplet, store_id, **options)
        return self.add_triplet(new_triplet, store_id, **options)

    def _validate_triplet(self, triplet: Triplet) -> bool:
        """Validate triplet structure."""
        if not triplet.subject or not triplet.predicate or not triplet.object:
            return False

        if triplet.confidence < 0 or triplet.confidence > 1:
            return False

        return True

    def _get_store(self, store_id: Optional[str] = None) -> TripletStore:
        """Get store by ID."""
        store_id = store_id or self.default_store_id

        if not store_id:
            raise ValidationError("No store specified and no default store configured")

        if store_id not in self.stores:
            raise ValidationError(f"Store not found: {store_id}")

        return self.stores[store_id]

    def _get_adapter(self, store: TripletStore) -> Any:
        """Get adapter for store type."""
        store_type = store.store_type.lower()

        if store_type == "blazegraph":
            from .blazegraph_adapter import BlazegraphAdapter

            return BlazegraphAdapter(endpoint=store.endpoint, **store.config)
        elif store_type == "jena":
            from .jena_adapter import JenaAdapter

            return JenaAdapter(**store.config)
        elif store_type == "rdf4j":
            from .rdf4j_adapter import RDF4JAdapter

            return RDF4JAdapter(endpoint=store.endpoint, **store.config)
        elif store_type == "virtuoso":
            from .virtuoso_adapter import VirtuosoAdapter

            return VirtuosoAdapter(endpoint=store.endpoint, **store.config)
        else:
            raise ValidationError(f"Unsupported store type: {store_type}")

    def get_store(self, store_id: str) -> Optional[TripletStore]:
        """Get store by ID."""
        return self.stores.get(store_id)

    def list_stores(self) -> List[str]:
        """List all store IDs."""
        return list(self.stores.keys())
