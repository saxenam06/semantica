"""
YAML Exporter Module

This module provides comprehensive YAML export capabilities for the Semantica
framework, enabling human-readable export of semantic networks and ontologies.

Key Features:
    - Semantic network export to YAML
    - Ontology schema export for human editing
    - Pipeline-ready YAML format
    - Entity, relationship, and triple export
    - Class and property definition export

Example Usage:
    >>> from semantica.export import SemanticNetworkYAMLExporter
    >>> exporter = SemanticNetworkYAMLExporter()
    >>> exporter.export(semantic_network, "network.yaml")
    >>> exporter.export_for_pipeline(extracted_data, pipeline_stage=2)

Author: Semantica Contributors
License: MIT
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from datetime import datetime

from ..utils.exceptions import ValidationError, ProcessingError
from ..utils.logging import get_logger
from ..utils.helpers import ensure_directory


class SemanticNetworkYAMLExporter:
    """
    Exports semantic networks to YAML format.
    
    This class provides YAML export functionality for semantic networks, enabling
    human-readable representation and intermediate processing in ontology
    generation pipelines.
    
    Part of the 6-stage ontology generation pipeline:
    1. Document parsing
    2. Semantic network extraction (YAML) ← This module
    3. Definition generation
    4. Type mapping
    5. Hierarchy building
    6. TTL export
    
    Example Usage:
        >>> exporter = SemanticNetworkYAMLExporter()
        >>> exporter.export(semantic_network, "network.yaml")
    """
    
    def __init__(self, **config):
        """
        Initialize YAML exporter.
        
        Sets up the exporter with YAML serialization support.
        
        Args:
            **config: Configuration options (currently unused)
        
        Raises:
            ImportError: If PyYAML is not installed
        """
        self.logger = get_logger("yaml_exporter")
        self.config = config or {}
        
        try:
            import yaml
            self.yaml = yaml
        except ImportError:
            raise ImportError(
                "PyYAML not installed. Install with: pip install pyyaml"
            )
        
        self.logger.debug("Semantic network YAML exporter initialized")
    
    def export_semantic_network(
        self,
        semantic_network: Dict[str, Any],
        **options
    ) -> str:
        """
        Export semantic network to YAML string.
        
        This method converts a semantic network (entities, relationships, triples)
        to YAML format with metadata and provenance information.
        
        Args:
            semantic_network: Semantic network dictionary containing:
                - entities: List of entity dictionaries
                - relationships: List of relationship dictionaries
                - triples: List of triple dictionaries (optional)
                - metadata: Metadata dictionary (optional)
            **options: Additional export options (unused)
        
        Returns:
            String containing YAML representation of semantic network
        
        Example:
            >>> network = {
            ...     "entities": [...],
            ...     "relationships": [...],
            ...     "triples": [...]
            ... }
            >>> yaml_str = exporter.export_semantic_network(network)
        """
        yaml_data = {
            "metadata": {
                "exported_at": datetime.now().isoformat(),
                "version": "1.0",
                **semantic_network.get("metadata", {})
            },
            "entities": semantic_network.get("entities", []),
            "relationships": semantic_network.get("relationships", []),
            "triples": semantic_network.get("triples", [])
        }
        
        return self.yaml.dump(
            yaml_data,
            default_flow_style=False,
            sort_keys=False
        )
    
    def export(
        self,
        data: Dict[str, Any],
        file_path: Union[str, Path],
        **options
    ) -> None:
        """
        Export data to YAML file.
        
        Args:
            data: Data to export
            file_path: Output file path
            **options: Additional options
        """
        file_path = Path(file_path)
        ensure_directory(file_path.parent)
        
        yaml_content = self.export_semantic_network(data, **options)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        
        self.logger.info(f"Exported YAML to: {file_path}")
    
    def export_entities(self, entities: List[Dict[str, Any]], include_metadata: bool = True, **options) -> str:
        """
        Export entities to YAML format.
        
        • Format entity properties
        • Include entity types and labels
        • Add confidence scores
        • Return YAML representation
        """
        yaml_data = {
            "entities": entities
        }
        
        if include_metadata:
            yaml_data["metadata"] = {
                "exported_at": datetime.now().isoformat(),
                "entity_count": len(entities)
            }
        
        return self.yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    def export_relationships(self, relationships: List[Dict[str, Any]], include_properties: bool = True, **options) -> str:
        """
        Export relationships to YAML format.
        
        • Format relationship triples
        • Include relationship types
        • Add directional information
        • Return YAML representation
        """
        yaml_data = {
            "relationships": relationships
        }
        
        if include_properties:
            yaml_data["metadata"] = {
                "exported_at": datetime.now().isoformat(),
                "relationship_count": len(relationships)
            }
        
        return self.yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    def export_triples(self, triples: List[Dict[str, Any]], include_confidence: bool = True, **options) -> str:
        """
        Export RDF triples to YAML format.
        
        • Format subject-predicate-object triples
        • Include namespace information
        • Add confidence and provenance
        • Return YAML representation
        """
        yaml_data = {
            "triples": [
                {
                    "subject": t.get("subject") or t.get("s"),
                    "predicate": t.get("predicate") or t.get("p"),
                    "object": t.get("object") or t.get("o"),
                    **({"confidence": t.get("confidence")} if include_confidence and "confidence" in t else {}),
                    **({"provenance": t.get("provenance")} if "provenance" in t else {})
                }
                for t in triples
            ]
        }
        
        yaml_data["metadata"] = {
            "exported_at": datetime.now().isoformat(),
            "triple_count": len(triples)
        }
        
        return self.yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    def export_for_pipeline(self, extracted_data: Dict[str, Any], pipeline_stage: int = 2, **options) -> str:
        """
        Export in format suitable for ontology generation pipeline.
        
        • Format for stage 2 (semantic network extraction)
        • Structure for definition generation
        • Include extraction metadata
        • Return pipeline-ready YAML
        """
        yaml_data = {
            "pipeline_stage": pipeline_stage,
            "metadata": {
                "extracted_at": datetime.now().isoformat(),
                **extracted_data.get("metadata", {})
            },
            "semantic_network": {
                "entities": extracted_data.get("entities", []),
                "relationships": extracted_data.get("relationships", []),
                "triples": extracted_data.get("triples", [])
            }
        }
        
        return self.yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)


class YAMLSchemaExporter:
    """
    Exports ontology schemas to YAML for human editing.
    
    Enables domain expert refinement by exporting schemas in
    human-readable YAML format.
    """
    
    def __init__(self, **config):
        """Initialize schema exporter."""
        self.logger = get_logger("yaml_schema_exporter")
        self.config = config or {}
        
        try:
            import yaml
            self.yaml = yaml
        except ImportError:
            raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
    
    def export_ontology_schema(self, ontology: Dict[str, Any], **options) -> str:
        """
        Export ontology schema to YAML.
        
        • Format classes and properties
        • Include hierarchies and constraints
        • Structure for easy editing
        • Return YAML schema
        """
        yaml_data = {
            "ontology": {
                "uri": ontology.get("uri", ""),
                "title": ontology.get("title", ""),
                "description": ontology.get("description", ""),
                "version": ontology.get("version", "1.0")
            },
            "classes": ontology.get("classes", []),
            "properties": ontology.get("properties", []),
            "namespaces": ontology.get("namespaces", {})
        }
        
        return self.yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    def export_class_definitions(self, classes: List[Dict[str, Any]], include_hierarchy: bool = True, **options) -> str:
        """Export class definitions to YAML."""
        yaml_data = {
            "classes": classes
        }
        
        if include_hierarchy:
            yaml_data["hierarchy"] = self._extract_hierarchy(classes)
        
        return self.yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    def export_property_definitions(self, properties: List[Dict[str, Any]], include_domain_range: bool = True, **options) -> str:
        """Export property definitions to YAML."""
        yaml_data = {
            "properties": properties
        }
        
        if include_domain_range:
            yaml_data["domain_range"] = self._extract_domain_range(properties)
        
        return self.yaml.dump(yaml_data, default_flow_style=False, sort_keys=False)
    
    def _extract_hierarchy(self, classes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract class hierarchy."""
        hierarchy = {}
        
        for cls in classes:
            class_id = cls.get("id") or cls.get("uri", "")
            parent = cls.get("parent") or cls.get("subClassOf")
            
            if class_id:
                hierarchy[class_id] = {
                    "label": cls.get("label", ""),
                    "parent": parent,
                    "children": []
                }
        
        # Build children relationships
        for class_id, class_info in hierarchy.items():
            parent = class_info.get("parent")
            if parent and parent in hierarchy:
                hierarchy[parent]["children"].append(class_id)
        
        return hierarchy
    
    def _extract_domain_range(self, properties: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract property domain and range."""
        domain_range = {}
        
        for prop in properties:
            prop_id = prop.get("id") or prop.get("uri", "")
            if prop_id:
                domain_range[prop_id] = {
                    "label": prop.get("label", ""),
                    "domain": prop.get("domain", []),
                    "range": prop.get("range", [])
                }
        
        return domain_range
