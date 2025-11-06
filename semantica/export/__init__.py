"""
Export and Reporting Module

This module provides comprehensive export and reporting capabilities for the
Semantica framework, supporting multiple formats and use cases.

Key Features:
    - Multiple export formats (RDF, JSON, CSV, Graph, YAML, OWL, Vector)
    - Knowledge graph export
    - Report generation (HTML, Markdown, JSON, Text)
    - Vector store integration
    - Batch export processing
    - Metadata and provenance tracking

Main Classes:
    - RDFExporter: RDF format export (Turtle, RDF/XML, JSON-LD)
    - JSONExporter: JSON and JSON-LD format export
    - CSVExporter: CSV format export for tabular data
    - GraphExporter: Graph format export (GraphML, GEXF, DOT)
    - YAMLExporter: YAML format export for semantic networks
    - OWLExporter: OWL format export for ontologies
    - VectorExporter: Vector embedding export for vector stores
    - ReportGenerator: Report generation (HTML, Markdown, JSON, Text)

Example Usage:
    >>> from semantica.export import JSONExporter, CSVExporter
    >>> json_exporter = JSONExporter()
    >>> json_exporter.export_knowledge_graph(kg, "output.json")
    >>> csv_exporter = CSVExporter()
    >>> csv_exporter.export_entities(entities, "entities.csv")

Author: Semantica Contributors
License: MIT
"""

from .rdf_exporter import (
    RDFExporter,
    RDFSerializer,
    RDFValidator,
    NamespaceManager,
)
from .json_exporter import JSONExporter
from .csv_exporter import CSVExporter
from .graph_exporter import GraphExporter
from .yaml_exporter import (
    SemanticNetworkYAMLExporter,
    YAMLSchemaExporter,
)
from .report_generator import ReportGenerator
from .owl_exporter import OWLExporter
from .vector_exporter import VectorExporter

__all__ = [
    "RDFExporter",
    "RDFSerializer",
    "RDFValidator",
    "NamespaceManager",
    "JSONExporter",
    "CSVExporter",
    "GraphExporter",
    "SemanticNetworkYAMLExporter",
    "YAMLSchemaExporter",
    "ReportGenerator",
    "OWLExporter",
    "VectorExporter",
]
