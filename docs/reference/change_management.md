# Change Management API Reference

Comprehensive API documentation for the Enhanced Change Management module in Semantica.

## Overview

The `semantica.change_management` module provides enterprise-grade version control, audit trails, and compliance tracking for knowledge graphs and ontologies. It includes persistent storage backends, detailed change tracking, data integrity verification, and standardized metadata structures.

## Module Structure

```
semantica.change_management/
├── change_log.py              # Standardized metadata structures
├── version_storage.py         # Storage abstraction and implementations
├── managers.py                # Enhanced version managers
├── ontology_version_manager.py # Ontology version management
└── change_management_usage.md # Usage guide
```

## Quick Import

```python
from semantica.change_management import (
    # Metadata
    ChangeLogEntry,
    
    # Storage
    VersionStorage,
    InMemoryVersionStorage,
    SQLiteVersionStorage,
    
    # Utilities
    compute_checksum,
    verify_checksum,
    
    # Version Managers
    BaseVersionManager,
    TemporalVersionManager,
    OntologyVersionManager,
    VersionManager,
    OntologyVersion
)
```

---

## Core Classes

### ChangeLogEntry

Standardized metadata structure for version changes with validation.

#### Class Definition

```python
@dataclass
class ChangeLogEntry:
    """
    Standardized change log entry with validation.
    
    Attributes:
        timestamp: ISO 8601 formatted timestamp
        author: Email address of the change author
        description: Change description (max 500 characters)
        change_id: Optional ID linking to external systems
    """
    timestamp: str
    author: str
    description: str
    change_id: Optional[str] = None
```

#### Methods

##### `__post_init__()`

Validates all fields after initialization.

**Raises:**
- `ValidationError`: If any field validation fails

**Example:**
```python
entry = ChangeLogEntry(
    timestamp="2024-01-30T12:00:00Z",
    author="user@example.com",
    description="Updated entity relationships",
    change_id="TICKET-123"
)
```

##### `create_now(author, description, change_id=None)` (classmethod)

Creates a change log entry with the current timestamp.

**Parameters:**
- `author` (str): Email address of the change author
- `description` (str): Change description (max 500 characters)
- `change_id` (str, optional): ID linking to external systems

**Returns:**
- `ChangeLogEntry`: New instance with current timestamp

**Example:**
```python
entry = ChangeLogEntry.create_now(
    author="developer@company.com",
    description="Fixed entity resolution bug",
    change_id="JIRA-1234"
)
```

#### Validation Rules

- **Timestamp**: Must be valid ISO 8601 format with 'T' separator
- **Author**: Must be valid email format (RFC 5322)
- **Description**: Maximum 500 characters
- **Change ID**: Optional, no validation

---

### VersionStorage

Abstract base class for storage implementations.

#### Class Definition

```python
class VersionStorage(ABC):
    """
    Abstract base class for version storage backends.
    
    Provides interface for saving, retrieving, and managing version snapshots.
    """
```

#### Abstract Methods

##### `save(snapshot)`

Save a version snapshot.

**Parameters:**
- `snapshot` (Dict[str, Any]): Version snapshot dictionary with metadata

**Raises:**
- `ValidationError`: If snapshot data is invalid
- `ProcessingError`: If save operation fails

**Example:**
```python
snapshot = {
    "label": "v1.0",
    "timestamp": "2024-01-30T12:00:00Z",
    "author": "user@example.com",
    "description": "Initial version",
    "data": {...}
}
storage.save(snapshot)
```

##### `get(label)`

Retrieve a version snapshot by label.

**Parameters:**
- `label` (str): Version label to retrieve

**Returns:**
- `Optional[Dict[str, Any]]`: Snapshot dictionary or None if not found

**Example:**
```python
snapshot = storage.get("v1.0")
if snapshot:
    print(f"Retrieved: {snapshot['label']}")
```

##### `list_all()`

List all version snapshots.

**Returns:**
- `List[Dict[str, Any]]`: List of snapshot metadata dictionaries

**Example:**
```python
versions = storage.list_all()
for v in versions:
    print(f"{v['label']}: {v['description']}")
```

##### `exists(label)`

Check if a version exists.

**Parameters:**
- `label` (str): Version label to check

**Returns:**
- `bool`: True if version exists, False otherwise

**Example:**
```python
if storage.exists("v1.0"):
    print("Version exists")
```

##### `delete(label)`

Delete a version snapshot.

**Parameters:**
- `label` (str): Version label to delete

**Returns:**
- `bool`: True if deleted, False if not found

**Example:**
```python
if storage.delete("v1.0"):
    print("Version deleted")
```

---

### InMemoryVersionStorage

In-memory version storage implementation.

#### Class Definition

```python
class InMemoryVersionStorage(VersionStorage):
    """
    In-memory version storage implementation.
    
    Fast, volatile storage for development and testing.
    Data is lost when the process ends.
    """
```

#### Constructor

```python
def __init__(self):
    """Initialize in-memory storage."""
```

**Example:**
```python
storage = InMemoryVersionStorage()
```

#### Performance Characteristics

- **Save**: 0.37-16ms (10-1000 entities)
- **Get**: 0.20-16ms (10-1000 entities)
- **List**: <0.03ms
- **Thread-safe**: Yes (uses RLock)

#### Use Cases

- Development and testing
- Temporary version tracking
- High-performance scenarios where persistence is not required

---

### SQLiteVersionStorage

SQLite-based persistent version storage implementation.

#### Class Definition

```python
class SQLiteVersionStorage(VersionStorage):
    """
    SQLite-based persistent version storage implementation.
    
    Provides persistence across process restarts with ACID guarantees.
    """
```

#### Constructor

```python
def __init__(self, storage_path: str):
    """
    Initialize SQLite storage.
    
    Args:
        storage_path: Path to SQLite database file
    """
```

**Parameters:**
- `storage_path` (str): Path to SQLite database file (created if doesn't exist)

**Example:**
```python
storage = SQLiteVersionStorage("versions.db")
```

#### Database Schema

```sql
CREATE TABLE versions (
    label TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    author TEXT NOT NULL,
    description TEXT,
    checksum TEXT,
    snapshot_data TEXT NOT NULL,
    created_at TEXT NOT NULL
)
```

#### Performance Characteristics

- **Save**: 7-25ms (10-1000 entities)
- **Get**: 2-8ms (10-1000 entities)
- **List**: 0.6-13ms
- **Thread-safe**: Yes (uses RLock)
- **ACID**: Full transaction support

#### Use Cases

- Production deployments
- Long-term version storage
- Compliance and audit requirements
- Multi-process environments

---

### BaseVersionManager

Abstract base class for version managers.

#### Class Definition

```python
class BaseVersionManager(ABC):
    """
    Abstract base class for version managers.
    
    Provides common functionality for version management across
    different data types (knowledge graphs, ontologies, etc.).
    """
```

#### Constructor

```python
def __init__(self, storage_path: Optional[str] = None):
    """
    Initialize base version manager.
    
    Args:
        storage_path: Path to SQLite database file for persistent storage.
                     If None, uses in-memory storage.
    """
```

**Parameters:**
- `storage_path` (str, optional): Path to SQLite database file

**Example:**
```python
# In-memory storage
manager = BaseVersionManager()

# Persistent storage
manager = BaseVersionManager(storage_path="versions.db")
```

#### Abstract Methods

##### `create_snapshot(data, version_label, author, description, **options)`

Create a versioned snapshot of the data.

**Parameters:**
- `data` (Any): Data to snapshot
- `version_label` (str): Version label
- `author` (str): Email address of the author
- `description` (str): Change description
- `**options`: Additional options

**Returns:**
- `Dict[str, Any]`: Snapshot with metadata and checksum

##### `compare_versions(version1, version2, **options)`

Compare two versions and return detailed differences.

**Parameters:**
- `version1` (Any): First version (label or snapshot)
- `version2` (Any): Second version (label or snapshot)
- `**options`: Comparison options

**Returns:**
- `Dict[str, Any]`: Detailed differences

#### Concrete Methods

##### `list_versions()`

List all version snapshots.

**Returns:**
- `List[Dict[str, Any]]`: List of version metadata

**Example:**
```python
versions = manager.list_versions()
for v in versions:
    print(f"{v['label']}: {v['description']}")
```

##### `get_version(label)`

Retrieve specific version by label.

**Parameters:**
- `label` (str): Version label

**Returns:**
- `Optional[Dict[str, Any]]`: Version snapshot or None

**Example:**
```python
version = manager.get_version("v1.0")
```

##### `verify_checksum(snapshot)`

Verify data integrity using checksum.

**Parameters:**
- `snapshot` (Dict[str, Any]): Snapshot to verify

**Returns:**
- `bool`: True if checksum is valid

**Example:**
```python
is_valid = manager.verify_checksum(snapshot)
```

---

### TemporalVersionManager

Enhanced temporal version management engine for knowledge graphs.

#### Class Definition

```python
class TemporalVersionManager(BaseVersionManager):
    """
    Enhanced temporal version management engine for knowledge graphs.
    
    Features:
        - Persistent snapshot storage (SQLite or in-memory)
        - Detailed change tracking with entity-level diffs
        - SHA-256 checksums for data integrity
        - Standardized metadata with author attribution
        - Version comparison with backward compatibility
        - Input validation and security features
    """
```

#### Constructor

```python
def __init__(self, storage_path: Optional[str] = None, **config):
    """
    Initialize enhanced temporal version manager.
    
    Args:
        storage_path: Path to SQLite database file for persistent storage.
                     If None, uses in-memory storage
        **config: Additional configuration options
    """
```

**Parameters:**
- `storage_path` (str, optional): Path to SQLite database file
- `**config`: Additional configuration options

**Example:**
```python
# In-memory storage
manager = TemporalVersionManager()

# Persistent storage
manager = TemporalVersionManager(storage_path="kg_versions.db")
```

#### Methods

##### `create_snapshot(graph, version_label, author, description, **options)`

Create and store snapshot with checksum and metadata.

**Parameters:**
- `graph` (Dict[str, Any]): Knowledge graph dict with "entities" and "relationships"
- `version_label` (str): Version string (e.g., "v1.0")
- `author` (str): Email address of the change author
- `description` (str): Change description (max 500 chars)
- `**options`: Additional options

**Returns:**
- `Dict[str, Any]`: Snapshot with metadata and checksum

**Raises:**
- `ValidationError`: If input validation fails
- `ProcessingError`: If snapshot creation fails

**Example:**
```python
graph = {
    "entities": [
        {"id": "e1", "name": "Entity 1", "type": "Person"},
        {"id": "e2", "name": "Entity 2", "type": "Organization"}
    ],
    "relationships": [
        {"source": "e1", "target": "e2", "type": "works_for"}
    ]
}

snapshot = manager.create_snapshot(
    graph,
    version_label="v1.0",
    author="user@example.com",
    description="Initial knowledge graph"
)

print(f"Created: {snapshot['label']}")
print(f"Checksum: {snapshot['checksum']}")
```

##### `compare_versions(version1, version2, **options)`

Compare two versions with detailed entity and relationship diffs.

**Parameters:**
- `version1` (Union[str, Dict]): First version (label or snapshot dict)
- `version2` (Union[str, Dict]): Second version (label or snapshot dict)
- `**options`: Comparison options

**Returns:**
- `Dict[str, Any]`: Detailed differences including:
  - `summary`: Aggregate statistics
  - `entity_changes`: Entity-level changes
  - `relationship_changes`: Relationship-level changes

**Example:**
```python
diff = manager.compare_versions("v1.0", "v2.0")

print(f"Entities added: {diff['summary']['entities_added']}")
print(f"Entities modified: {diff['summary']['entities_modified']}")
print(f"Relationships added: {diff['summary']['relationships_added']}")

# Detailed entity changes
for entity_id, changes in diff['entity_changes'].items():
    print(f"Entity {entity_id}: {changes['status']}")
    if changes['status'] == 'modified':
        print(f"  Before: {changes['before']}")
        print(f"  After: {changes['after']}")
```

#### Performance

- **Snapshot Creation**: 1.40-54ms (50-2000 entities)
- **Version Retrieval**: 0.65-26ms (50-2000 entities)
- **Version Comparison**: 3.46-33ms (100-1000 entities)
- **Concurrent Throughput**: 500+ operations/second

---

### OntologyVersionManager

Enhanced version management for ontologies.

#### Class Definition

```python
class OntologyVersionManager(BaseVersionManager):
    """
    Enhanced version management for ontologies.
    
    Features:
        - Persistent ontology snapshot storage
        - Structural comparison (classes, properties, axioms)
        - SHA-256 checksums for data integrity
        - Standardized metadata with author attribution
    """
```

#### Constructor

```python
def __init__(self, storage_path: Optional[str] = None, **config):
    """
    Initialize enhanced version manager for ontologies.
    
    Args:
        storage_path: Path to SQLite database file for persistent storage.
                     If None, uses in-memory storage
        **config: Additional configuration options
    """
```

**Example:**
```python
manager = OntologyVersionManager(storage_path="ontology_versions.db")
```

#### Methods

##### `create_snapshot(ontology, version_label, author, description, **options)`

Create ontology snapshot with metadata.

**Parameters:**
- `ontology` (Dict[str, Any]): Ontology dict with structure information
- `version_label` (str): Version label
- `author` (str): Email address of the author
- `description` (str): Change description
- `**options`: Additional options

**Returns:**
- `Dict[str, Any]`: Ontology snapshot with metadata

**Example:**
```python
ontology = {
    "uri": "https://example.com/ontology",
    "version_info": {"version": "1.0", "date": "2024-01-30"},
    "structure": {
        "classes": ["Person", "Organization", "Location"],
        "properties": ["name", "address", "email"],
        "individuals": ["JohnDoe", "ACME_Corp"],
        "axioms": ["Person hasAddress exactly 1 Location"]
    }
}

snapshot = manager.create_snapshot(
    ontology,
    version_label="ont_v1.0",
    author="architect@example.com",
    description="Initial ontology design"
)
```

##### `compare_versions(version1, version2, **options)`

Compare ontology versions with structural analysis.

**Parameters:**
- `version1` (Union[str, Dict]): First version
- `version2` (Union[str, Dict]): Second version
- `**options`: Comparison options

**Returns:**
- `Dict[str, Any]`: Structural differences including:
  - `classes_added`, `classes_removed`
  - `properties_added`, `properties_removed`
  - `individuals_added`, `individuals_removed`
  - `axioms_added`, `axioms_removed`, `axioms_modified`

**Example:**
```python
diff = manager.compare_versions("ont_v1.0", "ont_v2.0")

print(f"Classes added: {diff['classes_added']}")
print(f"Properties added: {diff['properties_added']}")
print(f"Axioms modified: {diff['axioms_modified']}")
```

---

## Utility Functions

### compute_checksum

Compute SHA-256 checksum for data integrity.

#### Function Signature

```python
def compute_checksum(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 checksum for data.
    
    Args:
        data: Dictionary to compute checksum for
        
    Returns:
        SHA-256 checksum as hexadecimal string
    """
```

**Parameters:**
- `data` (Dict[str, Any]): Dictionary to compute checksum for

**Returns:**
- `str`: SHA-256 checksum as hexadecimal string

**Example:**
```python
from semantica.change_management import compute_checksum

data = {"entities": [...], "relationships": [...]}
checksum = compute_checksum(data)
print(f"Checksum: {checksum}")
```

**Performance:** 1.29-110ms (100-10,000 entities)

---

### verify_checksum

Verify data integrity using stored checksum.

#### Function Signature

```python
def verify_checksum(snapshot: Dict[str, Any]) -> bool:
    """
    Verify data integrity using checksum.
    
    Args:
        snapshot: Snapshot dictionary with 'checksum' field
        
    Returns:
        True if checksum is valid, False otherwise
    """
```

**Parameters:**
- `snapshot` (Dict[str, Any]): Snapshot dictionary with 'checksum' field

**Returns:**
- `bool`: True if checksum is valid, False otherwise

**Example:**
```python
from semantica.change_management import verify_checksum

snapshot = manager.get_version("v1.0")
is_valid = verify_checksum(snapshot)

if not is_valid:
    print("WARNING: Data integrity compromised!")
```

**Performance:** 0.82-96ms (100-10,000 entities)

---

## Legacy Classes

### VersionManager

Original ontology version manager (moved from `semantica.ontology`).

#### Import

```python
from semantica.change_management import VersionManager, OntologyVersion
```

**Note:** This class is maintained for backward compatibility. New projects should use `OntologyVersionManager`.

---

## Error Handling

### ValidationError

Raised when input validation fails.

**Common Causes:**
- Invalid email format
- Description exceeds 500 characters
- Invalid ISO 8601 timestamp
- Missing required fields

**Example:**
```python
from semantica.utils.exceptions import ValidationError

try:
    entry = ChangeLogEntry(
        timestamp="invalid",
        author="not-an-email",
        description="x" * 501
    )
except ValidationError as e:
    print(f"Validation failed: {e}")
```

### ProcessingError

Raised when operations fail.

**Common Causes:**
- Database connection issues
- File system errors
- Concurrent modification conflicts

**Example:**
```python
from semantica.utils.exceptions import ProcessingError

try:
    storage.save(snapshot)
except ProcessingError as e:
    print(f"Save failed: {e}")
```

---

## Performance Considerations

### Benchmarks

Based on comprehensive performance testing:

| Component | Small (100) | Medium (500) | Large (2000) |
|-----------|-------------|--------------|--------------|
| Snapshot Creation | 2.33ms | 10.70ms | 54.23ms |
| Version Retrieval | 1.88ms | 7.33ms | 26.04ms |
| Version Comparison | 3.46ms | 17.39ms | 32.83ms |
| Checksum Compute | 1.29ms | 5.48ms | 22.15ms |
| SQLite Save | 8.69ms | 13.37ms | 25.33ms |
| InMemory Save | 1.18ms | 10.60ms | 14.11ms |

### Optimization Tips

1. **Use appropriate storage backend:**
   - Development: `InMemoryVersionStorage`
   - Production: `SQLiteVersionStorage`

2. **Batch operations when possible:**
   ```python
   for data in batch:
       manager.create_snapshot(data, ...)
   ```

3. **Implement retention policies:**
   ```python
   # Delete old versions periodically
   for version in old_versions:
       storage.delete(version['label'])
   ```

4. **Use concurrent operations:**
   - Thread-safe: 500+ operations/second
   - No performance degradation under load

---

## Compliance Features

### HIPAA Compliance

- Complete audit trails with author attribution
- Timestamp tracking for all changes
- Data integrity verification with checksums
- Secure storage with access controls

### SOX Compliance

- Immutable change records
- Detailed change descriptions
- External system linking (change IDs)
- Comprehensive audit reports

### FDA 21 CFR Part 11

- Electronic signatures (author email)
- Data integrity verification
- Audit trail generation
- Tamper detection

---

## Examples

### Complete Healthcare Example

```python
from semantica.change_management import TemporalVersionManager

# Initialize with HIPAA-compliant storage
manager = TemporalVersionManager(storage_path="hipaa_records.db")

# Patient knowledge graph
patient_kg = {
    "entities": [
        {"id": "patient_001", "type": "Patient", "name": "Jane Smith"},
        {"id": "diagnosis_001", "type": "Diagnosis", "code": "I10"}
    ],
    "relationships": [
        {"source": "patient_001", "target": "diagnosis_001", "type": "has_diagnosis"}
    ]
}

# Create versioned record
snapshot = manager.create_snapshot(
    patient_kg,
    "patient_001_v1.0",
    "dr.williams@hospital.com",
    "Initial diagnosis - Essential hypertension"
)

# Verify integrity
assert manager.verify_checksum(snapshot), "Data integrity check failed"

# Generate audit report
for version in manager.list_versions():
    print(f"{version['timestamp']}: {version['label']} by {version['author']}")
```

---

## See Also

- **Usage Guide**: `semantica/change_management/change_management_usage.md`
- **Performance Tests**: `tests/change_management/test_performance.py`
- **CHANGELOG**: `CHANGELOG.md`
- **GitHub**: https://github.com/Hawksight-AI/semantica
