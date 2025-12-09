# Core

> **Framework infrastructure, lifecycle management, and plugin system.**

---

## 🎯 Overview

<div class="grid cards" markdown>

-   :material-cogs:{ .lg .middle } **Semantica**

    ---

    Main framework class coordinating all components and workflows

-   :material-lifecycle:{ .lg .middle } **Lifecycle Management**

    ---

    Manage initialization, startup, shutdown, and state transitions

-   :material-tune:{ .lg .middle } **Configuration**

    ---

    Unified configuration management via YAML and Environment variables

-   :material-puzzle:{ .lg .middle } **Plugin System**

    ---

    Extensible plugin registry for adding custom modules and capabilities

-   :material-console:{ .lg .middle } **Method Registry**

    ---

    Registry for custom orchestration methods and extensibility

</div>

!!! tip "When to Use"
    - **Application Startup**: Initializing the Semantica framework in your app
    - **Configuration**: Tuning global settings
    - **Extension**: Developing custom plugins or modules
    - **Orchestration**: Coordinating complex workflows across multiple modules

---

## ⚙️ Algorithms Used

### Lifecycle Management
- **State Machine**: `UNINITIALIZED` -> `INITIALIZING` -> `READY` -> `RUNNING` -> `STOPPING` -> `STOPPED`
- **Priority-based Hooks**: Startup and shutdown hooks executed in priority order (lower = earlier)
- **Graceful Shutdown**: Ensuring all resources (DB connections, thread pools) are closed properly

### Configuration
- **Layered Loading**: Defaults -> Config File -> Environment Variables (Priority order)
- **Schema Validation**: Validating config structure against defined schemas
- **Nested Access**: Dot notation for accessing nested configuration values

### Plugin System
- **Discovery**: Auto-discovery of plugins via directory scanning
- **Registration**: Dynamic registration of classes and functions
- **Dependency Resolution**: Automatic loading of plugin dependencies

---

## Main Classes

### Semantica

The main framework class that coordinates all components.

**Methods:**

| Method | Description |
|--------|-------------|
| `__init__(config=None, **kwargs)` | Initialize framework with optional configuration |
| `initialize()` | Initialize all framework components |
| `build_knowledge_base(sources, **kwargs)` | Build knowledge base from data sources |
| `run_pipeline(pipeline, data)` | Execute a processing pipeline |
| `get_status()` | Get system health and status |
| `shutdown(graceful=True)` | Shutdown the framework gracefully |

**Example:**

```python
from semantica.core import Semantica

# Initialize framework
framework = Semantica()
framework.initialize()

# Build knowledge base
result = framework.build_knowledge_base(
    sources=["doc1.pdf", "doc2.docx"],
    embeddings=True,
    graph=True
)

# Check status
status = framework.get_status()
print(f"System state: {status['state']}")

# Shutdown
framework.shutdown()
```

### ConfigManager

Manages global configuration loading, validation, and merging.

**Methods:**

| Method | Description |
|--------|-------------|
| `load_from_file(file_path, validate=True)` | Load config from YAML or JSON file |
| `load_from_dict(config_dict, validate=True)` | Load config from dictionary |
| `merge_configs(*configs, validate=True)` | Merge multiple configurations |
| `get_config()` | Get current configuration |
| `set_config(config, validate=True)` | Set current configuration |
| `reload(file_path=None)` | Reload configuration from file |

**Example:**

```python
from semantica.core import ConfigManager

manager = ConfigManager()
config = manager.load_from_file("config.yaml")

# Merge configurations
config1 = manager.load_from_file("base_config.yaml")
config2 = manager.load_from_file("override_config.yaml")
merged = manager.merge_configs(config1, config2)
```

### Config

Configuration data class with validation and nested access.

**Methods:**

| Method | Description |
|--------|-------------|
| `get(key_path, default=None)` | Get nested configuration value by key path |
| `set(key_path, value)` | Set nested configuration value |
| `update(updates, merge=True)` | Update configuration with new values |
| `validate()` | Validate configuration settings |
| `to_dict()` | Convert configuration to dictionary |

**Example:**

```python
from semantica.core import Config, ConfigManager

manager = ConfigManager()
config = manager.load_from_dict({"processing": {"batch_size": 32}})

# Access nested values
batch_size = config.get("processing.batch_size", default=16)

# Update values
config.set("processing.batch_size", 64)
config.update({"quality": {"min_confidence": 0.9}})

# Validate
config.validate()
```

### LifecycleManager

System lifecycle management with hooks and health monitoring.

**Methods:**

| Method | Description |
|--------|-------------|
| `startup()` | Execute startup sequence with registered hooks |
| `shutdown(graceful=True)` | Execute shutdown sequence |
| `register_startup_hook(hook_fn, priority=50)` | Register a startup hook |
| `register_shutdown_hook(hook_fn, priority=50)` | Register a shutdown hook |
| `register_component(name, component)` | Register component for health monitoring |
| `health_check()` | Perform comprehensive system health check |
| `get_health_summary()` | Get summary of system health |
| `get_state()` | Get current system state |
| `is_ready()` | Check if system is ready |
| `is_running()` | Check if system is running |

**Example:**

```python
from semantica.core import LifecycleManager

manager = LifecycleManager()

# Register hooks
def init_db():
    print("Initializing database...")

manager.register_startup_hook(init_db, priority=10)
manager.startup()

# Register component for health monitoring
class DatabaseComponent:
    def health_check(self):
        return {"healthy": True, "message": "Connected"}

db = DatabaseComponent()
manager.register_component("database", db)

# Check health
health = manager.health_check()
summary = manager.get_health_summary()

manager.shutdown(graceful=True)
```

### PluginRegistry

Plugin registry and management system for dynamic plugin discovery and loading.

**Methods:**

| Method | Description |
|--------|-------------|
| `__init__(plugin_paths=None)` | Initialize with optional plugin paths for auto-discovery |
| `register_plugin(plugin_name, plugin_class, version="1.0.0", **metadata)` | Manually register a plugin |
| `load_plugin(plugin_name, **config)` | Load and initialize a plugin |
| `unload_plugin(plugin_name)` | Unload a plugin |
| `list_plugins()` | List all available plugins |
| `get_plugin_info(plugin_name)` | Get information about a plugin |
| `is_plugin_loaded(plugin_name)` | Check if a plugin is loaded |
| `get_loaded_plugin(plugin_name)` | Get loaded plugin instance |

**Example:**

```python
from semantica.core import PluginRegistry

# Auto-discover plugins
registry = PluginRegistry(plugin_paths=["./plugins"])

# Load plugin with configuration
plugin = registry.load_plugin("my_plugin", api_key="xxx")

# List all plugins
plugins = registry.list_plugins()
for plugin_info in plugins:
    print(f"{plugin_info['name']}: {plugin_info['version']}")

# Get plugin info
info = registry.get_plugin_info("my_plugin")
```

### MethodRegistry

Registry for custom orchestration methods.

**Methods:**

| Method | Description |
|--------|-------------|
| `register(task, name, method_func)` | Register a custom orchestration method |
| `get(task, name)` | Get method by task and name |
| `list_all(task=None)` | List all registered methods |
| `unregister(task, name)` | Unregister a method |
| `clear(task=None)` | Clear all registered methods |

**Example:**

```python
from semantica.core import method_registry

def custom_kb_builder(sources, **kwargs):
    # Custom logic
    return {"knowledge_graph": {}}

method_registry.register("knowledge_base", "custom", custom_kb_builder)

# Use custom method
method = method_registry.get("knowledge_base", "custom")
result = method(sources=["doc.pdf"])
```

---

## Orchestration Methods

Convenience functions for common orchestration tasks.

### build_knowledge_base()

Build knowledge base from data sources.

```python
from semantica.core.methods import build_knowledge_base

result = build_knowledge_base(
    sources=["doc1.pdf", "doc2.docx"],
    method="default",
    embeddings=True,
    graph=True
)
```

### run_pipeline()

Execute a processing pipeline.

```python
from semantica.core.methods import run_pipeline

result = run_pipeline(
    pipeline={"steps": ["parse", "extract"]},
    data="sample text",
    method="default"
)
```

### initialize_framework()

Initialize Semantica framework.

```python
from semantica.core.methods import initialize_framework

framework = initialize_framework(
    config={"llm_provider": {"name": "openai"}},
    method="default"
)
```

### get_status()

Get system status.

```python
from semantica.core.methods import get_status

status = get_status(framework=my_framework, method="detailed")
```

### get_orchestration_method()

Get orchestration method by task and name.

```python
from semantica.core.methods import get_orchestration_method

method = get_orchestration_method("knowledge_base", "custom")
```

### list_available_methods()

List all available orchestration methods.

```python
from semantica.core.methods import list_available_methods

all_methods = list_available_methods()
kb_methods = list_available_methods("knowledge_base")
```

---

## Configuration

### Environment Variables

Configuration can be loaded from environment variables with `SEMANTICA_` prefix:

```bash
export SEMANTICA_PROCESSING_BATCH_SIZE=64
export SEMANTICA_LLM_PROVIDER_MODEL=gpt-4
export SEMANTICA_QUALITY_MIN_CONFIDENCE=0.8
```

### YAML Configuration

```yaml
llm_provider:
  name: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}

embedding_model:
  name: openai
  model: text-embedding-ada-002

processing:
  batch_size: 32
  max_workers: 4

quality:
  min_confidence: 0.7

logging:
  level: INFO

plugins:
  my_plugin:
    enabled: true
    config_key: config_value
```

### JSON Configuration

```json
{
  "llm_provider": {
    "name": "openai",
    "model": "gpt-4"
  },
  "processing": {
    "batch_size": 32
  }
}
```

---

## Integration Examples

### Basic Usage

```python
from semantica.core import Semantica, ConfigManager

# 1. Load configuration
config_manager = ConfigManager()
config = config_manager.load_from_file("config.yaml")

# 2. Initialize framework
framework = Semantica(config=config)
framework.initialize()

try:
    # 3. Build knowledge base
    result = framework.build_knowledge_base(
        sources=["doc1.pdf", "doc2.docx"],
        embeddings=True,
        graph=True
    )
    
    # 4. Check status
    status = framework.get_status()
    print(f"System state: {status['state']}")
    
finally:
    # 5. Shutdown gracefully
    framework.shutdown(graceful=True)
```

### Custom Plugin

```python
from semantica.core import PluginRegistry

class MyPlugin:
    def initialize(self):
        print("Plugin initialized")
    
    def execute(self, data):
        return {"processed": True}

registry = PluginRegistry()
registry.register_plugin(
    plugin_name="my_plugin",
    plugin_class=MyPlugin,
    version="1.0.0"
)

plugin = registry.load_plugin("my_plugin")
result = plugin.execute("sample data")
```

### Lifecycle Hooks

```python
from semantica.core import LifecycleManager

manager = LifecycleManager()

def init_database():
    print("Initializing database...")

def cleanup_database():
    print("Cleaning up database...")

manager.register_startup_hook(init_database, priority=10)
manager.register_shutdown_hook(cleanup_database, priority=10)

manager.startup()
# ... do work ...
manager.shutdown(graceful=True)
```

### Custom Orchestration Method

```python
from semantica.core import method_registry, Semantica

def fast_kb_builder(sources, **kwargs):
    framework = Semantica()
    framework.initialize()
    try:
        return framework.build_knowledge_base(
            sources=sources,
            embeddings=False,  # Skip for speed
            graph=True,
            **kwargs
        )
    finally:
        framework.shutdown()

method_registry.register("knowledge_base", "fast", fast_kb_builder)

# Use custom method
from semantica.core.methods import build_knowledge_base
result = build_knowledge_base(sources=["doc.pdf"], method="fast")
```

---

## Best Practices

1. **Always Initialize**: Always call `initialize()` after creating a `Semantica` instance before using it.

2. **Graceful Shutdown**: Always call `shutdown(graceful=True)` in a `finally` block to ensure proper cleanup.

3. **Configuration Management**: Use `ConfigManager` for loading and managing configurations. Prefer YAML files for complex configurations.

4. **Error Handling**: Wrap framework operations in try-except blocks to handle `ConfigurationError` and `ProcessingError` appropriately.

5. **Health Monitoring**: Register components with `LifecycleManager` for health monitoring and use `health_check()` regularly.

6. **Plugin Development**: Follow the plugin interface (must have `initialize()` and `execute()` methods) when creating custom plugins.

7. **Method Registration**: Use `MethodRegistry` for extensibility. Register custom methods for knowledge base building, pipeline execution, etc.

8. **Hook Priorities**: Use appropriate priorities for lifecycle hooks. Lower numbers execute first.

9. **Configuration Validation**: Always validate configurations using `config.validate()` before using them.

10. **Resource Cleanup**: Ensure all resources are properly cleaned up in shutdown hooks.

---

## See Also
- [Core Usage Guide](core.md) - Comprehensive usage guide with detailed examples
- [Pipeline Module](pipeline.md) - Executed by the Semantica framework
- [Utils Module](utils.md) - Shared utilities used by Core

## Cookbook
- [Welcome to Semantica](https://github.com/Hawksight-AI/semantica/blob/main/cookbook/introduction/01_Welcome_to_Semantica.ipynb)
