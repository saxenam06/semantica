"""
=================================================================================
COPY THE CODE BELOW INTO NEW CELLS IN YOUR NOTEBOOK
=================================================================================

Instructions:
1. Open your notebook in Jupyter/VSCode
2. Find the cell where you build the knowledge graph (kg = graph_builder.build(...))
3. Add NEW cells below that and paste each section below into separate cells

=================================================================================
"""

# ============================================================================
# CELL 1: Build NetworkX Graph and Filter
# ============================================================================
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Build NetworkX graph from relationships
G = nx.DiGraph()

for rel in kg['relationships']:
    source = rel.get('source', '')
    target = rel.get('target', '')
    rel_type = rel.get('type', '')

    if source and target:
        G.add_edge(source, target, relation=rel_type)

print(f"NetworkX Graph Created:")
print(f"  - Nodes: {G.number_of_nodes()}")
print(f"  - Edges: {G.number_of_edges()}")

# Filter to important nodes
MIN_CONNECTIONS = 5
degrees = dict(G.degree())
important_nodes = [node for node, degree in degrees.items() if degree >= MIN_CONNECTIONS]
G_filtered = G.subgraph(important_nodes).copy()

print(f"\nFiltered Graph (nodes with {MIN_CONNECTIONS}+ connections):")
print(f"  - Nodes: {G_filtered.number_of_nodes()}")
print(f"  - Edges: {G_filtered.number_of_edges()}")

if G_filtered.number_of_nodes() == 0:
    print(f"\nNo nodes with {MIN_CONNECTIONS}+ connections. Showing all nodes.")
    G_filtered = G


# ============================================================================
# CELL 2: Create Visualization with Edge Labels
# ============================================================================
fig, ax = plt.subplots(figsize=(24, 20))

# Layout - use Kamada-Kawai for better spacing
try:
    pos = nx.kamada_kawai_layout(G_filtered, scale=6)
except:
    pos = nx.spring_layout(G_filtered, k=3, iterations=50, seed=42)

# Color nodes by type
node_colors = []
node_types = {}
for node in G_filtered.nodes():
    node_lower = node.lower()
    if 'risk' in node_lower or node in ['High', 'Medium', 'Low', 'Broken']:
        node_colors.append('#ff6b6b')  # Red for risk levels
        node_types[node] = 'Risk Level'
    elif 'sensor' in node_lower or node == 'Vibration':
        node_colors.append('#4ecdc4')  # Teal for sensors
        node_types[node] = 'Sensor'
    elif 'machine' in node_lower or node in ['CNC', 'Machine']:
        node_colors.append('#95e1d3')  # Green for machines
        node_types[node] = 'Machine'
    elif 'iso' in node_lower or node == 'ISO' or 'standard' in node_lower:
        node_colors.append('#ffd93d')  # Yellow for standards
        node_types[node] = 'Standard'
    elif 'temperature' in node_lower or 'temp' in node_lower or '°c' in node_lower:
        node_colors.append('#ffb6c1')  # Pink for temperature
        node_types[node] = 'Temperature'
    elif 'pressure' in node_lower or 'bar' in node_lower:
        node_colors.append('#dda15e')  # Brown for pressure
        node_types[node] = 'Pressure'
    elif 'current' in node_lower or 'flc' in node_lower or 'amp' in node_lower:
        node_colors.append('#bc6c25')  # Dark brown for current
        node_types[node] = 'Current'
    elif 'failure' in node_lower or 'fault' in node_lower or 'wear' in node_lower:
        node_colors.append('#e63946')  # Dark red for failures
        node_types[node] = 'Failure Mode'
    elif 'action' in node_lower or 'shutdown' in node_lower or 'inspect' in node_lower:
        node_colors.append('#06ffa5')  # Bright green for actions
        node_types[node] = 'Action'
    else:
        node_colors.append('#a8dadc')  # Light blue for others
        node_types[node] = 'Other'

# Draw edges FIRST (so they appear behind nodes)
nx.draw_networkx_edges(
    G_filtered, pos,
    alpha=0.35,
    width=2.5,
    edge_color='#444444',
    arrows=True,
    arrowsize=18,
    arrowstyle='->',
    connectionstyle='arc3,rad=0.1',
    ax=ax
)

# Draw nodes
nx.draw_networkx_nodes(
    G_filtered, pos,
    node_color=node_colors,
    node_size=1500,
    alpha=0.95,
    edgecolors='black',
    linewidths=2.5,
    ax=ax
)

# Draw node labels
nx.draw_networkx_labels(
    G_filtered, pos,
    font_size=9,
    font_weight='bold',
    font_color='black',
    ax=ax
)

# Draw edge labels (relationship names) - PARALLEL to edges
edge_labels = nx.get_edge_attributes(G_filtered, 'relation')
nx.draw_networkx_edge_labels(
    G_filtered, pos,
    edge_labels=edge_labels,
    font_size=6,
    font_color='darkred',
    font_weight='bold',
    alpha=0.8,
    rotate=True,  # KEY: Rotates labels to be parallel with edges!
    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7),
    ax=ax
)

# Create legend
legend_elements = [
    Patch(facecolor='#ff6b6b', edgecolor='black', label='Risk Level'),
    Patch(facecolor='#4ecdc4', edgecolor='black', label='Sensor'),
    Patch(facecolor='#95e1d3', edgecolor='black', label='Machine'),
    Patch(facecolor='#ffd93d', edgecolor='black', label='Standard'),
    Patch(facecolor='#ffb6c1', edgecolor='black', label='Temperature'),
    Patch(facecolor='#dda15e', edgecolor='black', label='Pressure'),
    Patch(facecolor='#bc6c25', edgecolor='black', label='Current'),
    Patch(facecolor='#e63946', edgecolor='black', label='Failure Mode'),
    Patch(facecolor='#06ffa5', edgecolor='black', label='Action'),
    Patch(facecolor='#a8dadc', edgecolor='black', label='Other'),
]

ax.legend(
    handles=legend_elements,
    loc='upper left',
    fontsize=12,
    title='Node Types',
    title_fontsize=14,
    framealpha=0.9
)

ax.set_title(
    f"Knowledge Graph with Relationship Labels\n{G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges",
    fontsize=20,
    fontweight='bold',
    pad=20
)
ax.axis('off')
plt.tight_layout()
plt.show()


# ============================================================================
# CELL 3: Print Statistics
# ============================================================================
print(f"Top 15 Most Connected Nodes:")
top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:15]
for i, (node, degree) in enumerate(top_nodes, 1):
    node_type = node_types.get(node, 'Other')
    print(f"  {i:2d}. {node:40s} - {degree:3d} connections ({node_type})")

print(f"\nTop 10 Relationship Types:")
rel_types = {}
for edge in G_filtered.edges(data=True):
    rel = edge[2].get('relation', 'Unknown')
    rel_types[rel] = rel_types.get(rel, 0) + 1
for rel, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:10]:
    print(f"  - {rel}: {count} edges")
