"""
Knowledge Graph Visualization with Edge Labels
Run this after building your KG in the notebook
"""

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def visualize_kg_with_edges(kg, min_connections=5):
    """
    Visualize knowledge graph with relationship labels on edges

    Args:
        kg: Knowledge graph dict with 'entities' and 'relationships'
        min_connections: Minimum connections for filtered view
    """

    # Build NetworkX graph
    G = nx.DiGraph()

    for rel in kg['relationships']:
        source = rel.get('source', '')
        target = rel.get('target', '')
        rel_type = rel.get('type', '')

        if source and target:
            G.add_edge(source, target, relation=rel_type)

    print(f"Graph Created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Filter to important nodes
    degrees = dict(G.degree())
    important_nodes = [node for node, degree in degrees.items() if degree >= min_connections]
    G_filtered = G.subgraph(important_nodes).copy()

    print(f"Filtered: {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

    if G_filtered.number_of_nodes() == 0:
        print(f"No nodes with {min_connections}+ connections. Showing all nodes.")
        G_filtered = G

    # Create visualization
    fig, ax = plt.subplots(figsize=(22, 18))

    # Layout
    try:
        pos = nx.kamada_kawai_layout(G_filtered, scale=5)
    except:
        pos = nx.spring_layout(G_filtered, k=2.5, iterations=50, seed=42)

    # Color nodes by type
    node_colors = []
    for node in G_filtered.nodes():
        node_lower = node.lower()
        if 'risk' in node_lower or node in ['High', 'Medium', 'Low', 'Broken']:
            node_colors.append('#ff6b6b')  # Red - Risk Level
        elif 'sensor' in node_lower or node == 'Vibration':
            node_colors.append('#4ecdc4')  # Teal - Sensor
        elif 'machine' in node_lower or node in ['CNC', 'Machine']:
            node_colors.append('#95e1d3')  # Green - Machine
        elif 'iso' in node_lower or node == 'ISO' or 'standard' in node_lower:
            node_colors.append('#ffd93d')  # Yellow - Standard
        elif 'temperature' in node_lower or '°c' in node_lower:
            node_colors.append('#ffb6c1')  # Pink - Temperature
        elif 'pressure' in node_lower or 'bar' in node_lower:
            node_colors.append('#dda15e')  # Brown - Pressure
        elif 'current' in node_lower or 'flc' in node_lower:
            node_colors.append('#bc6c25')  # Dark brown - Current
        elif 'failure' in node_lower or 'fault' in node_lower or 'wear' in node_lower:
            node_colors.append('#e63946')  # Dark red - Failure
        elif 'action' in node_lower or 'shutdown' in node_lower or 'inspect' in node_lower:
            node_colors.append('#06ffa5')  # Bright green - Action
        else:
            node_colors.append('#a8dadc')  # Light blue - Other

    # Draw edges (first, so they appear behind)
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

    # Draw edge labels (relationships) - PARALLEL to edges
    edge_labels = nx.get_edge_attributes(G_filtered, 'relation')
    nx.draw_networkx_edge_labels(
        G_filtered, pos,
        edge_labels=edge_labels,
        font_size=6,
        font_color='darkred',
        font_weight='bold',
        alpha=0.8,
        rotate=True,  # Key parameter - rotates labels parallel to edges
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7),
        ax=ax
    )

    # Legend
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

    # Print statistics
    print(f"\nTop nodes by connections:")
    for node in sorted(G_filtered.nodes(), key=lambda n: degrees[n], reverse=True)[:10]:
        print(f"  - {node}: {degrees[node]} connections")

    print(f"\nTop relationship types:")
    rel_types = {}
    for edge in G_filtered.edges(data=True):
        rel = edge[2].get('relation', 'Unknown')
        rel_types[rel] = rel_types.get(rel, 0) + 1
    for rel, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {rel}: {count} edges")


# Usage example (paste this in your notebook):
"""
# In your notebook, after building kg:
from visualize_kg_with_edge_labels import visualize_kg_with_edges

# Show all highly connected nodes (5+ connections) with edge labels
visualize_kg_with_edges(kg, min_connections=5)

# Or show all nodes
visualize_kg_with_edges(kg, min_connections=1)
"""
