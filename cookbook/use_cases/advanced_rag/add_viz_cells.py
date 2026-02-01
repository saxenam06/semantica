"""
Script to add visualization cells to the notebook
"""
import json
import sys

# Load the backup
with open('03_Multimodal_RAG_Comparison_PDF.ipynb.backup', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell after KG building (cell with "Building temporal knowledge graph")
insert_index = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if 'Building temporal knowledge graph' in source or 'kg = graph_builder.build' in source:
            insert_index = i + 1
            break

if insert_index is None:
    print("Could not find insertion point. Please manually add cells.")
    sys.exit(1)

print(f"Found insertion point at cell {insert_index}")

# New cells to add
new_cells = [
    # Markdown header
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Visualize the Knowledge Graph Network with Relationship Labels\n"]
    },
    # Visualization code
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Knowledge Graph Visualization with Edge Labels\n",
            "import networkx as nx\n",
            "import matplotlib.pyplot as plt\n",
            "from matplotlib.patches import Patch\n",
            "\n",
            "# Build NetworkX graph from relationships\n",
            "G = nx.DiGraph()\n",
            "\n",
            "for rel in kg['relationships']:\n",
            "    source = rel.get('source', '')\n",
            "    target = rel.get('target', '')\n",
            "    rel_type = rel.get('type', '')\n",
            "    \n",
            "    if source and target:\n",
            "        G.add_edge(source, target, relation=rel_type)\n",
            "\n",
            "print(f\"NetworkX Graph Created:\")\n",
            "print(f\"  - Nodes: {G.number_of_nodes()}\")\n",
            "print(f\"  - Edges: {G.number_of_edges()}\")\n",
            "\n",
            "# Filter to important nodes\n",
            "MIN_CONNECTIONS = 5\n",
            "degrees = dict(G.degree())\n",
            "important_nodes = [node for node, degree in degrees.items() if degree >= MIN_CONNECTIONS]\n",
            "G_filtered = G.subgraph(important_nodes).copy()\n",
            "\n",
            "print(f\"\\nFiltered Graph (nodes with {MIN_CONNECTIONS}+ connections):\")\n",
            "print(f\"  - Nodes: {G_filtered.number_of_nodes()}\")\n",
            "print(f\"  - Edges: {G_filtered.number_of_edges()}\")\n",
            "\n",
            "if G_filtered.number_of_nodes() == 0:\n",
            "    print(f\"\\nNo nodes with {MIN_CONNECTIONS}+ connections. Showing all nodes.\")\n",
            "    G_filtered = G\n"
        ]
    },
    # Visualization cell
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create visualization with edge labels\n",
            "fig, ax = plt.subplots(figsize=(24, 20))\n",
            "\n",
            "# Layout - use Kamada-Kawai for better spacing\n",
            "try:\n",
            "    pos = nx.kamada_kawai_layout(G_filtered, scale=6)\n",
            "except:\n",
            "    pos = nx.spring_layout(G_filtered, k=3, iterations=50, seed=42)\n",
            "\n",
            "# Color nodes by type\n",
            "node_colors = []\n",
            "node_types = {}\n",
            "for node in G_filtered.nodes():\n",
            "    node_lower = node.lower()\n",
            "    if 'risk' in node_lower or node in ['High', 'Medium', 'Low', 'Broken']:\n",
            "        node_colors.append('#ff6b6b')  # Red for risk levels\n",
            "        node_types[node] = 'Risk Level'\n",
            "    elif 'sensor' in node_lower or node == 'Vibration':\n",
            "        node_colors.append('#4ecdc4')  # Teal for sensors\n",
            "        node_types[node] = 'Sensor'\n",
            "    elif 'machine' in node_lower or node in ['CNC', 'Machine']:\n",
            "        node_colors.append('#95e1d3')  # Green for machines\n",
            "        node_types[node] = 'Machine'\n",
            "    elif 'iso' in node_lower or node == 'ISO' or 'standard' in node_lower:\n",
            "        node_colors.append('#ffd93d')  # Yellow for standards\n",
            "        node_types[node] = 'Standard'\n",
            "    elif 'temperature' in node_lower or 'temp' in node_lower or '°c' in node_lower:\n",
            "        node_colors.append('#ffb6c1')  # Pink for temperature\n",
            "        node_types[node] = 'Temperature'\n",
            "    elif 'pressure' in node_lower or 'bar' in node_lower:\n",
            "        node_colors.append('#dda15e')  # Brown for pressure\n",
            "        node_types[node] = 'Pressure'\n",
            "    elif 'current' in node_lower or 'flc' in node_lower or 'amp' in node_lower:\n",
            "        node_colors.append('#bc6c25')  # Dark brown for current\n",
            "        node_types[node] = 'Current'\n",
            "    elif 'failure' in node_lower or 'fault' in node_lower or 'wear' in node_lower:\n",
            "        node_colors.append('#e63946')  # Dark red for failures\n",
            "        node_types[node] = 'Failure Mode'\n",
            "    elif 'action' in node_lower or 'shutdown' in node_lower or 'inspect' in node_lower:\n",
            "        node_colors.append('#06ffa5')  # Bright green for actions\n",
            "        node_types[node] = 'Action'\n",
            "    else:\n",
            "        node_colors.append('#a8dadc')  # Light blue for others\n",
            "        node_types[node] = 'Other'\n",
            "\n",
            "# Draw edges FIRST (so they appear behind nodes)\n",
            "nx.draw_networkx_edges(\n",
            "    G_filtered, pos, \n",
            "    alpha=0.35,\n",
            "    width=2.5,\n",
            "    edge_color='#444444',\n",
            "    arrows=True, \n",
            "    arrowsize=18,\n",
            "    arrowstyle='->',\n",
            "    connectionstyle='arc3,rad=0.1',\n",
            "    ax=ax\n",
            ")\n",
            "\n",
            "# Draw nodes\n",
            "nx.draw_networkx_nodes(\n",
            "    G_filtered, pos, \n",
            "    node_color=node_colors, \n",
            "    node_size=1500,\n",
            "    alpha=0.95, \n",
            "    edgecolors='black',\n",
            "    linewidths=2.5,\n",
            "    ax=ax\n",
            ")\n",
            "\n",
            "# Draw node labels\n",
            "nx.draw_networkx_labels(\n",
            "    G_filtered, pos, \n",
            "    font_size=9,\n",
            "    font_weight='bold',\n",
            "    font_color='black',\n",
            "    ax=ax\n",
            ")\n",
            "\n",
            "# Draw edge labels (relationship names) - PARALLEL to edges\n",
            "edge_labels = nx.get_edge_attributes(G_filtered, 'relation')\n",
            "nx.draw_networkx_edge_labels(\n",
            "    G_filtered, pos,\n",
            "    edge_labels=edge_labels,\n",
            "    font_size=6,\n",
            "    font_color='darkred',\n",
            "    font_weight='bold',\n",
            "    alpha=0.8,\n",
            "    rotate=True,  # KEY: Rotates labels to be parallel with edges!\n",
            "    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.7),\n",
            "    ax=ax\n",
            ")\n",
            "\n",
            "# Create legend\n",
            "legend_elements = [\n",
            "    Patch(facecolor='#ff6b6b', edgecolor='black', label='Risk Level'),\n",
            "    Patch(facecolor='#4ecdc4', edgecolor='black', label='Sensor'),\n",
            "    Patch(facecolor='#95e1d3', edgecolor='black', label='Machine'),\n",
            "    Patch(facecolor='#ffd93d', edgecolor='black', label='Standard'),\n",
            "    Patch(facecolor='#ffb6c1', edgecolor='black', label='Temperature'),\n",
            "    Patch(facecolor='#dda15e', edgecolor='black', label='Pressure'),\n",
            "    Patch(facecolor='#bc6c25', edgecolor='black', label='Current'),\n",
            "    Patch(facecolor='#e63946', edgecolor='black', label='Failure Mode'),\n",
            "    Patch(facecolor='#06ffa5', edgecolor='black', label='Action'),\n",
            "    Patch(facecolor='#a8dadc', edgecolor='black', label='Other'),\n",
            "]\n",
            "\n",
            "ax.legend(\n",
            "    handles=legend_elements, \n",
            "    loc='upper left', \n",
            "    fontsize=12,\n",
            "    title='Node Types',\n",
            "    title_fontsize=14,\n",
            "    framealpha=0.9\n",
            ")\n",
            "\n",
            "ax.set_title(\n",
            "    f\"Knowledge Graph with Relationship Labels\\n{G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges\",\n",
            "    fontsize=20, \n",
            "    fontweight='bold',\n",
            "    pad=20\n",
            ")\n",
            "ax.axis('off')\n",
            "plt.tight_layout()\n",
            "plt.show()\n"
        ]
    },
    # Statistics cell
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Print statistics\n",
            "print(f\"Top 15 Most Connected Nodes:\")\n",
            "top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:15]\n",
            "for i, (node, degree) in enumerate(top_nodes, 1):\n",
            "    node_type = node_types.get(node, 'Other')\n",
            "    print(f\"  {i:2d}. {node:40s} - {degree:3d} connections ({node_type})\")\n",
            "\n",
            "print(f\"\\nTop 10 Relationship Types:\")\n",
            "rel_types = {}\n",
            "for edge in G_filtered.edges(data=True):\n",
            "    rel = edge[2].get('relation', 'Unknown')\n",
            "    rel_types[rel] = rel_types.get(rel, 0) + 1\n",
            "for rel, count in sorted(rel_types.items(), key=lambda x: x[1], reverse=True)[:10]:\n",
            "    print(f\"  - {rel}: {count} edges\")\n"
        ]
    }
]

# Insert the new cells
for i, cell in enumerate(new_cells):
    nb['cells'].insert(insert_index + i, cell)

# Save the modified notebook
with open('03_Multimodal_RAG_Comparison_PDF.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"✅ Successfully added {len(new_cells)} visualization cells at position {insert_index}")
print("The notebook has been updated with edge label visualizations!")
