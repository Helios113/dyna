import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def load_graph(json_file):
    """Load graph from JSON file."""
    with open(json_file) as f:
        return json.load(f)


def create_layout(nodes, edges):
    """Create a simple hierarchical layout for the graph."""
    positions = {}

    if not nodes:
        return positions

    # Build adjacency list
    children = {node["node_id"]: [] for node in nodes}
    parents = {node["node_id"]: [] for node in nodes}

    for edge in edges:
        children[edge["from"]].append(edge["to"])
        parents[edge["to"]].append(edge["from"])

    # Find root nodes (no parents)
    roots = [node["node_id"] for node in nodes if not parents[node["node_id"]]]
    if not roots:
        roots = [nodes[0]["node_id"]]  # fallback to first node

    # Assign levels using BFS
    levels = {}
    queue = [(root, 0) for root in roots]
    visited = set()

    while queue:
        node_id, level = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        levels[node_id] = level

        for child in children[node_id]:
            if child not in visited:
                queue.append((child, level + 1))

    # Position nodes
    level_counts = {}
    level_indices = {}

    for node_id in levels:
        level = levels[node_id]
        level_counts[level] = level_counts.get(level, 0) + 1
        level_indices[level] = level_indices.get(level, 0)

    for node_id in levels:
        level = levels[node_id]
        total_at_level = level_counts[level]
        index_at_level = level_indices[level]
        level_indices[level] += 1

        x = (index_at_level - (total_at_level - 1) / 2) * 2
        y = -level * 2
        positions[node_id] = (x, y)

    return positions


def visualize_graph(graph_data, title="Computation Graph"):
    """Create matplotlib graph visualization."""
    plt.figure(figsize=(12, 8))
    ax = plt.gca()

    # Handle different graph formats
    if "nodes" in graph_data:
        nodes = graph_data["nodes"]
        edges = graph_data.get("edges", [])
    else:
        # Old format
        ops = graph_data.get("operations", [])
        nodes = [{"node_id": i, "op_name": op} for i, op in enumerate(ops)]
        edges = [{"from": src, "to": dst} for src, dst in graph_data.get("edges", [])]

    if not nodes:
        plt.text(
            0.5, 0.5, "Empty Graph", ha="center", va="center", transform=ax.transAxes
        )
        plt.title(title)
        return plt.gcf()

    # Create layout
    positions = create_layout(nodes, edges)

    # Draw edges first
    for edge in edges:
        src_id, dst_id = edge["from"], edge["to"]
        if src_id in positions and dst_id in positions:
            x1, y1 = positions[src_id]
            x2, y2 = positions[dst_id]

            # Draw arrow
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops={"arrowstyle": "->", "color": "gray", "lw": 1.5},
            )

    # Draw nodes
    for node in nodes:
        node_id = node["node_id"]
        if node_id not in positions:
            continue

        x, y = positions[node_id]

        # Create label
        label = node["op_name"]
        if node.get("shape"):
            label += f"\n{node['shape']}"

        # Choose color
        color = "lightcoral" if node.get("is_leaf", False) else "lightblue"

        # Draw node
        bbox = FancyBboxPatch(
            (x - 0.8, y - 0.3),
            1.6,
            0.6,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor="black",
            linewidth=1,
        )
        ax.add_patch(bbox)

        # Add text
        ax.text(x, y, label, ha="center", va="center", fontsize=8, weight="bold")

    # Set axis properties
    ax.set_aspect("equal")
    ax.axis("off")

    # Auto-scale
    if positions:
        xs, ys = zip(*positions.values(), strict=False)
        margin = 1.5
        ax.set_xlim(min(xs) - margin, max(xs) + margin)
        ax.set_ylim(min(ys) - margin, max(ys) + margin)

    plt.title(title, fontsize=14, weight="bold")
    plt.tight_layout()
    return plt.gcf()


def main():
    """Visualize all graph JSON files and save as PDFs."""
    json_file = ""

    try:
        graph_data = load_graph(json_file)
        title = Path(json_file).stem.replace("_", " ").title()

        fig = visualize_graph(graph_data, title)
        pdf_name = f"{Path(json_file).stem}_vis.png"
        fig.savefig(pdf_name, format="pdf", bbox_inches="tight", dpi=150)
        fig.savefig(pdf_name, format="png", bbox_inches="tight", dpi=150)

        plt.close(fig)

        print(f"Saved: {pdf_name}")

    except Exception as e:
        print(f"Error processing {json_file}: {e}")


if __name__ == "__main__":
    main()
