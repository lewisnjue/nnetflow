"""Visualization module for nnetflow using Graphviz.

Provides a single powerful function `draw_dot` that visualizes:
- Any Tensor's computation graph (individual operations, additions, matmuls, etc.)
- Full model forward passes (when you call `model(input)` and then `draw_dot(output)`)

The graph shows:
- Ellipse nodes = tensors / parameters / inputs
- Box nodes   = operations (Conv2d, @, +, relu, etc.)
- Edges       = data flow (exactly the autograd DAG)

Requires: `pip install graphviz` (and the system Graphviz binary).
"""

from graphviz import Digraph
from .engine import Tensor
from typing import Optional


def draw_dot(
    root: Tensor,
    filename: str = "nnetflow_graph",
    format: str = "png",
    rankdir: str = "LR",           # LR = left-to-right (easier to read), TB = top-to-bottom
    show_grad: bool = True,
) -> Digraph:
    """
    Draw the full computation graph of a Tensor (including full model graphs).

    Args:
        root: The final Tensor (e.g. loss or model output after forward pass).
        filename: Base name of the saved file (without extension).
        format: Output format ('png', 'pdf', 'svg', etc.).
        rankdir: Graph direction ('LR' or 'TB').
        show_grad: Whether to show "requires_grad" / "Parameter" labels.

    Returns:
        The Digraph object (you can also call .view() or .render() again).
    """
    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})
    dot.attr("node", fontsize="10")
    dot.attr("edge", fontsize="9")

    seen = set()

    def add_node(t: Tensor) -> None:
        if t in seen:
            return
        seen.add(t)

        uid = str(id(t))

        # === Node styling & label ===
        if t._op:                                   # Operation node
            label = f"{t._op}\n{t.shape}"
            if show_grad and t.requires_grad:
                label += "\n(requires_grad)"
            dot.node(uid, label, shape="box", style="filled", fillcolor="#A8DADC")
        else:                                       # Tensor / Parameter / Input node
            label = f"Tensor\n{t.shape}"
            if t.requires_grad:
                label += "\nParameter"
            elif len(t._prev) == 0:
                label += "\nInput"
            if show_grad and t.grad is not None:
                label += "\n(grad)"
            dot.node(uid, label, shape="ellipse", style="filled", fillcolor="#E0F7FA")

        # === Recurse on children + draw edges ===
        for child in t._prev:
            add_node(child)
            child_uid = str(id(child))
            dot.edge(child_uid, uid)   # forward direction (data flow)

    add_node(root)

    # Render & save
    dot.render(filename, cleanup=True, format=format)
    print(f"✅ Graph saved as: {filename}.{format}")
    print(f"   (Open it with any image viewer or browser)")

    return dot


def visualize_model(model_output: Tensor, **kwargs) -> Digraph:
    """
    Convenience alias for full model visualization.

    Usage:
        output = model(dummy_input)
        visualize_model(output, filename="my_model")
    """
    return draw_dot(model_output, **kwargs)


__all__ = ["draw_dot", "visualize_model"]