"""Qwen3-VL module hierarchy visualizer."""

import sys
from collections import defaultdict
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent
SHARED_EXPORT = MODEL_ROOT / "shared" / "stage0_export"
sys.path.insert(0, str(SHARED_EXPORT))

from stage_common import load_model_and_processor, create_prefill_example_inputs  # pyright: ignore
from settings import MODEL_NAME, IMAGE_SIZE  # pyright: ignore

# ============================================================
# CONFIG
# ============================================================
MAX_DEPTH = None  # Max hierarchy depth to show (None = all)
OUTPUT_NAME = "model_hierarchy.html"
AUTO_OPEN = True  # Auto open in browser


def collect_io_shapes(model, example_inputs):
    """Run forward pass with hooks to collect input/output shapes for each module."""
    import torch

    io_shapes = {}  # module_name -> {"inputs": [...], "outputs": [...]}

    def make_hook(name):
        def hook(module, inputs, outputs):
            def get_shape(x):
                if isinstance(x, torch.Tensor):
                    return list(x.shape)
                elif isinstance(x, (tuple, list)):
                    return [get_shape(i) for i in x if i is not None]
                elif hasattr(x, "shape"):
                    return list(x.shape)
                return None

            input_shapes = []
            for inp in inputs:
                s = get_shape(inp)
                if s is not None:
                    input_shapes.append(s)

            output_shapes = []
            if isinstance(outputs, torch.Tensor):
                output_shapes.append(list(outputs.shape))
            elif isinstance(outputs, (tuple, list)):
                for out in outputs:
                    s = get_shape(out)
                    if s is not None:
                        output_shapes.append(s)
            elif hasattr(outputs, "shape"):
                output_shapes.append(list(outputs.shape))

            io_shapes[name] = {"inputs": input_shapes, "outputs": output_shapes}

        return hook

    handles = []
    for name, module in model.named_modules():
        if name:
            handles.append(module.register_forward_hook(make_hook(name)))

    print("  Running forward pass to collect I/O shapes...")
    with torch.no_grad():
        model(**example_inputs)

    for h in handles:
        h.remove()

    return io_shapes


def build_hierarchy_tree(model):
    """Build nested dict representing module hierarchy."""
    tree = {}
    for name, module in model.named_modules():
        if not name:
            continue
        parts = name.split(".")
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {"_module": None, "_children": {}}
            current = current[part]["_children"]
        # Walk back to set module
        current = tree
        for part in parts[:-1]:
            current = current[part]["_children"]
        current[parts[-1]]["_module"] = module
    return tree


def print_hierarchy_tree(tree, prefix="", depth=0, max_depth=None):
    """Print hierarchy tree with nice formatting."""
    if max_depth is not None and depth >= max_depth:
        return

    items = list(tree.items())
    for i, (name, node) in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        extension = "    " if is_last else "│   "

        module = node["_module"]
        module_type = type(module).__name__ if module else "?"

        # Count parameters if leaf
        param_count = sum(p.numel() for p in module.parameters(recurse=False)) if module else 0
        param_str = f" ({param_count:,} params)" if param_count > 0 else ""

        print(f"{prefix}{connector}{name}: {module_type}{param_str}")

        children = node["_children"]
        if children:
            print_hierarchy_tree(children, prefix + extension, depth + 1, max_depth)


def print_module_summary(model):
    """Print summary of unique module types."""
    type_counts = defaultdict(int)
    type_params = defaultdict(int)

    for name, module in model.named_modules():
        t = type(module).__name__
        type_counts[t] += 1
        type_params[t] += sum(p.numel() for p in module.parameters(recurse=False))

    print("\n" + "=" * 60)
    print("MODULE TYPE SUMMARY")
    print("=" * 60)
    for t, count in sorted(type_counts.items(), key=lambda x: -type_params[x[0]]):
        params = type_params[t]
        if params > 0:
            print(f"  {t}: {count}x, {params:,} params")
        else:
            print(f"  {t}: {count}x")


def tree_to_d3_json(tree, io_shapes):
    """Convert internal tree to D3.js hierarchical JSON format with I/O shape info."""

    def format_shape(s):
        """Format shape list as string like [1,128,1536]."""
        if isinstance(s, list) and s and isinstance(s[0], list):
            return "[" + ", ".join(format_shape(x) for x in s) + "]"
        return "×".join(str(d) for d in s)

    def get_param_shapes(module):
        """Get list of parameter shapes."""
        shapes = []
        for pname, p in module.named_parameters(recurse=False):
            shape_str = "×".join(str(d) for d in p.shape)
            total = p.numel()
            shapes.append(f"⚙ {pname}: {shape_str}={total:,}")
        return shapes

    def convert_node(name, node, path=""):
        module = node["_module"]
        module_type = type(module).__name__ if module else "?"
        param_count = sum(p.numel() for p in module.parameters(recurse=False)) if module else 0

        full_path = f"{path}.{name}" if path else name
        io = io_shapes.get(full_path, {"inputs": [], "outputs": []})

        lines = []
        # Input shapes
        for i, inp in enumerate(io["inputs"]):
            lines.append(f"→ in[{i}]: {format_shape(inp)}")
        # Output shapes
        for i, out in enumerate(io["outputs"]):
            lines.append(f"← out[{i}]: {format_shape(out)}")
        # Parameter shapes
        lines.extend(get_param_shapes(module) if module else [])

        result = {
            "name": name,
            "type": module_type,
            "params": param_count,
            "shapes": lines,
        }

        children = node["_children"]
        if children:
            result["children"] = [convert_node(k, v, full_path) for k, v in children.items()]

        return result

    return {
        "name": "model",
        "type": "Qwen2_5_VLForConditionalGeneration",
        "params": 0,
        "shapes": [],
        "children": [convert_node(k, v, "") for k, v in tree.items()],
    }


def generate_interactive_html(tree, io_shapes, output_path: Path):
    """Generate interactive HTML with D3.js collapsible tree (top-down, box nodes)."""
    import json

    d3_data = tree_to_d3_json(tree, io_shapes)

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Qwen3-VL Module Hierarchy (Tree, not DAG)</title>
<style>
body {{
  font-family: "JetBrains Mono", "Fira Code", monospace;
  background: #0d1117;
  color: #c9d1d9;
  margin: 0;
  overflow: hidden;
}}
.node rect {{
  cursor: pointer;
  stroke: #30363d;
  stroke-width: 1px;
}}
.node text {{
  font-size: 9px;
  fill: #c9d1d9;
}}
.node .name {{ font-weight: bold; fill: #f0f6fc; font-size: 10px; }}
.node .type {{ fill: #8b949e; }}
.node .shape {{ font-size: 8px; }}
.node .shape-in {{ fill: #79c0ff; }}
.node .shape-out {{ fill: #ffa657; }}
.node .shape-param {{ fill: #7ee787; }}
.node--collapsed rect {{ stroke: #f85149; stroke-width: 2px; }}
.link {{
  fill: none;
  stroke: #30363d;
  stroke-width: 1px;
}}
#controls {{
  position: fixed;
  top: 10px;
  left: 10px;
  z-index: 100;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  max-width: 400px;
}}
#controls button {{
  background: #21262d;
  color: #c9d1d9;
  border: 1px solid #30363d;
  padding: 5px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-family: inherit;
  font-size: 11px;
}}
#controls button:hover {{ background: #30363d; }}
#info {{
  position: fixed;
  bottom: 10px;
  left: 10px;
  font-size: 10px;
  color: #484f58;
}}
</style>
</head>
<body>
<div id="controls">
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
  <button onclick="expandLevel(1)">L1</button>
  <button onclick="expandLevel(2)">L2</button>
  <button onclick="expandLevel(3)">L3</button>
  <button onclick="expandLevel(4)">L4</button>
  <button onclick="expandLevel(5)">L5</button>
  <button onclick="expandLevel(6)">L6</button>
</div>
<div id="info">nn.Module hierarchy tree (not computation DAG) | Click to expand/collapse | Scroll zoom | Drag pan</div>
<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
const data = {json.dumps(d3_data)};

const width = window.innerWidth;
const height = window.innerHeight;

const svg = d3.select("body").append("svg")
  .attr("width", width)
  .attr("height", height);

const g = svg.append("g").attr("transform", `translate(${{width/2}},40)`);

const zoom = d3.zoom()
  .scaleExtent([0.02, 3])
  .on("zoom", (e) => g.attr("transform", e.transform));
svg.call(zoom);

const tree = d3.tree().nodeSize([280, 120]);
const root = d3.hierarchy(data);

root.x0 = 0;
root.y0 = 0;

root.children?.forEach(collapse);

function collapse(d) {{
  if (d.children) {{
    d._children = d.children;
    d._children.forEach(collapse);
    d.children = null;
  }}
}}

function expand(d) {{
  if (d._children) {{
    d.children = d._children;
    d._children = null;
  }}
}}

function expandAll() {{
  root.descendants().forEach(d => {{
    if (d._children) {{ d.children = d._children; d._children = null; }}
  }});
  update(root);
}}

function collapseAll() {{
  root.children?.forEach(collapse);
  update(root);
}}

function expandLevel(level) {{
  collapseAll();
  function go(d, lvl) {{
    if (lvl < level) {{ expand(d); d.children?.forEach(c => go(c, lvl + 1)); }}
  }}
  go(root, 0);
  update(root);
}}

function getColor(d) {{
  const t = d.data.type;
  if (t.includes("Attention")) return "#1f6feb";
  if (t.includes("MLP")) return "#d29922";
  if (t.includes("Embedding")) return "#a371f7";
  if (t.includes("Norm") || t.includes("LayerNorm") || t.includes("RMSNorm")) return "#db61a2";
  if (t.includes("Linear")) return "#3fb950";
  if (t.includes("Conv")) return "#f78166";
  if (t.includes("Dropout") || t.includes("Identity")) return "#484f58";
  return "#21262d";
}}

function getBoxHeight(d) {{
  const baseH = 36;
  const shapeLines = d.data.shapes ? d.data.shapes.length : 0;
  return baseH + shapeLines * 11;
}}

const boxW = 260;

function update(source) {{
  const duration = 200;
  const treeData = tree(root);
  const nodes = treeData.descendants();
  const links = treeData.links();

  // Top-down: swap x and y, use y for depth
  nodes.forEach(d => {{
    const tmp = d.x;
    d.x = d.y;
    d.y = tmp;
    d.x = d.depth * 110;
  }});

  const node = g.selectAll(".node").data(nodes, d => d.id || (d.id = Math.random()));

  const nodeEnter = node.enter().append("g")
    .attr("class", "node")
    .attr("transform", d => `translate(${{source.y0 || 0}},${{source.x0 || 0}})`)
    .on("click", (event, d) => {{
      if (d.children) {{ d._children = d.children; d.children = null; }}
      else if (d._children) {{ d.children = d._children; d._children = null; }}
      update(d);
    }});

  nodeEnter.each(function(d) {{
    const boxH = getBoxHeight(d);
    const g = d3.select(this);

    g.append("rect")
      .attr("width", boxW)
      .attr("height", boxH)
      .attr("x", -boxW / 2)
      .attr("y", -16)
      .attr("rx", 4)
      .attr("fill", getColor(d));

    g.append("text")
      .attr("class", "name")
      .attr("x", -boxW / 2 + 6)
      .attr("y", 0)
      .text(d.data.name);

    g.append("text")
      .attr("class", "type")
      .attr("x", -boxW / 2 + 6)
      .attr("y", 12)
      .text(d.data.type);

    if (d.data.shapes && d.data.shapes.length > 0) {{
      d.data.shapes.forEach((s, i) => {{
        let cls = "shape";
        if (s.startsWith("→")) cls += " shape-in";
        else if (s.startsWith("←")) cls += " shape-out";
        else if (s.startsWith("⚙")) cls += " shape-param";
        g.append("text")
          .attr("class", cls)
          .attr("x", -boxW / 2 + 6)
          .attr("y", 24 + i * 11)
          .text(s);
      }});
    }}

    g.append("text")
      .attr("class", "indicator")
      .attr("x", boxW / 2 - 6)
      .attr("y", 0)
      .attr("text-anchor", "end")
      .attr("fill", "#f85149")
      .attr("font-size", "12px")
      .text(d._children ? `▼${{d._children.length}}` : "");
  }});

  const nodeUpdate = nodeEnter.merge(node);

  nodeUpdate.transition().duration(duration)
    .attr("transform", d => `translate(${{d.y}},${{d.x}})`);

  nodeUpdate.select("rect")
    .attr("class", d => d._children ? "node--collapsed" : "");

  nodeUpdate.select(".indicator")
    .text(d => d._children ? `▼${{d._children.length}}` : "");

  node.exit().transition().duration(duration)
    .attr("transform", d => `translate(${{source.y}},${{source.x}})`)
    .remove();

  const link = g.selectAll(".link").data(links, d => d.target.id);

  const linkEnter = link.enter().insert("path", "g")
    .attr("class", "link")
    .attr("d", d => {{
      const o = {{x: source.x0 || 0, y: source.y0 || 0}};
      return vline(o, o);
    }});

  linkEnter.merge(link).transition().duration(duration)
    .attr("d", d => vline(d.source, d.target));

  link.exit().transition().duration(duration)
    .attr("d", d => {{
      const o = {{x: source.x, y: source.y}};
      return vline(o, o);
    }}).remove();

  nodes.forEach(d => {{ d.x0 = d.x; d.y0 = d.y; }});
}}

function vline(s, d) {{
  const sBoxH = getBoxHeight(s);
  const sy = s.x + sBoxH - 16;
  return `M${{s.y}},${{sy}} V${{(sy + d.x - 16) / 2}} H${{d.y}} V${{d.x - 16}}`;
}}

update(root);
svg.call(zoom.transform, d3.zoomIdentity.translate(width / 2, 40));
</script>
</body>
</html>'''

    output_path.write_text(html)
    print(f"[HTML] Saved: {output_path}")


def main():
    import webbrowser

    print("=" * 60)
    print("Qwen3-VL Model Hierarchy Visualizer")
    print("=" * 60)

    print("\n[1] Loading model and processor...")
    model, processor = load_model_and_processor(MODEL_NAME)

    print("\n[2] Creating example inputs...")
    example_inputs = create_prefill_example_inputs(processor, IMAGE_SIZE)

    print("\n[3] Collecting I/O shapes...")
    io_shapes = collect_io_shapes(model, example_inputs)

    print("\n" + "=" * 60)
    print("MODULE HIERARCHY TREE")
    print("=" * 60)
    tree = build_hierarchy_tree(model)
    print_hierarchy_tree(tree, max_depth=MAX_DEPTH)

    print_module_summary(model)

    print("\n[4] Generating interactive HTML...")
    html_path = MODEL_ROOT / OUTPUT_NAME
    generate_interactive_html(tree, io_shapes, html_path)

    if AUTO_OPEN:
        webbrowser.open(f"file://{html_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
