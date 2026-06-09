# ruff: noqa: E501

"""Interactive HTML dashboard for exploring model layers and their metrics."""

from __future__ import annotations

from pathlib import Path

from torch import nn

from torchinspector.utils import classify_architecture, list_conv_layers, list_mha_layers


def _build_dashboard_html(model: nn.Module, log_dir: Path) -> str:
    """Generate a self-contained HTML dashboard.

    Shows model structure as an interactive tree. Click any layer
    to see available metrics with deep links to TensorBoard.
    """
    layers = _collect_layer_info(model)
    tree_json = _build_tree_json(layers)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>TorchInspector Dashboard</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; display:flex; height:100vh; background:#1a1a2e; color:#e0e0e0; }}
#sidebar {{ width:320px; background:#16213e; overflow-y:auto; padding:16px; border-right:1px solid #0f3460; }}
#sidebar h2 {{ color:#e94560; margin-bottom:16px; font-size:18px; }}
#sidebar h3 {{ color:#0f3460; background:#e94560; padding:4px 8px; border-radius:4px; margin:12px 0 4px; font-size:12px; text-transform:uppercase; }}
#main {{ flex:1; padding:24px; overflow-y:auto; }}
#main h2 {{ color:#e94560; margin-bottom:8px; }}
.meta {{ color:#888; font-size:13px; margin-bottom:16px; }}
.layer {{ padding:6px 10px; cursor:pointer; border-radius:4px; margin:2px 0; font-size:13px; display:flex; justify-content:space-between; }}
.layer:hover {{ background:#0f3460; }}
.layer.active {{ background:#e94560; color:#fff; }}
.layer .type {{ color:#888; font-size:11px; }}
.layer .pri {{ font-size:10px; padding:1px 5px; border-radius:3px; }}
.pri-high {{ background:#e94560; color:#fff; }}
.pri-med {{ background:#f0a500; color:#000; }}
.pri-low {{ background:#333; color:#888; }}
.metric-card {{ background:#16213e; border-radius:8px; padding:16px; margin:8px 0; }}
.metric-card h3 {{ color:#f0a500; }}
.metric-card a {{ color:#4fc3f7; text-decoration:none; font-size:13px; }}
.metric-card a:hover {{ text-decoration:underline; }}
.tag-list {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:8px; }}
.tag {{ background:#0f3460; padding:3px 8px; border-radius:3px; font-size:11px; font-family:monospace; }}
.block-section {{ margin-bottom:16px; }}
.empty {{ color:#666; font-style:italic; padding:20px; text-align:center; }}
</style>
</head>
<body>
<div id="sidebar">
  <h2>TorchInspector</h2>
  <div id="tree"></div>
</div>
<div id="main">
  <div class="empty">Click a layer in the sidebar to see its metrics</div>
</div>
<script>
const data = {tree_json};

const blocks = {{}};
data.forEach(l => {{
  const b = l.block || 'unknown';
  if (!blocks[b]) blocks[b] = [];
  blocks[b].push(l);
}});

const priorityNames = {{3:'HIGH', 2:'MEDIUM', 1:'LOW', 0:''}};
const priorityClasses = {{3:'pri-high', 2:'pri-med', 1:'pri-low', 0:''}};
const tree = document.getElementById('tree');
const main = document.getElementById('main');

// Build sidebar tree grouped by block type
const blockOrder = ['conv_block', 'linear_block', 'transformer_block', 'rnn_block', 'residual', 'norm', 'pool', 'activation', 'dropout', 'unknown'];
const blockLabels = {{
  conv_block: 'Conv Blocks', linear_block: 'Linear Blocks',
  transformer_block: 'Transformer Blocks', rnn_block: 'RNN Blocks',
  residual: 'Residual', norm: 'Normalization', pool: 'Pooling',
  activation: 'Activation', dropout: 'Dropout', unknown: 'Other'
}};

blockOrder.forEach(blockType => {{
  if (!blocks[blockType] || blocks[blockType].length === 0) return;
  const section = document.createElement('div');
  section.className = 'block-section';
  section.innerHTML = '<h3>' + (blockLabels[blockType] || blockType) + '</h3>';
  blocks[blockType].forEach(l => {{
    const div = document.createElement('div');
    div.className = 'layer';
    const pri = l.priority || 0;
    div.innerHTML = '<span>' + l.name.split('.').pop() + ' <span class="type">' + l.type + '</span></span>' +
      (pri > 0 ? '<span class="pri ' + (priorityClasses[pri]||'') + '">' + (priorityNames[pri]||'') + '</span>' : '');
    div.onclick = () => {{
      document.querySelectorAll('.layer').forEach(d => d.classList.remove('active'));
      div.classList.add('active');
      showLayer(l);
    }};
    section.appendChild(div);
  }});
  tree.appendChild(section);
}});

function showLayer(l) {{
  let html = '<h2>' + l.name + '</h2>';
  html += '<div class="meta">' + l.type + ' | ' + l.params.toLocaleString() + ' params';
  if (l.shape) html += ' | shape ' + l.shape;
  html += '</div>';

  // Scalars
  if (l.scalar_tags.length > 0) {{
    html += '<div class="metric-card"><h3>Scalars (' + l.scalar_tags.length + ')</h3><div class="tag-list">';
    l.scalar_tags.forEach(t => {{
      html += '<a class="tag" href="http://localhost:6006/#scalars&regexInput=' + encodeURIComponent(t) + '" target="_blank">' + t + '</a>';
    }});
    html += '</div></div>';
  }}

  // Images
  if (l.image_tags.length > 0) {{
    html += '<div class="metric-card"><h3>Images (' + l.image_tags.length + ')</h3><div class="tag-list">';
    l.image_tags.forEach(t => {{
      html += '<a class="tag" href="http://localhost:6006/#images&regexInput=' + encodeURIComponent(t) + '" target="_blank">' + t + '</a>';
    }});
    html += '</div></div>';
  }}

  // Histograms
  if (l.hist_tags.length > 0) {{
    html += '<div class="metric-card"><h3>Histograms (' + l.hist_tags.length + ')</h3><div class="tag-list">';
    l.hist_tags.forEach(t => {{
      html += '<a class="tag" href="http://localhost:6006/#histograms&regexInput=' + encodeURIComponent(t) + '" target="_blank">' + t + '</a>';
    }});
    html += '</div></div>';
  }}

  if (l.scalar_tags.length === 0 && l.image_tags.length === 0 && l.hist_tags.length === 0) {{
    html += '<div class="metric-card"><h3>No metrics</h3><p class="meta">This layer is not watched. Use ins.watch(["' + l.name + '"]) to enable monitoring.</p></div>';
  }}

  // Link to explain for conv/MHA layers
  if (l.is_conv) {{
    html += '<div class="metric-card"><h3>Explainability</h3><a href="http://localhost:6006/#images&regexInput=explain%2F' + encodeURIComponent(l.name) + '" target="_blank">Grad-CAM for ' + l.name + '</a></div>';
  }}
  if (l.is_mha) {{
    html += '<div class="metric-card"><h3>Explainability</h3><a href="http://localhost:6006/#images&regexInput=attention%2F' + encodeURIComponent(l.name) + '" target="_blank">Attention heatmaps for ' + l.name + '</a></div>';
  }}

  main.innerHTML = html;
}}
</script>
</body>
</html>"""


def _collect_layer_info(model: nn.Module) -> list[dict[str, object]]:
    """Collect structured info about every layer in the model."""
    arch = classify_architecture(model)
    conv_names = set(list_conv_layers(model))
    mha_names = set(list_mha_layers(model))
    result = []

    for name, module in model.named_modules():
        if name == "":
            continue
        block_type, priority = arch.get(name, ("unknown", 0))
        params = sum(p.numel() for p in module.parameters())
        shape = ""
        w = getattr(module, "weight", None)
        if w is not None:
            shape = str(tuple(w.shape))
        elif hasattr(module, "in_features"):
            shape = f"({getattr(module, 'in_features')}, {getattr(module, 'out_features')})"

        typ = type(module).__name__
        scalar_tags = [
            f"activations/{name}/(mean|std|min|max|sparsity)",
            f"activations/{name}/dead_neuron_ratio",
            f"gradients/{name}\\.(weight|bias)/norm",
            f"bn/{name}/",
            f"pool/{name}/",
            f"rnn/{name}/",
        ]
        image_tags = [
            f"features/{name}/channels",
            f"weights/{name}/matrix",
            f"explain/{name}/gradcam",
            f"attention/{name}/",
        ]
        hist_tags = [
            f"params/{name}\\.",
            f"grads/{name}\\.",
        ]

        result.append({
            "name": name,
            "type": typ,
            "block": block_type,
            "priority": priority,
            "params": params,
            "shape": shape,
            "is_conv": name in conv_names,
            "is_mha": name in mha_names,
            "scalar_tags": scalar_tags,
            "image_tags": image_tags,
            "hist_tags": hist_tags,
        })

    return result


def _build_tree_json(layers: list[dict[str, object]]) -> str:
    """Build a JSON string for the layer tree data."""
    import json
    return json.dumps(layers)
