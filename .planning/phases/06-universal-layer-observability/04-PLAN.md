---
id: "04-PLAN"
plan: "04"
objective: "EmbeddingCollector — PCA visualization of embedding matrices; RNNCollector — LSTM/GRU gate monitoring"
wave: 2
depends_on: ["01-PLAN", "02-PLAN", "03-PLAN"]
files_modified:
  - "src/torchinspector/collectors/embedding.py"
  - "src/torchinspector/collectors/rnn.py"
  - "src/torchinspector/collectors/__init__.py"
  - "src/torchinspector/inspector.py"
  - "src/torchinspector/utils.py"
  - "tests/test_collectors/test_embedding.py"
  - "tests/test_collectors/test_rnn.py"
autonomous: true
requirements: ["UNIV-04", "UNIV-05"]
---

# Plan 04: Embedding PCA + RNN Gate Monitoring

**Wave:** 2
**Objective:** EmbeddingCollector uses `torch.pca_lowrank` to project embedding weights to 2D, renders as image. RNNCollector hooks LSTM/GRU to monitor gate activations and hidden state norms.

## Tasks

### Task 06-04-01: Create EmbeddingCollector
New file `collectors/embedding.py`. Detect `nn.Embedding` layers. Extract `weight.data` (vocab_size, embed_dim). If vocab > 256, sample first 256. Use `torch.pca_lowrank(matrix, q=2)` to project to 2D. Render as scatter-like heatmap image. Tag: `"embedding/{layer}/pca"`. Interval default 2000.

### Task 06-04-02: Create RNNCollector
New file `collectors/rnn.py`. Detect `nn.LSTM`, `nn.GRU` via isinstance. Register forward hooks to capture `(output, (h_n, c_n))` tuple. Log: `rnn/{layer}/hidden_norm`, `rnn/{layer}/cell_norm`, `rnn/{layer}/hidden_std`. For GRU: `hidden_norm` only (no cell state). All scalars.

### Task 06-04-03: Add utility functions
`list_embedding_layers(model)`, `list_rnn_layers(model)` in utils.py.

### Task 06-04-04: Integrate into Inspector
Add `embedding_interval=2000`, `rnn_interval` (default: log_interval) kwargs. Create collectors. Call in step().

### Task 06-04-05: Write tests
Embedding PCA projection, RNN hook capture, interval gating.

<automated>
```bash
pytest tests/test_collectors/test_embedding.py tests/test_collectors/test_rnn.py -x -q || exit 1
ruff check src/ tests/ || exit 1
```
</automated>
