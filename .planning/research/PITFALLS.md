# Pitfalls Research

**Domain:** PyTorch Training Observation Library
**Researched:** 2026-06-08
**Confidence:** HIGH

## Critical Pitfalls

### Pitfall 1: Forward Hook Memory Leak

**What goes wrong:**
Detached activation tensors accumulate in the HookManager's cache dict. If the user watches 10 layers at batch_size=32 on a ResNet, each forward pass adds ~50MB of detached CPU tensors. After 1000 steps, that's 50GB of stale activations. The Python GC can't free them because they're referenced by the cache dict.

**Why it happens:**
Forward hooks fire on every `model(x)` call. The naive implementation caches every activation without eviction. Detached tensors are new allocations, not views — they consume fresh memory each time.

**How to avoid:**
Overwrite, don't append. The activation cache is a dict of `name → latest_tensor`. Each forward pass replaces the previous value for the same layer name. Python GC frees the old tensor immediately since refcount drops to zero.

**Warning signs:**
- CPU RAM usage grows monotonically during training
- Memory profiler shows large `dict` of `torch.Tensor` objects
- Training doesn't crash immediately — OOM after N steps (N depends on model size)

**Phase to address:**
Phase 1 — implement activation cache as overwrite, not append. Add a `max_cached` limit as safety.

---

### Pitfall 2: Silent Hook Bypass with `model.forward(x)`

**What goes wrong:**
User calls `model.forward(x)` instead of `model(x)`. All registered hooks are silently skipped. Training appears to work normally, but no activation data is logged. User concludes "TorchInspector is broken."

**Why it happens:**
`nn.Module.__call__` invokes hooks; `model.forward()` does not. This is a PyTorch design decision, not a bug. Many tutorials and legacy code use `.forward()` directly.

**How to avoid:**
- Document prominently: "Always use `model(x)`, never `model.forward(x)`"
- Add a runtime check: wrap the model's `forward` with a warning if `__call__` wasn't the entry point (difficult to do reliably)
- In examples and quickstart, use `output = model(x)` explicitly
- Add a FAQ entry for "No activation data appearing"
- There's even a `flake8-no-module-forward-call` plugin — recommend it in docs

**Warning signs:**
- Activation cache is always empty
- `inspector.step()` runs without errors but TensorBoard shows no activation data
- User's code uses `.forward()` pattern

**Phase to address:**
Phase 1 — documentation and examples. Phase 2 — consider runtime detection via `warnings.warn()` if possible.

---

### Pitfall 3: CUDA Synchronization on Every Step

**What goes wrong:**
Every call to `tensor.detach().cpu()` triggers a CUDA synchronization point. If the training loop calls this on every step for parameter histograms, it serializes GPU→CPU transfers and kills training throughput. A 5ms GPU kernel now waits 10ms for the CPU copy.

**Why it happens:**
PyTorch operations are asynchronous on CUDA. `.cpu()` forces a sync to ensure the tensor is ready. Doing this in the hot path (every training step) pays the sync cost every time.

**How to avoid:**
- Parameter and activation logging only at `log_interval` (e.g., every 10-100 steps), not every step
- Scalar logging (loss, acc) uses Python floats already on CPU — no sync needed
- Document the trade-off: "Set log_interval higher for large models to minimize sync overhead"
- Consider `torch.cuda.synchronize()` only at collection points, not per-tensor

**Warning signs:**
- Training throughput drops significantly when TorchInspector is attached vs. unattached
- GPU utilization shows gaps (GPU idle waiting for CPU)
- nvprof/pytorch profiler shows large CPU-GPU sync bubbles

**Phase to address:**
Phase 1 — design interval-based collection from the start. Phase 2 — add async collection benchmark in CI.

---

### Pitfall 4: TensorBoard Event File Proliferation

**What goes wrong:**
Each `SummaryWriter` instance creates a new event file directory. If users create a new `Inspector` per experiment run without cleaning up, or if the API creates multiple writers internally, the `runs/` directory fills with orphaned event files.

**Why it happens:**
`SummaryWriter` creates `events.out.tfevents.*` files eagerly. No automatic cleanup. Multiple writers in the same process = multiple file handles = potential corruption.

**How to avoid:**
- Single `SummaryWriter` per `Inspector` instance (enforced)
- `Inspector.close()` must call `writer.close()` — use context manager (`__enter__`/`__exit__`)
- Document `with Inspector(...) as ins:` as the recommended usage pattern
- Add `log_dir` cleanup option: `Inspector(..., clean_existing=True)` to wipe previous run

**Warning signs:**
- `runs/` directory contains dozens of subdirectories after a few experiments
- "Event file corrupted" errors in TensorBoard
- Multiple TensorBoard processes reading the same directory show inconsistent data

**Phase to address:**
Phase 1 — implement context manager, single-writer enforcement.

---

### Pitfall 5: `torch.compile` Hook Incompatibility

**What goes wrong:**
When the user wraps their model with `torch.compile()`, PyTorch may fuse or reorder operations. Forward hooks registered on the original module may fire on the compiled graph's internal representation with different tensor shapes, different module names, or may not fire at all for fused operations.

**Why it happens:**
`torch.compile` rewrites the computation graph. Dynamo traces through hooks but may inline or fuse them in ways that change hook behavior. Some hooks that fire in eager mode may be optimized away.

**How to avoid:**
- Test with `torch.compile` from Day 1 (add to CI matrix)
- Document known limitations: "Hook behavior may differ under torch.compile; verify activation data matches expectations"
- Warn if `torch.compile` is detected on a watched model
- For v1: support eager mode fully; torch.compile is best-effort
- Stay updated with PyTorch releases — compile/hook interaction is actively improving

**Warning signs:**
- Activations data shape/size differs between eager and compiled mode
- Some watched layers produce no data under compile
- Errors about "cannot call register_forward_hook on compiled graph"

**Phase to address:**
Phase 1 — test with compile, document limitations. Phase 3+ — full compile support.

---

### Pitfall 6: Over-Engineering the Backend Abstraction

**What goes wrong:**
Spending weeks designing a perfectly generic Backend protocol with 25 methods before a single backend works. The abstraction ends up wrong because it's designed in a vacuum — it optimizes for hypothetical future backends rather than the one that ships first.

**Why it happens:**
"Make it extensible" is a seductive goal. It's easier to design abstractions than to ship working code. The temptation to support "any backend" before validating "one backend" is strong.

**How to avoid:**
- Start with ONE backend (TensorBoard). Make it work end-to-end.
- Extract the Backend protocol AFTER TensorBoard works — from real code, not speculation.
- Rule of thumb: the protocol should have ≤8 methods for v1
- If a method doesn't have a concrete TensorBoard implementation, it doesn't belong in the protocol yet
- YAGNI: You Ain't Gonna Need It — future backends can extend the protocol

**Warning signs:**
- Backend protocol has 15+ methods but only 1 backend implements it
- Protocol is in a separate file before the concrete backend exists
- Discussions about "what if someone wants to log to Kafka?" before TensorBoard works

**Phase to address:**
Phase 1 — concrete TensorBoard backend. Extract protocol at end of Phase 2 when activation logging is stable.

---

### Pitfall 7: Breaking on Non-Standard Training Loops

**What goes wrong:**
Assuming all training loops look like `for epoch: for batch:`. Breaking on GAN training (two models, alternating updates), RL loops (no epochs), or fine-tuning loops (frozen layers, different LR per group).

**Why it happens:**
The draft's example code uses a standard classification loop. It's tempting to bake assumptions about loop structure into the API (auto-epoch detection, auto-batch counting).

**How to avoid:**
- `Inspector.step()` is a manual call — the user controls when it's called
- No auto-epoch detection; provide optional `inspector.epoch()` marker only
- Don't assume one model / one optimizer — support `Inspector(models=[model], optimizers=[opt])`
- Test with at least: standard classification, GAN training loop, and a HuggingFace Trainer loop

**Warning signs:**
- API has `on_epoch_start` / `on_epoch_end` as required lifecycle methods
- Errors when model has multiple optimizers or vice versa
- Documentation only shows a single training loop pattern

**Phase to address:**
Phase 1 — keep step() manual and loop-agnostic. Phase 3 — add convenience auto-detection that can be disabled.

## Technical Debt Patterns

Shortcuts that seem reasonable but create long-term problems.

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Skip tests for collectors | Faster MVP | Regressions when refactoring hook logic | Never — collectors are the core logic |
| Hard-code TensorBoard in collectors | Don't need Backend protocol yet | Every collector needs rewrite when adding second backend | Acceptable in Phase 1; extract protocol in Phase 2 |
| No type hints on public API | Faster to write | IDE autocomplete broken; users confused | Never for a library — types are table stakes |
| Skip docs for v1 | Ship faster | No adoption; "this looks unmaintained" | Never — quickstart + API docs are minimum |
| Monolithic inspector.py (500+ lines) | Less file navigation | Hard to test individual components; hard to contribute | Acceptable in Phase 1; split into collectors/ in Phase 2 |

## Integration Gotchas

Common mistakes when connecting to external services.

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| TensorBoard | Creating one `SummaryWriter` per metric instead of one per run | Single writer; all metrics go through it |
| PyTorch hooks | Not removing hooks when Inspector is closed/deleted | `Inspector.close()` removes all handles; use context manager |
| ONNX export | Exporting during training (model in train mode, dropout active) | Export in eval mode; warn if `model.training` is True |
| GPU tensors | Passing GPU tensors to `add_histogram` (expects CPU) | Always `.detach().cpu()` before logging |

## Performance Traps

Patterns that work at small scale but fail as usage grows.

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Every-step histogram logging | Training 5-10x slower with observer attached | `log_interval` defaults to 10-50; users can tune | >100 params, interval=1 |
| Full parameter logging on large models | OOM from storing detached param copies | Sample large tensors (>1M elements) before histogram | >10M total parameters |
| Synchronous file I/O | GPU idle during TensorBoard flush | Batch writes at interval boundaries; async thread if needed | I/O latency > step time |
| Forward hook on all layers | Training slowdown; memory pressure from cache | Require explicit layer selection; suggest ≤5 layers | >50 layers watched |

## Security Mistakes

Domain-specific security issues beyond general web security.

| Mistake | Risk | Prevention |
|---------|------|------------|
| TensorBoard event files world-readable | Training data leakage if on shared filesystem | Set `log_dir` permissions; document security considerations |
| ONNX model contains weights | Intellectual property leakage if shared | Warn on export that ONNX contains full model weights |
| `torch.load` on untrusted files | Arbitrary code execution (pickle) | Never `torch.load` for logging; use safe formats |

## UX Pitfalls

Common user experience mistakes in this domain.

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Requiring 20 lines of setup | "I'll just use raw TensorBoard" vs reading docs | `<5 lines to get started`: `Inspector(model, opt).watch(["conv1"])` |
| Silent failures | User thinks it's working but nothing is logged | Warn if no layers matched by watch pattern; log which layers were found |
| Confusing error messages | "KeyError: 'layer1'" with no context | "Layer 'layer1' not found. Available layers: conv1, conv2, fc1, ..." |
| No default layer suggestions | "Which layers should I watch?" paralysis | `inspector.suggest_layers()` — auto-detect conv/attention/classifier layers |

## "Looks Done But Isn't" Checklist

- [ ] **Context manager:** Often missing — user must remember to call `.close()`. Verify `with Inspector(...) as ins:` works.
- [ ] **Error on double-close:** Calling `.close()` twice should be safe (idempotent), not crash.
- [ ] **Hooks removed on close:** Stale hooks on a deleted model cause hard-to-debug segfaults. Verify via `len(model._forward_hooks)` after close.
- [ ] **Works with DataParallel/DDP:** Hooks registered on the wrapped module may not fire as expected. Verify with at least DataParallel.
- [ ] **TensorBoard directory exists:** `SummaryWriter` silently creates parent dirs but fails on permission errors. Test with read-only filesystem.
- [ ] **Zero-parameter modules:** `nn.ReLU`, `nn.Dropout` have no parameters — param collector should skip them gracefully.

## Recovery Strategies

When pitfalls occur despite prevention, how to recover.

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| Hook memory leak | LOW | Call `.close()` to remove all hooks; restart Python process if OOM already hit |
| CUDA sync slowdown | LOW | Increase `log_interval`; reduce watched layers |
| TensorBoard file corruption | LOW | Delete the event file directory; restart run with clean `log_dir` |
| torch.compile incompatibility | MEDIUM | Fall back to eager mode for watched model; file a PyTorch issue |
| Over-engineered backend | HIGH | Refactor: extract protocol from working code, discard speculative methods |

## Pitfall-to-Phase Mapping

How roadmap phases should address these pitfalls.

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| Forward hook memory leak | Phase 1 | Unit test: verify activation cache uses overwrite, not append |
| `model.forward(x)` bypass | Phase 1 | Documentation + FAQ entry |
| CUDA sync overhead | Phase 1 | Benchmark: training throughput with/without Inspector at interval=10 |
| Event file proliferation | Phase 1 | Unit test: verify single SummaryWriter per Inspector |
| torch.compile incompatibility | Phase 1 | CI test: run with `torch.compile` on a small model |
| Over-engineered backend | Phase 1-2 | Review: ≤8 methods in Backend protocol |
| Non-standard loop breakage | Phase 1 | Integration test: GAN training loop, HF Trainer loop |
| Every-step histogram | Phase 1 | Default interval ≥10; document performance trade-off |
| Full tensor logging | Phase 2 | Test with large model (>10M params); verify stats-only mode |
| Silent failures / bad UX | Phase 2 | User testing: can a new user get started in <5 minutes? |

## Sources

- [PyTorch Hook Docs](https://pytorch.org/docs/stable/nn.html#hooks) — Official hook lifecycle documentation
- [PyTorch Issue #132700](https://github.com/pytorch/pytorch/issues/132700) — Forward hooks not firing under eval()+no_grad()
- [torch.compile FAQ](https://pytorch.org/docs/stable/torch.compiler_faq.html) — Compile/hook interaction notes
- [TensorBoard Performance](https://github.com/tensorflow/tensorboard) — Event file I/O patterns
- [torch-audit](https://pypi.org/project/torch-audit/) — Reference for training-time checks and pitfalls

---
*Pitfalls research for: PyTorch Training Observation Library*
*Researched: 2026-06-08*
