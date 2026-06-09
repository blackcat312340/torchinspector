---
id: "01-PLAN"
plan: "01"
objective: "Run bench_overhead.py on real hardware, collect data, verify <5% overhead"
wave: 1
depends_on: []
files_modified:
  - "benchmarks/bench_overhead.py"
  - "docs/faq.md"
autonomous: true
requirements: ["ECOS-01"]
---

# Plan 01: Performance Benchmarks

**Wave:** 1
**Objective:** Run bench_overhead.py on 3 models (MNIST CNN, CIFAR ResNet, Transformer), collect timing data, verify <5% overhead at default settings. Update FAQ with real numbers.

## Tasks

### Task 08-01-01: Run benchmarks
`python benchmarks/bench_overhead.py` — collect real timing data. If torchvision not available, use inline model definitions.

### Task 08-01-02: Verify <5% overhead
Check output: for each model, overhead should be <5%. If not, identify bottleneck and document.

### Task 08-01-03: Update FAQ with real numbers
Replace placeholder "typically <5%" with actual measured numbers.

<automated>
```bash
python benchmarks/bench_overhead.py
```
</automated>
