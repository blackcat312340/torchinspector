# Phase 15: Utils + TrendMonitor Extensions - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.

**Date:** 2026-06-15
**Phase:** 15-UtilsTrendMonitorExtensions
**Areas discussed:** Transformer 层检测, FlashAttention 兼容策略, TrendMonitor 扩展细节, 架构自动检测

---

## Transformer 层检测

### Q1: MHA 检测范围

| Option | Description | Selected |
|--------|-------------|----------|
| 只检测 nn.MHA | PyTorch 原生，简单可靠 | ✓ |
| nn.MHA + HuggingFace | 同时检测 HF 自定义 attention | |
| 属性名匹配 | 检测所有包含 Q/K/V 投影的模块 | |

**User's choice:** 只检测 nn.MHA (推荐)

### Q2: 检测方式

| Option | Description | Selected |
|--------|-------------|----------|
| isinstance 检测 | isinstance(module, nn.MultiheadAttention) | ✓ |
| 用户手动指定 | 用户指定要监控的 attention 层名称 | |

**User's choice:** isinstance 检测 (推荐)

---

## FlashAttention 兼容策略

### Q1: 强制时机

| Option | Description | Selected |
|--------|-------------|----------|
| 采集时强制 | 只在 log_interval 时强制 math backend | ✓ |
| 全程强制 | 整个训练过程都用 math backend | |
| 检测并跳过 | 检测 FlashAttention 则跳过采集 | |

**User's choice:** 采集时强制 (推荐)

### Q2: 强制机制

| Option | Description | Selected |
|--------|-------------|----------|
| sdpa_kernel 上下文管理器 | torch.nn.attention.sdpa_kernel(SDPBackend.MATH) | ✓ |
| forward_pre_hook 修改 | 修改 MHA 的 need_weights 参数 | |

**User's choice:** sdpa_kernel 上下文管理器 (推荐)

---

## TrendMonitor 扩展细节

### Q1: check 方法设计

| Option | Description | Selected |
|--------|-------------|----------|
| 单一 check_attention() | 同时处理 entropy 趋势和 head 健康 | ✓ |
| 分开两个方法 | check_attention_entropy() + check_head_health() | |

**User's choice:** 单一 check_attention() (推荐)

### Q2: QKV 检查方法

| Option | Description | Selected |
|--------|-------------|----------|
| 单一 check_qkv() | 处理条件数、谱范数、有效秩 | ✓ |
| 合并到 check_attention() | 减少方法数但职责混合 | |

**User's choice:** 单一 check_qkv() (推荐)

### Q3: 相关性规则

| Option | Description | Selected |
|--------|-------------|----------|
| 2 条规则 | attention_collapse + convergence_slow → WARN；qkv_condition_high + gradient_anomaly → WARN | ✓ |
| 只做 attention 规则 | QKV 规则留给 Phase 17 | |

**User's choice:** 2 条规则 (推荐)

---

## 架构自动检测

### Q1: 自动检测方式

| Option | Description | Selected |
|--------|-------------|----------|
| 扩展 classify_architecture() | 检测 Transformer 模型，自动启用分析 | ✓ |
| 用户手动指定 | transformer=True 参数 | |

**User's choice:** 扩展 classify_architecture() (推荐)

---

## Claude's Discretion

- list_transformer_layers() 放在 utils.py 中
- check_attention() 和 check_qkv() 签名与 check_wgr() 一致

## Deferred Ideas

None
