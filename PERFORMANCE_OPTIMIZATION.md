# 性能优化指南

## 概述

本指南说明如何优化 LLM Bench Shower 的性能，特别是在 RTX 4090 (24GB) 等高性能 GPU 上的使用。

## 已实施的优化

### 1. 模型量化（默认启用）
- **默认使用 float16**：模型加载时默认使用 `float16` 精度，相比 `float32` 可以：
  - 减少 50% 的显存占用
  - 提升约 2x 的推理速度
  - 在 RTX 4090 上完全支持 float16 加速

### 2. 贪心解码（选择题场景）
- **C-Eval 数据集**：使用贪心解码（`temperature=0.0`, `do_sample=False`）
  - 速度提升：约 2-3x
  - 对于选择题，贪心解码既快速又准确
  - `max_new_tokens` 从 512 降至 128（选择题答案很短）

### 3. 性能监控
- 自动记录每个样本的处理时间
- 显示总体性能统计（总时间、平均时间、处理速度）

## 环境变量配置

### 推荐配置（RTX 4090）

```bash
# 使用 float16 量化（默认已启用）
export LBS_TORCH_DTYPE=float16

# 或者使用 bfloat16（某些模型可能效果更好）
# export LBS_TORCH_DTYPE=bfloat16

# 使用自动设备映射（默认）
export LBS_LOCAL_DEVICE_MAP=auto

# 启用模型缓存（默认已启用）
export LBS_USE_MODEL_CACHE=1
export LBS_MAX_CACHED_LOCAL_MODELS=2  # 根据显存调整

# GPU 内存限制（可选，0.0 表示无限制）
export LBS_GPU_MEMORY_LIMIT=0.9  # 使用 90% 的显存
```

### 性能对比

| 配置 | 显存占用 | 推理速度 | 适用场景 |
|------|---------|---------|---------|
| float32 | 100% | 1x | 需要最高精度 |
| float16 (默认) | 50% | ~2x | **推荐：平衡速度和精度** |
| bfloat16 | 50% | ~2x | 某些模型可能更好 |
| 贪心解码 | - | +2-3x | 选择题场景 |

## 进一步优化建议

### 1. 批处理（未来计划）
当前实现是逐个处理样本。如果数据集支持，可以实施批处理：
- 批处理大小：根据显存调整（建议 4-8）
- 预期提升：2-4x 速度

### 2. 使用量化模型
如果模型支持，可以使用 8-bit 或 4-bit 量化：
- 8-bit：显存减少 75%，速度提升 1.5-2x
- 4-bit：显存减少 87.5%，速度提升 2-3x
- 注意：可能略微降低精度

### 3. 使用 Flash Attention（如果模型支持）
- 对于长上下文场景，Flash Attention 可以显著提升速度
- 需要模型和 transformers 库支持

### 4. 多 GPU 推理
如果有多张 GPU，可以：
- 使用 `device_map="auto"` 自动分配
- 或手动指定：`LBS_LOCAL_DEVICE_MAP=cuda:0,cuda:1`

## 性能监控

运行测试时，日志会显示：
```
[C-Eval] Model device: cuda:0
[C-Eval] Model dtype: torch.float16
[C-Eval] Item 1 processed in 0.45s
[C-Eval] Item 10 processed in 0.42s
...
[C-Eval] ⏱️  Performance: Total time: 45.2s, Average: 0.45s/item, Speed: 2.21 items/s
```

## 常见问题

### Q: 为什么速度还是很慢？
A: 检查以下几点：
1. 确认模型使用 GPU：查看日志中的 `Model device: cuda:0`
2. 确认使用 float16：查看日志中的 `Model dtype: torch.float16`
3. 检查 GPU 利用率：使用 `nvidia-smi` 查看 GPU 使用率
4. 模型大小：大模型（>7B）本身就需要更多时间

### Q: 如何确认优化生效？
A: 查看日志：
- `Model device: cuda:0` - 使用 GPU
- `Model dtype: torch.float16` - 使用 float16
- `Item X processed in X.XXs` - 每个样本的处理时间
- `Speed: X.XX items/s` - 总体处理速度

### Q: 可以同时运行多个测试吗？
A: 可以，但需要注意：
- 显存限制：每个模型会占用显存
- 建议：使用 `LBS_MAX_CACHED_LOCAL_MODELS=2` 限制缓存模型数量
- 或者：等待一个测试完成后再开始下一个

## 性能基准（参考）

在 RTX 4090 上，使用 float16 + 贪心解码：
- **小模型（<3B）**：~5-10 items/s
- **中等模型（3-7B）**：~2-5 items/s
- **大模型（>7B）**：~0.5-2 items/s

*注：实际速度取决于模型架构、输入长度、输出长度等因素*

