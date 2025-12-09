# GPU 内存限制配置指南

## 功能说明

现在系统支持两种方式来防止 GPU OOM（内存不足）错误：

1. **GPU 内存使用百分比限制**：限制模型加载时使用的 GPU 内存百分比
2. **PyTorch 内存管理**：启用 expandable_segments 来减少内存碎片

## 环境变量配置

### 1. 限制 GPU 内存使用百分比

```bash
# 使用 80% 的 GPU 内存（推荐）
export LBS_GPU_MEMORY_LIMIT=0.8

# 使用 70% 的 GPU 内存（更保守）
export LBS_GPU_MEMORY_LIMIT=0.7

# 使用 50% 的 GPU 内存（非常保守）
export LBS_GPU_MEMORY_LIMIT=0.5

# 禁用限制（默认，使用全部可用内存）
export LBS_GPU_MEMORY_LIMIT=0.0
```

### 2. 启用 PyTorch 内存管理

```bash
# 启用 expandable_segments（默认启用，有助于减少内存碎片）
export LBS_ENABLE_MEMORY_MANAGEMENT=1

# 禁用
export LBS_ENABLE_MEMORY_MANAGEMENT=0
```

### 3. 其他相关配置

```bash
# 禁用模型缓存（推荐用于 NeedleInAHaystack，因为会自动使用 uncached 模式）
export LBS_USE_MODEL_CACHE=0

# 使用 float16 减少内存使用
export LBS_TORCH_DTYPE=float16

# 组合使用（推荐配置）
export LBS_GPU_MEMORY_LIMIT=0.8
export LBS_ENABLE_MEMORY_MANAGEMENT=1
export LBS_TORCH_DTYPE=float16
export LBS_USE_MODEL_CACHE=0
```

## 使用示例

### 示例 1: 限制 GPU 内存为 80%

```bash
export LBS_GPU_MEMORY_LIMIT=0.8
export LBS_ENABLE_MEMORY_MANAGEMENT=1
cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

### 示例 2: 非常保守的配置（适合内存紧张的情况）

```bash
export LBS_GPU_MEMORY_LIMIT=0.6
export LBS_ENABLE_MEMORY_MANAGEMENT=1
export LBS_TORCH_DTYPE=float16
export LBS_USE_MODEL_CACHE=0
cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

### 示例 3: 完全禁用限制（不推荐，可能导致 OOM）

```bash
export LBS_GPU_MEMORY_LIMIT=0.0
cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

## 工作原理

1. **max_memory 参数**：
   - 当 `LBS_GPU_MEMORY_LIMIT > 0` 时，系统会计算每个 GPU 的最大可用内存
   - 在加载模型时，会传递 `max_memory` 参数给 `AutoModelForCausalLM.from_pretrained()`
   - Transformers 库会自动将模型分配到 GPU 和 CPU，确保不超过限制

2. **expandable_segments**：
   - PyTorch 的内存分配器会使用可扩展的内存段
   - 这有助于减少内存碎片，提高内存利用率
   - 即使内存使用接近限制，也能更好地分配内存

3. **NeedleInAHaystack 特殊处理**：
   - NeedleInAHaystack 数据集会自动使用 uncached 模式
   - 模型在使用后立即释放，不会保留在缓存中
   - 这确保了每次测试都有足够的内存

## 注意事项

1. **内存限制是硬限制**：
   - 如果模型需要的内存超过限制，可能会失败
   - 建议根据模型大小设置合适的限制

2. **模型缓存 vs 内存限制**：
   - 如果启用了模型缓存，多个模型可能会累积占用内存
   - 对于 NeedleInAHaystack，建议禁用缓存（会自动使用 uncached 模式）

3. **float16 vs float32**：
   - 使用 float16 可以将内存使用减少一半
   - 但可能会影响模型精度（通常影响很小）

4. **监控内存使用**：
   - 查看日志中的 `[Runner] GPU memory limit set: ...` 信息
   - 使用 `nvidia-smi` 监控实际内存使用

## 故障排除

如果仍然遇到 OOM 错误：

1. **降低内存限制**：
   ```bash
   export LBS_GPU_MEMORY_LIMIT=0.5  # 降低到 50%
   ```

2. **使用 float16**：
   ```bash
   export LBS_TORCH_DTYPE=float16
   ```

3. **禁用模型缓存**：
   ```bash
   export LBS_USE_MODEL_CACHE=0
   ```

4. **清理 GPU 内存**：
   ```bash
   cd /root/LLM-Bench-Shower
   /root/miniconda3/envs/bench/bin/python clear_gpu_memory.py
   ```

5. **重启服务器**：
   - 停止当前服务器进程
   - 清理 GPU 内存
   - 使用新的配置重新启动


