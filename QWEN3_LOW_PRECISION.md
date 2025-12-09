# Qwen3 低精度启动指南

## 快速开始

### 方法 1：使用 bfloat16（推荐）

bfloat16 是 Qwen3 模型的最佳选择，因为：
- ✅ 数值稳定性好（比 float16 更好）
- ✅ 减少约 50% 的显存使用
- ✅ 推理速度更快
- ✅ Qwen3 模型通常使用 bfloat16 训练

```bash
# 设置环境变量
export LBS_TORCH_DTYPE=bfloat16
export LBS_TRUST_REMOTE_CODE=1
export LBS_ENABLE_MEMORY_MANAGEMENT=1
export LBS_GPU_MEMORY_LIMIT=0.8

# 启动服务器
cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

### 方法 2：使用 float16（兼容性更好）

如果 GPU 不支持 bfloat16（较老的 GPU），使用 float16：

```bash
export LBS_TORCH_DTYPE=float16
export LBS_TRUST_REMOTE_CODE=1
export LBS_ENABLE_MEMORY_MANAGEMENT=1
export LBS_GPU_MEMORY_LIMIT=0.8

cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

## 完整配置示例

### 推荐配置（Qwen3-8B，128K 上下文）

```bash
# 精度设置（必需）
export LBS_TORCH_DTYPE=bfloat16  # 或 float16

# Qwen3 必需设置
export LBS_TRUST_REMOTE_CODE=1

# 内存管理（推荐）
export LBS_ENABLE_MEMORY_MANAGEMENT=1
export LBS_GPU_MEMORY_LIMIT=0.8  # 使用 80% GPU 内存

# 对于长上下文测试，建议禁用缓存
export LBS_USE_MODEL_CACHE=0

# 启动服务器
cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

### 保守配置（显存有限时）

如果显存不足，可以进一步优化：

```bash
export LBS_TORCH_DTYPE=float16  # float16 通常比 bfloat16 更省显存
export LBS_TRUST_REMOTE_CODE=1
export LBS_ENABLE_MEMORY_MANAGEMENT=1
export LBS_GPU_MEMORY_LIMIT=0.6  # 只使用 60% GPU 内存
export LBS_USE_MODEL_CACHE=0

cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

## 精度对比

| 精度 | 显存使用（Qwen3-8B） | 速度 | 数值稳定性 | 推荐场景 |
|------|---------------------|------|-----------|---------|
| **float32** | ~16 GB | 慢 | 最高 | 需要最高精度时 |
| **bfloat16** | ~8 GB | 快 | 好 | **推荐**（Ampere+ GPU） |
| **float16** | ~8 GB | 快 | 中等 | 兼容性更好（所有 GPU） |

## 验证配置

启动服务器后，查看日志确认：

### 1. 检查 dtype 设置
```
[Runner] Configuring model with torch_dtype: bfloat16
```

### 2. 检查实际使用的 dtype
```
[Runner] Model loaded successfully on device: auto
[Runner] Actual model dtype: torch.bfloat16
```

### 3. 检查显存使用
```
[Runner] GPU memory after loading: ~8 GB / 23.64 GB
```
如果是 16 精度，显存应该约为 float32 的一半。

## GPU 兼容性检查

### 检查 GPU 是否支持 bfloat16

```python
import torch

# 检查 CUDA 是否可用
print(f"CUDA available: {torch.cuda.is_available()}")

# 检查 GPU 计算能力（需要 8.0+ 才完全支持 bfloat16）
if torch.cuda.is_available():
    capability = torch.cuda.get_device_capability(0)
    print(f"GPU capability: {capability}")
    if capability >= (8, 0):
        print("✅ GPU supports bfloat16 natively")
    else:
        print("⚠️  GPU may not fully support bfloat16, consider using float16")
```

### GPU 支持情况

- **RTX 30 系列（Ampere）及以上**：✅ 完全支持 bfloat16
- **A100, H100**：✅ 完全支持 bfloat16
- **RTX 20 系列（Turing）**：⚠️ 部分支持，建议使用 float16
- **RTX 10 系列（Pascal）及更早**：❌ 不支持，必须使用 float16

## 常见问题

### Q1: 如何知道当前使用的是哪种精度？

查看后端日志中的这两行：
```
[Runner] Configuring model with torch_dtype: bfloat16
[Runner] Actual model dtype: torch.bfloat16
```

### Q2: bfloat16 和 float16 有什么区别？

- **bfloat16**：数值范围与 float32 相同，但精度降低。适合训练和推理。
- **float16**：数值范围和精度都降低。兼容性更好，但可能在某些情况下数值不稳定。

### Q3: 为什么推荐 bfloat16？

1. Qwen3 模型通常使用 bfloat16 训练
2. 数值稳定性更好，不容易出现 NaN 或 Inf
3. 在支持的 GPU 上性能更好

### Q4: 如果遇到 OOM（显存不足）怎么办？

1. **降低 GPU 内存限制**：
   ```bash
   export LBS_GPU_MEMORY_LIMIT=0.5  # 只使用 50%
   ```

2. **使用 float16 而不是 bfloat16**：
   ```bash
   export LBS_TORCH_DTYPE=float16
   ```

3. **禁用模型缓存**：
   ```bash
   export LBS_USE_MODEL_CACHE=0
   ```

4. **减少测试组合数量**（在代码中调整）

### Q5: 精度会影响测试结果吗？

对于大多数任务，16 精度对结果影响很小（通常 < 1%）。如果发现结果异常，可以尝试使用 float32 进行对比。

## 一键启动脚本

创建 `start_qwen3_low_precision.sh`：

```bash
#!/bin/bash

# Qwen3 低精度启动脚本

# 设置精度（bfloat16 或 float16）
export LBS_TORCH_DTYPE=bfloat16

# Qwen3 必需
export LBS_TRUST_REMOTE_CODE=1

# 内存管理
export LBS_ENABLE_MEMORY_MANAGEMENT=1
export LBS_GPU_MEMORY_LIMIT=0.8

# 禁用缓存（长上下文测试）
export LBS_USE_MODEL_CACHE=0

# 启动服务器
cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

使用方法：
```bash
chmod +x start_qwen3_low_precision.sh
./start_qwen3_low_precision.sh
```

## 性能优化建议

1. **首次加载**：如果模型文件是 float32，首次加载时会自动转换，需要一些时间
2. **后续加载**：如果启用了缓存，后续加载会更快
3. **长上下文**：对于 128K 上下文，建议禁用缓存以避免显存问题
4. **批量测试**：使用低精度可以显著加快批量测试速度

## 总结

对于 Qwen3-8B：
- ✅ **推荐**：`LBS_TORCH_DTYPE=bfloat16`（如果 GPU 支持）
- ✅ **备选**：`LBS_TORCH_DTYPE=float16`（兼容性更好）
- ❌ **不推荐**：`LBS_TORCH_DTYPE=float32`（显存占用大，速度慢）

记住同时设置 `LBS_TRUST_REMOTE_CODE=1`，这是 Qwen3 模型加载的必需选项。

