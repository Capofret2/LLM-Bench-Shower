# 使用 16 精度 Llama-2-7b 模型配置指南

## 推荐配置（bfloat16）

对于 Llama-2 模型，推荐使用 **bfloat16**，因为：
- 数值稳定性更好（比 float16）
- Llama-2 模型通常使用 bfloat16 训练
- 可以减少内存使用约 50%
- 推理速度更快

### 启动命令

```bash
# 设置环境变量
export LBS_TORCH_DTYPE=bfloat16
export LBS_ENABLE_MEMORY_MANAGEMENT=1
export LBS_GPU_MEMORY_LIMIT=0.8
export LBS_USE_MODEL_CACHE=0

# 启动服务器
cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

## 备选配置（float16）

如果 bfloat16 不可用（某些 GPU 不支持），可以使用 float16：

```bash
export LBS_TORCH_DTYPE=float16
export LBS_ENABLE_MEMORY_MANAGEMENT=1
export LBS_GPU_MEMORY_LIMIT=0.8
export LBS_USE_MODEL_CACHE=0

cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

## 内存对比

| 精度 | 模型大小 | 内存使用（约） | 速度 |
|------|----------|----------------|------|
| float32 | 13.4 GB | ~14 GB | 慢 |
| float16 | 6.7 GB | ~7 GB | 快 |
| bfloat16 | 6.7 GB | ~7 GB | 快（推荐） |

## 验证配置

启动服务器后，查看日志应该看到：

```
[Runner] Loading model from /path/to/model...
[Runner] Model loaded successfully on device: auto
[Runner] GPU memory after loading: ~7 GB / 23.64 GB  # 如果是 16 精度，内存应该减半
```

## 注意事项

1. **GPU 支持**：
   - bfloat16 需要 Ampere 架构或更新的 GPU（RTX 30 系列及以上，A100 等）
   - 如果 GPU 不支持 bfloat16，会自动降级到 float16 或 float32

2. **模型文件**：
   - 如果模型已经是 16 精度格式，加载会更快
   - 如果模型是 float32，加载时会自动转换（需要一些时间）

3. **精度影响**：
   - 16 精度对大多数任务影响很小
   - 对于需要高精度的任务，可能需要使用 float32

## 完整配置示例

```bash
# 停止当前服务器（如果需要）

# 设置所有推荐的环境变量
export LBS_TORCH_DTYPE=bfloat16
export LBS_ENABLE_MEMORY_MANAGEMENT=1
export LBS_GPU_MEMORY_LIMIT=0.8
export LBS_USE_MODEL_CACHE=0
export LBS_LOCAL_DEVICE_MAP=auto

# 启动服务器
cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

## 故障排除

如果遇到问题：

1. **检查 GPU 是否支持 bfloat16**：
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_capability(0))  # 需要 (8, 0) 或更高
   ```

2. **如果 bfloat16 不可用，使用 float16**：
   ```bash
   export LBS_TORCH_DTYPE=float16
   ```

3. **查看模型加载日志**：
   - 检查是否有 dtype 相关的警告
   - 确认实际使用的 dtype

