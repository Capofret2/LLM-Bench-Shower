# Qwen3 模型配置指南

## 问题修复

已修复代码中硬编码的 4096 tokens 限制，现在可以正确识别和使用 Qwen3-8B 的完整上下文长度（128K tokens）。

## 配置步骤

### 1. 启用 trust_remote_code

Qwen3 模型需要 `trust_remote_code=True` 才能正确加载。设置环境变量：

```bash
export LBS_TRUST_REMOTE_CODE=1
```

### 2. 推荐配置（低精度启动）

对于 Qwen3-8B（128K 上下文），**强烈推荐使用低精度**以节省显存和加快推理：

#### 快速配置（推荐）

```bash
# 启用 trust_remote_code（必需）
export LBS_TRUST_REMOTE_CODE=1

# 使用 bfloat16 低精度（推荐，节省约 50% 显存）
export LBS_TORCH_DTYPE=bfloat16
# 如果 GPU 不支持 bfloat16，使用 float16：
# export LBS_TORCH_DTYPE=float16

# GPU 内存限制（根据你的 GPU 容量调整）
export LBS_GPU_MEMORY_LIMIT=0.8  # 使用 80% GPU 内存

# 启用内存管理
export LBS_ENABLE_MEMORY_MANAGEMENT=1

# 禁用模型缓存（对于长上下文测试，建议禁用以避免内存问题）
export LBS_USE_MODEL_CACHE=0
```

**详细说明请参考：[QWEN3_LOW_PRECISION.md](./QWEN3_LOW_PRECISION.md)**

### 3. 启动服务器

```bash
cd /root/LLM-Bench-Shower/LLMBenchShower/backend
python server.py
```

### 4. 在前端使用

在前端界面中，输入模型路径：
```
/root/autodl-tmp/models/Qwen/Qwen3-8B
```

## 模型最大长度检测

代码现在会自动检测模型的最大上下文长度：

1. **优先使用 `tokenizer.model_max_length`**
2. **其次使用 `model.config.max_position_embeddings`**
3. **也检查 `model.config.seq_length`**（某些模型如 Qwen 使用此属性）
4. **过滤异常大的值**（如 `int(1e30)`）
5. **不再硬编码 4096 限制**

对于 Qwen3-8B，应该能正确检测到 128K（131072 tokens）的最大长度。

## 注意事项

1. **Qwen3-VL-8B-Instruct 不支持**：这是视觉语言模型，当前代码只支持纯文本模型。请使用 `Qwen3-8B` 而不是 `Qwen3-VL-8B-Instruct`。

2. **长上下文测试**：Qwen3-8B 支持 128K 上下文，测试时会自动调整上下文长度以适应模型能力。

3. **显存需求**：128K 上下文需要大量显存。如果遇到 OOM，可以：
   - 降低 `LBS_GPU_MEMORY_LIMIT`
   - 使用 `float16` 而不是 `bfloat16`
   - 减少测试组合数量

4. **Transformers 版本**：确保使用最新版本的 transformers：
   ```bash
   pip install --upgrade transformers
   ```
   如果仍然报错，可能需要从源码安装：
   ```bash
   pip install git+https://github.com/huggingface/transformers.git
   ```

## 验证配置

运行测试后，查看后端日志，应该能看到：
```
[NeedleBench] Detected model max length: 131072 tokens
[NeedleBench] Model max length: 131072, adjusting test parameters accordingly
```

如果看到 4096 或更小的值，说明检测失败，请检查：
1. 是否设置了 `LBS_TRUST_REMOTE_CODE=1`
2. Transformers 版本是否足够新
3. 模型路径是否正确

