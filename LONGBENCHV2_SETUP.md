# LongBenchV2 数据集导入指南

## 当前配置

**数据集路径**: `/root/longbench/LongBenchv2`

**配置文件**: `configs/default_dataset_paths.json`
```json
{
    "LongBenchV2": "/root/longbench/LongBenchv2"
}
```

## 数据文件要求

LongBench v2 使用**单个 JSON 文件**格式，包含所有域的数据。

**必需文件**: `/root/longbench/LongBenchv2/data.json`

## 数据获取方式

### 方式 1: 从 Hugging Face 下载（推荐）

根据 [LongBench v2 README](https://github.com/THUDM/LongBench-v2)，数据可以从 Hugging Face 下载：

```bash
# 使用 wget 下载
cd /root/longbench/LongBenchv2
wget https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json

# 或者使用 curl
curl -L https://huggingface.co/datasets/THUDM/LongBench-v2/resolve/main/data.json -o data.json
```

### 方式 2: 使用 Python datasets 库

```python
from datasets import load_dataset

# 下载数据
dataset = load_dataset('THUDM/LongBench-v2', split='train')

# 保存为 JSON 文件
import json
with open('/root/longbench/LongBenchv2/data.json', 'w', encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=2)
```

## 数据格式

LongBench v2 的数据格式如下：

```json
[
    {
        "_id": "Unique identifier",
        "domain": "The primary domain category",
        "sub_domain": "The specific sub-domain category",
        "difficulty": "easy" or "hard",
        "length": "short", "medium", or "long",
        "question": "The input/command for the task",
        "choice_A": "Option A",
        "choice_B": "Option B",
        "choice_C": "Option C",
        "choice_D": "Option D",
        "answer": "A", "B", "C", or "D",
        "context": "The long context required for the task"
    },
    ...
]
```

## 支持的域（Sub-datasets）

根据配置，系统支持以下域：

1. `Code_Repository_Understanding` - 代码仓库理解
2. `Long-dialogue_History_Understanding` - 长对话历史理解
3. `Long_In-context_Learning` - 长上下文学习

## 代码实现

代码已更新以支持：

1. **自动路径检测**: 优先从 `/root/longbench/LongBenchv2/data.json` 加载
2. **域过滤**: 根据 `domain` 和 `sub_domain` 字段自动过滤数据
3. **路径回退**: 如果主路径不存在，会尝试其他路径
4. **格式兼容**: 支持 JSON 和 JSONL 两种格式

## 验证数据

下载数据后，可以运行以下命令验证：

```bash
# 检查文件是否存在
ls -lh /root/longbench/LongBenchv2/data.json

# 检查数据格式
python3 -c "
import json
with open('/root/longbench/LongBenchv2/data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f'Total items: {len(data)}')
print(f'Sample item keys: {list(data[0].keys()) if len(data) > 0 else \"No data\"}')
print(f'Domains: {set(item.get(\"domain\", \"unknown\") for item in data[:100])}')
"
```

## 使用说明

1. **下载数据文件**到 `/root/longbench/LongBenchv2/data.json`
2. **重启后端服务**（如果正在运行）
3. **在前端选择数据集**：
   - 选择 `LongBenchV2` 加载所有域
   - 或选择特定域（如 `Code_Repository_Understanding`）

## 故障排除

### 问题 1: 文件不存在

**错误**: `FileNotFoundError: Failed to load dataset...`

**解决**: 
- 确认数据文件已下载到正确位置
- 检查文件权限：`chmod 644 /root/longbench/LongBenchv2/data.json`

### 问题 2: JSON 格式错误

**错误**: `JSONDecodeError`

**解决**:
- 验证 JSON 格式：`python3 -m json.tool /root/longbench/LongBenchv2/data.json > /dev/null`
- 重新下载数据文件

### 问题 3: 域数据为空

**错误**: 加载特定域时返回空列表

**解决**:
- 检查数据中的 `domain` 字段值
- 查看日志中的域映射信息
- 可能需要调整域名称映射

## 参考链接

- [LongBench v2 GitHub](https://github.com/THUDM/LongBench-v2)
- [LongBench v2 Hugging Face](https://huggingface.co/datasets/THUDM/LongBench-v2)
- [LongBench v2 Paper](https://arxiv.org/abs/2412.15204)

