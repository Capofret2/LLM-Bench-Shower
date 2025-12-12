# LongBench 和 LongBenchV2 数据集完整性报告

**生成时间**: 2025-12-11  
**检查范围**: LongBench 和 LongBenchV2 数据集

---

## 执行摘要

### 总体状态
- ✅ **测试数据完整**: 所有配置的子数据集文件都存在且格式正确
- ❌ **生产数据缺失**: 生产环境数据路径中未找到任何数据文件
- ⚠️ **数据量较小**: 测试数据每个子数据集仅包含 5 条样本（可能仅为测试用途）

---

## 一、LongBench 数据集

### 1.1 配置信息

**子数据集列表**（来自 `configs/sub_datasets.json`）:
- `LongBench` (主数据集，包含所有子数据集)
- `2wikimqa`
- `dureader`
- `gov_report`
- `hotpotqa`
- `narrativeqa`

**配置路径**:
- 生产路径: `/root/share/datasets/LongBench/data`
- 测试路径: `/root/LLM-Bench-Shower/tests/test_data/LongBench/`

### 1.2 测试数据完整性检查

| 子数据集 | 文件路径 | 状态 | 行数 | JSON 格式 | 必需字段 |
|---------|---------|------|------|-----------|---------|
| `2wikimqa` | `tests/test_data/LongBench/2wikimqa.jsonl` | ✅ 存在 | 5 | ✅ 有效 | ✅ 完整 |
| `dureader` | `tests/test_data/LongBench/dureader.jsonl` | ✅ 存在 | 5 | ✅ 有效 | ✅ 完整 |
| `gov_report` | `tests/test_data/LongBench/gov_report.jsonl` | ✅ 存在 | 5 | ✅ 有效 | ✅ 完整 |
| `hotpotqa` | `tests/test_data/LongBench/hotpotqa.jsonl` | ✅ 存在 | 5 | ✅ 有效 | ✅ 完整 |
| `narrativeqa` | `tests/test_data/LongBench/narrativeqa.jsonl` | ✅ 存在 | 5 | ✅ 有效 | ✅ 完整 |

**测试数据统计**:
- ✅ 所有 5 个子数据集文件都存在
- ✅ 所有文件 JSON 格式正确
- ✅ 所有文件包含必需字段：`question`, `context`, `answers`, `qid`
- ⚠️ 每个文件仅包含 5 条样本（可能仅为测试用途）

**样本数据结构**:
```json
{
    "question": "Sample question 1 for 2wikimqa",
    "context": "This is a long context document...",
    "answers": ["Answer to question 1"],
    "qid": "2wikimqa-1"
}
```

### 1.3 生产数据完整性检查

**路径**: `/root/share/datasets/LongBench/data/`

**状态**: ❌ **数据文件不存在**

- 目录存在但为空
- 未找到任何 `.jsonl` 文件
- 代码会回退到测试数据路径

### 1.4 代码加载逻辑

**当前实现** (`backend/bench/longbench/benchmarker.py`):
1. 仅从测试数据路径加载：`tests/test_data/LongBench/{subdataset}.jsonl`
2. 如果文件不存在，抛出 `FileNotFoundError`
3. **不检查生产数据路径**

**问题**:
- ⚠️ 代码硬编码只检查测试路径，未实现生产路径的回退机制
- ⚠️ 如果测试数据被删除，将无法加载数据

---

## 二、LongBenchV2 数据集

### 2.1 配置信息

**子数据集列表**（来自 `configs/sub_datasets.json`）:
- `LongBenchV2` (主数据集，包含所有域)
- `Code_Repository_Understanding`
- `Long-dialogue_History_Understanding`
- `Long_In-context_Learning`

**配置路径**:
- 生产路径: `/root/share/datasets/LongBenchV2/domains`
- 测试路径: `/root/LLM-Bench-Shower/tests/test_data/LongBenchV2/`

### 2.2 测试数据完整性检查

| 域名称 | 文件路径 | 状态 | 行数 | JSON 格式 | 必需字段 |
|-------|---------|------|------|-----------|---------|
| `Code_Repository_Understanding` | `tests/test_data/LongBenchV2/Code_Repository_Understanding/data.jsonl` | ✅ 存在 | 5 | ✅ 有效 | ✅ 完整 |
| `Long-dialogue_History_Understanding` | `tests/test_data/LongBenchV2/Long-dialogue_History_Understanding/data.jsonl` | ✅ 存在 | 5 | ✅ 有效 | ✅ 完整 |
| `Long_In-context_Learning` | `tests/test_data/LongBenchV2/Long_In-context_Learning/data.jsonl` | ✅ 存在 | 5 | ✅ 有效 | ✅ 完整 |

**测试数据统计**:
- ✅ 所有 3 个域的数据文件都存在
- ✅ 所有文件 JSON 格式正确
- ✅ 所有文件包含必需字段：`question`, `context`, `answer`, `id`
- ⚠️ 每个文件仅包含 5 条样本（可能仅为测试用途）

**样本数据结构**:
```json
{
    "question": "Sample question 1 for Code_Repository_Understanding",
    "instruction": "Instruction for Code_Repository_Understanding task 1",
    "context": "This is a long context for Code_Repository_Understanding...",
    "answer": "Answer to Code_Repository_Understanding question 1",
    "id": "Code_Repository_Understanding-1"
}
```

### 2.3 生产数据完整性检查

**路径**: `/root/share/datasets/LongBenchV2/domains/`

**状态**: ❌ **数据文件不存在**

- 目录存在但为空
- 未找到任何 `.jsonl` 文件
- 代码会回退到测试数据路径

### 2.4 代码加载逻辑

**当前实现** (`backend/bench/longbench_v2/benchmarker.py`):
1. 仅从测试数据路径加载：`tests/test_data/LongBenchV2/{domain}/data.jsonl`
2. 如果文件不存在，抛出 `FileNotFoundError`
3. **不检查生产数据路径**

**问题**:
- ⚠️ 代码硬编码只检查测试路径，未实现生产路径的回退机制
- ⚠️ 如果测试数据被删除，将无法加载数据

---

## 三、问题总结

### 3.1 严重问题

1. **生产数据完全缺失**
   - 两个数据集的生产路径都为空
   - 如果测试数据被删除，系统将无法工作

2. **代码未实现路径回退**
   - 代码只检查测试路径，未检查生产路径
   - 不符合其他数据集（如 C-Eval、CMMMU）的实现模式

### 3.2 潜在问题

1. **测试数据量过小**
   - 每个子数据集仅 5 条样本
   - 可能仅为测试用途，不适合实际评测

2. **路径硬编码**
   - 代码中硬编码了测试数据路径
   - 未使用 `get_dataset_path()` 获取生产路径

---

## 四、建议修复方案

### 4.1 短期修复（推荐）

**修改代码以支持路径回退**:

1. **LongBench** (`backend/bench/longbench/benchmarker.py`):
   ```python
   local_paths = [
       # 测试数据路径（优先）
       os.path.join(test_data_base, "LongBench", f"{subdataset_name}.jsonl"),
       # 生产数据路径（回退）
       os.path.join(self.dataset_path, f"{subdataset_name}.jsonl"),
   ]
   ```

2. **LongBenchV2** (`backend/bench/longbench_v2/benchmarker.py`):
   ```python
   local_paths = [
       # 测试数据路径（优先）
       os.path.join(test_data_base, "LongBenchV2", subdataset_name, "data.jsonl"),
       # 生产数据路径（回退）
       os.path.join(self.dataset_path, subdataset_name, "data.jsonl"),
   ]
   ```

### 4.2 长期修复

1. **补充生产数据**
   - 从官方仓库下载完整数据集
   - 放置到 `/root/share/datasets/LongBench/data/` 和 `/root/share/datasets/LongBenchV2/domains/`

2. **统一代码风格**
   - 参考 C-Eval 和 CMMMU 的实现
   - 统一路径检查逻辑

3. **数据验证**
   - 添加数据完整性验证
   - 检查必需字段是否存在

---

## 五、数据获取建议

### 5.1 LongBench

**官方仓库**: https://github.com/THUDM/LongBench

**数据下载**:
```bash
# 克隆仓库
git clone https://github.com/THUDM/LongBench.git

# 数据应放置在
/root/share/datasets/LongBench/data/
```

**预期文件结构**:
```
/root/share/datasets/LongBench/data/
├── 2wikimqa.jsonl
├── dureader.jsonl
├── gov_report.jsonl
├── hotpotqa.jsonl
└── narrativeqa.jsonl
```

### 5.2 LongBenchV2

**官方仓库**: https://github.com/THUDM/LongBench

**数据下载**:
```bash
# 数据应放置在
/root/share/datasets/LongBenchV2/domains/
```

**预期文件结构**:
```
/root/share/datasets/LongBenchV2/domains/
├── Code_Repository_Understanding/
│   └── data.jsonl
├── Long-dialogue_History_Understanding/
│   └── data.jsonl
└── Long_In-context_Learning/
    └── data.jsonl
```

---

## 六、测试建议

### 6.1 功能测试

1. **测试数据加载**
   ```python
   # 测试每个子数据集是否能正常加载
   benchmarker = LongBenchBenchmarker()
   for subdataset in ["2wikimqa", "dureader", "gov_report", "hotpotqa", "narrativeqa"]:
       data = benchmarker._load_dataset(subdataset)
       assert len(data) > 0
   ```

2. **测试路径回退**
   - 删除测试数据，验证是否能从生产路径加载
   - 或创建生产数据，验证优先级

### 6.2 数据质量测试

1. **字段完整性**
   - 检查每个样本是否包含必需字段
   - 验证字段类型是否正确

2. **数据量验证**
   - 确认生产数据包含足够的样本
   - 建议每个子数据集至少 100+ 样本

---

## 七、结论

### 7.1 当前状态

- ✅ **测试数据**: 完整且格式正确
- ❌ **生产数据**: 完全缺失
- ⚠️ **代码实现**: 未实现路径回退机制

### 7.2 风险评估

- **高风险**: 如果测试数据被删除，系统将无法工作
- **中风险**: 测试数据量过小，不适合实际评测
- **低风险**: 代码风格不一致，但不影响功能

### 7.3 优先级建议

1. **高优先级**: 修改代码支持路径回退（短期修复）
2. **中优先级**: 补充生产数据（长期修复）
3. **低优先级**: 统一代码风格（代码优化）

---

## 附录：检查命令

### A.1 快速检查脚本

```bash
#!/bin/bash
# 检查 LongBench 数据集
echo "=== LongBench ==="
for file in 2wikimqa dureader gov_report hotpotqa narrativeqa; do
    path="/root/LLM-Bench-Shower/tests/test_data/LongBench/${file}.jsonl"
    if [ -f "$path" ]; then
        lines=$(wc -l < "$path")
        echo "✅ $file: $lines lines"
    else
        echo "❌ $file: NOT FOUND"
    fi
done

# 检查 LongBenchV2 数据集
echo -e "\n=== LongBenchV2 ==="
for domain in Code_Repository_Understanding Long-dialogue_History_Understanding Long_In-context_Learning; do
    path="/root/LLM-Bench-Shower/tests/test_data/LongBenchV2/${domain}/data.jsonl"
    if [ -f "$path" ]; then
        lines=$(wc -l < "$path")
        echo "✅ $domain: $lines lines"
    else
        echo "❌ $domain: NOT FOUND"
    fi
done
```

### A.2 JSON 格式验证

```python
import json
import os

def validate_jsonl(filepath):
    """验证 JSONL 文件格式"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                if line.strip():
                    json.loads(line)
        return True, None
    except Exception as e:
        return False, str(e)

# 使用示例
file = "/root/LLM-Bench-Shower/tests/test_data/LongBench/2wikimqa.jsonl"
valid, error = validate_jsonl(file)
print(f"Valid: {valid}, Error: {error}")
```

---

**报告结束**

