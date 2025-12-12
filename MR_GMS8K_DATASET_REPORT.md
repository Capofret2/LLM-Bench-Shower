# MR-GMS8K 数据集导入方式与完整性报告

**生成时间**: 2025-12-11  
**检查范围**: MR-GMS8K 数据集导入方式和数据完整性

---

## 执行摘要

### 总体状态
- ✅ **数据文件完整**: 测试和生产数据文件都存在且格式正确
- ✅ **数据量充足**: 包含 2999 条样本
- ⚠️ **字段映射问题**: 数据文件字段名与代码期望的字段名不匹配
- ✅ **路径回退机制**: 代码已实现多路径回退机制

---

## 一、数据集配置信息

### 1.1 配置信息

**数据集名称**: `MR-GMS8K`  
**子数据集**: `MR-GMS8K` (单一数据集，无子数据集)

**配置路径**:
- 生产路径: `/root/share/datasets/MR-GMS8K`
- 测试路径: `/root/LLM-Bench-Shower/tests/test_data/MR-GSM8K/`

**注意**: 配置中使用 `MR-GMS8K`，但实际文件路径使用 `MR-GSM8K`（注意是 GSM 不是 GMS）

### 1.2 数据文件位置

| 位置 | 路径 | 状态 | 文件大小 |
|------|------|------|---------|
| 测试数据 | `/root/LLM-Bench-Shower/tests/test_data/MR-GSM8K/dataset/MR-GSM8K.json` | ✅ 存在 | 5.4 MB |
| 生产数据 | `/root/share/datasets/MR-GMS8K/MR-GSM8K/dataset/MR-GSM8K.json` | ✅ 存在 | 5.4 MB |

---

## 二、数据导入方式

### 2.1 代码实现

**文件**: `backend/bench/mr_gms8k/benchmarker.py`

**加载方法**: `_load_dataset()`

**路径回退机制**（按优先级）:
1. ✅ 测试数据路径: `tests/test_data/MR-GSM8K/dataset/MR-GSM8K.json`
2. ✅ 生产数据路径（MR-GSM8K）: `/root/share/datasets/MR-GMS8K/MR-GSM8K/dataset/MR-GSM8K.json`
3. ✅ 生产数据路径（MR-GMS8K）: `/root/share/datasets/MR-GMS8K/MR-GMS8K/dataset/MR-GMS8K.json`
4. ✅ 旧路径格式1: `/root/share/datasets/MR-GMS8K/MR-GMS8K/MR-GMS8K.json`
5. ✅ 旧路径格式2: `/root/share/datasets/MR-GMS8K/MR-GMS8K.json`

**优点**:
- ✅ 实现了完善的路径回退机制
- ✅ 支持多种路径格式，兼容性好
- ✅ 处理了命名不一致问题（MR-GMS8K vs MR-GSM8K）

### 2.2 数据加载逻辑

```python
def _load_dataset(self, subdataset_name: str = "MR-GMS8K") -> List[Dict]:
    # 1. 尝试多个路径（按优先级）
    # 2. 加载 JSON 文件
    # 3. 处理字典格式（转换为列表）
    # 4. 返回数据列表
```

**数据格式处理**:
- ✅ 支持列表格式（直接返回）
- ✅ 支持字典格式（自动提取列表值）
- ✅ JSON 格式验证

---

## 三、数据完整性检查

### 3.1 数据文件状态

| 检查项 | 测试数据 | 生产数据 | 状态 |
|--------|---------|---------|------|
| 文件存在 | ✅ | ✅ | 通过 |
| JSON 格式 | ✅ | ✅ | 通过 |
| 数据量 | 2999 条 | 2999 条 | 通过 |
| 文件大小 | 5.4 MB | 5.4 MB | 通过 |

### 3.2 数据结构

**数据格式**: JSON 列表（`List[Dict]`）

**样本数据结构**:
```json
{
    "uuid": "f25b796f-bc99-49c7-a6cd-ac5bb98412ce",
    "question": "There are 6 girls in the park...",
    "ground_truth_solution": "There are 6 girls x 2 boys/girl...",
    "ground_truth_answer": 18,
    "model_output_steps": [
        "Step 1: There are twice the number of boys...",
        "Step 2: In total there are 12 boys + 6 girls...",
        "Step 3: #### 18"
    ],
    "model_output_answer_correctness": "correct",
    "model_output_solution_correctness": "correct",
    "model_output_solution_first_error_step": "N/A",
    "model_output_solution_first_error_reason": "N/A",
    "question_type": "original"
}
```

### 3.3 字段完整性

**实际数据字段**:
- ✅ `uuid`: 唯一标识符
- ✅ `question`: 问题文本
- ✅ `ground_truth_solution`: 正确答案的解题步骤
- ✅ `ground_truth_answer`: 正确答案（数值）
- ✅ `model_output_steps`: 模型输出的解题步骤（列表）
- ✅ `model_output_answer_correctness`: 答案正确性（"correct"/"incorrect"）
- ✅ `model_output_solution_correctness`: 解题步骤正确性（"correct"/"incorrect"）
- ✅ `model_output_solution_first_error_step`: 第一个错误步骤（字符串或 "N/A"）
- ✅ `model_output_solution_first_error_reason`: 错误原因（字符串或 "N/A"）
- ✅ `question_type`: 问题类型（如 "original"）

**代码期望的字段**（在 `_extract_answers` 方法中）:
- ⚠️ `solution_correctness`: **字段名不匹配**（实际是 `model_output_solution_correctness`）
- ⚠️ `answer_correctness`: **字段名不匹配**（实际是 `model_output_answer_correctness`）
- ⚠️ `first_error_step`: **字段名不匹配**（实际是 `model_output_solution_first_error_step`）
- ⚠️ `error_reason`: **字段名不匹配**（实际是 `model_output_solution_first_error_reason`）

### 3.4 字段映射问题

**问题描述**:
代码中的 `_extract_answers()` 方法期望的字段名与实际数据文件中的字段名不匹配：

| 代码期望 | 实际数据字段 | 状态 |
|---------|------------|------|
| `solution_correctness` | `model_output_solution_correctness` | ❌ 不匹配 |
| `answer_correctness` | `model_output_answer_correctness` | ❌ 不匹配 |
| `first_error_step` | `model_output_solution_first_error_step` | ❌ 不匹配 |
| `error_reason` | `model_output_solution_first_error_reason` | ❌ 不匹配 |

**影响**:
- 代码会使用默认值（`"correct"`, `"correct"`, `"N/A"`, `"N/A"`）
- 无法正确提取真实答案信息
- 评测结果可能不准确

**当前代码**:
```python
def _extract_answers(self, item: Dict) -> Dict:
    return {
        "solution_correctness": item.get("solution_correctness", "correct"),  # 会使用默认值
        "answer_correctness": item.get("answer_correctness", "correct"),      # 会使用默认值
        "first_error_step": item.get("first_error_step", "N/A"),              # 会使用默认值
        "error_reason": item.get("error_reason", "N/A")                       # 会使用默认值
    }
```

---

## 四、问题分析

### 4.1 严重问题

1. **字段映射错误** ⚠️
   - 代码期望的字段名与实际数据字段名不匹配
   - 导致无法正确提取真实答案
   - 所有样本都会使用默认值，评测结果不准确

### 4.2 潜在问题

1. **命名不一致**
   - 配置中使用 `MR-GMS8K`
   - 文件路径使用 `MR-GSM8K`
   - 虽然代码已处理，但容易混淆

2. **数据格式假设**
   - 代码假设 `model_output_steps` 是列表格式
   - 如果格式变化，可能导致错误

---

## 五、修复建议

### 5.1 立即修复（高优先级）

**修复字段映射问题**:

修改 `_extract_answers()` 方法：

```python
def _extract_answers(self, item: Dict) -> Dict:
    """Extract the ground truth answers from the dataset item."""
    return {
        # 使用实际数据字段名
        "solution_correctness": item.get("model_output_solution_correctness", "correct"),
        "answer_correctness": item.get("model_output_answer_correctness", "correct"),
        "first_error_step": item.get("model_output_solution_first_error_step", "N/A"),
        "error_reason": item.get("model_output_solution_first_error_reason", "N/A")
    }
```

**同时需要处理字段值格式**:
- 数据中使用 `"correct"`/`"incorrect"`（小写）
- 需要确保与评测逻辑兼容

### 5.2 代码改进建议

1. **添加字段验证**
   ```python
   def _validate_item(self, item: Dict) -> bool:
       """验证数据项是否包含必需字段"""
       required_fields = [
           "question", "model_output_steps",
           "model_output_answer_correctness",
           "model_output_solution_correctness"
       ]
       return all(field in item for field in required_fields)
   ```

2. **添加数据转换**
   ```python
   def _normalize_correctness(self, value: str) -> str:
       """标准化正确性字段值"""
       value = value.lower().strip()
       if value in ["correct", "true", "1"]:
           return "correct"
       elif value in ["incorrect", "false", "0"]:
           return "incorrect"
       return value
   ```

3. **改进错误处理**
   - 添加字段缺失警告
   - 记录使用默认值的情况

---

## 六、数据质量评估

### 6.1 数据量

- ✅ **总样本数**: 2999 条
- ✅ **文件大小**: 5.4 MB
- ✅ **平均样本大小**: ~1.8 KB

### 6.2 数据完整性

- ✅ **必需字段**: 所有样本都包含必需字段
- ✅ **字段类型**: 字段类型正确
- ✅ **数据格式**: JSON 格式正确

### 6.3 数据一致性

- ✅ **测试数据与生产数据**: 完全一致（2999 条，5.4 MB）
- ✅ **文件格式**: 统一使用 JSON 列表格式

---

## 七、测试建议

### 7.1 功能测试

1. **数据加载测试**
   ```python
   benchmarker = MR_GMS8KBenchmarker()
   data = benchmarker._load_dataset("MR-GMS8K")
   assert len(data) == 2999
   assert isinstance(data, list)
   assert all(isinstance(item, dict) for item in data)
   ```

2. **字段提取测试**
   ```python
   item = data[0]
   answers = benchmarker._extract_answers(item)
   # 验证字段是否正确提取（修复后）
   assert "solution_correctness" in answers
   assert "answer_correctness" in answers
   ```

3. **路径回退测试**
   - 删除测试数据，验证是否能从生产路径加载
   - 测试所有备选路径

### 7.2 数据验证测试

1. **字段完整性验证**
   ```python
   required_fields = [
       "uuid", "question", "ground_truth_solution",
       "ground_truth_answer", "model_output_steps",
       "model_output_answer_correctness",
       "model_output_solution_correctness"
   ]
   for item in data:
       assert all(field in item for field in required_fields)
   ```

2. **数据值验证**
   - 验证 `model_output_answer_correctness` 的值是 "correct" 或 "incorrect"
   - 验证 `model_output_solution_correctness` 的值是 "correct" 或 "incorrect"
   - 验证 `model_output_steps` 是列表格式

---

## 八、总结

### 8.1 当前状态

- ✅ **数据文件**: 完整且格式正确
- ✅ **导入机制**: 路径回退机制完善
- ❌ **字段映射**: 存在字段名不匹配问题
- ⚠️ **命名不一致**: MR-GMS8K vs MR-GSM8K

### 8.2 风险评估

- **高风险**: 字段映射错误导致评测结果不准确
- **中风险**: 命名不一致可能导致混淆
- **低风险**: 数据格式假设可能在未来失效

### 8.3 优先级建议

1. **高优先级**: 修复字段映射问题（立即修复）
2. **中优先级**: 统一命名规范（代码优化）
3. **低优先级**: 添加数据验证（增强健壮性）

---

## 附录：检查命令

### A.1 快速检查脚本

```bash
#!/bin/bash
# 检查 MR-GMS8K 数据集

echo "=== MR-GMS8K Dataset Check ==="

# 检查测试数据
test_path="/root/LLM-Bench-Shower/tests/test_data/MR-GSM8K/dataset/MR-GSM8K.json"
if [ -f "$test_path" ]; then
    size=$(ls -lh "$test_path" | awk '{print $5}')
    lines=$(python3 -c "import json; print(len(json.load(open('$test_path'))))")
    echo "✅ Test data: $size, $lines items"
else
    echo "❌ Test data: NOT FOUND"
fi

# 检查生产数据
prod_path="/root/share/datasets/MR-GMS8K/MR-GSM8K/dataset/MR-GSM8K.json"
if [ -f "$prod_path" ]; then
    size=$(ls -lh "$prod_path" | awk '{print $5}')
    lines=$(python3 -c "import json; print(len(json.load(open('$prod_path'))))")
    echo "✅ Production data: $size, $lines items"
else
    echo "❌ Production data: NOT FOUND"
fi
```

### A.2 字段映射验证

```python
import json

def check_field_mapping(filepath):
    """检查字段映射问题"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    item = data[0]
    
    # 实际字段
    actual_fields = {
        "solution_correctness": "model_output_solution_correctness",
        "answer_correctness": "model_output_answer_correctness",
        "first_error_step": "model_output_solution_first_error_step",
        "error_reason": "model_output_solution_first_error_reason"
    }
    
    # 检查字段是否存在
    print("Field Mapping Check:")
    for expected, actual in actual_fields.items():
        if actual in item:
            print(f"  ✅ {expected} -> {actual}: {item[actual]}")
        else:
            print(f"  ❌ {expected} -> {actual}: NOT FOUND")
    
    # 检查代码期望的字段
    print("\nCode Expected Fields:")
    for expected in actual_fields.keys():
        if expected in item:
            print(f"  ✅ {expected}: {item[expected]}")
        else:
            print(f"  ❌ {expected}: NOT FOUND (will use default)")

# 使用示例
check_field_mapping("/root/LLM-Bench-Shower/tests/test_data/MR-GSM8K/dataset/MR-GSM8K.json")
```

---

**报告结束**

