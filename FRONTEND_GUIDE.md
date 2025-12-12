# LLM Bench Shower 前端实现指南

## 概述

LLM Bench Shower 前端采用纯原生 HTML/CSS/JavaScript 实现，无任何框架依赖，提供了简洁直观的 Web 界面用于配置、提交和查看 LLM 基准测试任务。

## 技术栈

- **HTML5**: 页面结构
- **CSS3**: 样式和布局（使用 Flexbox 和 Grid）
- **原生 JavaScript (ES6+)**: 业务逻辑和 API 交互
- **Fetch API**: HTTP 请求
- **localStorage**: API 地址配置存储

## 文件结构

```
LLMBenchShower/frontend/
├── index.html          # 主页面
├── css/
│   └── styles.css      # 样式文件
└── js/
    └── app.js          # 核心逻辑
```

## 核心功能

### 1. 数据集列表加载

页面加载时自动从后端获取可用数据集列表，并以分组形式展示：

- **单子数据集**: 直接显示数据集名称
- **多子数据集**: 显示格式为 `父数据集 / 子数据集`

**实现位置**: `app.js` 的 `loadDatasets()` 函数

**API 端点**: `GET /api/datasets`

### 2. 模型配置

用户可输入本地模型路径或 HuggingFace 模型名称：

- 支持本地路径：如 `/models/qwen`
- 支持模型名称：如 `Qwen/Qwen2-7B`

**实现位置**: `index.html` 的 `model-path` 输入框

### 3. 多数据集选择

支持同时选择多个数据集进行批量测试：

- 使用 `<select multiple>` 实现多选
- 每个数据集对应一个独立的测试任务

**实现位置**: `index.html` 的 `dataset-select` 下拉框

### 4. 任务提交

提交任务时的流程：

1. 验证输入（模型路径和数据集选择）
2. 为每个选中的数据集生成唯一的 `req_id`（使用 UUID）
3. 构建请求负载并提交到后端
4. 将任务添加到待处理列表
5. 启动结果轮询

**实现位置**: `app.js` 的 `handleSubmit()` 和 `submitBench()` 函数

**API 端点**: `POST /api/submit`

**请求格式**:
```json
[
  {
    "req_id": "uuid-string",
    "model_type": "local",
    "model_name_or_path": "/models/qwen",
    "dataset_name": "LongBench/needle"
  }
]
```

### 5. 结果轮询

采用轮询机制实时获取任务结果：

- **轮询间隔**: 1.2 秒
- **轮询策略**: 每次请求获取 `pending.size` 个结果（至少 1 个）
- **超时设置**: 0.2 秒（避免长时间阻塞）
- **自动停止**: 当所有任务完成时停止轮询

**实现位置**: `app.js` 的 `pollResults()` 和 `startPolling()` 函数

**API 端点**: `GET /api/results?timeout=0.2&count={count}`

### 6. 结果展示

支持两种类型的结果展示：

#### 6.1 普通测试结果

直接以 JSON 格式展示原始结果。

#### 6.2 参数化测试结果（Parameterized Comprehensive）

针对 `evaluation_type === "parameterized_comprehensive"` 的结果，提供详细的可视化展示：

- **总体统计**:
  - 总测试数、成功率
  - 平均得分、最高分、最低分
  - 平均长度分数、长度加权平均分
  - 最大测试长度、长度奖励

- **按测试方法统计**（四种方法）:
  - 单针检索 (Single Needle Retrieval)
  - 多针检索 (Multi Needle Retrieval)
  - 多针推理 (Multi Needle Reasoning)
  - 祖先追踪挑战 (Ancestral Trace Challenge)

- **按上下文长度统计**: 不同 token 长度下的测试表现

- **按埋针深度统计**: 不同深度百分比下的测试表现

- **详细结果**: 可展开查看完整的 JSON 数据

**实现位置**: `app.js` 的 `renderResults()` 函数

## 状态管理

前端使用简单的状态对象管理应用状态：

```javascript
const state = {
  datasets: {},        // 数据集列表
  pending: new Map(),  // 待处理任务（req_id -> {dataset}）
  results: [],        // 已完成结果
  pollTimer: null     // 轮询定时器
};
```

## API 配置

### 默认配置

默认后端地址为 `http://127.0.0.1:5000`。

### 前后端分离配置

当前后端分离部署时，可通过 `localStorage` 配置后端地址：

```javascript
// 在浏览器控制台执行
localStorage.setItem('lbs_api_base', 'http://your-backend-server:5000');
```

配置后刷新页面即可生效。

**实现位置**: `index.html` 的 `<script>` 标签

## UI 组件

### 布局结构

使用 CSS Grid 实现响应式布局：

- **主容器**: `main` 使用 `grid-template-columns: repeat(auto-fit, minmax(320px, 1fr))`
- **卡片式设计**: 每个功能模块使用 `.card` 样式

### 主要组件

1. **配置卡片**: 模型输入、数据集选择、操作按钮
2. **进度卡片**: 状态显示、待处理任务列表
3. **结果卡片**: 测试结果展示区域

### 样式特点

- 现代化的卡片式设计
- 柔和的阴影和圆角
- 清晰的视觉层次
- 响应式布局，适配不同屏幕尺寸

## 错误处理

### 数据集加载失败

- 显示错误信息在状态框
- 在下拉框中显示错误提示

### 任务提交失败

- 显示错误提示
- 清空待处理列表
- 恢复按钮状态

### 结果轮询失败

- 在状态框显示错误信息
- 自动继续轮询（下次轮询时重试）

## 使用流程

1. **启动后端服务**: 确保后端服务运行在 `http://127.0.0.1:5000`（或配置的地址）

2. **打开前端页面**: 在浏览器中打开 `index.html`

3. **等待数据集加载**: 页面自动加载可用数据集列表

4. **配置测试**:
   - 输入模型路径或名称
   - 选择一个或多个数据集

5. **提交任务**: 点击"开始测评"按钮

6. **查看进度**: 在"进度"卡片中查看待处理任务

7. **查看结果**: 任务完成后，结果自动显示在"结果"卡片中

8. **重置**: 点击"重置"按钮清空所有状态

## 浏览器兼容性

- **现代浏览器**: Chrome, Firefox, Edge, Safari（最新版本）
- **UUID 生成**: 优先使用 `crypto.randomUUID()`，不支持时使用兼容实现
- **Fetch API**: 需要支持 ES6+ 的浏览器

## 扩展建议

### 功能扩展

1. **历史记录查看**: 可调用 `/api/history` 端点展示历史测试记录
2. **统计信息**: 可调用 `/api/stats` 端点展示数据库统计
3. **任务取消**: 可添加取消进行中任务的功能
4. **结果导出**: 可添加导出结果到 JSON/CSV 的功能

### 性能优化

1. **虚拟滚动**: 当结果数量很大时，可使用虚拟滚动优化性能
2. **WebSocket**: 可考虑使用 WebSocket 替代轮询，实现实时推送
3. **结果缓存**: 可在 localStorage 中缓存历史结果

### UI 改进

1. **加载动画**: 添加更丰富的加载状态指示
2. **进度条**: 显示任务完成进度百分比
3. **图表可视化**: 使用 Chart.js 等库展示统计图表

## 开发调试

### 控制台日志

前端在关键操作处输出日志：

- `[Frontend] API_BASE: ...` - API 地址
- `[Frontend] Loading datasets from: ...` - 数据集加载
- `[Frontend] Loaded datasets: ...` - 数据集加载结果

### 常见问题

1. **无法加载数据集**: 检查后端服务是否运行，API 地址是否正确
2. **任务提交失败**: 检查网络连接和后端日志
3. **结果不更新**: 检查轮询是否正常启动，查看浏览器控制台错误

## 总结

LLM Bench Shower 前端采用简洁的原生技术栈实现，提供了完整的基准测试工作流。代码结构清晰，易于维护和扩展。通过合理的状态管理和轮询机制，实现了良好的用户体验。

