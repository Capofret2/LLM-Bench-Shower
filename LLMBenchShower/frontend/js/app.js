(() => {
  const API_BASE = window.API_BASE || "http://127.0.0.1:5000";
  console.log(`[Frontend] API_BASE: ${API_BASE}`);
  const datasetSelect = document.getElementById("dataset-select");
  const modelInput = document.getElementById("model-path");
  const form = document.getElementById("bench-form");
  const statusBox = document.getElementById("status-box");
  const pendingList = document.getElementById("pending-list");
  const resultsBox = document.getElementById("results");
  const runBtn = document.getElementById("run-btn");
  const resetBtn = document.getElementById("reset-btn");

  const state = {
    datasets: {},
    pending: new Map(),
    results: [],
    pollTimer: null,
  };

  const setStatus = (text) => {
    statusBox.textContent = text;
  };

  const uuid = () => {
    if (crypto.randomUUID) return crypto.randomUUID();
    return "xxxxxx4xyx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  };

  const renderDatasetOptions = (data) => {
    datasetSelect.innerHTML = "";
    Object.entries(data).forEach(([sup, subs]) => {
      const group = document.createElement("optgroup");
      group.label = sup;
      subs.forEach((sub) => {
        const option = document.createElement("option");
        option.value = `${sup}/${sub}`;
        // 如果只有一个子数据集且名称和主数据集相同，只显示主数据集名称
        if (subs.length === 1 && sub === sup) {
          option.textContent = sup;
        } else {
          option.textContent = `${sup} / ${sub}`;
        }
        group.appendChild(option);
      });
      datasetSelect.appendChild(group);
    });
  };

  const renderPending = () => {
    pendingList.innerHTML = "";
    state.pending.forEach(({ dataset }, reqId) => {
      const li = document.createElement("li");
      li.className = "status-item";
      li.textContent = `${dataset} · ${reqId}`;
      pendingList.appendChild(li);
    });
    if (!state.pending.size) {
      pendingList.innerHTML = "<li class='status-item done'>没有进行中的任务</li>";
    }
  };

  const renderResults = () => {
    resultsBox.innerHTML = "";
    if (!state.results.length) {
      resultsBox.textContent = "暂无结果";
      return;
    }

    state.results.forEach((item) => {
      const block = document.createElement("div");
      block.className = "result-block";

      const header = document.createElement("div");
      header.className = "result-header";
      header.innerHTML = `<span>${item.dataset}</span><span>${item.reqId}</span>`;

      const body = document.createElement("div");
      body.className = "result-content";
      
      if (item.error) {
        body.textContent = `❌ Error: ${item.error}`;
      } else if (item.result && item.result.evaluation_type === "parameterized_comprehensive") {
        // Render parameterized test results with better visualization
        const stats = item.result.overall_statistics || {};
        const taskStats = item.result.task_type_statistics || {};
        const lengthStats = item.result.context_length_statistics || {};
        const depthStats = item.result.depth_statistics || {};
        const totalTests = item.result.total_tests || 0;
        const successRate = ((item.result.success_rate || 0) * 100).toFixed(1);
        
        let html = `<div class="param-results">`;
        html += `<h3>参数化测试结果汇总</h3>`;
        
        // Overall statistics
        html += `<div class="summary-stats">`;
        html += `<h4>总体统计</h4>`;
        html += `<p><strong>总测试数:</strong> ${totalTests} | <strong>成功率:</strong> ${successRate}%</p>`;
        html += `<p><strong>平均得分:</strong> ${stats.avg_score?.toFixed(3) || 'N/A'} | <strong>最高分:</strong> ${stats.max_score?.toFixed(3) || 'N/A'} | <strong>最低分:</strong> ${stats.min_score?.toFixed(3) || 'N/A'}</p>`;
        html += `<p><strong>平均长度分数:</strong> ${stats.avg_length_score?.toFixed(3) || 'N/A'} <small>(所有测试长度的平均表现)</small> | <strong>长度加权平均分:</strong> ${stats.length_weighted_avg_score?.toFixed(3) || 'N/A'} <small>(更长上下文权重更高)</small></p>`;
        html += `<p><strong>最大测试长度:</strong> ${stats.max_tested_length || 'N/A'} tokens <small>(实际测试的最大长度)</small> | <strong>长度奖励:</strong> ${((stats.length_bonus || 0) * 100).toFixed(1)}% <small>(支持更长上下文的奖励)</small></p>`;
        html += `</div>`;
        
        // Task type statistics (four test methods)
        if (Object.keys(taskStats).length > 0) {
          // Map task type names to Chinese
          const taskTypeNames = {
            'single_needle_retrieval': '单针检索 (Single Needle Retrieval)',
            'multi_needle_retrieval': '多针检索 (Multi Needle Retrieval)',
            'multi_needle_reasoning': '多针推理 (Multi Needle Reasoning)',
            'ancestral_trace_challenge': '祖先追踪挑战 (Ancestral Trace Challenge)'
          };
          
          html += `<div class="task-stats"><h4>按测试方法统计（四种方法）:</h4><ul>`;
          for (const [taskType, taskStat] of Object.entries(taskStats)) {
            const displayName = taskTypeNames[taskType] || taskType;
            html += `<li><strong>${displayName}:</strong> ${taskStat.count} 个测试, 平均得分: ${taskStat.avg_score?.toFixed(3) || 'N/A'}, 最高: ${taskStat.max_score?.toFixed(3) || 'N/A'}, 最低: ${taskStat.min_score?.toFixed(3) || 'N/A'}</li>`;
          }
          html += `</ul></div>`;
        }
        
        // Context length statistics
        if (Object.keys(lengthStats).length > 0) {
          html += `<div class="length-stats"><h4>按上下文长度统计:</h4><ul>`;
          for (const [length, lengthStat] of Object.entries(lengthStats)) {
            html += `<li><strong>${length} tokens:</strong> ${lengthStat.count} 个测试, 平均得分: ${lengthStat.avg_score?.toFixed(3) || 'N/A'}, 最高: ${lengthStat.max_score?.toFixed(3) || 'N/A'}, 最低: ${lengthStat.min_score?.toFixed(3) || 'N/A'}</li>`;
          }
          html += `</ul></div>`;
        }
        
        // Depth statistics
        if (Object.keys(depthStats).length > 0) {
          html += `<div class="depth-stats"><h4>按埋针深度统计:</h4><ul>`;
          for (const [depth, depthStat] of Object.entries(depthStats)) {
            html += `<li><strong>深度 ${depth}%:</strong> ${depthStat.count} 个测试, 平均得分: ${depthStat.avg_score?.toFixed(3) || 'N/A'}</li>`;
          }
          html += `</ul></div>`;
        }
        
        html += `<details><summary>查看详细结果 (${item.result.all_results?.length || 0} 条)</summary>`;
        html += `<pre>${JSON.stringify(item.result, null, 2)}</pre>`;
        html += `</details>`;
        html += `</div>`;
        
        body.innerHTML = html;
      } else {
        // Single test result
        body.textContent = JSON.stringify(item.result, null, 2);
      }

      block.appendChild(header);
      block.appendChild(body);
      resultsBox.appendChild(block);
    });
  };

  const stopPolling = () => {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  };

  const pollResults = () => {
    if (!state.pending.size) {
      stopPolling();
      setStatus("全部任务完成");
      return;
    }

    const count = Math.max(1, state.pending.size);
    fetch(`${API_BASE}/api/results?timeout=0.2&count=${count}`)
      .then((res) => res.json())
      .then((data) => {
        if (!Array.isArray(data)) return;
        data.forEach((item) => {
          const pending = state.pending.get(item.req_id);
          if (!pending) return;
          state.pending.delete(item.req_id);
          state.results.unshift({
            reqId: item.req_id,
            dataset: pending.dataset,
            result: item.result,
            error: item.error,
          });
        });
        renderPending();
        renderResults();
        if (!state.pending.size) {
          setStatus("全部任务完成");
        } else {
          setStatus(`进行中：${state.pending.size} 个任务`);
        }
      })
      .catch((err) => {
        console.error(err);
        setStatus("获取结果失败，稍后自动重试…");
      });
  };

  const startPolling = () => {
    stopPolling();
    state.pollTimer = setInterval(pollResults, 1200);
  };

  const submitBench = async (payload) => {
    const res = await fetch(`${API_BASE}/api/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || "提交失败");
    }
    return res.json();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const modelPath = modelInput.value.trim();
    const selected = Array.from(datasetSelect.selectedOptions).map(
      (opt) => opt.value
    );

    if (!modelPath) {
      alert("请输入本地模型路径或名称");
      return;
    }
    if (!selected.length) {
      alert("请至少选择一个数据集");
      return;
    }

    runBtn.disabled = true;
    setStatus("正在提交任务…");

    state.pending.clear();
    state.results = [];
    renderResults();

    const payload = selected.map((dataset) => {
      const id = uuid();
      state.pending.set(id, { dataset });
      return {
        req_id: id,
        model_type: "local",
        model_name_or_path: modelPath,
        dataset_name: dataset,
      };
    });
    renderPending();

    try {
      await submitBench(payload);
      setStatus("任务已提交，开始轮询结果…");
      startPolling();
    } catch (err) {
      console.error(err);
      setStatus("提交失败，请检查后端或网络");
      alert(err.message || "提交失败");
      state.pending.clear();
      renderPending();
    } finally {
      runBtn.disabled = false;
    }
  };

  const resetForm = () => {
    form.reset();
    state.pending.clear();
    state.results = [];
    stopPolling();
    renderPending();
    renderResults();
    setStatus("已重置，等待开始…");
  };

  const loadDatasets = () => {
    setStatus("加载数据集列表…");
    const apiUrl = `${API_BASE}/api/datasets`;
    console.log(`[Frontend] Loading datasets from: ${apiUrl}`);
    fetch(apiUrl)
      .then((res) => {
        console.log(`[Frontend] Response status: ${res.status}`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        return res.json();
      })
      .then((data) => {
        console.log(`[Frontend] Loaded datasets:`, data);
        state.datasets = data || {};
        renderDatasetOptions(state.datasets);
        setStatus("请选择模型和数据集后开始测评");
      })
      .catch((err) => {
        console.error("[Frontend] Error loading datasets:", err);
        setStatus(`加载数据集失败: ${err.message}。请检查后端是否运行在 ${API_BASE}`);
        // Show error in dataset select
        datasetSelect.innerHTML = `<option disabled>加载失败: ${err.message}</option>`;
      });
  };

  form.addEventListener("submit", handleSubmit);
  resetBtn.addEventListener("click", resetForm);
  loadDatasets();
})();

