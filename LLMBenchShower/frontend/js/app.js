(() => {
  const API_BASE = window.API_BASE || "http://127.0.0.1:5000";
  console.log(`[Frontend] API_BASE: ${API_BASE}`);
  const datasetSelect = document.getElementById("dataset-select");
  const modelFolderInput = document.getElementById("model-folder-path");
  const modelSelect = document.getElementById("model-select");
  const scanModelsBtn = document.getElementById("scan-models-btn");
  const form = document.getElementById("bench-form");
  const statusBox = document.getElementById("status-box");
  const pendingList = document.getElementById("pending-list");
  const resultsBox = document.getElementById("results");
  const runBtn = document.getElementById("run-btn");
  const resetBtn = document.getElementById("reset-btn");

  const state = {
    datasets: {},
    models: [], // 扫描到的模型列表
    pending: new Map(), // req_id -> {dataset, progress}
    results: [],
    pollTimer: null,
    progressTimer: null, // 进度轮询定时器
    accuracyHistory: new Map(), // dataset -> [{modelName, accuracy, timestamp}]
    accuracyCharts: new Map(), // dataset -> Chart instance
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

  const formatTime = (seconds) => {
    if (!seconds || seconds < 0) return "计算中...";
    if (seconds < 60) return `${Math.round(seconds)}秒`;
    const minutes = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    if (minutes < 60) return `${minutes}分${secs}秒`;
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}小时${mins}分`;
  };

  const renderPending = () => {
    pendingList.innerHTML = "";
    state.pending.forEach(({ dataset, progress }, reqId) => {
      const li = document.createElement("li");
      li.className = "status-item";
      
      let content = `<strong>${dataset}</strong><br>`;
      content += `<small>任务ID: ${reqId.substring(0, 8)}...</small><br>`;
      
      if (progress) {
        const current = progress.current_item || 0;
        const total = progress.total_items || 0;
        const percentage = progress.progress_percentage || 0;
        const currentQuestion = progress.current_question || "";
        const remaining = progress.estimated_remaining_seconds;
        
        if (total > 0) {
          content += `<div style="margin-top: 8px;">`;
          content += `<div style="font-size: 12px; margin-bottom: 4px;">进度: ${current}/${total} (${percentage.toFixed(1)}%)</div>`;
          content += `<div style="width: 100%; background: #e5e7eb; border-radius: 4px; height: 8px; overflow: hidden;">`;
          content += `<div style="width: ${percentage}%; background: #2563eb; height: 100%; transition: width 0.3s;"></div>`;
          content += `</div>`;
          
          if (currentQuestion) {
            const questionPreview = currentQuestion.length > 50 
              ? currentQuestion.substring(0, 50) + "..." 
              : currentQuestion;
            content += `<div style="font-size: 11px; color: #6b7280; margin-top: 4px;">当前题目: ${questionPreview}</div>`;
          }
          
          if (remaining !== null && remaining !== undefined) {
            content += `<div style="font-size: 11px; color: #059669; margin-top: 4px;">预计剩余时间: ${formatTime(remaining)}</div>`;
          }
          content += `</div>`;
        } else {
          content += `<div style="font-size: 12px; color: #6b7280; margin-top: 4px;">初始化中...</div>`;
        }
      } else {
        content += `<div style="font-size: 12px; color: #6b7280; margin-top: 4px;">等待开始...</div>`;
      }
      
      li.innerHTML = content;
      pendingList.appendChild(li);
    });
    if (!state.pending.size) {
      pendingList.innerHTML = "<li class='status-item done'>没有进行中的任务</li>";
    }
  };

  // Chart instances storage (for result comparison charts)
  const chartInstances = new Map();

  // Main datasets for accuracy visualization (6 main datasets)
  const MAIN_DATASETS = [
    'LongBench',
    'LongBenchV2',
    'MR-GMS8K',
    'C-Eval',
    'CMMMU',
    'NeedleInAHaystack'
  ];

  /**
   * Extract accuracy/score from result object
   * @param {Object} result - The result object
   * @param {string} datasetName - The dataset name (optional, for dataset-specific logic)
   * @returns {number|null} - Accuracy/score value, or null if not available
   */
  const extractAccuracy = (result, datasetName = null) => {
    if (!result) return null;

    // Special handling for LongBenchV2: use metrics.score directly (not as percentage)
    const baseDatasetName = datasetName ? getDatasetBaseName(datasetName) : null;
    if (baseDatasetName === 'LongBenchV2') {
      if (result.metrics?.score !== undefined && result.metrics?.score !== null) {
        const score = result.metrics.score;
        // LongBenchV2 score is already in 0-1 range, return as-is (will be displayed as percentage)
        return typeof score === 'number' ? score * 100 : score;
      }
      // Fallback to other score fields for LongBenchV2
      if (result.score !== undefined && result.score !== null) {
        return typeof result.score === 'number' ? result.score * 100 : result.score;
      }
      return null;
    }

    // For parameterized_comprehensive evaluation
    if (result.evaluation_type === "parameterized_comprehensive") {
      const successRate = result.success_rate;
      if (successRate !== undefined && successRate !== null) {
        return successRate * 100;
      }
      const avgScore = result.overall_statistics?.avg_score;
      if (avgScore !== undefined && avgScore !== null) {
        return avgScore * 100; // Assuming score is 0-1 range
      }
    }

    // For other evaluation types
    if (result.accuracy !== undefined && result.accuracy !== null) {
      return typeof result.accuracy === 'number' ? result.accuracy * 100 : result.accuracy;
    }
    if (result.score !== undefined && result.score !== null) {
      return typeof result.score === 'number' ? result.score * 100 : result.score;
    }
    if (result.avg_score !== undefined && result.avg_score !== null) {
      return typeof result.avg_score === 'number' ? result.avg_score * 100 : result.avg_score;
    }
    if (result.metrics?.score !== undefined && result.metrics?.score !== null) {
      const score = result.metrics.score;
      return typeof score === 'number' ? score * 100 : score;
    }

    return null;
  };

  /**
   * Get dataset base name (e.g., "LongBench/2wikimqa" -> "LongBench")
   */
  const getDatasetBaseName = (datasetName) => {
    if (!datasetName) return null;
    const parts = datasetName.split('/');
    return parts[0];
  };

  /**
   * Extract the last folder name from a model path
   * @param {string} modelPath - Full model path
   * @returns {string} - Last folder name
   */
  const getModelDisplayName = (modelPath) => {
    if (!modelPath) return "unknown";
    // Remove trailing slashes and split by '/'
    const parts = modelPath.replace(/\/+$/, '').split('/');
    // Return the last non-empty part
    return parts[parts.length - 1] || modelPath;
  };

  /**
   * Check if dataset is one of the main datasets
   */
  const isMainDataset = (datasetName) => {
    const baseName = getDatasetBaseName(datasetName);
    return baseName && MAIN_DATASETS.includes(baseName);
  };

  /**
   * Load accuracy history from localStorage
   */
  const loadAccuracyHistory = () => {
    try {
      const stored = localStorage.getItem('llm_bench_accuracy_history');
      if (stored) {
        const parsed = JSON.parse(stored);
        Object.entries(parsed).forEach(([dataset, entries]) => {
          // Migrate old format to new format (with modelPath and modelName)
          const migratedEntries = entries.map(entry => {
            if (entry.modelPath && entry.modelName) {
              // Already in new format
              return entry;
            } else {
              // Old format: only modelName, treat it as modelPath
              const modelPath = entry.modelName || entry.modelPath || '';
              return {
                modelPath: modelPath,
                modelName: getModelDisplayName(modelPath),
                accuracy: entry.accuracy,
                timestamp: entry.timestamp || Date.now()
              };
            }
          });
          state.accuracyHistory.set(dataset, migratedEntries);
        });
      }
    } catch (err) {
      console.warn('[Frontend] Failed to load accuracy history:', err);
    }
  };

  /**
   * Save accuracy history to localStorage
   */
  const saveAccuracyHistory = () => {
    try {
      const toStore = {};
      state.accuracyHistory.forEach((entries, dataset) => {
        toStore[dataset] = entries;
      });
      localStorage.setItem('llm_bench_accuracy_history', JSON.stringify(toStore));
    } catch (err) {
      console.warn('[Frontend] Failed to save accuracy history:', err);
    }
  };

  /**
   * Add accuracy record for a dataset
   */
  const addAccuracyRecord = (datasetName, modelName, accuracy) => {
    if (!isMainDataset(datasetName) || accuracy === null || accuracy === undefined) {
      return;
    }

    const baseName = getDatasetBaseName(datasetName);
    if (!state.accuracyHistory.has(baseName)) {
      state.accuracyHistory.set(baseName, []);
    }

    const history = state.accuracyHistory.get(baseName);
    
    // Extract display name (last folder) but keep full path for matching
    const displayName = getModelDisplayName(modelName);
    
    // Check if this model already has a record for this dataset (match by full path)
    const existingIndex = history.findIndex(entry => entry.modelPath === modelName);
    const newEntry = {
      modelPath: modelName, // Store full path for matching
      modelName: displayName, // Store display name for chart
      accuracy: accuracy,
      timestamp: Date.now()
    };

    if (existingIndex >= 0) {
      // Update existing record
      history[existingIndex] = newEntry;
    } else {
      // Add new record
      history.push(newEntry);
    }

    // Sort by timestamp (newest first)
    history.sort((a, b) => b.timestamp - a.timestamp);

    saveAccuracyHistory();
    updateAccuracyChart(baseName);
  };

  /**
   * Load history from backend API
   */
  const loadHistoryFromBackend = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/history`, {
        method: 'GET',
        mode: 'cors',
        credentials: 'omit',
        cache: 'no-cache',
      });

      if (!response.ok) {
        console.warn('[Frontend] Failed to load history from backend');
        return;
      }

      const history = await response.json();
      
      // Process history entries
      history.forEach(item => {
        const datasetName = item.dataset_name;
        const modelName = item.model_name;
        const result = item.results;

        if (isMainDataset(datasetName) && result) {
          const accuracy = extractAccuracy(result, datasetName);
          if (accuracy !== null) {
            addAccuracyRecord(datasetName, modelName, accuracy);
          }
        }
      });
    } catch (err) {
      console.warn('[Frontend] Error loading history from backend:', err);
    }
  };

  /**
   * Update accuracy chart for a specific dataset
   */
  const updateAccuracyChart = (datasetBaseName) => {
    const history = state.accuracyHistory.get(datasetBaseName) || [];
    
    if (history.length === 0) {
      // Show "no data" message
      const chartId = `accuracy-chart-${datasetBaseName}`;
      const chartCard = document.getElementById(chartId);
      if (chartCard) {
        const wrapper = chartCard.querySelector('.chart-wrapper');
        if (wrapper) {
          wrapper.innerHTML = '<div class="no-data">暂无数据</div>';
        }
      }
      return;
    }

    // Prepare chart data
    const labels = history.map(entry => entry.modelName);
    const data = history.map(entry => entry.accuracy);

    // Get or create chart canvas
    const chartId = `accuracy-chart-${datasetBaseName}`;
    let chartCard = document.getElementById(chartId);
    
    if (!chartCard) {
      // Create chart card if it doesn't exist
      const container = document.getElementById('accuracy-charts-container');
      chartCard = document.createElement('div');
      chartCard.id = chartId;
      chartCard.className = 'accuracy-chart-card';
      // Use different label for LongBenchV2 (score instead of accuracy)
      const yAxisLabel = datasetBaseName === 'LongBenchV2' ? '分数 (%)' : '正确率 (%)';
      chartCard.innerHTML = `
        <h3>${datasetBaseName}</h3>
        <div class="chart-wrapper">
          <canvas id="canvas-${datasetBaseName}"></canvas>
        </div>
        <div class="chart-legend">${yAxisLabel}</div>
      `;
      container.appendChild(chartCard);
    }

    // Update chart
    setTimeout(() => {
      const canvasId = `canvas-${datasetBaseName}`;
      const canvas = document.getElementById(canvasId);
      if (!canvas) return;

      // Destroy existing chart if any
      if (state.accuracyCharts.has(datasetBaseName)) {
        state.accuracyCharts.get(datasetBaseName).destroy();
      }

      // Use different labels for LongBenchV2 (score instead of accuracy)
      const isLongBenchV2 = datasetBaseName === 'LongBenchV2';
      const datasetLabel = isLongBenchV2 ? '分数 (%)' : '正确率 (%)';
      const tooltipLabel = isLongBenchV2 ? '分数' : '正确率';
      const yAxisTitle = isLongBenchV2 ? '分数 (%)' : '正确率 (%)';

      // Create new chart
      const chart = new Chart(canvas, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: datasetLabel,
            data: data,
            backgroundColor: 'rgba(37, 99, 235, 0.6)',
            borderColor: 'rgba(37, 99, 235, 1)',
            borderWidth: 2,
            borderRadius: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              display: false
            },
            tooltip: {
              callbacks: {
                label: function(context) {
                  return `${tooltipLabel}: ${context.parsed.y.toFixed(2)}%`;
                }
              }
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              max: 100,
              title: {
                display: true,
                text: yAxisTitle
              },
              ticks: {
                callback: function(value) {
                  return value + '%';
                }
              }
            },
            x: {
              title: {
                display: true,
                text: '模型'
              },
              ticks: {
                maxRotation: 45,
                minRotation: 45
              }
            }
          }
        }
      });

      state.accuracyCharts.set(datasetBaseName, chart);
    }, 100);
  };

  /**
   * Render all accuracy charts
   */
  const renderAccuracyCharts = () => {
    MAIN_DATASETS.forEach(dataset => {
      updateAccuracyChart(dataset);
    });
  };

  const renderResults = () => {
    resultsBox.innerHTML = "";
    if (!state.results.length) {
      resultsBox.textContent = "暂无结果";
      return;
    }

    // Group results by dataset
    const resultsByDataset = new Map();
    state.results.forEach((item) => {
      const dataset = item.dataset || "unknown";
      if (!resultsByDataset.has(dataset)) {
        resultsByDataset.set(dataset, []);
      }
      resultsByDataset.get(dataset).push(item);
    });

    // Render each dataset group
    resultsByDataset.forEach((items, dataset) => {
      const datasetBlock = document.createElement("div");
      datasetBlock.className = "dataset-group";
      
      const header = document.createElement("div");
      header.className = "dataset-header";
      header.innerHTML = `<h2>${dataset}</h2><span class="model-count">${items.length} 个模型结果</span>`;
      datasetBlock.appendChild(header);

      // Filter out errors
      const validResults = items.filter(item => !item.error && item.result);
      const errorResults = items.filter(item => item.error);

      if (validResults.length > 0) {
        // Create comparison section
        const comparisonSection = document.createElement("div");
        comparisonSection.className = "comparison-section";
        
        // Extract metrics for comparison
        const comparisonData = validResults.map(item => {
          const result = item.result;
          if (result.evaluation_type === "parameterized_comprehensive") {
            const stats = result.overall_statistics || {};
            return {
              modelName: item.modelName || "unknown",
              reqId: item.reqId,
              avgScore: stats.avg_score || 0,
              successRate: (result.success_rate || 0) * 100,
              totalTests: result.total_tests || 0,
              lengthStats: result.context_length_statistics || {},
              taskStats: result.task_type_statistics || {},
              fullResult: result
            };
          } else {
            // For other evaluation types, try to extract score
            let score = 0;
            if (result.accuracy !== undefined) score = result.accuracy;
            else if (result.score !== undefined) score = result.score;
            else if (result.avg_score !== undefined) score = result.avg_score;
            
            return {
              modelName: item.modelName || "unknown",
              reqId: item.reqId,
              avgScore: score,
              successRate: score * 100,
              totalTests: 1,
              lengthStats: {},
              taskStats: {},
              fullResult: result
            };
          }
        });

        // Model comparison chart
        if (comparisonData.length > 1 && typeof Chart !== 'undefined') {
          const chartContainer = document.createElement("div");
          chartContainer.className = "chart-container";
          chartContainer.innerHTML = `<h3>模型得分对比</h3><canvas id="chart-${dataset.replace(/\s+/g, '-')}"></canvas>`;
          comparisonSection.appendChild(chartContainer);
          
          // Destroy existing chart if any
          const chartId = `chart-${dataset.replace(/\s+/g, '-')}`;
          if (chartInstances.has(chartId)) {
            chartInstances.get(chartId).destroy();
          }
          
          // Create new chart
          setTimeout(() => {
            const ctx = document.getElementById(chartId);
            if (ctx) {
              const chart = new Chart(ctx, {
                type: 'bar',
                data: {
                  labels: comparisonData.map(d => d.modelName),
                  datasets: [{
                    label: '平均得分',
                    data: comparisonData.map(d => d.avgScore),
                    backgroundColor: 'rgba(37, 99, 235, 0.6)',
                    borderColor: 'rgba(37, 99, 235, 1)',
                    borderWidth: 1
                  }, {
                    label: '成功率 (%)',
                    data: comparisonData.map(d => d.successRate),
                    backgroundColor: 'rgba(16, 185, 129, 0.6)',
                    borderColor: 'rgba(16, 185, 129, 1)',
                    borderWidth: 1,
                    yAxisID: 'y1'
                  }]
                },
                options: {
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      display: true,
                      position: 'top'
                    },
                    tooltip: {
                      callbacks: {
                        label: function(context) {
                          let label = context.dataset.label || '';
                          if (label) {
                            label += ': ';
                          }
                          if (context.parsed.y !== null) {
                            label += context.parsed.y.toFixed(3);
                          }
                          return label;
                        }
                      }
                    }
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: '平均得分'
                      }
                    },
                    y1: {
                      type: 'linear',
                      display: true,
                      position: 'right',
                      beginAtZero: true,
                      max: 100,
                      title: {
                        display: true,
                        text: '成功率 (%)'
                      },
                      grid: {
                        drawOnChartArea: false
                      }
                    }
                  }
                }
              });
              chartInstances.set(chartId, chart);
            }
          }, 100);
        }

        // Context length comparison chart (if available)
        const hasLengthStats = comparisonData.some(d => Object.keys(d.lengthStats).length > 0);
        if (hasLengthStats && comparisonData.length > 1 && typeof Chart !== 'undefined') {
          const lengthChartContainer = document.createElement("div");
          lengthChartContainer.className = "chart-container";
          lengthChartContainer.innerHTML = `<h3>不同上下文长度表现对比</h3><canvas id="chart-length-${dataset.replace(/\s+/g, '-')}"></canvas>`;
          comparisonSection.appendChild(lengthChartContainer);
          
          setTimeout(() => {
            const ctx = document.getElementById(`chart-length-${dataset.replace(/\s+/g, '-')}`);
            if (ctx) {
              // Collect all unique lengths
              const allLengths = new Set();
              comparisonData.forEach(d => {
                Object.keys(d.lengthStats).forEach(len => allLengths.add(len));
              });
              const sortedLengths = Array.from(allLengths).sort((a, b) => parseInt(a) - parseInt(b));
              
              const datasets = comparisonData.map((d, idx) => {
                const colors = [
                  'rgba(37, 99, 235, 0.6)',
                  'rgba(16, 185, 129, 0.6)',
                  'rgba(245, 158, 11, 0.6)',
                  'rgba(239, 68, 68, 0.6)',
                  'rgba(139, 92, 246, 0.6)'
                ];
                return {
                  label: d.modelName,
                  data: sortedLengths.map(len => d.lengthStats[len]?.avg_score || 0),
                  borderColor: colors[idx % colors.length],
                  backgroundColor: colors[idx % colors.length].replace('0.6', '0.2'),
                  tension: 0.1
                };
              });
              
              const chart = new Chart(ctx, {
                type: 'line',
                data: {
                  labels: sortedLengths.map(l => `${l} tokens`),
                  datasets: datasets
                },
                options: {
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: {
                    legend: {
                      display: true,
                      position: 'top'
                    }
                  },
                  scales: {
                    y: {
                      beginAtZero: true,
                      title: {
                        display: true,
                        text: '平均得分'
                      }
                    },
                    x: {
                      title: {
                        display: true,
                        text: '上下文长度'
                      }
                    }
                  }
                }
              });
              chartInstances.set(`chart-length-${dataset.replace(/\s+/g, '-')}`, chart);
            }
          }, 200);
        }

        datasetBlock.appendChild(comparisonSection);

        // Individual model results
        const modelsSection = document.createElement("div");
        modelsSection.className = "models-section";
        modelsSection.innerHTML = `<h3>各模型详细结果</h3>`;
        
        validResults.forEach((item) => {
          const modelBlock = document.createElement("div");
          modelBlock.className = "model-result-block";
          
          const modelHeader = document.createElement("div");
          modelHeader.className = "model-header";
          modelHeader.innerHTML = `<span class="model-name">${item.modelName}</span><span class="req-id">${item.reqId.substring(0, 8)}...</span>`;
          modelBlock.appendChild(modelHeader);
          
          const modelBody = document.createElement("div");
          modelBody.className = "model-result-content";
          
          if (item.result && item.result.evaluation_type === "parameterized_comprehensive") {
            const stats = item.result.overall_statistics || {};
            const taskStats = item.result.task_type_statistics || {};
            const lengthStats = item.result.context_length_statistics || {};
            const totalTests = item.result.total_tests || 0;
            const successRate = ((item.result.success_rate || 0) * 100).toFixed(1);
            
            let html = `<div class="param-results">`;
            html += `<div class="summary-stats">`;
            html += `<p><strong>总测试数:</strong> ${totalTests} | <strong>成功率:</strong> ${successRate}%</p>`;
            html += `<p><strong>平均得分:</strong> ${stats.avg_score?.toFixed(3) || 'N/A'} | <strong>最高分:</strong> ${stats.max_score?.toFixed(3) || 'N/A'} | <strong>最低分:</strong> ${stats.min_score?.toFixed(3) || 'N/A'}</p>`;
            html += `</div>`;
            
            if (Object.keys(taskStats).length > 0) {
              const taskTypeNames = {
                'single_needle_retrieval': '单针检索',
                'multi_needle_retrieval': '多针检索',
                'multi_needle_reasoning': '多针推理',
                'ancestral_trace_challenge': '祖先追踪'
              };
              html += `<div class="task-stats"><h4>测试方法统计:</h4><ul>`;
              for (const [taskType, taskStat] of Object.entries(taskStats)) {
                const displayName = taskTypeNames[taskType] || taskType;
                html += `<li><strong>${displayName}:</strong> ${taskStat.count} 个测试, 平均得分: ${taskStat.avg_score?.toFixed(3) || 'N/A'}</li>`;
              }
              html += `</ul></div>`;
            }
            
            if (Object.keys(lengthStats).length > 0) {
              html += `<div class="length-stats"><h4>上下文长度统计:</h4><ul>`;
              for (const [length, lengthStat] of Object.entries(lengthStats)) {
                html += `<li><strong>${length} tokens:</strong> ${lengthStat.count} 个测试, 平均得分: ${lengthStat.avg_score?.toFixed(3) || 'N/A'}</li>`;
              }
              html += `</ul></div>`;
            }
            
            html += `<details><summary>查看完整JSON数据</summary>`;
            html += `<pre>${JSON.stringify(item.result, null, 2)}</pre>`;
            html += `</details>`;
            html += `</div>`;
            
            modelBody.innerHTML = html;
          } else {
            // For other result types, show formatted JSON
            modelBody.innerHTML = `<pre class="json-result">${JSON.stringify(item.result, null, 2)}</pre>`;
          }
          
          modelBlock.appendChild(modelBody);
          modelsSection.appendChild(modelBlock);
        });
        
        datasetBlock.appendChild(modelsSection);
      }

      // Show errors if any
      if (errorResults.length > 0) {
        const errorSection = document.createElement("div");
        errorSection.className = "error-section";
        errorSection.innerHTML = `<h3>错误结果</h3>`;
        errorResults.forEach(item => {
          const errorBlock = document.createElement("div");
          errorBlock.className = "error-block";
          errorBlock.innerHTML = `<strong>${item.modelName || 'Unknown'}</strong>: ${item.error}`;
          errorSection.appendChild(errorBlock);
        });
        datasetBlock.appendChild(errorSection);
      }

      resultsBox.appendChild(datasetBlock);
    });
  };

  const stopPolling = () => {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
    if (state.progressTimer) {
      clearInterval(state.progressTimer);
      state.progressTimer = null;
    }
  };

  const pollResults = () => {
    if (!state.pending.size) {
      stopPolling();
      setStatus("全部任务完成");
      return;
    }

    const count = Math.max(1, state.pending.size);
    fetch(`${API_BASE}/api/results?timeout=0.2&count=${count}`, {
      method: 'GET',
      mode: 'cors', // Explicitly set CORS mode
      credentials: 'omit',
      cache: 'no-cache',
    })
      .then((res) => {
        // Check response status before parsing JSON
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        return res.json();
      })
      .then((data) => {
        if (!Array.isArray(data)) {
          console.warn('[Frontend] Unexpected response format:', data);
          return;
        }
        data.forEach((item) => {
          const pending = state.pending.get(item.req_id);
          if (!pending) return;
          state.pending.delete(item.req_id);
          
          const modelName = item.model_name || "unknown";
          const datasetName = item.dataset_name || pending.dataset;
          
          state.results.unshift({
            reqId: item.req_id,
            modelName: modelName,
            dataset: datasetName,
            result: item.result,
            error: item.error,
          });

          // Update accuracy history if result is available and no error
          if (!item.error && item.result) {
            const accuracy = extractAccuracy(item.result, datasetName);
            if (accuracy !== null) {
              addAccuracyRecord(datasetName, modelName, accuracy);
            }
          }
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
        console.error('[Frontend] Error polling results:', err);
        // Don't show error message for every poll failure, only log it
        // The polling will continue and retry automatically
      });
  };

  const pollProgress = () => {
    if (!state.pending.size) {
      return;
    }

    const reqIds = Array.from(state.pending.keys());
    const reqIdsParam = reqIds.map(id => `req_id=${id}`).join('&');
    
    fetch(`${API_BASE}/api/progress?${reqIdsParam}`, {
      method: 'GET',
      mode: 'cors',
      credentials: 'omit',
      cache: 'no-cache',
    })
      .then((res) => {
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }
        return res.json();
      })
      .then((progressData) => {
        // 更新每个任务的进度信息
        Object.entries(progressData).forEach(([reqId, progress]) => {
          const pending = state.pending.get(reqId);
          if (pending) {
            pending.progress = progress;
          }
        });
        renderPending();
      })
      .catch((err) => {
        // 静默失败，不影响主流程
        console.warn('[Frontend] Error polling progress:', err);
      });
  };

  const startPolling = () => {
    stopPolling();
    // 降低轮询频率：本地模型处理速度较慢，使用 5 秒间隔
    state.pollTimer = setInterval(pollResults, 5000);
    // 进度轮询：每 3 秒更新一次进度（比结果轮询更频繁，因为进度信息更轻量）
    state.progressTimer = setInterval(pollProgress, 3000);
    // 立即获取一次进度
    pollProgress();
  };

  const submitBench = async (payload) => {
    try {
      const res = await fetch(`${API_BASE}/api/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        mode: 'cors', // Explicitly set CORS mode
        credentials: 'omit',
        cache: 'no-cache',
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        let errorText = "提交失败";
        try {
          errorText = await res.text();
        } catch (e) {
          errorText = `HTTP ${res.status}: ${res.statusText}`;
        }
        throw new Error(errorText || "提交失败");
      }
      return await res.json();
    } catch (err) {
      // Re-throw with more context if it's a network error
      if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
        throw new Error(`无法连接到后端服务器 ${API_BASE}。请检查后端是否正在运行。`);
      }
      throw err;
    }
  };

  const scanModels = async () => {
    const folderPath = modelFolderInput.value.trim();
    if (!folderPath) {
      alert("请输入模型存储文件夹路径");
      return;
    }

    scanModelsBtn.disabled = true;
    setStatus("正在递归扫描模型文件夹（可能包含多层子目录）…");
    modelSelect.innerHTML = '<option value="">扫描中，请稍候…</option>';

    try {
      const response = await fetch(`${API_BASE}/api/scan-models`, {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        mode: 'cors',
        credentials: 'omit',
        cache: 'no-cache',
        body: JSON.stringify({ 
          folder_path: folderPath,
          max_depth: 10  // 可以从前端配置，默认10层
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: `HTTP ${response.status}` }));
        throw new Error(errorData.error || "扫描失败");
      }

      const data = await response.json();
      state.models = data.models || [];

      // 更新模型选择下拉框
      modelSelect.innerHTML = '';
      if (state.models.length === 0) {
        modelSelect.innerHTML = '<option value="">未找到模型，请检查文件夹路径</option>';
        setStatus("未找到模型，请检查文件夹路径是否正确");
      } else {
        state.models.forEach((model) => {
          const option = document.createElement("option");
          option.value = model.path;
          option.textContent = `${model.name} (${model.path})`;
          modelSelect.appendChild(option);
        });
        setStatus(`找到 ${state.models.length} 个模型，请选择要使用的模型`);
      }
    } catch (err) {
      console.error("[Frontend] Error scanning models:", err);
      modelSelect.innerHTML = '<option value="">扫描失败</option>';
      setStatus(`扫描失败: ${err.message}`);
      alert(`扫描模型失败: ${err.message}`);
    } finally {
      scanModelsBtn.disabled = false;
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const modelPath = modelSelect.value.trim();
    const selected = Array.from(datasetSelect.selectedOptions).map(
      (opt) => opt.value
    );

    if (!modelPath) {
      alert("请先扫描模型文件夹并选择一个模型");
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
      state.pending.set(id, { dataset, progress: null });
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
    state.models = [];
    modelSelect.innerHTML = '<option value="">请先扫描模型文件夹</option>';
    stopPolling();
    renderPending();
    renderResults();
    setStatus("已重置，等待开始…");
  };

  const loadDatasets = () => {
    setStatus("加载数据集列表…");
    const apiUrl = `${API_BASE}/api/datasets`;
    console.log(`[Frontend] Loading datasets from: ${apiUrl}`);
    console.log(`[Frontend] Current origin: ${window.location.origin}`);
    console.log(`[Frontend] API_BASE: ${API_BASE}`);
    
    fetch(apiUrl, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      mode: 'cors', // Explicitly set CORS mode
      credentials: 'omit', // Don't send credentials for CORS
      cache: 'no-cache', // Don't cache the request
    })
      .then((res) => {
        console.log(`[Frontend] Response status: ${res.status}`);
        console.log(`[Frontend] Response headers:`, [...res.headers.entries()]);
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
        console.error("[Frontend] Error details:", {
          name: err.name,
          message: err.message,
          stack: err.stack,
        });
        let errorMsg = err.message;
        if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
          errorMsg = `无法连接到后端服务器 ${API_BASE}。请检查：\n1. 后端服务是否正在运行\n2. 端口是否正确（应该是 5000）\n3. 防火墙是否阻止了连接\n4. 浏览器控制台是否有 CORS 错误\n\n原始错误: ${err.message}`;
        } else if (err.message.includes('HTTP')) {
          errorMsg = `后端返回错误: ${err.message}`;
        }
        setStatus(`加载数据集失败: ${errorMsg}`);
        // Show error in dataset select
        datasetSelect.innerHTML = `<option disabled>加载失败: ${errorMsg}</option>`;
      });
  };

  form.addEventListener("submit", handleSubmit);
  resetBtn.addEventListener("click", resetForm);
  scanModelsBtn.addEventListener("click", scanModels);
  
  // Load datasets on page load (loadDatasets already has error handling)
  // No need for separate connection test - loadDatasets will handle errors
  loadDatasets();

  // Initialize accuracy visualization
  loadAccuracyHistory();
  loadHistoryFromBackend().then(() => {
    renderAccuracyCharts();
  });
})();

