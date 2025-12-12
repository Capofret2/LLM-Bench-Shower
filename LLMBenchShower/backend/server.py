import uuid
import sys
import os
from pathlib import Path

# 确保可以正确导入模块
# 获取当前文件的目录（backend/）
backend_dir = Path(__file__).parent.absolute()
# 获取项目根目录（LLMBenchShower/）
project_root = backend_dir.parent

# 将项目根目录添加到 Python 路径（这样可以使用 backend.runner, backend.bench 等）
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# 也将 backend 目录添加到路径（为了向后兼容，支持直接导入 runner, bench 等）
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from flask import Flask, request, jsonify
from flask_cors import CORS

# 尝试两种导入方式，确保兼容性
try:
    from runner import get_llm_bench_runner
    from bench.utils import get_available_datasets
except ImportError:
    # 如果直接导入失败，尝试从 backend 包导入
    from backend.runner import get_llm_bench_runner
    from backend.bench.utils import get_available_datasets

app = Flask(__name__)
# Enable CORS for all routes with more permissive settings
CORS(app, 
     resources={r"/api/*": {"origins": "*"}},
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
     methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])

# Initialize the runner
try:
    print("[Server] Initializing runner...")
    runner = get_llm_bench_runner()
    print(f"[Server] Runner initialized. Available datasets: {list(runner.available_datasets.keys())}")
except Exception as e:
    print(f"[Server] Error initializing runner: {e}")
    import traceback
    traceback.print_exc()
    raise

@app.route('/api/datasets', methods=['GET'])
def get_datasets():
    """Get available datasets."""
    try:
        # 每次请求时重新读取配置，确保获取最新的数据集列表
        datasets = get_available_datasets()
        print(f"[Server] Returning datasets: {list(datasets.keys())}")
        # 同时更新 runner 的缓存（可选，用于一致性）
        runner.available_datasets = datasets
        return jsonify(datasets)
    except Exception as e:
        print(f"[Server] Error getting datasets: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/scan-models', methods=['POST'])
def scan_models():
    """Scan a folder for model directories (recursively)."""
    try:
        data = request.json
        folder_path = data.get('folder_path')
        max_depth = data.get('max_depth', 10)  # 默认最大深度为10层
        
        if not folder_path:
            return jsonify({"error": "folder_path is required"}), 400
        
        import os
        from pathlib import Path
        
        folder_path = Path(folder_path)
        if not folder_path.exists():
            return jsonify({"error": f"Folder does not exist: {folder_path}"}), 404
        
        if not folder_path.is_dir():
            return jsonify({"error": f"Path is not a directory: {folder_path}"}), 400
        
        # 识别模型文件夹的特征文件
        # 优先级：先检查最重要的文件（config.json），再检查其他文件
        model_indicators = [
            'config.json',  # Transformers 模型配置文件（最重要）
            'tokenizer.json',  # Tokenizer 文件
            'tokenizer_config.json',  # Tokenizer 配置文件
            'generation_config.json',  # 生成配置
            'model.safetensors',  # SafeTensors 格式模型文件
            'pytorch_model.bin',  # PyTorch 格式模型文件
            'model.pt',  # PyTorch 格式模型文件
            'adapter_config.json',  # LoRA/Adapter 配置
            'special_tokens_map.json',  # 特殊 token 映射
            'vocab.json',  # 词汇表
        ]
        
        def is_model_directory(dir_path):
            """检查目录是否是模型目录（包含特征文件）"""
            for indicator in model_indicators:
                if (dir_path / indicator).exists():
                    return True
            return False
        
        def scan_recursive(current_path, depth=0, visited=None):
            """递归扫描目录查找模型"""
            if visited is None:
                visited = set()
            
            # 防止无限循环（处理符号链接）
            real_path = current_path.resolve()
            if real_path in visited:
                return []
            visited.add(real_path)
            
            # 检查深度限制
            if depth > max_depth:
                return []
            
            models = []
            
            try:
                # 首先检查当前目录是否是模型目录
                if is_model_directory(current_path):
                    model_path = str(current_path.absolute())
                    # 使用相对于根目录的路径作为显示名称，更清晰
                    try:
                        rel_path = current_path.relative_to(folder_path)
                        if str(rel_path) == '.':
                            model_name = current_path.name
                        else:
                            model_name = f"{folder_path.name}/{rel_path}"
                    except ValueError:
                        model_name = current_path.name
                    
                    models.append({
                        "name": model_name,
                        "path": model_path
                    })
                    print(f"[Server] Found model: {model_name} at {model_path} (depth: {depth})")
                    # 如果当前目录是模型目录，不再继续扫描子目录（避免重复）
                    return models
                
                # 如果不是模型目录，继续扫描子目录
                for item in current_path.iterdir():
                    if not item.is_dir():
                        continue
                    
                    # 跳过一些常见的非模型目录（可选，提高效率）
                    if item.name.startswith('.') or item.name in ['__pycache__', 'node_modules']:
                        continue
                    
                    # 递归扫描子目录
                    sub_models = scan_recursive(item, depth + 1, visited)
                    models.extend(sub_models)
                    
            except PermissionError:
                print(f"[Server] Permission denied: {current_path}")
            except Exception as e:
                print(f"[Server] Error scanning {current_path}: {e}")
            
            return models
        
        try:
            # 开始递归扫描
            models = scan_recursive(folder_path)
            
            # 如果没有找到模型，检查根目录本身是否是模型文件夹
            if not models and is_model_directory(folder_path):
                models.append({
                    "name": folder_path.name,
                    "path": str(folder_path.absolute())
                })
                print(f"[Server] Root folder is a model: {folder_path.name}")
            
            # 按路径排序，使结果更有序
            models.sort(key=lambda x: x['path'])
            
            print(f"[Server] Scanned {folder_path} (max_depth={max_depth}), found {len(models)} models")
            return jsonify({
                "models": models,
                "count": len(models)
            })
        except PermissionError:
            return jsonify({"error": f"Permission denied: {folder_path}"}), 403
        except Exception as e:
            print(f"[Server] Error scanning models: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500
            
    except Exception as e:
        print(f"[Server] Error in scan_models: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/submit', methods=['POST'])
def submit_task():
    """Submit benchmark tasks."""
    data = request.json
    print(f"[Server] ========== SUBMIT REQUEST RECEIVED ==========")
    print(f"[Server] Request data: {data}")
    
    if not data:
        print(f"[Server] Error: No data provided")
        return jsonify({"error": "No data provided"}), 400
    
    if not isinstance(data, list):
        data = [data]

    requests_to_submit = []
    for item in data:
        # Generate req_id if not present
        if 'req_id' not in item:
            item['req_id'] = str(uuid.uuid4())
        
        # Convert model_type to bytes as required by runner
        if 'model_type' in item and isinstance(item['model_type'], str):
            item['model_type'] = item['model_type'].encode('utf-8')
        
        requests_to_submit.append(item)
        print(f"[Server] Prepared request: req_id={item.get('req_id')}, model={item.get('model_name_or_path')}, dataset={item.get('dataset_name')}")

    try:
        print(f"[Server] Submitting {len(requests_to_submit)} requests to runner...")
        print(f"[Server] Input queue size before submit: {runner.input_queue.qsize()}")
        runner.submit_requests(requests_to_submit)
        print(f"[Server] Requests submitted successfully")
        print(f"[Server] Input queue size after submit: {runner.input_queue.qsize()}")
        print(f"[Server] ========== SUBMIT REQUEST COMPLETED ==========")
        return jsonify({
            "status": "success",
            "submitted_count": len(requests_to_submit),
            "req_ids": [r['req_id'] for r in requests_to_submit]
        })
    except Exception as e:
        print(f"[Server] Error submitting requests: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    """Get benchmark results (polling)."""
    count = request.args.get('count', type=int)
    timeout = request.args.get('timeout', default=0.1, type=float)
    
    # Only log every 10th poll to reduce noise
    import random
    should_log = random.random() < 0.1  # Log 10% of polls
    
    if should_log:
        print(f"[Server] Polling results: count={count}, timeout={timeout}")
        print(f"[Server] Input queue size: {runner.input_queue.qsize()}, Output queue size: {runner.output_queue.qsize()}")
    
    results = runner.get_results(count=count, timeout=timeout)
    
    if results:
        print(f"[Server] Retrieved {len(results)} results")
        for res in results:
            print(f"[Server] Result for req_id={res.req_id}, has_error={res.error is not None}, has_result={res.result is not None}")
    elif should_log:
        print(f"[Server] Retrieved 0 results (task still processing)")
    
    # Convert NamedTuple to dict for JSON serialization
    # Also include model_name and dataset_name from progress_tracker
    json_results = []
    for res in results:
        # Get model and dataset info from progress tracker
        progress_info = runner.get_progress([res.req_id]).get(res.req_id, {})
        json_results.append({
            "req_id": res.req_id,
            "model_name": progress_info.get("model", "unknown"),
            "dataset_name": progress_info.get("dataset", "unknown"),
            "result": res.result,
            "error": res.error
        })
        
    return jsonify(json_results)

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get all benchmark history."""
    history = runner.get_all_history()
    # history is list of tuples: (model_name, dataset_name, results, created_at, updated_at)
    formatted_history = []
    for item in history:
        formatted_history.append({
            "model_name": item[0],
            "dataset_name": item[1],
            "results": item[2],
            "created_at": item[3],
            "updated_at": item[4]
        })
    return jsonify(formatted_history)

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """Clear benchmark history."""
    model_name = request.args.get('model_name')
    dataset_name = request.args.get('dataset_name')
    
    try:
        count = runner.clear_history(model_name=model_name, dataset_name=dataset_name)
        return jsonify({"status": "success", "deleted_count": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/progress', methods=['GET'])
def get_progress():
    """Get progress information for pending requests."""
    req_ids = request.args.getlist('req_id')
    if not req_ids:
        # Return all progress if no specific req_ids requested
        progress = runner.get_progress()
    else:
        progress = runner.get_progress(req_ids)
    
    # Calculate estimated completion time for each request
    import time
    for req_id, prog in progress.items():
        if prog.get("status") == "processing" and prog.get("current_item", 0) > 0:
            total_items = prog.get("total_items", 0)
            current_item = prog.get("current_item", 0)
            if total_items > 0 and current_item > 0:
                elapsed_time = time.time() - prog.get("start_time", time.time())
                avg_time_per_item = elapsed_time / current_item
                remaining_items = total_items - current_item
                estimated_remaining = avg_time_per_item * remaining_items
                prog["estimated_completion_time"] = time.time() + estimated_remaining
                prog["estimated_remaining_seconds"] = estimated_remaining
                prog["progress_percentage"] = (current_item / total_items) * 100
            else:
                prog["estimated_completion_time"] = None
                prog["estimated_remaining_seconds"] = None
                prog["progress_percentage"] = 0
        else:
            prog["estimated_completion_time"] = None
            prog["estimated_remaining_seconds"] = None
            prog["progress_percentage"] = 100 if prog.get("status") == "completed" else 0
    
    return jsonify(progress)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics."""
    return jsonify(runner.get_database_stats())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
