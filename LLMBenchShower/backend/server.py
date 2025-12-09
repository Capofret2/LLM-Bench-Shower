import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from runner import get_llm_bench_runner
from bench.utils import get_available_datasets

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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
    json_results = []
    for res in results:
        json_results.append({
            "req_id": res.req_id,
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

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics."""
    return jsonify(runner.get_database_stats())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
