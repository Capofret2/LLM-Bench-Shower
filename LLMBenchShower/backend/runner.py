import atexit
import gc
import time
import torch
import threading
import queue
from collections import defaultdict
from typing import Dict, Tuple, List, Any, NamedTuple, Set
from openai import Client
from transformers import AutoModelForCausalLM, AutoTokenizer
import envs
from bench import init_all_benchmarkers
from bench.utils import get_available_datasets
from db import BenchmarkDatabase
from model_cache import ModelCache


# NOTE(haukzero): dataset_name format: dataset_name/subdataset_name. For example, "LongBench/2wikimqa"
class ModelDatasetPair(NamedTuple):
    model_name: str
    dataset_name: str


class BenchResponse(NamedTuple):
    req_id: str
    result: Dict[str, Any]
    error: str | None = None


_LLM_BENCHMARKER_RUNNER = None


class LLMBenchRunner:
    def __init__(self):
        self.available_datasets = get_available_datasets()
        self.benchmarkers = init_all_benchmarkers()

        self.db = BenchmarkDatabase(envs.LBS_DB_PATH)
        # Keep in-memory cache for fast access during runtime
        self.bench_history: Dict[ModelDatasetPair, Dict] = defaultdict(dict)
        # Track which results need to be written to database
        self._dirty_results: Set[ModelDatasetPair] = set()
        self._dirty_lock = threading.Lock()

        # Load existing results from database into memory
        self._load_history_from_db()

        self.input_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.output_queue: queue.Queue[BenchResponse] = queue.Queue()
        
        # Progress tracking: req_id -> progress info
        self.progress_tracker: Dict[str, Dict[str, Any]] = {}
        self.progress_lock = threading.Lock()

        self._closed = False

        # Start background write-back thread
        self._stop_writeback = threading.Event()
        self._writeback_interval = envs.LBS_DB_WRITEBACK_S
        self._writeback_thread = threading.Thread(
            target=self._writeback_worker, daemon=True, name="DBWriteBackThread"
        )
        self._writeback_thread.start()

        self._stop_consumer = threading.Event()
        self._consumer_thread = threading.Thread(
            target=self._consumer_worker, daemon=True, name="BenchConsumerThread"
        )
        self._consumer_thread.start()

        self.device_map = envs.LBS_LOCAL_DEVICE_MAP
        self.use_model_cache = envs.LBS_USE_MODEL_CACHE
        if self.use_model_cache:
            self.max_cached_local_models = envs.LBS_MAX_CACHED_LOCAL_MODELS
            self.max_gpu_utilization = envs.LBS_GPU_MAX_UTILIZATION
            self.max_cpu_utilization = envs.LBS_CPU_MAX_UTILIZATION
            self.model_cache = ModelCache(
                max_cached_models=self.max_cached_local_models,
                gpu_max_utilization=self.max_gpu_utilization,
                cpu_max_utilization=self.max_cpu_utilization,
                device_map=self.device_map,
            )
            self.eval_local_model_fn = self.eval_local_model_cached
        else:
            self.eval_local_model_fn = self.eval_local_model_uncached

    def _load_history_from_db(self):
        all_results = self.db.get_all_results()
        for model_name, dataset_name, results, _, _ in all_results:
            pair = ModelDatasetPair(model_name, dataset_name)
            self.bench_history[pair] = results

    def _writeback_worker(self):
        while not self._stop_writeback.wait(timeout=self._writeback_interval):
            self._flush_dirty_results()

    def _consumer_worker(self):
        while not self._stop_consumer.is_set():
            try:
                request = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Extract req_id from request
            req_id = request.get("req_id")
            if not req_id:
                print("Warning: Request without req_id received, skipping")
                self.input_queue.task_done()
                continue

            dataset_name = request.get("dataset_name", "unknown")
            model_name = request.get("model_name_or_path", "unknown")
            print(f"[Runner] ========== STARTING REQUEST ==========")
            print(f"[Runner] Request ID: {req_id}")
            print(f"[Runner] Dataset: {dataset_name}")
            print(f"[Runner] Model: {model_name}")
            print(f"[Runner] Input queue remaining: {self.input_queue.qsize()}")

            # Initialize progress tracking
            with self.progress_lock:
                self.progress_tracker[req_id] = {
                    "req_id": req_id,
                    "dataset": dataset_name,
                    "model": model_name,
                    "status": "processing",
                    "current_item": 0,
                    "total_items": 0,
                    "current_question": "",
                    "start_time": time.time(),
                    "last_update_time": time.time(),
                    "estimated_completion_time": None
                }

            try:
                # Process the request (exclude req_id from processing params)
                result = self._process_single_request(request, req_id=req_id)
                print(f"[Runner] Request {req_id} completed successfully")
                print(f"[Runner] Result keys: {list(result.keys()) if isinstance(result, dict) else 'not_dict'}")
                
                # Update progress to completed
                with self.progress_lock:
                    if req_id in self.progress_tracker:
                        self.progress_tracker[req_id]["status"] = "completed"
                        self.progress_tracker[req_id]["current_item"] = self.progress_tracker[req_id]["total_items"]
                        self.progress_tracker[req_id]["last_update_time"] = time.time()
                
                # Put result in output queue
                response = BenchResponse(req_id=req_id, result=result, error=None)
                self.output_queue.put(response)
                print(f"[Runner] Result added to output queue. Output queue size: {self.output_queue.qsize()}")
                print(f"[Runner] ========== REQUEST COMPLETED ==========")
                
                # Clean up progress tracking after a delay (to allow frontend to fetch final status)
                def cleanup_progress():
                    time.sleep(5)  # Keep progress for 5 seconds after completion
                    with self.progress_lock:
                        self.progress_tracker.pop(req_id, None)
                threading.Thread(target=cleanup_progress, daemon=True).start()
            except Exception as e:
                # Put error response in output queue
                error_msg = f"Error processing request: {str(e)}"
                print(f"[Runner] Request {req_id} failed with error: {error_msg}")
                import traceback
                traceback.print_exc()
                
                # Update progress to failed
                with self.progress_lock:
                    if req_id in self.progress_tracker:
                        self.progress_tracker[req_id]["status"] = "failed"
                        self.progress_tracker[req_id]["error"] = error_msg
                        self.progress_tracker[req_id]["last_update_time"] = time.time()
                
                response = BenchResponse(req_id=req_id, result={}, error=error_msg)
                self.output_queue.put(response)
                print(f"[Runner] Error response added to output queue. Output queue size: {self.output_queue.qsize()}")
                print(f"[Runner] ========== REQUEST FAILED ==========")
            finally:
                self.input_queue.task_done()

    def _process_single_request(self, request: Dict[str, Any], req_id: str = None) -> Dict:
        """Process a single benchmark request."""
        model_type: bytes = request.get("model_type", b"local")
        dataset_name = request.get("dataset_name", "unknown")
        model_path = request.get("model_name_or_path", "unknown")
        print(f"[Runner] Processing request: dataset={dataset_name}, model={model_path}, type={model_type}")
        try:
            match model_type:
                case b"local":
                    # Store req_id in request for use in eval functions
                    if req_id:
                        request["_req_id"] = req_id
                    result = self.eval_local_model_fn(**request)
                    print(f"[Runner] Request completed: dataset={dataset_name}, result_keys={list(result.keys()) if isinstance(result, dict) else 'not_dict'}")
                    return result
                case b"api":
                    result = self.eval_api_model(**request)
                    print(f"[Runner] Request completed: dataset={dataset_name}, result_keys={list(result.keys()) if isinstance(result, dict) else 'not_dict'}")
                    return result
                case _:
                    raise ValueError(f"Unknown model_type: {model_type}")
        except KeyError as e:
            # Provide more helpful error message for missing benchmarker
            if "NeedleInAHaystack" in str(e) or "NeedleInHaystack" in str(e):
                available = list(self.benchmarkers.keys())
                raise ValueError(
                    f"Benchmarker not found. Available benchmarkers: {available}. "
                    f"Requested: {request.get('dataset_name', 'unknown')}"
                ) from e
            raise

    def _mark_dirty(self, pair: ModelDatasetPair):
        with self._dirty_lock:
            self._dirty_results.add(pair)

    def _flush_dirty_results(self):
        with self._dirty_lock:
            if not self._dirty_results:
                return

            # Copy and clear the dirty set
            dirty_pairs = list(self._dirty_results)
            self._dirty_results.clear()

        # Prepare batch data
        batch_data = []
        for pair in dirty_pairs:
            if pair in self.bench_history:
                batch_data.append(
                    (pair.model_name, pair.dataset_name, self.bench_history[pair])
                )

        # Write to database
        if batch_data:
            try:
                self.db.save_results_batch(batch_data)
            except Exception as e:
                # If write fails, mark them as dirty again
                print(f"Warning: Failed to write results to database: {e}")
                with self._dirty_lock:
                    self._dirty_results.update(dirty_pairs)

    def _split_dataset_name(self, dataset_name: str) -> Tuple[str, str]:
        try:
            supdataset_name, subdataset_name = dataset_name.split("/")
        except ValueError:
            raise ValueError(
                f"Dataset name '{dataset_name}' is not in the correct format 'dataset_name/subdataset_name'."
            )
        if (
            supdataset_name not in self.available_datasets
            or subdataset_name not in self.available_datasets[supdataset_name]
        ):
            raise ValueError(f"Dataset {dataset_name} not found in available datasets.")
        return supdataset_name, subdataset_name

    def eval_local_model_uncached(
        self,
        model_name_or_path: str,
        dataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        pair = ModelDatasetPair(model_name_or_path, dataset_name)
        print(f"[Runner] eval_local_model_uncached: model={model_name_or_path}, dataset={dataset_name}")
        if pair in self.bench_history:
            cached_result = self.bench_history[pair]
            print(f"[Runner] Found cached result for {pair}, returning from cache")
            print(f"[Runner] Cached result keys: {list(cached_result.keys()) if isinstance(cached_result, dict) else 'not_dict'}")
            print(f"[Runner] Cached result evaluation_type: {cached_result.get('evaluation_type', 'not_set') if isinstance(cached_result, dict) else 'not_dict'}")
            # For NeedleInAHaystack, we want to force re-run to get parameterized results
            if "NeedleInAHaystack" in dataset_name:
                print(f"[Runner] NeedleInAHaystack detected - forcing re-run to get parameterized results (ignoring cache)")
                # Don't return cached result, continue to run new test
            else:
                return cached_result
        print(f"[Runner] No cached result (or cache bypassed), loading model and running benchmark...")
        supdataset_name, subdataset_name = self._split_dataset_name(dataset_name)
        print(f"[Runner] Split dataset: sup={supdataset_name}, sub={subdataset_name}")

        # Prepare model loading kwargs
        model_kwargs = {
            "device_map": self.device_map,
        }
        
        # Add torch_dtype if specified
        torch_dtype = envs.get_torch_dtype()
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype
            dtype_name = str(torch_dtype).replace("torch.", "") if isinstance(torch_dtype, type) else str(torch_dtype)
            print(f"[Runner] Configuring model with torch_dtype: {dtype_name}")
        else:
            print(f"[Runner] Using torch_dtype: auto (will use model's default, likely float32)")
        
        # Add max_memory if GPU memory limit is set
        max_memory = envs.get_max_memory()
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
            if torch.cuda.is_available():
                total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                limit_mem = max_memory[0] / 1024**3
                print(f"[Runner] GPU memory limit set: {limit_mem:.2f} GB / {total_mem:.2f} GB ({envs.LBS_GPU_MEMORY_LIMIT*100:.1f}%)")
        
        # Add trust_remote_code if enabled
        if envs.LBS_TRUST_REMOTE_CODE:
            model_kwargs["trust_remote_code"] = True
        
        # Load model with GPU fallback to CPU on OOM
        print(f"[Runner] Loading model from {model_name_or_path}...")
        if torch.cuda.is_available():
            print(f"[Runner] GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **model_kwargs
            )
            # Check actual dtype used
            actual_dtype = next(model.parameters()).dtype
            print(f"[Runner] Model loaded successfully on device: {self.device_map}")
            print(f"[Runner] Actual model dtype: {actual_dtype}")
            if torch.cuda.is_available():
                print(f"[Runner] GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        except torch.cuda.OutOfMemoryError as e:
            print(f"[Runner] GPU OOM during model loading: {e}")
            print(f"[Runner] Falling back to CPU...")
            # Fallback to CPU: remove GPU-specific kwargs and set device_map to CPU
            cpu_model_kwargs = model_kwargs.copy()
            cpu_model_kwargs["device_map"] = "cpu"
            if "max_memory" in cpu_model_kwargs:
                del cpu_model_kwargs["max_memory"]  # Remove max_memory for CPU
            # Clear GPU cache before retrying
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, **cpu_model_kwargs
            )
            print(f"[Runner] Model loaded successfully on CPU (fallback)")
            self.device_map = "cpu"  # Update device_map for this request
        except Exception as e:
            print(f"[Runner] Error loading model: {e}")
            raise
        
        # Load tokenizer
        print(f"[Runner] Loading tokenizer from {model_name_or_path}...")
        tokenizer_kwargs = {}
        if envs.LBS_TRUST_REMOTE_CODE:
            tokenizer_kwargs["trust_remote_code"] = True
        
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        print(f"[Runner] Tokenizer loaded successfully")

        # Set pad_token if not present (required for some models like Llama)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        # Get benchmarker (handle naming variations)
        if supdataset_name not in self.benchmarkers:
            # Try alternative naming (e.g., NeedleInAHaystack vs NeedleInHaystack)
            alt_name = supdataset_name.replace("NeedleInAHaystack", "NeedleInHaystack").replace("NeedleInHaystack", "NeedleInAHaystack")
            if alt_name in self.benchmarkers:
                benchmarker = self.benchmarkers[alt_name]
            else:
                available = list(self.benchmarkers.keys())
                raise KeyError(
                    f"Benchmarker '{supdataset_name}' not found. "
                    f"Available benchmarkers: {available}. "
                    f"Dataset name: {dataset_name}"
                )
        else:
            benchmarker = self.benchmarkers[supdataset_name]
        
        print(f"[Runner] Calling benchmarker.evaluate_local_llm: supdataset={supdataset_name}, subdataset={subdataset_name}")
        print(f"[Runner] Benchmarker type: {type(benchmarker).__name__}")
        benchmark_results = benchmarker.evaluate_local_llm(
            model=model,
            tokenizer=tokenizer,
            subdataset_name=subdataset_name,
            *args,
            **kwargs,
        )
        print(f"[Runner] Benchmarker returned: result_type={type(benchmark_results)}, keys={list(benchmark_results.keys()) if isinstance(benchmark_results, dict) else 'not_dict'}")
        if isinstance(benchmark_results, dict):
            print(f"[Runner] Result evaluation_type: {benchmark_results.get('evaluation_type', 'not_set')}")
            print(f"[Runner] Result total_tests: {benchmark_results.get('total_tests', 'not_set')}")
        # Save to memory and mark for write-back
        self.bench_history[pair] = benchmark_results
        self._mark_dirty(pair)

        print(f"[Runner] Releasing model and tokenizer...")
        if torch.cuda.is_available():
            print(f"[Runner] GPU memory before release: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        if torch.cuda.is_available():
            print(f"[Runner] GPU memory after release: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        return benchmark_results

    def eval_local_model_cached(
        self,
        model_name_or_path: str,
        dataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        # Extract req_id from kwargs if available (will be passed to benchmarker)
        req_id = kwargs.get("_req_id")
        pair = ModelDatasetPair(model_name_or_path, dataset_name)
        print(f"[Runner] eval_local_model_cached: model={model_name_or_path}, dataset={dataset_name}")
        if pair in self.bench_history:
            cached_result = self.bench_history[pair]
            print(f"[Runner] Found cached result for {pair}, returning from cache")
            print(f"[Runner] Cached result keys: {list(cached_result.keys()) if isinstance(cached_result, dict) else 'not_dict'}")
            print(f"[Runner] Cached result evaluation_type: {cached_result.get('evaluation_type', 'not_set') if isinstance(cached_result, dict) else 'not_dict'}")
            # For NeedleInAHaystack, we want to force re-run to get parameterized results
            if "NeedleInAHaystack" in dataset_name:
                print(f"[Runner] NeedleInAHaystack detected - forcing re-run to get parameterized results (ignoring cache)")
                # Don't return cached result, continue to run new test
                # For NeedleInAHaystack, use uncached version to ensure memory is released after use
                print(f"[Runner] Using uncached evaluation for NeedleInAHaystack to ensure memory release")
                return self.eval_local_model_uncached(model_name_or_path, dataset_name, *args, **kwargs)
            else:
                return cached_result
        print(f"[Runner] No cached result (or cache bypassed), using model cache...")
        supdataset_name, subdataset_name = self._split_dataset_name(dataset_name)
        print(f"[Runner] Split dataset: sup={supdataset_name}, sub={subdataset_name}")

        # Use model cache to get model and tokenizer
        print(f"[Runner] Getting model from cache: {model_name_or_path}")
        if torch.cuda.is_available():
            print(f"[Runner] Current GPU memory before loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            # Clear cache before loading if memory is low
            if torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 0.9:
                print(f"[Runner] GPU memory usage > 90%, clearing cache before loading...")
                torch.cuda.empty_cache()
                gc.collect()
        model, tokenizer = self.model_cache.get_model(model_name_or_path)
        print(f"[Runner] Model retrieved from cache")
        if torch.cuda.is_available():
            print(f"[Runner] Current GPU memory after loading: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Get benchmarker (handle naming variations)
        print(f"[Runner] Getting benchmarker: supdataset={supdataset_name}, subdataset={subdataset_name}")
        if supdataset_name not in self.benchmarkers:
            # Try alternative naming (e.g., NeedleInAHaystack vs NeedleInHaystack)
            alt_name = supdataset_name.replace("NeedleInAHaystack", "NeedleInHaystack").replace("NeedleInHaystack", "NeedleInAHaystack")
            if alt_name in self.benchmarkers:
                benchmarker = self.benchmarkers[alt_name]
            else:
                available = list(self.benchmarkers.keys())
                raise KeyError(
                    f"Benchmarker '{supdataset_name}' not found. "
                    f"Available benchmarkers: {available}. "
                    f"Dataset name: {dataset_name}"
                )
        else:
            benchmarker = self.benchmarkers[supdataset_name]

        print(f"[Runner] Calling benchmarker.evaluate_local_llm: supdataset={supdataset_name}, subdataset={subdataset_name}")
        print(f"[Runner] Benchmarker type: {type(benchmarker).__name__}")
        
        # Get req_id from kwargs if available
        req_id = kwargs.get("_req_id")
        
        # Create progress update callback if req_id is available
        def update_progress(current_item, total_items, current_question=""):
            """Update progress for the current request."""
            if req_id:
                with self.progress_lock:
                    if req_id in self.progress_tracker:
                        self.progress_tracker[req_id]["current_item"] = current_item
                        self.progress_tracker[req_id]["total_items"] = total_items
                        self.progress_tracker[req_id]["current_question"] = current_question
                        self.progress_tracker[req_id]["last_update_time"] = time.time()
                        # Log progress
                        percentage = (current_item / total_items * 100) if total_items > 0 else 0
                        question_preview = current_question[:50] + "..." if len(current_question) > 50 else current_question
                        print(f"[Runner] Progress [{req_id[:8]}...]: {current_item}/{total_items} ({percentage:.1f}%) - {question_preview if current_question else 'N/A'}")
        
        # Pass progress callback to benchmarker
        kwargs_with_progress = kwargs.copy()
        if req_id:
            kwargs_with_progress["progress_callback"] = update_progress
        
        benchmark_results = benchmarker.evaluate_local_llm(
            model=model,
            tokenizer=tokenizer,
            subdataset_name=subdataset_name,
            *args,
            **kwargs_with_progress,
        )
        print(f"[Runner] Benchmarker returned: result_type={type(benchmark_results)}, keys={list(benchmark_results.keys()) if isinstance(benchmark_results, dict) else 'not_dict'}")
        if isinstance(benchmark_results, dict):
            print(f"[Runner] Result evaluation_type: {benchmark_results.get('evaluation_type', 'not_set')}")
            print(f"[Runner] Result total_tests: {benchmark_results.get('total_tests', 'not_set')}")
        # Save to memory and mark for write-back
        self.bench_history[pair] = benchmark_results
        self._mark_dirty(pair)
        return benchmark_results

    def eval_api_model(
        self,
        model_name: str,
        dataset_name: str,
        openai_api_key: str,
        base_url: str | None = None,
        *args,
        **kwargs,
    ) -> Dict:
        api_model_name = f"api::{model_name}"
        pair = ModelDatasetPair(api_model_name, dataset_name)
        if pair in self.bench_history:
            return self.bench_history[pair]
        supdataset_name, subdataset_name = self._split_dataset_name(dataset_name)
        client = Client(api_key=openai_api_key, base_url=base_url)
        
        # Get benchmarker (handle naming variations)
        if supdataset_name not in self.benchmarkers:
            # Try alternative naming (e.g., NeedleInAHaystack vs NeedleInHaystack)
            alt_name = supdataset_name.replace("NeedleInAHaystack", "NeedleInHaystack").replace("NeedleInHaystack", "NeedleInAHaystack")
            if alt_name in self.benchmarkers:
                benchmarker = self.benchmarkers[alt_name]
            else:
                available = list(self.benchmarkers.keys())
                raise KeyError(
                    f"Benchmarker '{supdataset_name}' not found. "
                    f"Available benchmarkers: {available}. "
                    f"Dataset name: {dataset_name}"
                )
        else:
            benchmarker = self.benchmarkers[supdataset_name]
        
        benchmark_results = benchmarker.evaluate_api_llm(
            client=client,
            model=model_name,
            subdataset_name=subdataset_name,
            *args,
            **kwargs,
        )
        # Save to memory and mark for write-back
        self.bench_history[pair] = benchmark_results
        self._mark_dirty(pair)
        return benchmark_results

    def submit_request(self, request: Dict[str, Any]):
        if "req_id" not in request:
            raise ValueError("Request must contain 'req_id' field")
        elif "model_type" not in request or request["model_type"] not in [
            b"local",
            b"api",
        ]:
            raise ValueError("Request must contain valid 'model_type' field")
        print(f"[Runner] Adding request to input queue: req_id={request.get('req_id')}, queue_size_before={self.input_queue.qsize()}")
        self.input_queue.put(request)
        print(f"[Runner] Request added. Queue size after: {self.input_queue.qsize()}")

    def submit_requests(self, requests: List[Dict[str, Any]]):
        print(f"[Runner] submit_requests called with {len(requests)} requests")
        for request in requests:
            self.submit_request(request)

    def get_result(self, timeout: float = None) -> BenchResponse:
        return self.output_queue.get(timeout=timeout)

    def get_results(
        self,
        count: int | None = None,
        timeout: float | None = None,
    ) -> List[BenchResponse]:
        """Get multiple results from the output queue.

        Args:
            count: Number of results to retrieve (None = all available)
            timeout: Maximum time to wait for each result

        Returns:
            List of BenchResponse objects
        """
        results = []
        try:
            if count is None:
                # Get all available results without blocking
                while True:
                    try:
                        results.append(self.output_queue.get_nowait())
                    except queue.Empty:
                        break
            else:
                # Get specific number of results
                for _ in range(count):
                    results.append(self.output_queue.get(timeout=timeout))
        except queue.Empty:
            pass
        return results

    def get_progress(self, req_ids: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get progress information for one or more requests.
        
        Args:
            req_ids: List of request IDs to get progress for. If None, returns all.
            
        Returns:
            Dictionary mapping req_id to progress information
        """
        with self.progress_lock:
            if req_ids is None:
                # Return all progress
                return self.progress_tracker.copy()
            else:
                # Return only requested req_ids
                return {req_id: self.progress_tracker.get(req_id, {}) for req_id in req_ids}

    def get_database_stats(self) -> Dict:
        return self.db.get_stats()

    def get_all_history(self) -> List[Tuple[str, str, Dict, str, str]]:
        return self.db.get_all_results()

    def clear_history(self, model_name: str = None, dataset_name: str = None) -> int:
        """Clear benchmark history.

        Args:
            model_name: If specified, only clear results for this model
            dataset_name: If specified, only clear results for this dataset

        Returns:
            Number of results cleared
        """
        if model_name and dataset_name:
            # Clear specific result
            pair = ModelDatasetPair(model_name, dataset_name)
            if pair in self.bench_history:
                del self.bench_history[pair]
            # Remove from dirty set if present
            with self._dirty_lock:
                self._dirty_results.discard(pair)
            return 1 if self.db.delete_result(model_name, dataset_name) else 0
        elif model_name or dataset_name:
            # Clear by model or dataset - need to iterate
            count = 0
            to_delete = []
            for pair in self.bench_history.keys():
                if (model_name and pair.model_name == model_name) or (
                    dataset_name and pair.dataset_name == dataset_name
                ):
                    to_delete.append(pair)
            for pair in to_delete:
                del self.bench_history[pair]
                # Remove from dirty set if present
                with self._dirty_lock:
                    self._dirty_results.discard(pair)
                if self.db.delete_result(pair.model_name, pair.dataset_name):
                    count += 1
            return count
        else:
            # Clear all
            self.bench_history.clear()
            # Clear dirty set
            with self._dirty_lock:
                self._dirty_results.clear()
            return self.db.clear_all_results()

    def close(self):
        """Clean up resources and flush all pending writes."""
        if self._closed:
            return  # Already closed, avoid double cleanup

        self._closed = True

        # Stop the consumer thread
        self._stop_consumer.set()
        if self._consumer_thread.is_alive():
            self._consumer_thread.join(timeout=5.0)

        # Stop the write-back thread
        self._stop_writeback.set()
        if self._writeback_thread.is_alive():
            self._writeback_thread.join(timeout=5.0)

        # Flush any remaining dirty results
        self._flush_dirty_results()

        if self.use_model_cache:
            self.model_cache.clear_cache()

    def __del__(self):
        try:
            self.close()
        except Exception as e:
            print(f"Warning: Error during LLMBenchRunner cleanup: {e}")


def get_llm_bench_runner():
    global _LLM_BENCHMARKER_RUNNER
    if _LLM_BENCHMARKER_RUNNER is None:
        _LLM_BENCHMARKER_RUNNER = LLMBenchRunner()
    return _LLM_BENCHMARKER_RUNNER


def _cleanup_global_runner():
    global _LLM_BENCHMARKER_RUNNER
    if _LLM_BENCHMARKER_RUNNER is not None:
        try:
            _LLM_BENCHMARKER_RUNNER.close()
        except Exception as e:
            print(f"Warning: Error closing global LLMBenchRunner: {e}")


atexit.register(_cleanup_global_runner)
