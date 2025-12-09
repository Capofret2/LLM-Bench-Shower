import os
import torch

# load model
LBS_LOCAL_DEVICE_MAP = os.getenv("LBS_LOCAL_DEVICE_MAP", "auto")

# model loading parameters
LBS_TORCH_DTYPE = os.getenv("LBS_TORCH_DTYPE", "auto")  # auto, float16, bfloat16, float32
LBS_TRUST_REMOTE_CODE = bool(int(os.getenv("LBS_TRUST_REMOTE_CODE", "0")))  # Set to 1 to enable trust_remote_code

# GPU memory limit (percentage of total GPU memory, 0.0-1.0)
# Set to 0.8 to use 80% of GPU memory, or 0.0 to disable limit
LBS_GPU_MEMORY_LIMIT = float(os.getenv("LBS_GPU_MEMORY_LIMIT", "0.0"))  # Default: no limit (0.0)

# Enable PyTorch memory management to prevent OOM
# Set to 1 to enable expandable_segments (helps with fragmentation)
LBS_ENABLE_MEMORY_MANAGEMENT = bool(int(os.getenv("LBS_ENABLE_MEMORY_MANAGEMENT", "1")))

# model caching
LBS_USE_MODEL_CACHE = bool(int(os.getenv("LBS_USE_MODEL_CACHE", "1")))
LBS_MAX_CACHED_LOCAL_MODELS = int(os.getenv("LBS_MAX_CACHED_LOCAL_MODELS", "4"))
LBS_GPU_MAX_UTILIZATION = float(os.getenv("LBS_GPU_MAX_UTILIZATION", "0.5"))
LBS_CPU_MAX_UTILIZATION = float(os.getenv("LBS_CPU_MAX_UTILIZATION", "0.8"))

# database
LBS_DB_PATH = os.getenv(
    "LBS_DB_PATH",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        "benchmark_results.db",
    ),
)
LBS_DB_WRITEBACK_S = int(os.getenv("LBS_DB_WRITEBACK_S", "240"))

# Helper function to get torch dtype
def get_torch_dtype():
    """Get torch dtype from environment variable."""
    dtype_str = LBS_TORCH_DTYPE.lower()
    if dtype_str == "auto":
        return "auto"
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float32":
        return torch.float32
    else:
        return "auto"

# Helper function to get max_memory dict for model loading
def get_max_memory():
    """Get max_memory dict for model loading based on GPU memory limit."""
    if LBS_GPU_MEMORY_LIMIT <= 0.0 or not torch.cuda.is_available():
        return None  # No limit
    
    max_memory = {}
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory
        # Convert to bytes and apply limit
        max_memory[i] = int(total_memory * LBS_GPU_MEMORY_LIMIT)
    
    return max_memory

# Initialize PyTorch memory management if enabled
# Note: This is set at module import time, so torch.cuda may not be available yet
# The actual check happens in get_max_memory()
if LBS_ENABLE_MEMORY_MANAGEMENT:
    import os
    # Set PyTorch CUDA allocator config to use expandable segments
    # This helps reduce memory fragmentation
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
