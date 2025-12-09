#!/usr/bin/env python3
"""
清理 GPU 内存的脚本
用于在遇到 CUDA OOM 错误时释放 GPU 内存
"""

import torch
import gc

def clear_gpu_memory():
    """清理 GPU 内存"""
    if torch.cuda.is_available():
        print(f"清理前 GPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 清理 PyTorch 缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 再次清理
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"清理后 GPU 内存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print("GPU 内存已清理")
    else:
        print("CUDA 不可用")

if __name__ == "__main__":
    clear_gpu_memory()


