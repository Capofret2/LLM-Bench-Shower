#!/usr/bin/env python3
"""
测试benchmarker.py中的四个实验：
1. 单针检索 (SINGLE_NEEDLE_RETRIEVAL)
2. 多针检索 (MULTI_NEEDLE_RETRIEVAL) 
3. 多针推理 (MULTI_NEEDLE_REASONING)
4. 祖源追溯挑战 (ANCESTRAL_TRACE_CHALLENGE)

支持本地模型接口和DeepSeek API接口
"""

import os
import sys
import asyncio
from openai import Client
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加backend目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "LLMBenchShower", "backend")
sys.path.insert(0, backend_dir)

from bench.needle_in_haystack.benchmarker import NeedleInHaystackBenchmarker, TaskType

class BenchmarkerExperimentTester:
    def __init__(self):
        self.benchmarker = NeedleInHaystackBenchmarker()
        self.local_model = None
        self.local_tokenizer = None
        self.api_client = None
    
    async def setup_local_model(self, model_path: str):
        """设置本地模型"""
        try:
            print(f"加载本地模型: {model_path}")
            self.local_tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("本地模型加载成功")
            return True
        except Exception as e:
            print(f"本地模型加载失败: {e}")
            return False
    
    def setup_api_client(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        """设置DeepSeek API客户端"""
        try:
            print("设置DeepSeek API客户端")
            self.api_client = Client(
                api_key=api_key,
                base_url=base_url
            )
            print("API客户端设置成功")
            return True
        except Exception as e:
            print(f"API客户端设置失败: {e}")
            return False
    
    async def run_single_experiment(self, task_type: str, use_local: bool = True, params: dict = None):
        """运行单个实验"""
        if params is None:
            params = {
                "context_length": 32768,
                "depth_percent": 50,
                "num_needles": 3 if task_type in [TaskType.MULTI_NEEDLE_RETRIEVAL, TaskType.MULTI_NEEDLE_REASONING] else 5 if task_type == TaskType.ANCESTRAL_TRACE_CHALLENGE else 1,
                "language": "English",
                "task_type": task_type,
                "model_name": "deepseek-chat"  # DeepSeek模型名称
            }
        
        print(f"\n{'='*60}")
        print(f"运行实验: {task_type}")
        print(f"使用: {'本地模型' if use_local else 'DeepSeek API'}")
        print(f"参数: {params}")
        print(f"{'='*60}")
        
        try:
            if use_local:
                if self.local_model is None or self.local_tokenizer is None:
                    print("错误: 本地模型未设置")
                    return None
                
                result = await self.benchmarker.evaluate_local_llm(
                    self.local_model, self.local_tokenizer, params
                )
            else:
                if self.api_client is None:
                    print("错误: API客户端未设置")
                    return None
                
                result = await self.benchmarker.evaluate_api_llm(
                    self.api_client, params
                )
            
            print(f"实验结果:")
            print(f"  模型: {result.get('model', 'N/A')}")
            print(f"  数据集: {result.get('dataset', 'N/A')}")
            print(f"  语言: {result.get('language', 'N/A')}")
            print(f"  上下文长度: {result.get('context_length', 'N/A')}")
            print(f"  深度百分比: {result.get('depth_percent', 'N/A')}")
            print(f"  针数: {result.get('num_needles', 'N/A')}")
            print(f"  得分: {result.get('score', 'N/A'):.3f}")
            print(f"  任务类型: {result.get('task_type', 'N/A')}")
            print(f"  成功: {result.get('success', False)}")
            
            if result.get('response'):
                print(f"\n  模型响应 (前200字符): {result['response'][:200]}...")
            
            if 'expected_needle' in result:
                print(f"  预期针: {result['expected_needle']}")
            elif 'expected_answer' in result:
                print(f"  预期答案: {result['expected_answer']}")
            
            return result
            
        except Exception as e:
            print(f"实验执行失败: {e}")
            return None
    
    async def run_all_experiments_local(self, model_path: str):
        """使用本地模型运行所有四个实验"""
        print("\n" + "="*80)
        print("开始使用本地模型运行四个实验")
        print("="*80)
        
        # 设置本地模型
        if not await self.setup_local_model(model_path):
            return
        
        results = {}
        
        # 运行四个实验
        tasks = [
            TaskType.SINGLE_NEEDLE_RETRIEVAL,
            TaskType.MULTI_NEEDLE_RETRIEVAL,
            TaskType.MULTI_NEEDLE_REASONING,
            TaskType.ANCESTRAL_TRACE_CHALLENGE
        ]
        
        for task in tasks:
            result = await self.run_single_experiment(task, use_local=True)
            results[task] = result
        
        return results
    
    async def run_all_experiments_api(self, api_key: str):
        """使用DeepSeek API运行所有四个实验"""
        print("\n" + "="*80)
        print("开始使用DeepSeek API运行四个实验")
        print("="*80)
        
        # 设置API客户端
        if not self.setup_api_client(api_key):
            return
        
        results = {}
        
        # 运行四个实验
        tasks = [
            TaskType.SINGLE_NEEDLE_RETRIEVAL,
            TaskType.MULTI_NEEDLE_RETRIEVAL,
            TaskType.MULTI_NEEDLE_REASONING,
            TaskType.ANCESTRAL_TRACE_CHALLENGE
        ]
        
        for task in tasks:
            result = await self.run_single_experiment(task, use_local=False)
            results[task] = result
        
        return results
    
    async def run_needlebench_v2_comprehensive(self, model_path: str = None, api_key: str = None):
        """运行NeedleBench V2综合评估"""
        print("\n" + "="*80)
        print("开始运行NeedleBench V2综合评估")
        print("="*80)
        
        if model_path:
            # 使用本地模型
            if not await self.setup_local_model(model_path):
                return
            
            result = await self.benchmarker.evaluate_needlebench_v2(
                model=self.local_model, 
                tokenizer=self.local_tokenizer
            )
        elif api_key:
            # 使用API
            if not self.setup_api_client(api_key):
                return
            
            result = await self.benchmarker.evaluate_needlebench_v2(
                client=self.api_client
            )
        else:
            print("错误: 必须提供模型路径或API密钥")
            return
        
        print("NeedleBench V2综合评估结果:")
        print(f"  总体得分: {result.get('overall_score', 'N/A'):.3f}")
        print(f"  评分系统: {result.get('scoring_system', 'N/A')}")
        
        task_scores = result.get('task_scores', {})
        print(f"  各任务得分:")
        for task, score in task_scores.items():
            print(f"    {task}: {score:.3f}")
        
        return result

async def main():
    """主测试函数"""
    tester = BenchmarkerExperimentTester()
    
    # 配置选项 - 请根据实际情况修改
    # 方法1: 设置环境变量 DEEPSEEK_API_KEY="your_api_key"
    # 方法2: 直接在此处填写您的配置
    
    # 本地模型路径配置 - 请修改为您的实际路径
    LOCAL_MODEL_PATH = "/path/to/your/local/model"  # 示例: "/home/jikaining/models/deepseek-coder-7b"
    
    # DeepSeek API密钥配置 - 请修改为您的实际API密钥
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or "sk-40d956c427e54efabdf1190630c6840e"  # 示例: "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    
    print("Benchmarker实验测试器")
    print("支持四个实验:")
    print("1. 单针检索 (SINGLE_NEEDLE_RETRIEVAL)")
    print("2. 多针检索 (MULTI_NEEDLE_RETRIEVAL)")
    print("3. 多针推理 (MULTI_NEEDLE_REASONING)")
    print("4. 祖源追溯挑战 (ANCESTRAL_TRACE_CHALLENGE)")
    print("")
    print("接口位置:")
    print("- 本地模型接口: benchmarker.py中的evaluate_local_llm()方法")
    print("- DeepSeek API接口: benchmarker.py中的evaluate_api_llm()方法")
    print("- 综合评估接口: benchmarker.py中的evaluate_needlebench_v2()方法")
    print("")
    
    # 测试选项
    test_option = input("选择测试模式 (1=本地模型, 2=DeepSeek API, 3=NeedleBench V2综合): ")
    
    if test_option == "1":
        # 本地模型测试
        if LOCAL_MODEL_PATH == "/path/to/your/local/model":
            print("请先修改LOCAL_MODEL_PATH变量为您的本地模型路径")
            return
        
        await tester.run_all_experiments_local(LOCAL_MODEL_PATH)
    
    elif test_option == "2":
        # DeepSeek API测试
        if DEEPSEEK_API_KEY == "your_api_key_here":
            print("请先设置DEEPSEEK_API_KEY环境变量或修改代码中的API密钥")
            return
        
        await tester.run_all_experiments_api(DEEPSEEK_API_KEY)
    
    elif test_option == "3":
        # NeedleBench V2综合测试
        sub_option = input("选择模型类型 (1=本地模型, 2=DeepSeek API): ")
        
        if sub_option == "1":
            if LOCAL_MODEL_PATH == "/path/to/your/local/model":
                print("请先修改LOCAL_MODEL_PATH变量为您的本地模型路径")
                return
            await tester.run_needlebench_v2_comprehensive(model_path=LOCAL_MODEL_PATH)
        elif sub_option == "2":
            if DEEPSEEK_API_KEY == "your_api_key_here":
                print("请先设置DEEPSEEK_API_KEY环境变量或修改代码中的API密钥")
                return
            await tester.run_needlebench_v2_comprehensive(api_key=DEEPSEEK_API_KEY)
        else:
            print("无效选项")
    
    else:
        print("无效选项")

if __name__ == "__main__":
    # 确保导入torch
    import torch
    
    # 运行测试
    asyncio.run(main())