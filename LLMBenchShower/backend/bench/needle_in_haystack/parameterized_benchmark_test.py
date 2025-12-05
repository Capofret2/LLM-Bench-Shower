#!/usr/bin/env python3
"""
参数化基准测试框架
支持多维度参数组合测试，全面评估模型在不同条件下的性能
"""

import os
import sys
import asyncio
import json
import csv
from datetime import datetime
from typing import List, Dict, Any
from openai import Client
from transformers import AutoModelForCausalLM, AutoTokenizer

# 添加backend目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "LLMBenchShower", "backend")
sys.path.insert(0, backend_dir)

from bench.needle_in_haystack.benchmarker import NeedleInHaystackBenchmarker, TaskType

class ParameterizedBenchmarkTester:
    def __init__(self):
        self.benchmarker = NeedleInHaystackBenchmarker()
        self.local_model = None
        self.local_tokenizer = None
        self.api_client = None
        self.results = []
        
        # 定义参数空间
        self.parameter_space = {
            # 上下文长度 (tokens)
            "context_lengths": [
                4096,    # 短文本
                8192,    # 中等文本
                16384,   # 长文本
                32768,   # 超长文本
                65536    # 极长文本
            ],
            
            # 埋针深度百分比 (0-100)
            "depth_percents": [
                10,      # 开头附近
                25,      # 前四分之一
                50,      # 中间
                75,      # 后四分之一
                90       # 结尾附近
            ],
            
            # 针数配置
            "needle_configs": {
                TaskType.SINGLE_NEEDLE_RETRIEVAL: [1],
                TaskType.MULTI_NEEDLE_RETRIEVAL: [2, 3, 5],
                TaskType.MULTI_NEEDLE_REASONING: [3, 5, 7],
                TaskType.ANCESTRAL_TRACE_CHALLENGE: [5]
            },
            
            # 语言配置
            "languages": ["English", "Chinese"],
            
            # 任务类型
            "task_types": [
                TaskType.SINGLE_NEEDLE_RETRIEVAL,
                TaskType.MULTI_NEEDLE_RETRIEVAL,
                TaskType.MULTI_NEEDLE_REASONING,
                TaskType.ANCESTRAL_TRACE_CHALLENGE
            ]
        }
    
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
    
    def generate_test_combinations(self, max_combinations: int = 100):
        """生成测试参数组合"""
        combinations = []
        
        # 计算每个任务类型的最大测试数，确保均衡分布
        task_types = self.parameter_space["task_types"]
        tests_per_task = max_combinations // len(task_types)
        remaining_tests = max_combinations % len(task_types)
        
        # 为每个任务类型生成测试组合
        for i, task_type in enumerate(task_types):
            task_combinations = []
            num_needles_options = self.parameter_space["needle_configs"][task_type]
            
            # 计算该任务类型的测试数
            task_max_tests = tests_per_task
            if i < remaining_tests:
                task_max_tests += 1
            
            # 为每个针数配置生成测试
            for num_needles in num_needles_options:
                # 为每个上下文长度和深度组合生成测试
                for context_length in self.parameter_space["context_lengths"]:
                    for depth_percent in self.parameter_space["depth_percents"]:
                        for language in self.parameter_space["languages"]:
                            combination = {
                                "context_length": context_length,
                                "depth_percent": depth_percent,
                                "num_needles": num_needles,
                                "language": language,
                                "task_type": task_type,
                                "model_name": "deepseek-chat"
                            }
                            
                            task_combinations.append(combination)
                            
                            # 限制该任务类型的组合数量
                            if len(task_combinations) >= task_max_tests:
                                break
                        if len(task_combinations) >= task_max_tests:
                            break
                    if len(task_combinations) >= task_max_tests:
                        break
                if len(task_combinations) >= task_max_tests:
                    break
            
            # 如果该任务类型的组合数不足，添加一些基础测试
            if len(task_combinations) < task_max_tests:
                # 添加基础测试组合
                base_combination = {
                    "context_length": 16384,
                    "depth_percent": 50,
                    "num_needles": num_needles_options[0],
                    "language": "English",
                    "task_type": task_type,
                    "model_name": "deepseek-chat"
                }
                if base_combination not in task_combinations:
                    task_combinations.append(base_combination)
            
            combinations.extend(task_combinations)
        
        # 如果总数超过限制，截断
        return combinations[:max_combinations]
    
    async def run_single_test(self, params: Dict, use_local: bool = True):
        """运行单个测试"""
        try:
            if use_local:
                if self.local_model is None or self.local_tokenizer is None:
                    return None
                
                result = await self.benchmarker.evaluate_local_llm(
                    self.local_model, self.local_tokenizer, params
                )
            else:
                if self.api_client is None:
                    return None
                
                result = await self.benchmarker.evaluate_api_llm(
                    self.api_client, params
                )
            
            # 添加测试参数信息
            result.update({
                "test_timestamp": datetime.now().isoformat(),
                "test_params": params
            })
            
            return result
            
        except Exception as e:
            print(f"测试执行失败: {e}")
            return {
                "test_timestamp": datetime.now().isoformat(),
                "test_params": params,
                "error": str(e),
                "success": False,
                "score": 0.0
            }
    
    async def run_comprehensive_test(self, combinations: List[Dict], use_local: bool = True):
        """运行全面测试"""
        print(f"\n开始运行参数化测试 ({len(combinations)} 个组合)")
        print("="*80)
        
        total_tests = len(combinations)
        completed_tests = 0
        
        for i, params in enumerate(combinations, 1):
            print(f"\n测试 {i}/{total_tests}: {params['task_type']}")
            print(f"  上下文长度: {params['context_length']}")
            print(f"  埋针深度: {params['depth_percent']}%")
            print(f"  针数: {params['num_needles']}")
            print(f"  语言: {params['language']}")
            
            result = await self.run_single_test(params, use_local)
            
            if result:
                self.results.append(result)
                score = result.get('score', 0.0)
                print(f"  得分: {score:.3f}")
            else:
                print(f"  测试失败")
            
            completed_tests += 1
            progress = (completed_tests / total_tests) * 100
            print(f"  进度: {progress:.1f}%")
    
    def analyze_results(self):
        """分析测试结果"""
        if not self.results:
            print("没有测试结果可分析")
            return
        
        print("\n" + "="*80)
        print("测试结果分析")
        print("="*80)
        
        # 按任务类型分组
        task_results = {}
        for task_type in self.parameter_space["task_types"]:
            task_results[task_type] = [r for r in self.results if r.get('test_params', {}).get('task_type') == task_type]
        
        # 总体统计
        total_tests = len(self.results)
        successful_tests = len([r for r in self.results if r.get('success', False)])
        avg_score = sum(r.get('score', 0.0) for r in self.results) / total_tests
        
        print(f"总体统计:")
        print(f"  总测试数: {total_tests}")
        print(f"  成功测试数: {successful_tests}")
        print(f"  成功率: {successful_tests/total_tests*100:.1f}%")
        print(f"  平均得分: {avg_score:.3f}")
        
        # 按任务类型统计
        print(f"\n按任务类型统计:")
        for task_type, results in task_results.items():
            if results:
                task_scores = [r.get('score', 0.0) for r in results]
                avg_task_score = sum(task_scores) / len(task_scores)
                print(f"  {task_type}: {len(results)} 个测试, 平均得分: {avg_task_score:.3f}")
        
        # 按上下文长度统计
        print(f"\n按上下文长度统计:")
        for context_length in self.parameter_space["context_lengths"]:
            context_results = [r for r in self.results if r.get('test_params', {}).get('context_length') == context_length]
            if context_results:
                context_scores = [r.get('score', 0.0) for r in context_results]
                avg_context_score = sum(context_scores) / len(context_scores)
                print(f"  {context_length} tokens: {len(context_results)} 个测试, 平均得分: {avg_context_score:.3f}")
        
        # 按埋针深度统计
        print(f"\n按埋针深度统计:")
        for depth_percent in self.parameter_space["depth_percents"]:
            depth_results = [r for r in self.results if r.get('test_params', {}).get('depth_percent') == depth_percent]
            if depth_results:
                depth_scores = [r.get('score', 0.0) for r in depth_results]
                avg_depth_score = sum(depth_scores) / len(depth_scores)
                print(f"  {depth_percent}% 深度: {len(depth_results)} 个测试, 平均得分: {avg_depth_score:.3f}")
    
    def export_results(self, filename: str = None):
        """导出测试结果"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}"
        
        # JSON格式导出
        json_filename = f"{filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"结果已导出到: {json_filename}")
        
        # CSV格式导出（简化版）
        csv_filename = f"{filename}.csv"
        if self.results:
            fieldnames = ['timestamp', 'task_type', 'context_length', 'depth_percent', 
                         'num_needles', 'language', 'score', 'success', 'model_response']
            
            with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = {
                        'timestamp': result.get('test_timestamp', ''),
                        'task_type': result.get('task_type', ''),
                        'context_length': result.get('test_params', {}).get('context_length', ''),
                        'depth_percent': result.get('test_params', {}).get('depth_percent', ''),
                        'num_needles': result.get('test_params', {}).get('num_needles', ''),
                        'language': result.get('test_params', {}).get('language', ''),
                        'score': result.get('score', 0.0),
                        'success': result.get('success', False),
                        'model_response': result.get('response', '')[:200] if result.get('response') else ''
                    }
                    writer.writerow(row)
            
            print(f"CSV结果已导出到: {csv_filename}")

async def main():
    """主测试函数"""
    tester = ParameterizedBenchmarkTester()
    
    # 配置选项
    LOCAL_MODEL_PATH = "/path/to/your/local/model"  # 修改为实际路径
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY") or "sk-40d956c427e54efabdf1190630c6840e"
    
    print("参数化基准测试框架")
    print("="*80)
    print("支持多维度参数组合测试:")
    print("- 上下文长度: 4096, 8192, 16384, 32768, 65536 tokens")
    print("- 埋针深度: 10%, 25%, 50%, 75%, 90%")
    print("- 针数配置: 根据任务类型动态调整")
    print("- 语言: 中文/英文")
    print("- 任务类型: 单针检索/多针检索/多针推理/祖源追溯")
    print("")
    
    # 生成测试组合
    combinations = tester.generate_test_combinations(max_combinations=30)
    print(f"生成了 {len(combinations)} 个测试组合")
    
    # 选择测试模式
    test_mode = input("选择测试模式 (1=本地模型, 2=DeepSeek API): ")
    
    if test_mode == "1":
        # 本地模型测试
        if LOCAL_MODEL_PATH == "/path/to/your/local/model":
            print("请先修改LOCAL_MODEL_PATH变量为您的本地模型路径")
            return
        
        if await tester.setup_local_model(LOCAL_MODEL_PATH):
            await tester.run_comprehensive_test(combinations, use_local=True)
    elif test_mode == "2":
        # API测试
        if DEEPSEEK_API_KEY == "your_api_key_here":
            print("请先设置DEEPSEEK_API_KEY环境变量或修改代码中的API密钥")
            return
        
        if tester.setup_api_client(DEEPSEEK_API_KEY):
            await tester.run_comprehensive_test(combinations, use_local=False)
    else:
        print("无效的选择")
        return
    
    # 分析结果
    tester.analyze_results()
    
    # 导出结果
    export_choice = input("是否导出测试结果? (y/n): ")
    if export_choice.lower() == 'y':
        tester.export_results()

if __name__ == "__main__":
    import torch  # 确保torch在本地模型测试时可用
    asyncio.run(main())