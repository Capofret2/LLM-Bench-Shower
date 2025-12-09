import os
import json
import torch
import re
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from openai import Client
from ..benchbase import BaseBench
from ..utils import get_dataset_path, get_sub_datasets

class C_EvalBenchmarker(BaseBench):
    """Benchmarker for CEval dataset.
    
    C-Eval is a comprehensive Chinese evaluation suite for foundation models.
    It consists of multiple-choice questions across various disciplines.
    Reference: https://huggingface.co/datasets/ceval/ceval-exam
    """
    def __init__(self):
        """Initialize the CEval benchmarker."""
        self.dataset_path = get_dataset_path("C-Eval")
        self.sub_datasets = get_sub_datasets("C-Eval")

    def _load_dataset(self, subdataset_name: str) -> List[Dict]:
        """Load a subdataset from local files or Hugging Face.

        Args:
            subdataset_name (str): The name of the subdataset (subject) to load.

        Returns:
            List[Dict]: The loaded dataset as a list of dictionaries.
        """
        # 如果 subdataset_name 是 "C-Eval"，提示用户选择具体的科目
        if subdataset_name == "C-Eval":
            available_subjects = [f for f in os.listdir(self.dataset_path) 
                                 if f.endswith('.jsonl') and os.path.isfile(os.path.join(self.dataset_path, f))]
            available_subjects = [f.replace('.jsonl', '') for f in available_subjects]
            error_msg = (
                f"C-Eval requires a specific subject name, not 'C-Eval'.\n"
                f"Available subjects in {self.dataset_path}:\n"
                f"  {', '.join(sorted(available_subjects)[:10])}{'...' if len(available_subjects) > 10 else ''}\n"
                f"\n"
                f"Please use format: C-Eval/{{subject_name}}\n"
                f"Example: C-Eval/computer_network, C-Eval/high_school_mathematics"
            )
            raise ValueError(error_msg)
        
        # 优先尝试从测试数据路径加载
        # 从 benchmarker.py 回到项目根目录: backend/bench/c_eval -> backend/bench -> backend -> LLMBenchShower -> 项目根
        test_data_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "tests", "test_data")
        
        # 构建所有可能的路径（测试数据优先）
        local_paths = [
            # 测试数据路径：tests/test_data/C-Eval/{subject}.jsonl
            os.path.join(test_data_base, "C-Eval", f"{subdataset_name}.jsonl"),
            # 生产数据路径：/root/share/datasets/C-Eval/{subject}.jsonl
            os.path.join(self.dataset_path, f"{subdataset_name}.jsonl"),
        ]
        
        local_file = None
        for path in local_paths:
            if os.path.exists(path):
                local_file = path
                print(f"[C-Eval] ✅ Found dataset at: {local_file}")
                break
        
        if local_file:
            try:
                data_list = []
                with open(local_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            # 确保 subject_name 字段存在
                            if 'subject_name' not in item:
                                item['subject_name'] = subdataset_name
                            data_list.append(item)
                print(f"[C-Eval] ✅ Loaded {len(data_list)} items from local file: {local_file}")
                return data_list
            except Exception as e:
                print(f"[C-Eval] ⚠️  Failed to load from local file {local_file}: {e}")
                print(f"[C-Eval] Falling back to Hugging Face...")
        else:
            print(f"[C-Eval] ⚠️  Local file not found in any of the paths:")
            for path in local_paths:
                print(f"  - {path}")
            print(f"[C-Eval] Attempting to load from Hugging Face...")
        
        # 如果本地文件不存在或加载失败，尝试从 Hugging Face 加载
        # 注意：移除 trust_remote_code，因为新版本的 datasets 库不再支持
        hf_dataset_names = [
            "ceval/ceval-exam",  # Hugging Face 数据集名称
        ]
        
        for hf_path in hf_dataset_names:
            try:
                print(f"[C-Eval] 📥 Attempting to load from Hugging Face: {hf_path}")
                # path: 远程仓库地址
                # name: 具体科目 (e.g.'computer_network')
                # split: C-Eval通常验证用 'val' (test集无答案)
                # cache_dir: 指定本地缓存目录
                dataset = load_dataset(
                    path=hf_path,
                    name=subdataset_name,
                    split="val",
                    cache_dir=self.dataset_path
                )
                data_list = []
                for item in dataset:
                    item_dict = dict(item)
                    item_dict['subject_name'] = subdataset_name
                    data_list.append(item_dict)
                print(f"[C-Eval] ✅ Successfully loaded {len(data_list)} items from Hugging Face")
                return data_list
            except Exception as e:
                print(f"[C-Eval] ❌ Failed to load from {hf_path}: {e}")
                continue
        
        # 如果所有方法都失败，提供详细的错误信息
        error_msg = (
            f"Failed to load dataset '{subdataset_name}' for C-Eval.\n"
            f"Tried:\n"
            f"  1. Local file: {local_file} ({'not found' if not os.path.exists(local_file) else 'load failed'})\n"
            f"  2. Hugging Face dataset: {hf_dataset_names[0]} (failed)\n"
            f"\n"
            f"Solutions:\n"
            f"  - Ensure the local dataset file exists: {local_file}\n"
            f"  - Download the dataset using: python /root/share/datasets/C-Eval/download_ceval.py --subjects {subdataset_name}\n"
            f"  - Or ensure you have internet access to download from Hugging Face\n"
            f"  - Check that the dataset name '{subdataset_name}' is correct"
        )
        raise RuntimeError(error_msg)

    def _prepare_prompt(self, item: Dict) -> str:
        """Prepare the prompt from a dataset item (MCQ format).

        Args:
            item (Dict): A data item containing question and options.

        Returns:
            str: The formatted prompt.
        """
        question = item.get("question", "")
        opt_a = item.get("A", "")
        opt_b = item.get("B", "")
        opt_c = item.get("C", "")
        opt_d = item.get("D", "")
        
        # 简单的选择题prompt模板
        prompt = (
            f"\n\nQuestion:{question}\nA. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\n\nAnswer:"
        )
        print("[debug] prompt prepared:", prompt)
        return prompt

    def _extract_prediction_label(self, prediction: str) -> str:
        """Extract the option letter (A/B/C/D) from model output.
        
        Args:
            prediction (str): The raw model output.
            
        Returns:
            str: The extracted letter or empty string.
        """
        # 清理空白字符
        pred = prediction.strip()
        if not pred:
            return ""
            
        # 1. 如果第一个字符就是选项
        if pred[0].upper() in ["A", "B", "C", "D"]:
            return pred[0].upper()
            
        # 2. 正则
        match = re.search(r"(?:答案|选|Choice|Answer)[\s:：]*([ABCD])", pred, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
        # 3. 如果没找到明确标记，尝试找最后一个出现的选项字母
        matches = re.findall(r"([ABCD])", pred.upper())
        if matches:
            return matches[-1]
            
        return ""

    def _calculate_score(self, prediction: str, ground_truth: str) -> float:
        """Calculate score (0 or 1) for MCQ.

        Args:
            prediction (str): The model's raw prediction text.
            ground_truth (str): The correct option letter (e.g., "A").

        Returns:
            float: 1.0 if correct, 0.0 otherwise.
        """
        extracted_pred = self._extract_prediction_label(prediction)
        
        # 比较提取出的答案和标准答案
        if extracted_pred == ground_truth.strip().upper():
            return 1.0
        return 0.0

    def evaluate_local_llm(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        subdataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        """Evaluate a local LLM on CEval dataset."""
        dataset = self._load_dataset(subdataset_name)
        
        results = {
            "dataset": subdataset_name,
            "model_type": "local",
            "predictions": [],
            "metrics": {
                "total": len(dataset),
                "processed": 0,
                "score": 0.0,
            }
        }
        
        all_scores = []
        
        for item in dataset:
            try:
                prompt = self._prepare_prompt(item)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                # Extract only the generated text (excluding the prompt)
                # Method 1: Decode only new tokens (preferred)
                try:
                    if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                        input_length = inputs["input_ids"].shape[1]
                        # Decode only the newly generated tokens
                        response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                    else:
                        # Fallback: decode full output and remove prompt
                        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = full_response[len(prompt):] if full_response.startswith(prompt) else full_response
                except Exception as e:
                    # Fallback: decode full output and try to remove prompt
                    print(f"[C-Eval] Warning: Error extracting new tokens: {e}, using fallback method")
                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = full_response[len(prompt):] if full_response.startswith(prompt) else full_response
                
                response = response.strip()
                print(f"[C-Eval] Generated response: {response[:100]}..." if len(response) > 100 else f"[C-Eval] Generated response: {response}")
                
                ground_truth = item.get("answer", "")
                
                # Calculate score
                score = self._calculate_score(response, ground_truth)
                all_scores.append(score)
                
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": response, 
                    "extracted_answer": self._extract_prediction_label(response),
                    "ground_truth": ground_truth,
                    "score": score,
                })
                results["metrics"]["processed"] += 1
                
            except Exception as e:
                print(f"Error processing item: {e}")
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": f"Error: {str(e)}",
                    "ground_truth": item.get("answer", ""),
                    "score": 0.0,
                })
        
        if all_scores:
            results["metrics"]["score"] = sum(all_scores) / len(all_scores)
        
        return results

    def evaluate_api_llm(
        self,
        client: Client,
        model: str,
        subdataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        """Evaluate an API LLM on CEval dataset."""
        dataset = self._load_dataset(subdataset_name)
        
        results = {
            "dataset": subdataset_name,
            "model_type": "api",
            "model": model,
            "predictions": [],
            "metrics": {
                "total": len(dataset),
                "processed": 0,
                "score": 0.0,
            }
        }
        
        all_scores = []
        
        for item in dataset:
            try:
                prompt = self._prepare_prompt(item)
                
                response = client.messages.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    max_tokens=512,
                    temperature=0.7,
                )
                #print("[debug api] response:", response)
                prediction = response.choices[0].message.content.strip()
                ground_truth = item.get("answer", "")
                
                score = self._calculate_score(prediction, ground_truth)
                all_scores.append(score)
                
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": prediction,
                    "extracted_answer": self._extract_prediction_label(prediction),
                    "ground_truth": ground_truth,
                    "score": score,
                })
                results["metrics"]["processed"] += 1
                
            except Exception as e:
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": f"Error: {str(e)}",
                    "ground_truth": item.get("answer", ""),
                    "score": 0.0,
                })
        
        if all_scores:
            results["metrics"]["score"] = sum(all_scores) / len(all_scores)
        
        return results