import json
import os
import torch
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import Client
from ..benchbase import BaseBench
from ..utils import get_dataset_path, get_sub_datasets
from .metrics import get_metric_function


class LongBenchV2Benchmarker(BaseBench):
    """Benchmarker for LongBenchV2 dataset.
    
    LongBenchV2 is an improved benchmark for evaluating long-context understanding
    abilities with more diverse tasks across different domains. It contains tasks
    like code repository understanding, long dialogue history understanding, etc.
    Reference: https://github.com/THUDM/LongBench
    """

    def __init__(self):
        """Initialize the LongBenchV2 benchmarker."""
        self.dataset_path = get_dataset_path("LongBenchV2")
        self.sub_datasets = get_sub_datasets("LongBenchV2")

    def _load_dataset(self, subdataset_name: str) -> List[Dict]:
        """Load a subdataset from disk, with fallback options.

        Args:
            subdataset_name (str): The name of the subdataset to load.
                Can be domain name like 'Code_Repository_Understanding' or the
                main 'LongBenchV2' to load all domains.

        Returns:
            List[Dict]: The loaded dataset as a list of dictionaries.
        """
        if subdataset_name == "LongBenchV2":
            # Return all domain datasets combined
            all_data = []
            for domain_name in self.sub_datasets[1:]:  # Skip "LongBenchV2" itself
                try:
                    data = self._load_dataset(domain_name)
                    all_data.extend(data)
                except:
                    continue
            return all_data
        
        # 只从测试数据路径加载
        # 从 benchmarker.py 回到项目根目录: backend/bench/longbench_v2 -> backend/bench -> backend -> LLMBenchShower -> 项目根
        test_data_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "tests", "test_data")
        
        local_paths = [
            # 测试数据路径：tests/test_data/LongBenchV2/{domain}/data.jsonl
            os.path.join(test_data_base, "LongBenchV2", subdataset_name, "data.jsonl"),
        ]
        
        file_path = None
        for path in local_paths:
            if os.path.exists(path):
                file_path = path
                print(f"[LongBenchV2] ✅ Found dataset at: {file_path}")
                break
        
        if file_path is None:
            error_msg = (
                f"Failed to load dataset '{subdataset_name}' for LongBenchV2.\n"
                f"Tried paths:\n"
            )
            for path in local_paths:
                error_msg += f"  - {path}\n"
            error_msg += (
                f"\nSolutions:\n"
                f"  - Ensure the dataset file exists in one of the above paths\n"
                f"  - Check that the dataset name '{subdataset_name}' is correct"
            )
            raise FileNotFoundError(error_msg)
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        print(f"[LongBenchV2] ✅ Loaded {len(data)} items from {file_path}")
        return data

    def _prepare_prompt(self, item: Dict) -> str:
        """Prepare the prompt from a dataset item.

        Args:
            item (Dict): A data item. Different domains may have different structures,
                but generally contain context and question fields.

        Returns:
            str: The formatted prompt.
        """
        context = item.get("context", "")
        question = item.get("question", "")
        
        # Handle different possible field names
        if not context:
            context = item.get("input", "")
        if not question:
            question = item.get("instruction", "")
        
        if context and question:
            return f"{context}\n\nQuestion: {question}\n\nAnswer:"
        elif context:
            return f"{context}\n\nAnswer:"
        elif question:
            return f"{question}\n\nAnswer:"
        else:
            return "Please provide a response:\n\nAnswer:"

    def _extract_answer(self, item: Dict) -> str:
        """Extract the ground truth answer from a dataset item.

        Args:
            item (Dict): A data item containing the answer.

        Returns:
            str: The ground truth answer.
        """
        # Try different possible answer field names
        answers = item.get("answers", item.get("answer", item.get("output", "")))
        
        if isinstance(answers, list):
            return answers[0] if answers else ""
        return str(answers)

    def _calculate_score(self, prediction: str, ground_truths: List[str], subdataset_name: str, all_classes: List = None) -> float:
        """Calculate the evaluation score for a prediction.

        Args:
            prediction (str): The model's prediction.
            ground_truths (List[str]): List of ground truth answers.
            subdataset_name (str): The name of the domain (used to select metric).
            all_classes (List): List of all classes (for classification tasks).

        Returns:
            float: The evaluation score.
        """
        # For LongBenchV2, use QA F1 score as default (can be extended with domain-specific metrics)
        metric_fn = get_metric_function(subdataset_name.lower())
        
        # Calculate max score among all ground truths
        max_score = 0.0
        kwargs = {}
        if all_classes:
            kwargs["all_classes"] = all_classes
        
        for ground_truth in ground_truths:
            try:
                score = metric_fn(prediction, ground_truth, **kwargs)
                max_score = max(max_score, score)
            except Exception as e:
                # If metric calculation fails, return 0
                continue
        
        return max_score

    def evaluate_local_llm(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        subdataset_name: str,
        *args,
        max_samples: int = None,
        **kwargs,
    ) -> Dict:
        """Evaluate a local LLM on LongBenchV2 dataset.

        Args:
            model (AutoModelForCausalLM): The local LLM model to evaluate.
            tokenizer (AutoTokenizer): The tokenizer for the LLM model.
            subdataset_name (str): The name of the domain/subdataset to evaluate on.
            max_samples (int, optional): Maximum number of samples to evaluate. If None, evaluates all.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict: Evaluation results containing metrics and predictions.
        """
        dataset = self._load_dataset(subdataset_name)
        
        # Limit samples if specified (useful for testing)
        if max_samples is not None and max_samples > 0:
            dataset = dataset[:max_samples]
            print(f"[LongBenchV2] Limiting evaluation to {max_samples} samples (out of {len(self._load_dataset(subdataset_name))} total)")
        
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
                        top_p=0.9,
                    )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove the prompt from the response
                response = response[len(prompt):].strip()
                
                ground_truths = item.get("answers", item.get("answer", item.get("output", "")))
                if isinstance(ground_truths, str):
                    ground_truths = [ground_truths]
                
                # Calculate score
                all_classes = item.get("all_classes", None)
                score = self._calculate_score(response, ground_truths, subdataset_name, all_classes)
                all_scores.append(score)
                
                results["predictions"].append({
                    "question": item.get("question", item.get("instruction", "")),
                    "prediction": response,
                    "ground_truths": ground_truths,
                    "score": score,
                })
                results["metrics"]["processed"] += 1
            except Exception as e:
                ground_truths = item.get("answers", item.get("answer", item.get("output", "")))
                if isinstance(ground_truths, str):
                    ground_truths = [ground_truths]
                
                results["predictions"].append({
                    "question": item.get("question", item.get("instruction", "")),
                    "prediction": f"Error: {str(e)}",
                    "ground_truths": ground_truths,
                    "score": 0.0,
                })
        
        # Calculate average score
        if all_scores:
            results["metrics"]["score"] = sum(all_scores) / len(all_scores)
        
        return results

    def evaluate_api_llm(
        self,
        client: Client,
        model: str,
        subdataset_name: str,
        *args,
        max_samples: int = None,
        **kwargs,
    ) -> Dict:
        """Evaluate an API LLM on LongBenchV2 dataset.

        Args:
            client (Client): The OpenAI client for API calls.
            model (str): The model name to evaluate (e.g., "gpt-4").
            subdataset_name (str): The name of the domain/subdataset to evaluate on.
            max_samples (int, optional): Maximum number of samples to evaluate. If None, evaluates all.
            **kwargs: Additional keyword arguments for API calls.

        Returns:
            Dict: Evaluation results containing metrics and predictions.
        """
        dataset = self._load_dataset(subdataset_name)
        
        # Limit samples if specified (useful for testing)
        if max_samples is not None and max_samples > 0:
            dataset = dataset[:max_samples]
            print(f"[LongBenchV2] Limiting evaluation to {max_samples} samples (out of {len(self._load_dataset(subdataset_name))} total)")
        
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
                
                prediction = response.choices[0].message.content.strip()
                ground_truths = item.get("answers", item.get("answer", item.get("output", "")))
                if isinstance(ground_truths, str):
                    ground_truths = [ground_truths]
                
                # Calculate score
                all_classes = item.get("all_classes", None)
                score = self._calculate_score(prediction, ground_truths, subdataset_name, all_classes)
                all_scores.append(score)
                
                results["predictions"].append({
                    "question": item.get("question", item.get("instruction", "")),
                    "prediction": prediction,
                    "ground_truths": ground_truths,
                    "score": score,
                })
                results["metrics"]["processed"] += 1
            except Exception as e:
                ground_truths = item.get("answers", item.get("answer", item.get("output", "")))
                if isinstance(ground_truths, str):
                    ground_truths = [ground_truths]
                
                results["predictions"].append({
                    "question": item.get("question", item.get("instruction", "")),
                    "prediction": f"Error: {str(e)}",
                    "ground_truths": ground_truths,
                    "score": 0.0,
                })
        
        # Calculate average score
        if all_scores:
            results["metrics"]["score"] = sum(all_scores) / len(all_scores)
        
        return results
