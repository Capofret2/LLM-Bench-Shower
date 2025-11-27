import json
import os
import torch
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import Client
from ..benchbase import BaseBench
from ..utils import get_dataset_path, get_sub_datasets


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
        """Load a subdataset from disk.

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
        
        # Load specific domain dataset
        # The structure is /domains/{domain_name}/data.jsonl
        domain_path = os.path.join(self.dataset_path, subdataset_name)
        file_path = os.path.join(domain_path, "data.jsonl")
        
        if not os.path.exists(file_path):
            # Try alternative structure
            file_path = os.path.join(self.dataset_path, f"{subdataset_name}.jsonl")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found for {subdataset_name}")
        
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
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

    def evaluate_local_llm(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        subdataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        """Evaluate a local LLM on LongBenchV2 dataset.

        Args:
            model (AutoModelForCausalLM): The local LLM model to evaluate.
            tokenizer (AutoTokenizer): The tokenizer for the LLM model.
            subdataset_name (str): The name of the domain/subdataset to evaluate on.
            **kwargs: Additional keyword arguments.

        Returns:
            Dict: Evaluation results containing metrics and predictions.
        """
        dataset = self._load_dataset(subdataset_name)
        
        results = {
            "dataset": subdataset_name,
            "model_type": "local",
            "predictions": [],
            "metrics": {
                "total": len(dataset),
                "processed": 0,
            }
        }
        
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
                
                ground_truth = self._extract_answer(item)
                
                results["predictions"].append({
                    "question": item.get("question", item.get("instruction", "")),
                    "prediction": response,
                    "ground_truth": ground_truth,
                })
                results["metrics"]["processed"] += 1
            except Exception as e:
                results["predictions"].append({
                    "question": item.get("question", item.get("instruction", "")),
                    "prediction": f"Error: {str(e)}",
                    "ground_truth": self._extract_answer(item),
                })
        
        return results

    def evaluate_api_llm(
        self,
        client: Client,
        model: str,
        subdataset_name: str,
        *args,
        **kwargs,
    ) -> Dict:
        """Evaluate an API LLM on LongBenchV2 dataset.

        Args:
            client (Client): The OpenAI client for API calls.
            model (str): The model name to evaluate (e.g., "gpt-4").
            subdataset_name (str): The name of the domain/subdataset to evaluate on.
            **kwargs: Additional keyword arguments for API calls.

        Returns:
            Dict: Evaluation results containing metrics and predictions.
        """
        dataset = self._load_dataset(subdataset_name)
        
        results = {
            "dataset": subdataset_name,
            "model_type": "api",
            "model": model,
            "predictions": [],
            "metrics": {
                "total": len(dataset),
                "processed": 0,
            }
        }
        
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
                ground_truth = self._extract_answer(item)
                
                results["predictions"].append({
                    "question": item.get("question", item.get("instruction", "")),
                    "prediction": prediction,
                    "ground_truth": ground_truth,
                })
                results["metrics"]["processed"] += 1
            except Exception as e:
                results["predictions"].append({
                    "question": item.get("question", item.get("instruction", "")),
                    "prediction": f"Error: {str(e)}",
                    "ground_truth": self._extract_answer(item),
                })
        
        return results
