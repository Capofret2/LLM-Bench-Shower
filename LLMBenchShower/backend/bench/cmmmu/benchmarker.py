import os
import re
import json
import string
import torch
import ast
from typing import Dict, List, Any, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import Client
from ..benchbase import BaseBench
from ..utils import get_dataset_path
from PIL import Image

# Try to import AutoProcessor for multimodal models
try:
    from transformers import AutoProcessor
    HAS_PROCESSOR = True
except ImportError:
    HAS_PROCESSOR = False
    AutoProcessor = None

class CMMMUBenchmarker(BaseBench):
    """
    Benchmarker for CMMMU dataset (Text-Only Version).
    Integrated with robust evaluation logic similar to official CMMMU scripts.
    """
    def __init__(self):
        self.dataset_path = get_dataset_path("CMMMU")

    def _load_dataset(self, subdataset_name: str) -> List[Dict]:
        """Load dataset from local JSONL files, with fallback options."""
        # 优先尝试从测试数据路径加载: /root/LLM-Bench-Shower/tests/test_data/CMMMU/cmmmu-data-dev/{subject}/{subject}.jsonl
        # 从 benchmarker.py 回到项目根目录: backend/bench/cmmmu -> backend/bench -> backend -> LLMBenchShower -> 项目根
        test_data_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))), "tests", "test_data")
        
        local_paths = [
            # 测试数据路径：tests/test_data/CMMMU/cmmmu-data-dev/{subject}/{subject}.jsonl
            os.path.join(test_data_base, "CMMMU", "cmmmu-data-dev", subdataset_name, f"{subdataset_name}.jsonl"),
            os.path.join(test_data_base, "CMMMU", "cmmmu-data-test", subdataset_name, f"{subdataset_name}.jsonl"),
            os.path.join(test_data_base, "CMMMU", "cmmmu-data-val", subdataset_name, f"{subdataset_name}.jsonl"),
            # 生产数据路径：/root/share/datasets/CMMMU/CMMMU/cmmmu-data-dev/{subject}/{subject}.jsonl
            os.path.join(self.dataset_path, "CMMMU", "cmmmu-data-dev", subdataset_name, f"{subdataset_name}.jsonl"),
            os.path.join(self.dataset_path, "CMMMU", "cmmmu-data-test", subdataset_name, f"{subdataset_name}.jsonl"),
            os.path.join(self.dataset_path, "CMMMU", "cmmmu-data-val", subdataset_name, f"{subdataset_name}.jsonl"),
            # 旧路径：dataset_path/subdataset_name/subdataset_name.jsonl
            os.path.join(self.dataset_path, subdataset_name, f"{subdataset_name}.jsonl"),
        ]
        
        file_path = None
        for path in local_paths:
            if os.path.exists(path):
                file_path = path
                print(f"[CMMMU] ✅ Found dataset at: {file_path}")
                break
        
        if file_path is None:
            error_msg = (
                f"Failed to load dataset '{subdataset_name}' for CMMMU.\n"
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
        
        # 设置图片路径（使用找到的文件所在目录）
        self.image_path = os.path.dirname(file_path)
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        item['subject_name'] = subdataset_name
                        data.append(item)
            print(f"[CMMMU] ✅ Loaded {len(data)} items from {file_path}")
            return data

        except Exception as e:
            raise RuntimeError(f"Error loading dataset {subdataset_name} from {file_path}: {e}")

    def _clean_text(self, text: str) -> str:
        """Remove <img...> tags and extra whitespace."""
        if not text:
            return ""
        text = re.sub(r'<img="[^"]+">', '', str(text))
        return text.strip()
    
    def _process_question_with_image(self, question: str) -> Dict:
        """
        Args:
            question (str):original text contains <img="([^"]+)"。
        Returns:
            Dict: {"text": text,"images": image}
        """
        # image
        match = re.search(r'<img="([^"]+)"', question)
        image_paths = self.image_path + "/"+match.group(1)
        image = Image.open(image_paths).convert("RGB")
        # text
        text = re.sub(r'<img[^>]+>', '', question).strip()

        return {
            "text": text,
            "images": image
        }
    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for fuzzy matching (used in Fill-in-the-blank).
        Removes punctuation, lowers case, strips whitespace.
        """
        if not text:
            return ""
        text = str(text).lower()
        # remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        # remove extra spaces
        text = " ".join(text.split())
        return text

    def _prepare_prompt(self, item: Dict) -> Union[str, Dict]:
        """Prepare the prompt from a dataset item (MCQ format).

        Args:
            item (Dict): A data item containing question and options.

        Returns:
            Union[str, Dict]: The formatted prompt. If contains image, returns dict with 'text' and 'images' keys.
        """
        q_type = item.get("type")
        has_image = False
        image_data = None
        question_text = ""
        
        # Check if question contains image
        question = item.get("question", "")
        if "<img" in question:
            has_image = True
            image_data = self._process_question_with_image(question)
            question_text = image_data["text"]
            image_obj = image_data["images"]
        else:
            question_text = question
        
        if q_type == "选择":
            context = "[选择题]:"
            opt_a = item.get("option1", "")
            opt_b = item.get("option2", "")
            opt_c = item.get("option3", "")
            opt_d = item.get("option4", "")
            # 选择题prompt模板
            prompt_text = (
                f"\n\nQuestion:{context}{question_text}\nA. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\n\nAnswer:"
            )
        elif q_type == "判断":
            context = "[判断题]:"
            # 判断题prompt模板
            prompt_text = (
                f"\n\nQuestion:{context}{question_text}\n\nAnswer:"
            )
        elif q_type == "填空":
            context = "[填空题]:"
            # 填空题prompt模板
            prompt_text = (
                f"\n\nQuestion:{context}{question_text}\n\nAnswer:"
            )
        else:
            prompt_text = (
                f"\n\nQuestion:{question_text}\n\nAnswer:"
            )
        
        # Return dict if has image, otherwise return string
        if has_image:
            result = {
                "text": prompt_text,
                "images": image_obj
            }
            print(f"[CMMMU] Prompt prepared with image: {prompt_text[:100]}...")
            return result
        else:
            print(f"[CMMMU] Prompt prepared (text only): {prompt_text[:100]}...")
            return prompt_text

    def _extract_prediction_label(self, prediction: str, item: Dict) -> str:
        """
        Robust answer extraction based on question type.
        1. For MCQ (选择题), look for explicit letter (A/B/C/D) or match option content.
        2. For 判断题, look for "正确/错误" or "True/False".
        3. For 填空题, return the normalized prediction directly.
        """
        pred = prediction.strip()
        if not pred:
            return ""

        q_type = item.get("type", "")

        # 1. 选择题 (MCQ)
        if q_type == "选择":
            # 优先提取明确的选项字母
            match = re.search(r"(?:答案|选|Answer|Choice|^)[\s:：]*([ABCD])(?![a-zA-Z])", pred, re.IGNORECASE)
            if match:
                return match.group(1).upper()
            
            # 尝试匹配选项内容
            pred_normalized = self._normalize_text(pred)
            chars = "ABCD"
            for i, char in enumerate(chars):
                opt_text = ""
                if f"option{i+1}" in item:
                    opt_text = str(item[f"option{i+1}"])
                elif "options" in item:
                    opts = item["options"]
                    if isinstance(opts, str):
                        try: opts = ast.literal_eval(opts)
                        except: opts = []
                    if isinstance(opts, list) and i < len(opts):
                        opt_text = str(opts[i])
                if opt_text:
                    opt_clean = self._normalize_text(self._clean_text(opt_text))
                    if opt_clean and opt_clean == pred_normalized:
                        return char  # 找到了对应的选项字母

            # 兜底：寻找最后一个出现的字母
            matches = re.findall(r"([ABCD])", pred.upper())
            if matches:
                return matches[-1]

        # 2. 判断题 (True/False)
        elif q_type == "判断":
            # 尝试匹配 "正确/错误" 或 "True/False"
            if re.search(r"(正确|True)", pred, re.IGNORECASE):
                return "正确"
            elif re.search(r"(错误|False)", pred, re.IGNORECASE):
                return "错误"

        # 3. 填空题 (Fill-in-the-blank)
        elif q_type == "填空":
            # 返回标准化后的预测内容
            return self._normalize_text(pred)

        # 4. 其他类型
        return ""

    def _calculate_score(self, prediction: str, item: Dict, ground_truth: str) -> float:
        """Calculate score with type-specific logic."""
        q_type = item.get("type", "选择")

        if q_type == "选择":
            extracted_pred = self._extract_prediction_label(prediction, item)
            if extracted_pred == ground_truth.upper():
                return 1.0
        else:
            clean_pred = self._normalize_text(prediction)
            clean_gt = self._normalize_text(ground_truth)
            # 1. 完全匹配
            if clean_pred == clean_gt:
                return 1.0
            # 2. 包含匹配
            if clean_gt in clean_pred and len(clean_gt) > 0:
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
        """Evaluate a local LLM on CMMMU dataset.
        
        Supports both text-only and multimodal models.
        If the model supports images (has processor), uses multimodal processing.
        Otherwise, falls back to text-only processing (images are ignored).
        """
        dataset = self._load_dataset(subdataset_name)
        
        # Try to detect if model supports multimodal processing
        # Check if model has a processor attribute or if we can use AutoProcessor
        processor = None
        is_multimodal = False
        
        # Try to get processor from model or tokenizer
        # First, try to get model name for AutoProcessor
        model_name = None
        if hasattr(model, 'config') and hasattr(model.config, 'name_or_path'):
            model_name = model.config.name_or_path
        
        # Try different methods to get processor
        if HAS_PROCESSOR and model_name:
            # Method 1: Try AutoProcessor from model name (most reliable)
            try:
                processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                # Verify it's actually a processor, not just a tokenizer
                if hasattr(processor, 'image_processor') or hasattr(processor, 'tokenizer'):
                    is_multimodal = True
                    print(f"[CMMMU] Created processor from model name: {model_name}")
                else:
                    processor = None
                    print(f"[CMMMU] AutoProcessor created but doesn't appear to support images")
            except Exception as e:
                print(f"[CMMMU] Could not create processor from model name: {e}")
                processor = None
        
        # Method 2: Try to get processor from model or tokenizer attributes
        if not is_multimodal:
            if hasattr(model, 'processor'):
                proc = model.processor
                # Check if it's actually a processor (has image_processor attribute)
                if hasattr(proc, 'image_processor'):
                    processor = proc
                    is_multimodal = True
                    print("[CMMMU] Using model.processor for multimodal processing")
            elif hasattr(tokenizer, 'processor'):
                proc = tokenizer.processor
                if hasattr(proc, 'image_processor'):
                    processor = proc
                    is_multimodal = True
                    print("[CMMMU] Using tokenizer.processor for multimodal processing")
        
        if not is_multimodal:
            print("[CMMMU] Model does not support multimodal processing, using text-only mode (images will be ignored)")
        
        results = {
            "dataset": subdataset_name,
            "model_type": "local",
            "multimodal": is_multimodal,
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
                prompt_data = self._prepare_prompt(item)
                
                # Check if prompt contains image
                if isinstance(prompt_data, dict) and "images" in prompt_data:
                    # Multimodal processing
                    if is_multimodal and processor is not None:
                        try:
                            # Check if processor supports images parameter
                            # Some processors use 'image' (singular) instead of 'images' (plural)
                            image_obj = prompt_data["images"]
                            
                            # Try with 'images' parameter first
                            try:
                                inputs = processor(
                                    text=prompt_data["text"],
                                    images=image_obj,
                                    return_tensors="pt",
                                    padding=True
                                )
                            except TypeError:
                                # Try with 'image' parameter (singular)
                                try:
                                    inputs = processor(
                                        text=prompt_data["text"],
                                        image=image_obj,
                                        return_tensors="pt",
                                        padding=True
                                    )
                                except TypeError:
                                    # If processor doesn't support images at all, fall back to text-only
                                    print(f"[CMMMU] Warning: Processor doesn't support image parameter, using text only")
                                    inputs = tokenizer(prompt_data["text"], return_tensors="pt", truncation=True, max_length=4096)
                                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                            
                            # Move tensors to model device
                            if isinstance(inputs, dict):
                                inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                                         for k, v in inputs.items()}
                        except Exception as e:
                            print(f"[CMMMU] Error processing multimodal input: {e}, falling back to text-only")
                            inputs = tokenizer(prompt_data["text"], return_tensors="pt", truncation=True, max_length=4096)
                            inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    else:
                        # Fallback: use text only (ignore image)
                        print(f"[CMMMU] Warning: Image detected but model doesn't support multimodal, using text only")
                        inputs = tokenizer(prompt_data["text"], return_tensors="pt", truncation=True, max_length=4096)
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                else:
                    # Text-only processing
                    prompt_text = prompt_data if isinstance(prompt_data, str) else prompt_data.get("text", "")
                    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                # Extract only the generated text (excluding the prompt)
                try:
                    if "input_ids" in inputs and isinstance(inputs["input_ids"], torch.Tensor):
                        input_length = inputs["input_ids"].shape[1]
                        # Decode only the newly generated tokens
                        # For multimodal models, use tokenizer from processor if available
                        if processor is not None and is_multimodal:
                            # Processor usually has a tokenizer attribute
                            if hasattr(processor, 'tokenizer'):
                                decode_tokenizer = processor.tokenizer
                            else:
                                # If processor itself can decode, use it
                                decode_tokenizer = processor if hasattr(processor, 'decode') else tokenizer
                            response = decode_tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                        else:
                            response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                    else:
                        # Fallback: decode full output and remove prompt
                        prompt_text = prompt_data["text"] if isinstance(prompt_data, dict) else prompt_data
                        if processor is not None and is_multimodal:
                            if hasattr(processor, 'tokenizer'):
                                decode_tokenizer = processor.tokenizer
                            else:
                                decode_tokenizer = processor if hasattr(processor, 'decode') else tokenizer
                        else:
                            decode_tokenizer = tokenizer
                        full_response = decode_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        response = full_response[len(prompt_text):] if full_response.startswith(prompt_text) else full_response
                except Exception as e:
                    # Fallback: decode full output and try to remove prompt
                    print(f"[CMMMU] Warning: Error extracting new tokens: {e}, using fallback method")
                    prompt_text = prompt_data["text"] if isinstance(prompt_data, dict) else prompt_data
                    if processor is not None and is_multimodal:
                        if hasattr(processor, 'tokenizer'):
                            decode_tokenizer = processor.tokenizer
                        else:
                            decode_tokenizer = processor if hasattr(processor, 'decode') else tokenizer
                    else:
                        decode_tokenizer = tokenizer
                    full_response = decode_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    response = full_response[len(prompt_text):] if full_response.startswith(prompt_text) else full_response
                
                response = response.strip()
                print(f"[CMMMU] Generated response: {response[:100]}..." if len(response) > 100 else f"[CMMMU] Generated response: {response}")
                
                ground_truth = item.get("answer", "")
                
                # Calculate score
                score = self._calculate_score(response, item, ground_truth)
                all_scores.append(score)
                
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": response, 
                    "extracted_answer": self._extract_prediction_label(response, item),
                    "ground_truth": ground_truth,
                    "score": score,
                })
                results["metrics"]["processed"] += 1
                
            except Exception as e:
                print(f"[CMMMU] Error processing item: {e}")
                import traceback
                traceback.print_exc()
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
        """Evaluate an API LLM on CMMMU dataset."""
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
                
                score = self._calculate_score(prediction, item, ground_truth)
                all_scores.append(score)
                
                results["predictions"].append({
                    "question": item.get("question", ""),
                    "prediction": prediction,
                    "extracted_answer": self._extract_prediction_label(prediction, item),
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