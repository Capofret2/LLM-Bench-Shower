import os
import sys
import json
import random
import asyncio
import tiktoken
import re
import gc
import math
import torch
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from openai import Client
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchbase import BaseBench

# NeedleBench V2 Task Types
class TaskType:
    SINGLE_NEEDLE_RETRIEVAL = "single_needle_retrieval"
    MULTI_NEEDLE_RETRIEVAL = "multi_needle_retrieval" 
    MULTI_NEEDLE_REASONING = "multi_needle_reasoning"
    ANCESTRAL_TRACE_CHALLENGE = "ancestral_trace_challenge"

class NeedleInHaystackBenchmarker(BaseBench):
    def __init__(self):
        # Get the project root directory (go up 4 levels from backend/bench/needle_in_haystack)
        # From backend/bench/needle_in_haystack/benchmarker.py -> backend/bench/needle_in_haystack -> backend/bench -> backend -> LLMBenchShower -> project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        
        # Dataset paths using relative paths - datasets are in project_root/tests/test_data/
        self.english_dataset_dir = os.path.join(project_root, "tests", "test_data", "NeedleInHaystack", "PaulGrahamEssays")
        self.chinese_dataset_path = os.path.join(project_root, "tests", "test_data", "NeedleInHaystack", "Journey_to_the_West.txt")# ATC相关配置
        self.atc_names_file = None  # 暂时设为None，使用内置姓氏列表
        
        # Default needles for testing
        self.default_needles = {
            "English": [
                "The most important thing for a startup is to make something people want.",
                "Startups should focus on a small, specific market initially.",
                "The best way to get startup ideas is to look for problems you have yourself.",
                "Founders should be prepared for the emotional rollercoaster of building a company.",
                "The most successful startups often come from technical founders.",
                "Startups should aim for growth, not just profitability.",
                "The best time to start a startup is when you're young and have few responsibilities.",
                "Startups should avoid raising too much money too early.",
                "The most important quality in a founder is determination.",
                "Startups should focus on building a great product first, then worry about marketing."
            ],
            "Chinese": [
                "人生在世，应当有所作为。",
                "知足常乐是一种智慧。",
                "诚实守信是做人的根本。",
                "学而时习之，不亦说乎？",
                "千里之行，始于足下。",
                "书山有路勤为径，学海无涯苦作舟。",
                "天道酬勤，厚德载物。",
                "君子爱财，取之有道。",
                "三人行，必有我师焉。",
                "己所不欲，勿施于人。"
            ]
        }
        
        # Default retrieval questions
        self.default_questions = {
            "English": "Based on the provided text, what is the most important advice for startups?",
            "Chinese": "根据提供的文本，最重要的道理是什么？"
        }
        
        # ATC relationship templates and terms (from NeedleBench V2)
        self.relationship_templates_zh_CN = [
            '{A}是{B}的{relationship}。',
            '{B}的{relationship}是{A}。',
            '{A}作为{B}的{relationship}，对{B}的成长有重要影响。',
            '{A}不仅是{B}的{relationship}，还是{B}的榜样。',
            '{A}在{B}的成长过程中，不仅仅是{B}的{relationship}，还是{B}的监护人。',
            '{A}对{B}来说，不只是一个{relationship}，还是一个朋友。',
        ]
        
        self.relationship_templates_en = [
            "{A} is {B}'s {relationship}.",
            "{B}'s {relationship} is {A}.",
            ("{A}, as {B}'s {relationship}, "
             "has a significant impact on {B}'s upbringing."),
            ("{A} is not only {B}'s {relationship} "
             "but also {B}'s role model."),
            ("During {B}'s upbringing, {A} was not only {B}'s {relationship}, "
             "but also {B}'s guardian."),
            ('For {B}, {A} is not just a {relationship}, '
             'but also a friend.'),
            'For {B}, {A} is more than just a {relationship}; {A} is a lifelong mentor of {B}.',
        ]
        
        self.relationship_terms_zh_CN = [
            '父亲', '母亲', '爸爸', '妈妈', '爷爷', '奶奶', '姥姥', '姥爷', '外公', '外婆'
        ]
        
        self.relationship_terms_en = [
            'father', 'mother', 'dad', 'mom', 'grandfather', 'grandmother',
            'maternal grandmother', 'maternal grandfather', 'paternal grandfather', 'paternal grandmother'
        ]
        
        # Default test parameters
        self.default_params = {
            "context_length": 32768,
            "depth_percent": 50,
            "num_needles": 1,
            "language": "English",
            "task_type": TaskType.SINGLE_NEEDLE_RETRIEVAL
        }
        
        # Parameter space for comprehensive testing (from parameterized_benchmark_test.py)
        self.parameter_space = {
            "context_length_groups": {
                "short": [4096, 8192],
                "medium": [16384, 32768],
                "long": [65536, 128000],
                "extra_long": [256000, 512000],
                "extreme": [1000000]
            },
            "depth_percents": [0, 10, 21, 31, 42, 52, 63, 73, 84, 94, 100],
            "needle_configs": {
                TaskType.SINGLE_NEEDLE_RETRIEVAL: [1],
                TaskType.MULTI_NEEDLE_RETRIEVAL: [2, 4, 8],
                TaskType.MULTI_NEEDLE_REASONING: [4, 8, 16],
                TaskType.ANCESTRAL_TRACE_CHALLENGE: [8, 16, 32]
            },
            "language_configs": {
                "English": {"length_buffer": 3000, "guide": True},
                "Chinese": {"length_buffer": 200, "guide": True}
            },
            "task_types": [
                TaskType.SINGLE_NEEDLE_RETRIEVAL,
                TaskType.MULTI_NEEDLE_RETRIEVAL,
                TaskType.MULTI_NEEDLE_REASONING,
                TaskType.ANCESTRAL_TRACE_CHALLENGE
            ]
        }
    
    def _load_dataset(self, dataset_name: str) -> str:
        """Load the dataset content."""
        if dataset_name == "Journey_to_the_West":
            with open(self.chinese_dataset_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:  # PaulGrahamEssays
            content = ""
            essays_dir = Path(self.english_dataset_dir)
            for file_path in essays_dir.glob("*.txt"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content += f.read() + "\n\n"
            return content
    
    def _get_random_needle(self, language: str = "English") -> str:
        """Get a random needle from the default needles."""
        return random.choice(self.default_needles[language])
    
    def _get_needles_list(self, num_needles: int, language: str = "English") -> List[str]:
        """Get a list of needles for multi-needle testing."""
        needles_source = self.default_needles[language]
        if num_needles > len(needles_source):
            needles = needles_source * (num_needles // len(needles_source))
            needles.extend(needles_source[:num_needles % len(needles_source)])
            return needles
        else:
            return random.sample(needles_source, num_needles)
    
    def _insert_needle(self, context: str, needle: str, depth_percent: int) -> str:
        """Insert needle into context at specified depth percentage."""
        # Use tiktoken for token-level insertion
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Tokenize context and needle
        context_tokens = tokenizer.encode(context)
        needle_tokens = tokenizer.encode(needle)
        
        # Calculate insertion point
        insertion_point = int(len(context_tokens) * (depth_percent / 100))
        
        # Insert needle
        new_tokens = context_tokens[:insertion_point] + needle_tokens + context_tokens[insertion_point:]
        
        # Decode back to text
        return tokenizer.decode(new_tokens)
    
    def _insert_multiple_needles(self, context: str, needles: List[str], depth_percent: int) -> str:
        """Insert multiple needles into context with equal spacing."""
        tokenizer = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer.encode(context)
        
        # Calculate insertion points
        num_needles = len(needles)
        insertion_points = []
        for i in range(num_needles):
            point = int(len(context_tokens) * (depth_percent / 100) * (i + 1) / num_needles)
            insertion_points.append(point)
        
        # Insert needles in reverse order to maintain correct positions
        new_tokens = context_tokens
        for i, (needle, point) in enumerate(zip(reversed(needles), reversed(insertion_points))):
            needle_tokens = tokenizer.encode(needle)
            new_tokens = new_tokens[:point] + needle_tokens + new_tokens[point:]
        
        return tokenizer.decode(new_tokens)
    
    def _prepare_prompt(self, context: str, question: str, language: str = "English") -> str:
        """Prepare the prompt for the model."""
        if language == "Chinese":
            return f"""这是一个长文本能力测试。请仔细阅读下面的长文档，然后根据文档中的信息直接回答问题。

重要提示：
1. 请直接给出答案，不要添加额外的解释、分析或说明。
2. 只回答文档中明确提到的内容。
3. 如果文档中没有相关信息，请回答"未找到"。

长文档内容：

<文档>
{context}
</文档>

问题：{question}

请直接回答（只给出答案，不要解释）："""
        else:
            return f"""This is a long-text capability test. Please read the long document below carefully, then answer the question directly based on the information in the document.

Important instructions:
1. Provide the answer directly without additional explanations, analysis, or commentary.
2. Only answer based on information explicitly mentioned in the document.
3. If the information is not found in the document, answer "Not found".

Document content:

<Document>
{context}
</Document>

Question: {question}

Please answer directly (answer only, no explanation): """
    
    def _evaluate_response(self, response: str, needle: str, language: str = "English") -> float:
        """Evaluate the model's response against the expected needle."""
        needle_clean = needle.strip().lower()
        response_clean = response.strip().lower()
        
        # 1. Clean response: remove quotes and markdown formatting
        response_clean = re.sub(r'[\*\*\_\`\"]', '', response_clean)  # Remove markdown formatting
        response_clean = re.sub(r'\\boxed\{[^}]*\}', '', response_clean)  # Remove boxed answers
        
        # 2. Extract direct answer (first sentence or key phrase)
        # Try to extract the most relevant part of the response
        sentences = re.split(r'[.!?]', response_clean)
        if sentences:
            # Use the first sentence as the main answer
            main_answer = sentences[0].strip()
            
            # Check if main answer contains needle
            if needle_clean in main_answer:
                return 1.0
            
            # Check word overlap in main answer
            needle_words = set(needle_clean.split())
            answer_words = set(main_answer.split())
            
            if len(needle_words) > 0:
                overlap = len(needle_words.intersection(answer_words)) / len(needle_words)
                
                # Enhanced scoring based on overlap
                if overlap >= 0.8:
                    return 0.9  # High semantic similarity
                elif overlap >= 0.6:
                    return 0.7  # Good semantic similarity
                elif overlap >= 0.4:
                    return 0.5  # Moderate semantic similarity
                elif overlap >= 0.2:
                    return 0.3  # Low semantic similarity
        
        # 3. Check full response for exact match
        if needle_clean in response_clean:
            return 0.8  # Found in full response but not in main answer
        
        # 4. Check for key phrases in full response
        key_phrases = [
            "startup", "important", "focus", "market", "growth", 
            "product", "founder", "company", "advice", "best"
        ] if language == "English" else [
            "重要", "道理", "智慧", "诚实", "学习", "行动", "勤奋", "道德"
        ]
        
        key_phrase_count = sum(1 for phrase in key_phrases if phrase in response_clean)
        if key_phrase_count >= 3:
            return 0.4  # Contains relevant key phrases
        elif key_phrase_count >= 1:
            return 0.2  # Contains some relevant phrases
        
        return 0.0  # No meaningful match
    
    def _evaluate_atc_response(self, response: str, expected_answer: str) -> float:
        """Evaluate ATC response using NeedleBench V2's evaluation logic."""
        print(f"[NeedleBench] Evaluating ATC response:")
        print(f"[NeedleBench]   Expected answer: {expected_answer}")
        print(f"[NeedleBench]   Response (first 200 chars): {response[:200]}...")
        
        # Extract answer from boxed format
        # Try multiple patterns: \boxed{}, boxed{}, or just the answer
        boxed_patterns = [
            r'\\boxed\{([^}]+)\}',  # LaTeX format: \boxed{answer}
            r'boxed\{([^}]+)\}',    # Without backslash
            r'\\boxed\s*\{([^}]+)\}',  # With spaces
            r'boxed\s*\{([^}]+)\}',   # Without backslash, with spaces
        ]
        
        extracted_answer = None
        for pattern in boxed_patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).strip()
                print(f"[NeedleBench]   Extracted from boxed format: {extracted_answer}")
                break
        
        # If no boxed format found, try to extract answer from response directly
        if not extracted_answer:
            # Try to find the expected answer name in the response
            expected_answer_clean = expected_answer.strip().lower()
            response_lower = response.lower()
            
            # Check if expected answer appears in response
            if expected_answer_clean in response_lower:
                # Try to extract the sentence containing the answer
                sentences = re.split(r'[.!?。！？]', response)
                for sentence in sentences:
                    if expected_answer_clean in sentence.lower():
                        # Extract potential answer (could be the name itself or a phrase)
                        extracted_answer = sentence.strip()
                        print(f"[NeedleBench]   Extracted from sentence: {extracted_answer[:100]}...")
                        break
            
            # If still not found, try to extract last word or phrase that might be the answer
            if not extracted_answer:
                # Look for common answer patterns
                # For English: "is [name]", "the answer is [name]", "[name] is the..."
                # For Chinese: "是[name]", "答案是[name]", "[name]是..."
                answer_patterns = [
                    r'(?:is|are|was|were)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # English: "is John"
                    r'(?:the\s+)?answer\s+is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # English: "answer is John"
                    r'是\s*([\u4e00-\u9fa5]+)',  # Chinese: "是张伟"
                    r'答案是\s*([\u4e00-\u9fa5]+)',  # Chinese: "答案是张伟"
                ]
                
                for pattern in answer_patterns:
                    match = re.search(pattern, response, re.IGNORECASE)
                    if match:
                        extracted_answer = match.group(1).strip()
                        print(f"[NeedleBench]   Extracted from pattern: {extracted_answer}")
                        break
        
        if extracted_answer:
            extracted_answer_clean = extracted_answer.strip().lower()
            expected_answer_clean = expected_answer.strip().lower()
            
            print(f"[NeedleBench]   Comparing: '{extracted_answer_clean}' vs '{expected_answer_clean}'")
            
            # Exact match
            if extracted_answer_clean == expected_answer_clean:
                print(f"[NeedleBench]   Exact match! Score: 1.0")
                return 1.0
            
            # Check if expected answer is contained in extracted answer
            if expected_answer_clean in extracted_answer_clean:
                print(f"[NeedleBench]   Expected answer found in response! Score: 1.0")
                return 1.0
                
            # Check if extracted answer is contained in expected answer
            if extracted_answer_clean in expected_answer_clean:
                print(f"[NeedleBench]   Partial match! Score: 0.8")
                return 0.8
            
            # Check for word-level match (for multi-word names)
            expected_words = set(expected_answer_clean.split())
            extracted_words = set(extracted_answer_clean.split())
            if expected_words and extracted_words:
                overlap = len(expected_words.intersection(extracted_words)) / len(expected_words)
                if overlap >= 0.8:
                    print(f"[NeedleBench]   High word overlap ({overlap:.2f})! Score: 0.9")
                    return 0.9
                elif overlap >= 0.5:
                    print(f"[NeedleBench]   Moderate word overlap ({overlap:.2f})! Score: 0.6")
                    return 0.6
        
        print(f"[NeedleBench]   No match found. Score: 0.0")
        print(f"[NeedleBench]   Full response: {response[:500]}...")
        return 0.0
    
    def _generate_atc_needles(self, num_needles: int, language: str) -> Dict:
        """Generate ATC needles for multi-needle reasoning using NeedleBench V2's power-of-2 distribution."""
        # Use built-in Chinese and English names
        chinese_names = ["张伟", "王芳", "李娜", "刘洋", "陈静", "杨明", "黄强", "赵丽", "周杰", "吴婷", 
                       "孙涛", "朱敏", "徐辉", "林静", "赵强", "王芳", "刘娜", "陈涛", "杨丽", "吴杰"]
        english_names = ["John", "Mary", "David", "Sarah", "Michael", "Emily", "James", "Emma", "Robert", "Olivia",
                       "William", "Ava", "Alexander", "Sophia", "Daniel", "Isabella", "Matthew", "Charlotte", "Ethan", "Amelia"]
        
        names = chinese_names if language == "Chinese" else english_names
        
        # Generate family relationship chain using power-of-2 distribution as in NeedleBench V2
        # Ensure we have enough names for the requested number of needles
        if num_needles > len(names):
            # If not enough names, repeat the list with suffixes
            repeated_names = []
            for i in range((num_needles // len(names)) + 1):
                for name in names:
                    repeated_names.append(f"{name}{i+1}" if i > 0 else name)
            names = repeated_names
        
        # NeedleBench V2 uses power-of-2 distribution: 2, 4, 8, 16, 32, 64, etc.
        # Generate a chain with exact number of needles requested
        selected_names = names[:num_needles]
        
        # Create relationship chain with proper NeedleBench V2 structure
        relationship_chain = []
        for i in range(len(selected_names) - 1):
            if language == "Chinese":
                # Use proper Chinese relationship terms (from youngest to oldest)
                relationships_zh = ["父亲", "母亲", "祖父", "祖母", "曾祖父", "曾祖母", "高祖父", "高祖母"]
                relationship_type = relationships_zh[i % len(relationships_zh)]
                relationship = f"{selected_names[i]}是{selected_names[i+1]}的{relationship_type}"
            else:
                # Use proper English relationship terms (from youngest to oldest)
                relationships_en = ["father", "mother", "grandfather", "grandmother", "great-grandfather", "great-grandmother", 
                                  "great-great-grandfather", "great-great-grandmother"]
                relationship_type = relationships_en[i % len(relationships_en)]
                relationship = f"{selected_names[i]} is the {relationship_type} of {selected_names[i+1]}"
            relationship_chain.append(relationship)
        
        # Generate retrieval question based on NeedleBench V2 format
        if language == "Chinese":
            retrieval_question = f"在上面提供的文本中，'{selected_names[-1]}'的能够向上追溯到的最年长的亲人是谁？请使用 \\boxed{{答案}} 的格式直接给出答案。"
        else:
            retrieval_question = f"Given the context described above, who is the eldest relative that '{selected_names[-1]}' can trace back to in the context? Please provide the answer directly using the format \\boxed{{answer}}."
        
        return {
            'needles': relationship_chain,
            'answer': selected_names[0],
            'retrieval_question': retrieval_question,
            'last_person': selected_names[-1]
        }
    
    async def _call_local_model(self, model, tokenizer, prompt: str) -> str:
        """Call a local model with the given prompt."""
        # Ensure model is on GPU if available (don't use CPU fallback unless absolutely necessary)
        if torch.cuda.is_available():
            # Check if model is on CPU and move it back to GPU
            first_param = next(model.parameters())
            if first_param.device.type == 'cpu':
                print(f"[NeedleBench] Model is on CPU, moving back to GPU...")
                try:
                    model = model.to('cuda:0')
                    print(f"[NeedleBench] Model moved to GPU successfully")
                except Exception as e:
                    print(f"[NeedleBench] Warning: Failed to move model to GPU: {e}, continuing on CPU")
        
        try:
            print(f"[NeedleBench] Calling local model with prompt length: {len(prompt)} characters")
            
            # Check if model is Llama-2 chat model and use chat template
            # Try multiple ways to get model name
            model_name = (
                getattr(model.config, '_name_or_path', None) or 
                getattr(model.config, 'name_or_path', None) or 
                getattr(model.config, 'model_type', None) or 
                ""
            )
            # Also check model type from config
            model_type = getattr(model.config, 'model_type', '').lower() if hasattr(model.config, 'model_type') else ''
            # Check if it's a Llama model (from model_type or name)
            is_llama = "llama" in model_type or "llama" in model_name.lower()
            # Check if it's a chat model (from name or path)
            is_chat = "chat" in model_name.lower()
            is_llama2_chat = is_llama and is_chat
            
            print(f"[NeedleBench] Model name: {model_name}, model_type: {model_type}, is_llama2_chat: {is_llama2_chat}")
            
            # Use chat template for Llama-2 chat models if available
            if is_llama2_chat and hasattr(tokenizer, 'apply_chat_template'):
                try:
                    # Format as chat messages
                    messages = [{"role": "user", "content": prompt}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    print(f"[NeedleBench] Using Llama-2 chat template")
                    prompt = formatted_prompt
                except Exception as e:
                    print(f"[NeedleBench] Warning: Failed to apply chat template: {e}, using raw prompt")
            elif is_llama2_chat:
                # Fallback: Use Llama-2 chat format manually
                # Llama-2 chat format: [INST] {prompt} [/INST]
                if not prompt.startswith("[INST]"):
                    prompt = f"[INST] {prompt} [/INST]"
                print(f"[NeedleBench] Using manual Llama-2 chat format")
            
            # Ensure pad_token is set
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                else:
                    print(f"[NeedleBench] Warning: No pad_token or eos_token found in tokenizer")
            
            # Get model max length using the dedicated method
            model_max_length = self._get_model_max_length(model, tokenizer)
            
            # First, check token count before truncation
            temp_inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            original_token_count = temp_inputs['input_ids'].shape[1]
            print(f"[NeedleBench] Prompt token count (before truncation): {original_token_count} tokens")
            
            # Tokenize the prompt (truncate if necessary)
            # IMPORTANT: If truncation happens here, the needle might be cut off!
            # This should be rare if context_length was properly adjusted
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=model_max_length)
            input_length = inputs['input_ids'].shape[1]
            print(f"[NeedleBench] Tokenized input shape: {inputs['input_ids'].shape}, length: {input_length} tokens")
            
            if original_token_count > model_max_length:
                truncated_tokens = original_token_count - input_length
                print(f"[NeedleBench] ⚠️  WARNING: Prompt was truncated from {original_token_count} to {input_length} tokens (removed {truncated_tokens} tokens)")
                print(f"[NeedleBench] ⚠️  This may have cut off the needle or context! Test accuracy may be affected.")
                print(f"[NeedleBench] ⚠️  Consider using a model with longer context support or reducing context_length further")
            
            # Move inputs to the same device as the model
            # Handle both single device and device_map="auto" cases
            if hasattr(model, 'device'):
                device = model.device
            elif hasattr(model, 'hf_device_map'):
                # For models with device_map="auto", find the device of the first parameter
                first_param = next(model.parameters())
                device = first_param.device
            else:
                # Fallback: try to get device from first parameter
                first_param = next(model.parameters())
                device = first_param.device
            
            print(f"[NeedleBench] Model device: {device}")
            # Store original device for potential fallback
            original_device = device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get pad_token_id
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = tokenizer.eos_token_id
            if pad_token_id is None:
                print(f"[NeedleBench] Warning: pad_token_id is None, using 0 as fallback")
                pad_token_id = 0
            
            print(f"[NeedleBench] Generating response with pad_token_id={pad_token_id}")
            
            # Generate response
            # Ensure attention_mask exists
            if "attention_mask" not in inputs:
                inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
            
            # Prepare generation kwargs
            # Use smaller max_new_tokens and batch_size=1 to reduce memory usage
            # For long contexts, use greedy decoding (do_sample=False) for faster generation
            input_length = inputs['input_ids'].shape[1]
            use_greedy = input_length > 10000  # Use greedy for very long contexts
            
            generation_kwargs = {
                "max_new_tokens": 128 if input_length > 10000 else 256,  # Further reduce for very long contexts
                "temperature": 0.7 if not use_greedy else 1.0,
                "do_sample": not use_greedy,  # Use greedy decoding for long contexts
                "pad_token_id": pad_token_id,
                "num_return_sequences": 1,  # Ensure only one sequence
            }
            
            if use_greedy:
                print(f"[NeedleBench] Using greedy decoding for long context (input_length={input_length})")
            
            # Add eos_token_id if available
            if tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
            
            print(f"[NeedleBench] Generation kwargs: max_new_tokens={generation_kwargs['max_new_tokens']}, pad_token_id={generation_kwargs['pad_token_id']}, eos_token_id={generation_kwargs.get('eos_token_id', 'None')}")
            print(f"[NeedleBench] Starting generation... (this may take a while for long contexts)")
            
            import time
            start_time = time.time()
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        **generation_kwargs
                    )
                    elapsed = time.time() - start_time
                    print(f"[NeedleBench] Generation completed in {elapsed:.2f} seconds")
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[NeedleBench] GPU OOM during generation: {e}")
                    # Try to clear cache first before falling back to CPU
                    print(f"[NeedleBench] Clearing GPU cache and retrying...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Try reducing max_new_tokens and retry
                    if generation_kwargs.get('max_new_tokens', 128) > 64:
                        print(f"[NeedleBench] Reducing max_new_tokens from {generation_kwargs['max_new_tokens']} to 64 and retrying...")
                        generation_kwargs['max_new_tokens'] = 64
                        try:
                            outputs = model.generate(**inputs, **generation_kwargs)
                            elapsed = time.time() - start_time
                            print(f"[NeedleBench] Generation completed in {elapsed:.2f} seconds (with reduced tokens)")
                        except torch.cuda.OutOfMemoryError as e2:
                            print(f"[NeedleBench] Still OOM after reducing tokens, falling back to CPU...")
                            # Only attempt CPU fallback if model is not already on CPU
                            if original_device.type != 'cpu' and hasattr(model, 'to'):
                                try:
                                    model = model.to('cpu')
                                    inputs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                                    print(f"[NeedleBench] Model moved to CPU, retrying generation...")
                                    outputs = model.generate(**inputs, **generation_kwargs)
                                    print(f"[NeedleBench] Generation successful on CPU")
                                except Exception as cpu_e:
                                    print(f"[NeedleBench] Failed to generate on CPU: {cpu_e}")
                                    raise RuntimeError(f"Generation failed on both GPU and CPU: GPU error: {e2}, CPU error: {cpu_e}")
                            else:
                                raise RuntimeError(f"Generation failed: {e2}")
                    else:
                        # Already at minimum, fall back to CPU
                        if original_device.type != 'cpu' and hasattr(model, 'to'):
                            try:
                                model = model.to('cpu')
                                inputs = {k: v.to('cpu') if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                                print(f"[NeedleBench] Model moved to CPU, retrying generation...")
                                outputs = model.generate(**inputs, **generation_kwargs)
                                print(f"[NeedleBench] Generation successful on CPU")
                            except Exception as cpu_e:
                                print(f"[NeedleBench] Failed to generate on CPU: {cpu_e}")
                                raise RuntimeError(f"Generation failed on both GPU and CPU: GPU error: {e}, CPU error: {cpu_e}")
                        else:
                            raise RuntimeError(f"Generation failed: {e}")
            
            print(f"[NeedleBench] Generated output shape: {outputs.shape}")
            
            # Extract only the new text (after the prompt)
            # Use input_ids length to extract new tokens more reliably
            prompt_tokens = inputs['input_ids'][0]
            input_length = prompt_tokens.shape[0] if hasattr(prompt_tokens, 'shape') else len(prompt_tokens)
            output_length = outputs[0].shape[0] if hasattr(outputs[0], 'shape') else len(outputs[0])
            
            print(f"[NeedleBench] Input tokens: {input_length}, Output tokens: {output_length}")
            
            # Extract new tokens (everything after the input)
            if output_length > input_length:
                new_tokens = outputs[0][input_length:]
                print(f"[NeedleBench] Extracted {len(new_tokens)} new tokens")
                
                # Decode only the new tokens
                try:
                    # Convert to list if it's a tensor
                    if hasattr(new_tokens, 'cpu'):
                        new_tokens_list = new_tokens.cpu().tolist()
                    elif hasattr(new_tokens, 'tolist'):
                        new_tokens_list = new_tokens.tolist()
                    else:
                        new_tokens_list = list(new_tokens)
                    
                    # Filter out invalid tokens (negative or too large)
                    vocab_size = getattr(tokenizer, 'vocab_size', 32000)
                    valid_tokens = [t for t in new_tokens_list if isinstance(t, int) and 0 <= t < vocab_size]
                    
                    if len(valid_tokens) > 0:
                        response = tokenizer.decode(valid_tokens, skip_special_tokens=True)
                        # Clean up response: remove any remaining special tokens or formatting
                        response = response.strip()
                        # Remove common chat template artifacts
                        response = response.replace("[INST]", "").replace("[/INST]", "").strip()
                    else:
                        print(f"[NeedleBench] Warning: No valid tokens found in new_tokens")
                        response = ""
                except Exception as decode_error:
                    print(f"[NeedleBench] Error decoding new tokens: {decode_error}")
                    # Fallback: decode full output and try to extract
                    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Try to find where the prompt ends
                    if prompt in full_response:
                        response = full_response[len(prompt):].strip()
                    else:
                        # If prompt not found, try to find common markers
                        if "[/INST]" in full_response:
                            response = full_response.split("[/INST]")[-1].strip()
                        else:
                            response = full_response.strip()
            else:
                print(f"[NeedleBench] Warning: No new tokens generated (output_length <= input_length)")
                response = ""
            
            print(f"[NeedleBench] Final response length: {len(response)}")
            print(f"[NeedleBench] Final response preview (first 200 chars): {response[:200] if response else '(empty)'}")
            
            if not response:
                print(f"[NeedleBench] Warning: Empty response generated")
                print(f"[NeedleBench] Debug info: prompt_length={len(prompt)}, output_shape={outputs.shape}, prompt_tokens={len(prompt_tokens)}, new_tokens={len(new_tokens)}")
            
            return response
        except Exception as e:
            print(f"[NeedleBench] Error calling local model: {e}")
            import traceback
            traceback.print_exc()
            raise  # Re-raise the exception so it can be caught by the caller
    
    async def _call_api_model(self, client: Client, model: str, prompt: str) -> str:
        """Call an API model with the given prompt."""
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling API model: {e}")
            return ""
    
    def evaluate_local_llm(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        subdataset_name_or_params=None,
        *args,
        **kwargs,
    ) -> Dict:
        """Evaluate a local LLM using the needle-in-haystack benchmark.
        
        Supports two calling conventions:
        1. Standard interface (from runner): Runs comprehensive parameterized tests
        2. Parameterized interface (from parameterized_benchmark_test.py): evaluate_local_llm(model, tokenizer, params: Dict)
        
        Args:
            model: The local LLM model to evaluate.
            tokenizer: The tokenizer for the LLM model.
            subdataset_name_or_params: Either a string (subdataset name) or a Dict (params dict).
                - If string: "NeedleInAHaystack" (defaults to PaulGrahamEssays), "PaulGrahamEssays", or "Journey_to_the_West"
                - If Dict: Full parameter dictionary with task_type, context_length, etc.
            *args: Additional positional arguments (ignored if params is Dict).
            **kwargs: Additional keyword arguments that can override default params.
                - If 'subdataset_name' is in kwargs, it takes precedence over subdataset_name_or_params.
                - If 'run_parameterized' is False in kwargs, runs single test instead of parameterized suite.
        
        Returns:
            Dict: Evaluation results. For standard interface, returns aggregated results from parameterized tests.
        """
        # Check if we should run parameterized tests (default True for standard interface)
        run_parameterized = kwargs.pop("run_parameterized", True)
        print(f"[NeedleBench] evaluate_local_llm called: run_parameterized={run_parameterized}, kwargs keys: {list(kwargs.keys())}, subdataset_name_or_params type: {type(subdataset_name_or_params)}")
        
        # IMPORTANT: Check if third argument is a dict FIRST (parameterized interface from _run_parameterized_evaluation)
        # This must be checked before checking kwargs to handle the case where params dict is passed directly
        if isinstance(subdataset_name_or_params, dict):
            # Parameterized interface: params dict is passed directly
            # This is called from _run_parameterized_evaluation, so run_parameterized should be False
            print(f"[NeedleBench] Detected parameterized interface: params dict passed as third argument")
            params = self.default_params.copy()
            params.update(subdataset_name_or_params)
            # Ensure subdataset_name is set
            if "subdataset_name" not in params:
                params["subdataset_name"] = "PaulGrahamEssays"
            params.update(kwargs)
            # Don't run parameterized again - we're already in a parameterized test
            run_parameterized = False
            # Continue to single test execution below
        
        # Check if subdataset_name is provided as a keyword argument (from runner)
        elif "subdataset_name" in kwargs:
            subdataset_name = kwargs.pop("subdataset_name")
            # Standard interface: subdataset_name is a string
            # Map subdataset_name to actual dataset name
            if subdataset_name == "NeedleInAHaystack":
                actual_dataset_name = "PaulGrahamEssays"
            else:
                actual_dataset_name = subdataset_name
            
            # If run_parameterized is True, run comprehensive parameterized tests
            if run_parameterized:
                try:
                    print(f"[NeedleBench] ========== STARTING PARAMETERIZED EVALUATION ==========")
                    print(f"[NeedleBench] Dataset: {actual_dataset_name}")
                    print(f"[NeedleBench] Model: {getattr(model.config, '_name_or_path', None) or getattr(model.config, 'name_or_path', None) or 'unknown'}")
                    print(f"[NeedleBench] Additional kwargs: {list(kwargs.keys())}")
                    result = self._run_parameterized_evaluation(model, tokenizer, actual_dataset_name, **kwargs)
                    print(f"[NeedleBench] ========== PARAMETERIZED EVALUATION COMPLETED ==========")
                    print(f"[NeedleBench] Total tests: {result.get('total_tests', 0)}")
                    print(f"[NeedleBench] Successful tests: {result.get('successful_tests', 0)}")
                    print(f"[NeedleBench] Result keys: {list(result.keys())}")
                    print(f"[NeedleBench] Evaluation type: {result.get('evaluation_type', 'unknown')}")
                    if result.get('evaluation_type') != 'parameterized_comprehensive':
                        print(f"[NeedleBench] WARNING: Result does not have expected evaluation_type!")
                    return result
                except Exception as e:
                    print(f"[NeedleBench] ========== PARAMETERIZED EVALUATION FAILED ==========")
                    print(f"[NeedleBench] Error: {e}")
                    import traceback
                    traceback.print_exc()
                    # Fall back to single test if parameterized fails
                    print(f"[NeedleBench] Falling back to single test due to error above")
                    params = self.default_params.copy()
                    params["subdataset_name"] = actual_dataset_name
                    params.update(kwargs)
                    task_type = params.get("task_type", TaskType.SINGLE_NEEDLE_RETRIEVAL)
                    try:
                        loop = asyncio.get_running_loop()
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(self._run_sync_evaluation, model, tokenizer, params, task_type)
                            return future.result()
                    except RuntimeError:
                        return self._run_sync_evaluation(model, tokenizer, params, task_type)
            
            # Otherwise run single test
            params = self.default_params.copy()
            params["subdataset_name"] = actual_dataset_name
            params.update(kwargs)
            task_type = params.get("task_type", TaskType.SINGLE_NEEDLE_RETRIEVAL)
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_sync_evaluation, model, tokenizer, params, task_type)
                    return future.result()
            except RuntimeError:
                return self._run_sync_evaluation(model, tokenizer, params, task_type)
        
        elif subdataset_name_or_params is not None:
            # Standard interface: subdataset_name is a string (positional argument)
            subdataset_name = subdataset_name_or_params
            # Map subdataset_name to actual dataset name
            if subdataset_name == "NeedleInAHaystack":
                actual_dataset_name = "PaulGrahamEssays"
            else:
                actual_dataset_name = subdataset_name
            
            # If run_parameterized is True, run comprehensive parameterized tests
            if run_parameterized:
                return self._run_parameterized_evaluation(model, tokenizer, actual_dataset_name, **kwargs)
            
            # Otherwise run single test
            params = self.default_params.copy()
            params["subdataset_name"] = actual_dataset_name
            params.update(kwargs)
        else:
            # No subdataset_name provided, use defaults
            if run_parameterized:
                return self._run_parameterized_evaluation(model, tokenizer, "PaulGrahamEssays", **kwargs)
            params = self.default_params.copy()
            params["subdataset_name"] = "PaulGrahamEssays"
            params.update(kwargs)
        
        # Get task type for single test
        task_type = params.get("task_type", TaskType.SINGLE_NEEDLE_RETRIEVAL)
        
        # Check if we're in an async context (for parameterized_benchmark_test.py)
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to use a different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_sync_evaluation, model, tokenizer, params, task_type)
                return future.result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            return self._run_sync_evaluation(model, tokenizer, params, task_type)
    
    def _run_sync_evaluation(self, model, tokenizer, params: Dict, task_type: str) -> Dict:
        """Run synchronous evaluation by creating a new event loop."""
        # Run async evaluation
        if task_type == TaskType.SINGLE_NEEDLE_RETRIEVAL:
            return asyncio.run(self._evaluate_single_needle_retrieval(model, tokenizer, params))
        elif task_type == TaskType.MULTI_NEEDLE_RETRIEVAL:
            return asyncio.run(self._evaluate_multi_needle_retrieval(model, tokenizer, params))
        elif task_type == TaskType.MULTI_NEEDLE_REASONING:
            return asyncio.run(self._evaluate_multi_needle_reasoning(model, tokenizer, params))
        elif task_type == TaskType.ANCESTRAL_TRACE_CHALLENGE:
            return asyncio.run(self._evaluate_ancestral_trace_challenge(model, tokenizer, params))
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _get_model_max_length(self, model, tokenizer) -> int:
        """Get the actual maximum context length for the model."""
        tokenizer_max = getattr(tokenizer, 'model_max_length', None)
        config_max = getattr(model.config, 'max_position_embeddings', None)
        
        # Also check for other common config attributes
        # Some models use different attribute names (e.g., Qwen uses seq_length)
        seq_length = getattr(model.config, 'seq_length', None)
        
        # Filter out unreasonably large values (some tokenizers set model_max_length to int(1e30))
        # Use the smallest reasonable value found
        candidates = []
        if tokenizer_max and tokenizer_max < 1e10:
            candidates.append(tokenizer_max)
        if config_max and config_max < 1e10:
            candidates.append(config_max)
        if seq_length and seq_length < 1e10:
            candidates.append(seq_length)
        
        if candidates:
            # Use the minimum of all valid candidates (most conservative)
            model_max_length = min(candidates)
        else:
            # Default to 4096 for older models (e.g., Llama-2-7b)
            # But don't cap it - let models with longer contexts use their full capacity
            model_max_length = 4096
            print(f"[NeedleBench] Warning: Could not detect model max length, defaulting to {model_max_length}")
        
        print(f"[NeedleBench] Detected model max length: {model_max_length} tokens")
        return model_max_length
    
    def _run_parameterized_evaluation(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        subdataset_name: str,
        max_combinations: int = 30,
        length_group: str = "medium",
        **kwargs
    ) -> Dict:
        """Run comprehensive parameterized evaluation suite.
        
        This method implements the core testing logic from parameterized_benchmark_test.py,
        running multiple test combinations with different parameters and aggregating results.
        
        Args:
            model: The local LLM model to evaluate.
            tokenizer: The tokenizer for the LLM model.
            subdataset_name: The dataset name to use.
            max_combinations: Maximum number of test combinations to run (default: 30).
            length_group: Context length group to use (default: "medium").
            **kwargs: Additional parameters.
        
        Returns:
            Dict: Aggregated evaluation results with statistics.
        """
        # Generate test combinations
        try:
            combinations = self._generate_test_combinations(max_combinations, length_group, subdataset_name)
            print(f"[NeedleBench] Generated {len(combinations)} test combinations")
        except Exception as e:
            print(f"[NeedleBench] Error generating test combinations: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        if not combinations:
            print(f"[NeedleBench] Warning: No test combinations generated, falling back to single test")
            # Fall back to single test
            params = self.default_params.copy()
            params["subdataset_name"] = subdataset_name
            task_type = params.get("task_type", TaskType.SINGLE_NEEDLE_RETRIEVAL)
            return self._run_sync_evaluation(model, tokenizer, params, task_type)
        
        print(f"[NeedleBench] Running parameterized evaluation: {len(combinations)} test combinations")
        
        # Get model's actual max length and adjust test parameters
        model_max_length = self._get_model_max_length(model, tokenizer)
        print(f"[NeedleBench] Model max length: {model_max_length}, adjusting test parameters accordingly")
        
        # Estimate overhead more accurately (use conservative estimates):
        # - Prompt template (question + formatting): ~150-250 tokens (English prompt is ~150 chars = ~40 tokens, but with formatting can be more)
        # - Chat template (if Llama-2): ~30-50 tokens (can add significant overhead)
        # - Needle(s): ~20-50 tokens per needle (estimate max)
        # - Generation space: ~128 tokens (for max_new_tokens)
        # - Safety margin: ~100 tokens (for variations in tokenization)
        # Total overhead: ~500-600 tokens (very conservative to avoid truncation)
        prompt_template_overhead = 250  # Question + formatting (conservative)
        chat_template_overhead = 50     # Llama-2 chat template (conservative)
        max_needle_overhead = 50        # Max needle tokens
        generation_overhead = 128       # max_new_tokens
        safety_margin = 100             # Safety margin for tokenization variations
        total_overhead = prompt_template_overhead + chat_template_overhead + max_needle_overhead + generation_overhead + safety_margin
        
        available_context_length = model_max_length - total_overhead
        print(f"[NeedleBench] Reserved {total_overhead} tokens for overhead:")
        print(f"[NeedleBench]   - Prompt template: {prompt_template_overhead} tokens")
        print(f"[NeedleBench]   - Chat template: {chat_template_overhead} tokens")
        print(f"[NeedleBench]   - Needle(s): {max_needle_overhead} tokens")
        print(f"[NeedleBench]   - Generation: {generation_overhead} tokens")
        print(f"[NeedleBench]   - Safety margin: {safety_margin} tokens")
        print(f"[NeedleBench] Available context length: {available_context_length} tokens")
        
        # Adjust combinations to fit model's max length
        adjusted_combinations = []
        for combo in combinations:
            original_length = combo.get("context_length", 32768)
            depth_percent = combo.get("depth_percent", 50)
            
            if original_length > available_context_length:
                # Scale down context_length to fit model
                adjusted_length = max(1024, available_context_length)  # Minimum 1024 tokens
                
                # IMPORTANT: Adjust depth_percent to ensure needle stays within adjusted context
                # If depth was 50% of 32768, it should be 50% of adjusted_length too
                # But we need to ensure the needle position is still valid
                # For now, keep the same depth_percent (relative position)
                
                combo["context_length"] = adjusted_length
                combo["original_context_length"] = original_length  # Keep original for scoring
                print(f"[NeedleBench] Adjusted context_length: {original_length} -> {adjusted_length} tokens (depth: {depth_percent}%, needle will be at ~{int(adjusted_length * depth_percent / 100)} tokens)")
            else:
                combo["original_context_length"] = original_length
                print(f"[NeedleBench] Context length {original_length} fits within model limit (depth: {depth_percent}%)")
            
            adjusted_combinations.append(combo)
        
        combinations = adjusted_combinations
        
        # Run all tests (following parameterized_benchmark_test.py pattern)
        all_results = []
        for i, params in enumerate(combinations, 1):
            try:
                task_type = params.get("task_type", TaskType.SINGLE_NEEDLE_RETRIEVAL)
                print(f"[NeedleBench] Running test {i}/{len(combinations)}: {task_type}, context={params.get('context_length')}, depth={params.get('depth_percent')}%, needles={params.get('num_needles')}, language={params.get('language')}")
                
                # Call evaluate_local_llm with params dict (parameterized interface)
                # This matches the pattern from parameterized_benchmark_test.py line 274-276
                # IMPORTANT: Pass run_parameterized=False to avoid infinite recursion
                result = self.evaluate_local_llm(model, tokenizer, params, run_parameterized=False)
                
                # Add test metadata (matching parameterized_benchmark_test.py line 286-289)
                result["test_index"] = i
                result["test_params"] = params.copy()
                # Ensure task_type in test_params matches the actual task_type from result
                if "task_type" in result:
                    result["test_params"]["task_type"] = result["task_type"]
                if "test_timestamp" not in result:
                    from datetime import datetime
                    result["test_timestamp"] = datetime.now().isoformat()
                
                all_results.append(result)
                score = result.get('score', 0.0)
                success = result.get('success', False)
                print(f"[NeedleBench] Test {i}/{len(combinations)} completed: {task_type}, score: {score:.3f}, success: {success}")
                
                # Ensure model is back on GPU after each test (in case it was moved to CPU)
                if torch.cuda.is_available():
                    first_param = next(model.parameters())
                    if first_param.device.type == 'cpu':
                        print(f"[NeedleBench] Moving model back to GPU after test {i}...")
                        try:
                            model = model.to('cuda:0')
                            print(f"[NeedleBench] Model moved back to GPU successfully")
                        except Exception as e:
                            print(f"[NeedleBench] Warning: Failed to move model back to GPU: {e}")
                    
                    # Clear GPU cache periodically to prevent OOM (every 10 tests instead of 3 to reduce overhead)
                    if i % 3 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()
                        print(f"[NeedleBench] GPU memory after {i} tests: {torch.cuda.memory_allocated() / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            except Exception as e:
                print(f"[NeedleBench] Test {i}/{len(combinations)} failed: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({
                    "test_index": i,
                    "test_params": params.copy(),
                    "error": str(e),
                    "success": False,
                    "score": 0.0,
                    "test_timestamp": datetime.now().isoformat() if 'datetime' in dir() else None
                })
        
        print(f"[NeedleBench] All tests completed. Total: {len(all_results)}, Successful: {len([r for r in all_results if r.get('success', False)])}")
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results, subdataset_name)
        print(f"[NeedleBench] Aggregation completed. Result keys: {list(aggregated.keys())}")
        return aggregated
    
    def _generate_test_combinations(
        self, 
        max_combinations: int = 30, 
        length_group: str = "medium",
        subdataset_name: str = "PaulGrahamEssays"
    ) -> List[Dict]:
        """Generate test parameter combinations (based on parameterized_benchmark_test.py logic)."""
        combinations = []
        
        # Get context lengths
        if length_group and length_group in self.parameter_space["context_length_groups"]:
            context_lengths = self.parameter_space["context_length_groups"][length_group]
        else:
            # Use medium by default
            context_lengths = self.parameter_space["context_length_groups"]["medium"]
        
        # Calculate tests per task type (ensure balanced distribution)
        task_types = self.parameter_space["task_types"]
        tests_per_task = max_combinations // len(task_types)
        remaining_tests = max_combinations % len(task_types)
        
        # Generate combinations for each task type
        for i, task_type in enumerate(task_types):
            task_combinations = []
            num_needles_options = self.parameter_space["needle_configs"][task_type]
            
            # Calculate max tests for this task type
            task_max_tests = tests_per_task
            if i < remaining_tests:
                task_max_tests += 1
            
            # Generate combinations for this task type
            for num_needles in num_needles_options:
                for language, lang_config in self.parameter_space["language_configs"].items():
                    for context_length in context_lengths:
                        # Select optimal depths for this task type
                        selected_depths = self._select_optimal_depths(task_type, num_needles, max_depths=5)
                        
                        for depth_percent in selected_depths:
                            combination = {
                                "context_length": context_length,
                                "depth_percent": depth_percent,
                                "num_needles": num_needles,
                                "language": language,
                                "task_type": task_type,
                                "subdataset_name": subdataset_name,
                                "length_buffer": lang_config.get("length_buffer", 1000),
                                "guide": lang_config.get("guide", True)
                            }
                            task_combinations.append(combination)
                            
                            if len(task_combinations) >= task_max_tests:
                                break
                        if len(task_combinations) >= task_max_tests:
                            break
                    if len(task_combinations) >= task_max_tests:
                        break
                if len(task_combinations) >= task_max_tests:
                    break
            
            combinations.extend(task_combinations[:task_max_tests])
        
        # Trim to max_combinations
        result = combinations[:max_combinations]
        print(f"[NeedleBench] Generated {len(result)} test combinations from {len(combinations)} candidates")
        return result
    
    def _select_optimal_depths(self, task_type: str, num_needles: int, max_depths: int = 5) -> List[int]:
        """Select optimal depth points for testing."""
        all_depths = self.parameter_space["depth_percents"]
        
        if task_type == TaskType.SINGLE_NEEDLE_RETRIEVAL:
            key_depths = [0, 52, 100]
        elif task_type == TaskType.MULTI_NEEDLE_RETRIEVAL:
            step = max(1, len(all_depths) // max_depths)
            key_depths = all_depths[::step][:max_depths]
        elif task_type == TaskType.MULTI_NEEDLE_REASONING:
            key_depths = [52, 63, 84]
        else:  # ANCESTRAL_TRACE_CHALLENGE
            key_depths = [0, 31, 52]
        
        selected = []
        for depth in key_depths:
            if depth in all_depths and depth not in selected:
                selected.append(depth)
                if len(selected) >= max_depths:
                    break
        
        return selected if selected else [50]  # Fallback to middle
    
    def _aggregate_results(self, all_results: List[Dict], subdataset_name: str) -> Dict:
        """Aggregate results from multiple tests into summary statistics."""
        if not all_results:
            return {
                "dataset": subdataset_name,
                "success": False,
                "error": "No test results",
                "total_tests": 0
            }
        
        # Calculate statistics
        successful_tests = [r for r in all_results if r.get("success", False)]
        failed_tests = [r for r in all_results if not r.get("success", False)]
        
        scores = [r.get("score", 0.0) for r in successful_tests]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        
        # Group by task type
        # Normalize task_type values (handle both uppercase and lowercase)
        task_results = {}
        for task_type in self.parameter_space["task_types"]:
            # Try both exact match and case-insensitive match
            task_tests = []
            for r in successful_tests:
                # Check both result["task_type"] and result["test_params"]["task_type"]
                result_task_type = r.get("task_type", "")
                test_params_task_type = r.get("test_params", {}).get("task_type", "")
                
                # Normalize: convert to lowercase for comparison
                def normalize_task_type(t):
                    if isinstance(t, str):
                        return t.lower().strip()
                    return str(t).lower().strip()
                
                result_task_type_norm = normalize_task_type(result_task_type)
                test_params_task_type_norm = normalize_task_type(test_params_task_type)
                task_type_norm = normalize_task_type(task_type)
                
                # Match if either task_type matches
                if result_task_type_norm == task_type_norm or test_params_task_type_norm == task_type_norm:
                    task_tests.append(r)
            
            if task_tests:
                task_scores = [r.get("score", 0.0) for r in task_tests]
                task_results[task_type] = {
                    "count": len(task_tests),
                    "avg_score": sum(task_scores) / len(task_scores),
                    "max_score": max(task_scores),
                    "min_score": min(task_scores)
                }
                print(f"[NeedleBench] Task type {task_type}: {len(task_tests)} tests, avg_score: {task_results[task_type]['avg_score']:.3f}")
            else:
                print(f"[NeedleBench] Warning: No tests found for task_type {task_type}")
        
        # Group by context length (use original_context_length if available for scoring)
        length_results = {}
        length_weighted_scores = []
        length_weights = []
        max_tested_length = 0
        
        for result in successful_tests:
            # Use original_context_length for scoring, context_length for grouping
            original_length = result.get("test_params", {}).get("original_context_length")
            # Use actual tested context_length (not original) for max_tested_length
            actual_tested_length = result.get("test_params", {}).get("context_length", 32768)
            context_length = original_length or actual_tested_length  # Use original for scoring weight
            length_key = str(actual_tested_length)  # Use actual tested length for grouping
            
            if length_key not in length_results:
                length_results[length_key] = {"scores": [], "count": 0, "original_length": original_length or actual_tested_length}
            length_results[length_key]["scores"].append(result.get("score", 0.0))
            length_results[length_key]["count"] += 1
            
            # Calculate length-weighted score (longer contexts get higher weight)
            # Use original_length for weighting to reward models that can handle longer contexts
            score = result.get("score", 0.0)
            weight = max(1.0, math.log10(max(context_length, 1000)))
            length_weighted_scores.append(score * weight)
            length_weights.append(weight)
            # Use actual tested length for max_tested_length (not original)
            max_tested_length = max(max_tested_length, actual_tested_length)
        
        for length in length_results:
            scores = length_results[length]["scores"]
            length_results[length]["avg_score"] = sum(scores) / len(scores)
            length_results[length]["max_score"] = max(scores)
            length_results[length]["min_score"] = min(scores)
            del length_results[length]["scores"]
        
        # Calculate length-weighted average and length bonus
        length_weighted_avg = sum(length_weighted_scores) / sum(length_weights) if length_weights else 0.0
        # Bonus for supporting longer contexts (up to 10% bonus)
        length_bonus = min(0.1, max_tested_length / 100000.0)
        # Average length score (performance across all tested lengths)
        avg_length_score = sum(stat["avg_score"] for stat in length_results.values()) / len(length_results) if length_results else 0.0
        
        # Group by depth
        depth_results = {}
        for result in successful_tests:
            depth = result.get("test_params", {}).get("depth_percent", "unknown")
            if depth not in depth_results:
                depth_results[depth] = {"scores": [], "count": 0}
            depth_results[depth]["scores"].append(result.get("score", 0.0))
            depth_results[depth]["count"] += 1
        
        for depth in depth_results:
            scores = depth_results[depth]["scores"]
            depth_results[depth]["avg_score"] = sum(scores) / len(scores)
            del depth_results[depth]["scores"]
        
        # Determine actual dataset name from results (use the most common one from test_params)
        actual_dataset_in_results = subdataset_name
        if all_results:
            # Check what dataset was actually used in the tests
            dataset_counts = {}
            for result in all_results:
                test_params = result.get("test_params", {})
                used_dataset = test_params.get("subdataset_name", subdataset_name)
                # Also check if language was Chinese, which means Journey_to_the_West
                language = test_params.get("language", "English")
                if language == "Chinese":
                    used_dataset = "Journey_to_the_West"
                dataset_counts[used_dataset] = dataset_counts.get(used_dataset, 0) + 1
            if dataset_counts:
                actual_dataset_in_results = max(dataset_counts, key=dataset_counts.get)
        
        return {
            "dataset": actual_dataset_in_results,  # Use actual dataset name used in tests
            "success": True,
            "total_tests": len(all_results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests) / len(all_results) if all_results else 0.0,
            "overall_statistics": {
                "avg_score": avg_score,
                "max_score": max_score,
                "min_score": min_score,
                "length_weighted_avg_score": length_weighted_avg,  # 长度加权平均分（更长上下文权重更高）
                "avg_length_score": avg_length_score,  # 平均长度分数（所有测试长度的平均表现）
                "length_bonus": length_bonus,  # 长度奖励（支持更长上下文的奖励）
                "max_tested_length": max_tested_length  # 最大测试长度
            },
            "task_type_statistics": task_results,
            "context_length_statistics": length_results,
            "depth_statistics": depth_results,
            "all_results": all_results,  # Include individual results for detailed analysis
            "evaluation_type": "parameterized_comprehensive"
        }
    
    async def _evaluate_single_needle_retrieval(self, model, tokenizer, params: Dict) -> Dict:
        """Evaluate single needle retrieval task."""
        # Get parameters
        num_needles = params.get("num_needles", 1)
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        
        # Select dataset based on language
        if language == "Chinese":
            actual_dataset_name = "Journey_to_the_West"
        else:  # English
            actual_dataset_name = "PaulGrahamEssays"
        
        # Override subdataset_name if explicitly set, otherwise use language-based selection
        if subdataset_name and subdataset_name not in ["PaulGrahamEssays", "Journey_to_the_West"]:
            actual_dataset_name = subdataset_name
        elif subdataset_name == "Journey_to_the_West" or (language == "Chinese" and subdataset_name == "PaulGrahamEssays"):
            actual_dataset_name = "Journey_to_the_West" if language == "Chinese" else "PaulGrahamEssays"
        
        print(f"[NeedleBench] Language: {language}, Using dataset: {actual_dataset_name}")
        
        # Load dataset
        context = self._load_dataset(actual_dataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Get needles and question
        if num_needles > 1:
            needles = self._get_needles_list(num_needles, language)
            # Check needle token length
            needle_tokens_total = sum(len(tokenizer_tiktoken.encode(n)) for n in needles)
            print(f"[NeedleBench] Using {num_needles} needles, total token length: {needle_tokens_total} tokens")
            context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
            needle_for_eval = needles[0]  # Use first needle for evaluation
        else:
            needle = self._get_random_needle(language)
            needle_tokens = len(tokenizer_tiktoken.encode(needle))
            print(f"[NeedleBench] Using single needle, token length: {needle_tokens} tokens, needle: {needle[:50]}...")
            context_with_needles = self._insert_needle(context, needle, depth_percent)
            needle_for_eval = needle
        
        question = self.default_questions[language]
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        import torch
        try:
            response = await self._call_local_model(model, tokenizer, prompt)
            
            # Check if response is empty
            if not response or not response.strip():
                print(f"[NeedleBench] Warning: Empty response from model")
                model_name = getattr(model.config, '_name_or_path', None) or getattr(model.config, 'name_or_path', None) or "unknown_model"
                return {
                    "model": model_name,
                    "dataset": actual_dataset_name,  # Use actual dataset name based on language
                    "language": language,
                    "context_length": context_length,
                    "depth_percent": depth_percent,
                    "num_needles": num_needles,
                    "score": 0.0,
                    "response": "",
                    "expected_needle": needle_for_eval,
                    "success": False,
                    "error": "Empty response from model",
                    "task_type": TaskType.SINGLE_NEEDLE_RETRIEVAL
                }
            
            # Evaluate response
            score = self._evaluate_response(response, needle_for_eval, language)
            
            # Get model name safely
            model_name = getattr(model.config, '_name_or_path', None) or getattr(model.config, 'name_or_path', None) or "unknown_model"
            
            # Return results
            return {
                "model": model_name,
                "dataset": actual_dataset_name,  # Use actual dataset name based on language
                "language": language,
                "context_length": context_length,
                "depth_percent": depth_percent,
                "num_needles": num_needles,
                "score": score,
                "response": response,
                "expected_needle": needle_for_eval,
                "success": True,
                "task_type": "SINGLE_NEEDLE_RETRIEVAL"
            }
        except Exception as e:
            print(f"[NeedleBench] Error in _evaluate_single_needle_retrieval: {e}")
            import traceback
            traceback.print_exc()
            model_name = getattr(model.config, '_name_or_path', None) or getattr(model.config, 'name_or_path', None) or "unknown_model"
            return {
                "model": model_name,
                "dataset": subdataset_name,
                "language": language,
                "context_length": context_length,
                "depth_percent": depth_percent,
                "num_needles": num_needles,
                "score": 0.0,
                "response": "",
                "expected_needle": needle_for_eval,
                "success": False,
                "error": str(e),
                "task_type": "SINGLE_NEEDLE_RETRIEVAL"
            }
    
    async def _evaluate_multi_needle_retrieval(self, model, tokenizer, params: Dict) -> Dict:
        """Evaluate multi needle retrieval task."""
        # Get parameters
        num_needles = params.get("num_needles", 3)
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        
        # Select dataset based on language
        if language == "Chinese":
            actual_dataset_name = "Journey_to_the_West"
        else:  # English
            actual_dataset_name = "PaulGrahamEssays"
        
        # Override subdataset_name if explicitly set, otherwise use language-based selection
        if subdataset_name and subdataset_name not in ["PaulGrahamEssays", "Journey_to_the_West"]:
            actual_dataset_name = subdataset_name
        elif subdataset_name == "Journey_to_the_West" or (language == "Chinese" and subdataset_name == "PaulGrahamEssays"):
            actual_dataset_name = "Journey_to_the_West" if language == "Chinese" else "PaulGrahamEssays"
        
        print(f"[NeedleBench] Language: {language}, Using dataset: {actual_dataset_name}")
        
        # Load dataset
        context = self._load_dataset(actual_dataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Get needles and question
        needles = self._get_needles_list(num_needles, language)
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        needle_for_eval = needles[0]  # Use first needle for evaluation
        
        question = self.default_questions[language]
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        import torch
        response = await self._call_local_model(model, tokenizer, prompt)
        
        # Evaluate response
        score = self._evaluate_response(response, needle_for_eval, language)
        
        # Get model name safely
        model_name = getattr(model.config, '_name_or_path', None) or getattr(model.config, 'name_or_path', None) or "unknown_model"
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_needle": needle_for_eval,
            "needles": needles,
            "success": True,
                "task_type": TaskType.MULTI_NEEDLE_RETRIEVAL
        }
    
    async def _evaluate_multi_needle_reasoning(self, model, tokenizer, params: Dict) -> Dict:
        """Evaluate multi needle reasoning task."""
        # Get parameters
        num_needles = params.get("num_needles", 4)  # 使用基于2的幂次
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        
        # Select dataset based on language
        if language == "Chinese":
            actual_dataset_name = "Journey_to_the_West"
        else:  # English
            actual_dataset_name = "PaulGrahamEssays"
        
        # Override subdataset_name if explicitly set, otherwise use language-based selection
        if subdataset_name and subdataset_name not in ["PaulGrahamEssays", "Journey_to_the_West"]:
            actual_dataset_name = subdataset_name
        elif subdataset_name == "Journey_to_the_West" or (language == "Chinese" and subdataset_name == "PaulGrahamEssays"):
            actual_dataset_name = "Journey_to_the_West" if language == "Chinese" else "PaulGrahamEssays"
        
        print(f"[NeedleBench] Language: {language}, Using dataset: {actual_dataset_name}")
        
        # Load dataset
        context = self._load_dataset(actual_dataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Generate ATC needles for reasoning - 使用NeedleBench V2的基于2的幂次分布
        atc_data = self._generate_atc_needles(num_needles, language)
        needles = atc_data['needles']
        expected_answer = atc_data['answer']
        question = atc_data['retrieval_question']
        
        # Insert needles into context
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        import torch
        response = await self._call_local_model(model, tokenizer, prompt)
        
        # Evaluate response using ATC evaluation logic
        score = self._evaluate_atc_response(response, expected_answer)
        
        # Get model name safely
        model_name = getattr(model.config, '_name_or_path', None) or getattr(model.config, 'name_or_path', None) or "unknown_model"
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_answer": expected_answer,
            "needles": needles,
            "success": True,
            "task_type": TaskType.MULTI_NEEDLE_REASONING
        }
    
    async def _evaluate_ancestral_trace_challenge(self, model, tokenizer, params: Dict) -> Dict:
        """Evaluate ancestral trace challenge (ATC) task."""
        # Get parameters
        num_needles = params.get("num_needles", 8)  # ATC使用基于2的幂次的稀疏分布
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        
        # Select dataset based on language
        if language == "Chinese":
            actual_dataset_name = "Journey_to_the_West"
        else:  # English
            actual_dataset_name = "PaulGrahamEssays"
        
        # Override subdataset_name if explicitly set, otherwise use language-based selection
        if subdataset_name and subdataset_name not in ["PaulGrahamEssays", "Journey_to_the_West"]:
            actual_dataset_name = subdataset_name
        elif subdataset_name == "Journey_to_the_West" or (language == "Chinese" and subdataset_name == "PaulGrahamEssays"):
            actual_dataset_name = "Journey_to_the_West" if language == "Chinese" else "PaulGrahamEssays"
        
        print(f"[NeedleBench] Language: {language}, Using dataset: {actual_dataset_name}")
        
        # Load dataset
        context = self._load_dataset(actual_dataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Generate ATC needles
        atc_data = self._generate_atc_needles(num_needles, language)
        needles = atc_data['needles']
        expected_answer = atc_data['answer']
        question = atc_data['retrieval_question']
        
        # Insert needles into context
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        import torch
        response = await self._call_local_model(model, tokenizer, prompt)
        
        # Evaluate response using ATC evaluation logic
        score = self._evaluate_atc_response(response, expected_answer)
        
        # Get model name safely
        model_name = getattr(model.config, '_name_or_path', None) or getattr(model.config, 'name_or_path', None) or "unknown_model"
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_answer": expected_answer,
            "needles": needles,
            "success": True,
            "task_type": TaskType.ANCESTRAL_TRACE_CHALLENGE
        }
    
    def evaluate_api_llm(
        self,
        client: Client,
        model_or_params=None,
        subdataset_name: str = None,
        *args,
        **kwargs,
    ) -> Dict:
        """Evaluate an API-based LLM using the needle-in-haystack benchmark.
        
        Supports two calling conventions:
        1. Standard interface (from runner): evaluate_api_llm(client, model: str, subdataset_name: str, **kwargs)
        2. Parameterized interface (from parameterized_benchmark_test.py): evaluate_api_llm(client, params: Dict)
        
        Args:
            client: The OpenAI client to use for API calls.
            model_or_params: Either a string (model name) or a Dict (params dict).
                - If string: The name of the API LLM model to evaluate.
                - If Dict: Full parameter dictionary with task_type, context_length, model, etc.
            subdataset_name: The name of the sub-dataset (only used if model_or_params is a string).
                For NeedleInAHaystack, this can be "NeedleInAHaystack" (defaults to PaulGrahamEssays),
                "PaulGrahamEssays", or "Journey_to_the_West".
            *args: Additional positional arguments (ignored if params is Dict).
            **kwargs: Additional keyword arguments that can override default params.
                - If 'model' and 'subdataset_name' are in kwargs, they take precedence.
        
        Returns:
            Dict: Evaluation results.
        """
        # Check if model and subdataset_name are provided as keyword arguments (from runner)
        if "model" in kwargs and "subdataset_name" in kwargs:
            model = kwargs.pop("model")
            subdataset_name = kwargs.pop("subdataset_name")
            # Standard interface: model is a string, subdataset_name is a string
            # Map subdataset_name to actual dataset name
            if subdataset_name == "NeedleInAHaystack" or subdataset_name is None:
                actual_dataset_name = "PaulGrahamEssays"
            else:
                actual_dataset_name = subdataset_name
            
            # Merge default params with kwargs
            params = self.default_params.copy()
            params["subdataset_name"] = actual_dataset_name
            params["model"] = model
            params.update(kwargs)
        # Detect calling convention: if second argument is a Dict, use parameterized interface
        elif isinstance(model_or_params, dict):
            # Parameterized interface: params dict is passed directly
            params = self.default_params.copy()
            params.update(model_or_params)
            # Ensure subdataset_name and model are set
            if "subdataset_name" not in params:
                params["subdataset_name"] = "PaulGrahamEssays"
            if "model" not in params:
                params["model"] = "deepseek-chat"  # Default model name
            params.update(kwargs)
        elif model_or_params is not None:
            # Standard interface: model is a string (positional argument)
            model = model_or_params
            # Map subdataset_name to actual dataset name
            if subdataset_name == "NeedleInAHaystack" or subdataset_name is None:
                actual_dataset_name = "PaulGrahamEssays"
            else:
                actual_dataset_name = subdataset_name
            
            # Merge default params with kwargs
            params = self.default_params.copy()
            params["subdataset_name"] = actual_dataset_name
            params["model"] = model
            params.update(kwargs)
        else:
            # No model provided, use defaults
            params = self.default_params.copy()
            params["subdataset_name"] = "PaulGrahamEssays"
            params["model"] = kwargs.pop("model", "deepseek-chat")
            params.update(kwargs)
        
        # Get task type
        task_type = params.get("task_type", TaskType.SINGLE_NEEDLE_RETRIEVAL)
        
        # Check if we're in an async context (for parameterized_benchmark_test.py)
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context, need to use a different approach
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_sync_api_evaluation, client, params, task_type)
                return future.result()
        except RuntimeError:
            # No running event loop, safe to use asyncio.run
            return self._run_sync_api_evaluation(client, params, task_type)
    
    def _run_sync_api_evaluation(self, client: Client, params: Dict, task_type: str) -> Dict:
        """Run synchronous API evaluation by creating a new event loop."""
        # Run async evaluation
        if task_type == TaskType.SINGLE_NEEDLE_RETRIEVAL:
            return asyncio.run(self._evaluate_api_single_needle_retrieval(client, params))
        elif task_type == TaskType.MULTI_NEEDLE_RETRIEVAL:
            return asyncio.run(self._evaluate_api_multi_needle_retrieval(client, params))
        elif task_type == TaskType.MULTI_NEEDLE_REASONING:
            return asyncio.run(self._evaluate_api_multi_needle_reasoning(client, params))
        elif task_type == TaskType.ANCESTRAL_TRACE_CHALLENGE:
            return asyncio.run(self._evaluate_api_ancestral_trace_challenge(client, params))
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def evaluate_needlebench_v2(self, client: Client = None, model=None, tokenizer=None, params: Dict = None) -> Dict:
        """
        Evaluate using NeedleBench V2's balanced scoring system.
        Overall score is the simple average of three main tasks:
        - Single Needle Retrieval
        - Multi Needle Retrieval  
        - Multi Needle Reasoning
        Each task gets equal weight.
        """
        if params is None:
            params = self.default_params
        
        # Store original task type
        original_task_type = params.get("task_type", TaskType.SINGLE_NEEDLE_RETRIEVAL)
        
        # Evaluate each task type
        task_results = {}
        
        # Single Needle Retrieval
        params["task_type"] = TaskType.SINGLE_NEEDLE_RETRIEVAL
        if client:
            task_results["single_needle_retrieval"] = await self.evaluate_api_llm(client, params)
        else:
            task_results["single_needle_retrieval"] = await self.evaluate_local_llm(model, tokenizer, params)
        
        # Multi Needle Retrieval
        params["task_type"] = TaskType.MULTI_NEEDLE_RETRIEVAL
        if client:
            task_results["multi_needle_retrieval"] = await self.evaluate_api_llm(client, params)
        else:
            task_results["multi_needle_retrieval"] = await self.evaluate_local_llm(model, tokenizer, params)
        
        # Multi Needle Reasoning
        params["task_type"] = TaskType.MULTI_NEEDLE_REASONING
        if client:
            task_results["multi_needle_reasoning"] = await self.evaluate_api_llm(client, params)
        else:
            task_results["multi_needle_reasoning"] = await self.evaluate_local_llm(model, tokenizer, params)
        
        # Calculate overall score (simple average of three tasks)
        scores = [result["score"] for result in task_results.values()]
        overall_score = sum(scores) / len(scores)
        
        # Restore original task type
        params["task_type"] = original_task_type
        
        return {
            "overall_score": overall_score,
            "task_scores": {
                "single_needle_retrieval": task_results["single_needle_retrieval"]["score"],
                "multi_needle_retrieval": task_results["multi_needle_retrieval"]["score"],
                "multi_needle_reasoning": task_results["multi_needle_reasoning"]["score"]
            },
            "task_results": task_results,
            "scoring_system": "NeedleBench V2 Balanced Scoring (equal weight average)"
        }
    
    async def _evaluate_api_single_needle_retrieval(self, client: Client, params: Dict) -> Dict:
        """Evaluate single needle retrieval task for API models."""
        # Get parameters
        num_needles = params.get("num_needles", 1)
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        model_name = params.get("model_name", "gpt-3.5-turbo")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Get needles and question
        if num_needles > 1:
            needles = self._get_needles_list(num_needles, language)
            # Check needle token length
            needle_tokens_total = sum(len(tokenizer_tiktoken.encode(n)) for n in needles)
            print(f"[NeedleBench] Using {num_needles} needles, total token length: {needle_tokens_total} tokens")
            context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
            needle_for_eval = needles[0]  # Use first needle for evaluation
        else:
            needle = self._get_random_needle(language)
            needle_tokens = len(tokenizer_tiktoken.encode(needle))
            print(f"[NeedleBench] Using single needle, token length: {needle_tokens} tokens, needle: {needle[:50]}...")
            context_with_needles = self._insert_needle(context, needle, depth_percent)
            needle_for_eval = needle
        
        question = self.default_questions[language]
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        response = await self._call_api_model(client, model_name, prompt)
        
        # Evaluate response
        score = self._evaluate_response(response, needle_for_eval, language)
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_needle": needle_for_eval,
            "success": True,
            "task_type": "SINGLE_NEEDLE_RETRIEVAL"
        }
    
    async def _evaluate_api_multi_needle_retrieval(self, client: Client, params: Dict) -> Dict:
        """Evaluate multi needle retrieval task for API models."""
        # Get parameters
        num_needles = params.get("num_needles", 3)
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        model_name = params.get("model_name", "gpt-3.5-turbo")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Get needles and question
        needles = self._get_needles_list(num_needles, language)
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        needle_for_eval = needles[0]  # Use first needle for evaluation
        
        question = self.default_questions[language]
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        response = await self._call_api_model(client, model_name, prompt)
        
        # Evaluate response
        score = self._evaluate_response(response, needle_for_eval, language)
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_needle": needle_for_eval,
            "needles": needles,
            "success": True,
                "task_type": TaskType.MULTI_NEEDLE_RETRIEVAL
        }
    
    async def _evaluate_api_multi_needle_reasoning(self, client: Client, params: Dict) -> Dict:
        """Evaluate multi needle reasoning task for API models."""
        # Get parameters
        num_needles = params.get("num_needles", 3)
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        model_name = params.get("model_name", "gpt-3.5-turbo")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Generate ATC needles for reasoning
        atc_data = self._generate_atc_needles(num_needles, language)
        needles = atc_data['needles']
        expected_answer = atc_data['answer']
        question = atc_data['retrieval_question']
        
        # Insert needles into context
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        response = await self._call_api_model(client, model_name, prompt)
        
        # Evaluate response using ATC evaluation logic
        score = self._evaluate_atc_response(response, expected_answer)
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_answer": expected_answer,
            "needles": needles,
            "success": True,
            "task_type": TaskType.MULTI_NEEDLE_REASONING
        }
    
    async def _evaluate_api_ancestral_trace_challenge(self, client: Client, params: Dict) -> Dict:
        """Evaluate ancestral trace challenge (ATC) task for API models."""
        # Get parameters
        num_needles = params.get("num_needles", 5)  # ATC typically uses more needles
        language = params.get("language", "English")
        context_length = params.get("context_length", 32768)
        depth_percent = params.get("depth_percent", 50)
        subdataset_name = params.get("subdataset_name", "PaulGrahamEssays")
        model_name = params.get("model_name", "gpt-3.5-turbo")
        
        # Load dataset
        context = self._load_dataset(subdataset_name)
        
        # Truncate context to desired length
        tokenizer_tiktoken = tiktoken.get_encoding("cl100k_base")
        context_tokens = tokenizer_tiktoken.encode(context)
        if len(context_tokens) > context_length:
            context_tokens = context_tokens[:context_length]
            context = tokenizer_tiktoken.decode(context_tokens)
        
        # Generate ATC needles
        atc_data = self._generate_atc_needles(num_needles, language)
        needles = atc_data['needles']
        expected_answer = atc_data['answer']
        question = atc_data['retrieval_question']
        
        # Insert needles into context
        context_with_needles = self._insert_multiple_needles(context, needles, depth_percent)
        
        # Prepare prompt
        prompt = self._prepare_prompt(context_with_needles, question, language)
        
        # Call model
        response = await self._call_api_model(client, model_name, prompt)
        
        # Evaluate response using ATC evaluation logic
        score = self._evaluate_atc_response(response, expected_answer)
        
        # Return results
        return {
            "model": model_name,
            "dataset": subdataset_name,
            "language": language,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "num_needles": num_needles,
            "score": score,
            "response": response,
            "expected_answer": expected_answer,
            "needles": needles,
            "success": True,
            "task_type": TaskType.ANCESTRAL_TRACE_CHALLENGE
        }
    
    def evaluate_with_fallback(
        self,
        model_or_client: Any,
        model_name: str,
        subdataset_name: str = "PaulGrahamEssays",
        *args,
        **kwargs
    ) -> Dict:
        """Evaluate with automatic fallback from local model to API."""
        try:
            # Try local model evaluation first
            if hasattr(model_or_client, 'generate') and hasattr(model_or_client, 'config'):
                # This is likely a local model
                tokenizer = kwargs.get('tokenizer')
                if tokenizer is None:
                    raise ValueError("Tokenizer is required for local model evaluation")
                
                return self.evaluate_local_llm(model_or_client, tokenizer, subdataset_name, *args, **kwargs)
            
            # Fall back to API evaluation
            return self.evaluate_api_llm(model_or_client, model_name, subdataset_name, *args, **kwargs)
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return {
                "model": model_name,
                "dataset": subdataset_name,
                "error": str(e),
                "success": False
            }

# Example usage
if __name__ == "__main__":
    # Create benchmarker
    benchmarker = NeedleInHaystackBenchmarker()
    
    # Example: Test with a simple prompt
    context = "This is a test context. " * 100
    needle = "The secret needle is here."
    question = "What is the secret needle?"
    
    context_with_needle = benchmarker._insert_needle(context, needle, 50)
    prompt = benchmarker._prepare_prompt(context_with_needle, question)
    
    print("Test prompt:")
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("\n" + "="*50 + "\n")
    
    # Test evaluation
    test_response = "The secret needle is here."
    score = benchmarker._evaluate_response(test_response, needle)
    print(f"Test evaluation score: {score}")
