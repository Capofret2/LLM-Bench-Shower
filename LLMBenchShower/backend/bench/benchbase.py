from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM
from typing import Dict


class BaseBench(ABC):
    @abstractmethod
    def evaluate_local_llm(self, model: AutoModelForCausalLM, *args, **kwargs) -> Dict:
        """Evaluate a local LLM model.

        Args:
            model (AutoModelForCausalLM): The local LLM model to evaluate.
            (args, kwargs): Other arguments needed for evaluation.

        Returns:
            The evaluation results.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate_api_llm(self, *args, **kwargs) -> Dict:
        """Evaluate an API LLM model.

        Args:
            Arguments needed for evaluation.

        Returns:
            The evaluation results.
        """
        raise NotImplementedError
