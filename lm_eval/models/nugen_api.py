import os
import logging
import re
from functools import cached_property
from typing import Dict, List, Tuple, Union, Optional

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import OpenAICompletionsAPI

eval_logger = logging.getLogger(__name__)

def extract_letter_answer(text):
    """Extract the letter answer (A, B, C, or D) from model response"""
    if not text:
        return ""
    
    # First, try to extract "Answer is: X" pattern
    answer_is_match = re.search(r'Answer is:\s*([ABCD])', text)
    if answer_is_match:
        return answer_is_match.group(1)
    
    # If that fails, check if the response starts with a letter
    if text[0] in ['A', 'B', 'C', 'D']:
        return text[0]
    
    # If that fails, check if there's a letter after a space at the beginning
    if text.startswith(' ') and len(text) > 1 and text[1] in ['A', 'B', 'C', 'D']:
        return text[1]
    
    # If that fails, look for standalone A, B, C, or D in the text
    letter_match = re.search(r'\b([ABCD])\b', text)
    if letter_match:
        return letter_match.group(1)
    
    # If all else fails, return empty string
    return ""

@register_model("nugen")
class NugenAPI(OpenAICompletionsAPI):
    """API client for Nugen's legal model."""
    
    def __init__(
        self,
        model="llama-v3p1-405b-instruct",
        base_url="https://api.nugen.in/inference/completions",
        tokenizer_backend=None,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            tokenizer_backend=tokenizer_backend,
            **kwargs
        )
    
    @cached_property
    def api_key(self):
        key = os.environ.get("NUGEN_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the NUGEN_API_KEY environment variable."
            )
        return key
    
    @property
    def header(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def model_call(self, messages, **kwargs):
        eval_logger.info(f"Sending request to Nugen API: {messages}")
        try:
            response = super().model_call(messages, **kwargs)
            eval_logger.info(f"Received response from Nugen API: {response}")
            return response
        except Exception as e:
            eval_logger.error(f"Error from Nugen API: {e}")
            raise
    
    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """Parse generations from Nugen API responses and extract letter answers."""
        eval_logger.info(f"Parsing generations")
        res = []
        
        if not isinstance(outputs, list):
            outputs = [outputs]
            
        for out in outputs:
            if 'choices' in out and isinstance(out['choices'], list):
                for choice in out['choices']:
                    if 'text' in choice:
                        # Extract just the letter answer
                        letter = extract_letter_answer(choice['text'])
                        res.append(letter)
                    else:
                        res.append("")
            else:
                eval_logger.warning(f"Unexpected output format")
                res.append("")
                
        eval_logger.info(f"Extracted answers: {res}")
        return res