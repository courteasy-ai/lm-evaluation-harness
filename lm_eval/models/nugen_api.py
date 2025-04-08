import os
import logging
import re
import json
import time
from functools import cached_property
from typing import Dict, List, Tuple, Union, Optional

from lm_eval.api.registry import register_model
from lm_eval.models.openai_completions import OpenAICompletionsAPI
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_random_exponential,
    retry_if_exception_type,
    retry_if_result,
    before_sleep_log
)

# Import the Azure OpenAI client
from openai import AzureOpenAI
AZURE_AVAILABLE = True

eval_logger = logging.getLogger(__name__)

# Azure OpenAI configuration - can be overridden with environment variables
AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "05d8237b31e44ce9adebe8fd9fc1357e")
AZURE_ENDPOINT = os.environ.get("AZURE_ENDPOINT", "https://courteasy-azure-openai.openai.azure.com/")
AZURE_DEPLOYMENT = os.environ.get("AZURE_DEPLOYMENT", "courteasy-ai-gpt-4o-mini")
AZURE_API_VERSION = os.environ.get("AZURE_API_VERSION", "2024-02-01")
client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=AZURE_ENDPOINT,
            api_version=AZURE_API_VERSION
        )
def extract_answer_using_azure(response_text, question_context=None):
    """
    Use Azure OpenAI to extract the correct option (A, B, C, or D) from a response 
    when pattern matching fails.
    """
    try:
        # Log the context to debug what's being received
        eval_logger.debug(f"Azure extraction context: {question_context[:100] if question_context else 'None'}")
        
        # Enhanced prompt with full context
        full_context = ""
        if question_context:
            # More robust pattern to extract questions and options
            question_match = re.search(r'Question:\s*(.*?)(?=\nA\.|\n\s*A\.)', question_context, re.DOTALL)
            question_text = question_match.group(1).strip() if question_match else ""
            
            # Extract options with a more robust pattern
            options_pattern = r'([A-D])\.\s*(.*?)(?=\s*[A-D]\.|$)'
            options_matches = re.findall(options_pattern, question_context, re.DOTALL)
            
            options = {match[0]: match[1].strip() for match in options_matches}
            
            full_context = f"Question: {question_text}\n\nOptions:\n"
            for opt in ['A', 'B', 'C', 'D']:
                if opt in options:
                    full_context += f"{opt}. {options[opt]}\n"
        
        # Log the extracted context
        eval_logger.debug(f"Extracted context for Azure: {full_context[:200]}...")
        
        # Create a more explicit prompt for Azure
        system_prompt = """
        You are a specialized legal answer extractor with expertise in Indian law. Your task is to identify which option (A, B, C, D) is being selected as the answer in a model's response to a legal question.

        Focus ONLY on determining which option is indicated in the model's response by:
        1. Looking for explicit mentions like "The correct option is: X" or "Answer: X"
        2. Identifying which option (A, B, C, D) the content of the response is describing or supporting
        3. Matching key legal concepts or terminology between the response and a specific option

        If the model's response clearly discusses the content of an option, but doesn't explicitly state the letter, infer which option it's referring to.

        Output ONLY a single letter (A, B, C, or D) representing the answer.
        If you cannot determine with high confidence which option is being selected, respond with 'NO_ANSWER'.

        Examples:
        Example 1:
        - Response: "The principle of res judicata is dealt with in Section 11 of the Civil Procedure Code."
        - Option A: Section 9 of CPC deals with res judicata
        - Option B: Section 10 of CPC deals with res judicata
        - Option C: Section 11 of CPC deals with res judicata
        - Option D: Section 12 of CPC deals with res judicata
        Output: C

        Example 2:
        - Response: "The principle in Kesavananda Bharati established the basic structure doctrine."
        - Option A: Minerva Mills case
        - Option B: Kesavananda Bharati case
        - Option C: Golaknath case
        - Option D: S.R. Bommai case
        Output: B

        Example 3:
        - Response: "The Constitution of India was adopted on November 26, 1949."
        - (No option matches this information)
        Output: NO_ANSWER
        """
        
        # User content with full question and options
        user_content = f"{full_context}\n\nModel Response: {response_text}\n\nBased on the content of the model's response, which option (A, B, C, or D) is it selecting as the answer?"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Make the API call
        completion = client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=messages,
            temperature=0.0,
            max_tokens=50
        )
        
        extracted_answer = completion.choices[0].message.content.strip().upper()
        eval_logger.debug(f"Raw Azure extraction result: '{extracted_answer}'")
        
        # Check for NO_ANSWER response
        if "NO_ANSWER" in extracted_answer:
            eval_logger.info("Azure extraction determined no valid answer could be extracted")
            return "No Response"
            
        # Validate the extraction - look for just the letter
        letter_match = re.search(r'\b([ABCD])\b', extracted_answer)
        if letter_match:
            extracted_letter = letter_match.group(1)
            eval_logger.info(f"Azure extraction (found letter in response): '{extracted_letter}'")
            return extracted_letter
        elif len(extracted_answer) == 1 and extracted_answer in ['A', 'B', 'C', 'D']:
            eval_logger.info(f"Azure extraction (direct letter): '{extracted_answer}'")
            return extracted_answer
        else:
            eval_logger.warning(f"Invalid extraction from Azure: '{extracted_answer}'")
            return "No Response"
            
    except Exception as e:
        eval_logger.error(f"Error in Azure answer extraction: {e}", exc_info=True)
        return "No Response"

def extract_letter_answer(text, question_context=None, max_azure_retries=2):
    """
    Extract the letter answer (A, B, C, or D) from model response using multiple patterns.
    Falls back to Azure OpenAI for difficult cases.
    
    Args:
        text: The model's response to extract from
        question_context: The original question and options (if available)
        
    Returns:
        Uppercase letter or empty string if no match found
    """
    if not text:
        return "No Text Found"
    
    # Check for API failure messages first and return empty string immediately
    if any(failure_msg in text for failure_msg in [
        "Failed to get response", 
        "Failed to get response from intermediate API"
    ]):
        eval_logger.warning(f"API failure detected in response: '{text}'")
        # Return a special marker that indicates this should be retried
        return "_RETRY_NEEDED_"
    
    # Define comprehensive patterns for matching answers
    patterns = [
        # System prompt format pattern (primary pattern to check first)
        r'(?:t|T)he correct answer is ([ABCD])',
        
        # Other common patterns
        r'(?:a|A)nswer(?:\s+is)?:\s*([ABCD])',
        r'(?:a|A)nswer(?:\s+is)?:\s*([abcd])',
        r'(?:f|F)ound:\s*([ABCD])',
        r'(?:a|A)nswer(?:\s+is)?:\s*([ABCD])\s*-?',
        r'(?:a|A)nswer(?:\s+is)?:\s*(?:o|O)ption\s+([ABCD])',
        r'(?:a|A)nswer(?:\s+is)?:\s*(?:o|O)ption\s+([ABCD])\s*-?',
        r'(?:a|A)nswer(?:\s+is)?:\s*(?:o|O)ption\s+([ABCD])\s+(?:r|R)easons:',
        r'(?:a|A)nswer(?:\s+is)?:\s*([abcd])\)',
        r'(?:a|A)nswer(?:\s+is)?:\s*(?:o|O)ption\s+([ABCD])\s+',
        r'(?:a|A)nswer(?:\s+is)?:\s*(?:o|O)ption\s+([ABCD])\s+\(.+?\)\s+(?:r|R)easons:',
        
        # Simpler patterns
        r'(?:o|O)ption\s+([ABCD])\s+is correct',
        r'(?:o|O)ption\s+([ABCD])',
        r'\b([ABCD])\b',  # Standalone A, B, C, or D
        r'^([ABCD])$',    # Just a single letter response
        r'^([abcd])$',    # Lowercase single letter
    ]

    # Try each pattern in order
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            # Return uppercase version of the matched letter
            return match.group(1).upper()
    
    # If nothing found with regex patterns, try first character approach
    # (only if it's a valid option)
    if text and text[0] in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']:
        return text[0].upper()
    
    # If text starts with a space, try second character
    if len(text) > 1 and text[0].isspace() and text[1] in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']:
        return text[1].upper()
    
    # If all pattern matching fails, try Azure OpenAI extraction
    if AZURE_AVAILABLE:
        azure_retries = 0
        while azure_retries <= max_azure_retries:
            eval_logger.info(f"Pattern matching failed, trying Azure extraction (attempt {azure_retries+1}/{max_azure_retries+1}) for: '{text[:100]}...'")
            
            # For retries, try with different temperature values
            temperature = min(0.2 * azure_retries, 0.7)  # Increase temperature slightly on retries
            
            azure_result = extract_answer_using_azure(text, question_context, temperature=temperature)
            
            if azure_result and azure_result != "No Response":
                return azure_result
                
            azure_retries += 1
            
            if azure_retries <= max_azure_retries:
                eval_logger.info(f"Azure extraction returned 'No Response', retrying with temperature={temperature}")
                time.sleep(1)  # Small delay between retries
    
    # If all else fails, return empty string
    return ""

# Global dictionary to store question contexts for each request
question_contexts = {}

@register_model("nugen")
class NugenAPI(OpenAICompletionsAPI):
    """API client for Nugen's legal model with Azure fallback for answer extraction."""
    
    def __init__(
        self,
        model="nugen-legal-india",
        base_url="https://api.nugen.in/inference/completions",
        tokenizer_backend=None,
        max_retries=3,
        min_retry_wait=2,
        max_retry_wait=60,
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            model=model,
            tokenizer_backend=tokenizer_backend,
            **kwargs
        )
        self.max_retries = max_retries
        self.min_retry_wait = min_retry_wait
        self.max_retry_wait = max_retry_wait
    
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
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=1, min=1, max=20),
        retry=(
            retry_if_exception_type((ConnectionError, TimeoutError, ValueError)) |
            retry_if_result(lambda x: isinstance(x, str) and "rate_limit_exceeded" in x.lower())
        ),
        before_sleep=before_sleep_log(eval_logger, logging.INFO),
    )
    def model_call(self, messages, **kwargs):
        """Make API call with rate limiting and retry logic."""
        eval_logger.debug(f"Sending request to Nugen API: {messages}")
        request_id = kwargs.get('request_id', None)
        eval_logger.debug(f"Request ID in model_call: {request_id}")
        
        try:
            # Reduce sleep time to speed up processing
            time.sleep(0.2)  # Reduced from 1.0
            
            response = super().model_call(messages, **kwargs)
            
            # Check for API failure in the response content
            if isinstance(response, dict) and response.get('choices') and response['choices'][0].get('text'):
                text = response['choices'][0]['text']
                if "Failed to get response from intermediate API" in text:
                    eval_logger.warning("Intermediate API failure detected, will retry")
                    raise ValueError("Intermediate API failure")  # This will trigger a retry
            
            # Check for rate limit errors in the response content
            if isinstance(response, dict) and response.get('error', {}).get('type') == 'rate_limit_exceeded':
                eval_logger.warning("Rate limit exceeded, will retry after backoff")
                return "rate_limit_exceeded"  # This will trigger a retry
                
            eval_logger.debug(f"Received response from Nugen API: {response}")
            return response
        except Exception as e:
            # Check for rate limit keywords in the error
            if isinstance(e, ValueError) and any(
                kw in str(e).lower() for kw in ["rate", "limit", "too many", "requests", "throttle", "intermediate api failure"]
            ):
                eval_logger.warning(f"Error requiring retry: {e}")
                raise  # This will be caught by retry decorator
            
            eval_logger.error(f"Error from Nugen API: {e}")
            raise
    
    def loglikelihood(self, requests):
        """Capture question context when processing loglikelihood requests."""
        eval_logger.info(f"Processing {len(requests)} loglikelihood requests")
        for request in requests:
            if hasattr(request, 'args') and hasattr(request, 'request_id'):
                args = request.args
                if args and len(args) >= 1:
                    question_contexts[request.request_id] = args[0]
                    eval_logger.debug(f"Stored context for request {request.request_id}: {args[0][:100]}...")
        
        return super().loglikelihood(requests)
    
    def generate_until(self, requests):
        """Capture question context when processing generate_until requests."""
        eval_logger.info(f"Processing {len(requests)} generate_until requests")
        for i, request in enumerate(requests):
            # Add debugging for request structure
            eval_logger.debug(f"Request {i} type: {type(request)}")
            eval_logger.debug(f"Request {i} has args: {hasattr(request, 'args')}")
            eval_logger.debug(f"Request {i} has request_id: {hasattr(request, 'request_id')}")
            
            if hasattr(request, 'args'):
                args = request.args
                # Ensure request_id exists - if not, generate one
                request_id = getattr(request, 'request_id', f"req_{i}")
                if not hasattr(request, 'request_id'):
                    request.request_id = request_id
                    eval_logger.info(f"Created new request_id: {request_id}")
                
                if args and len(args) >= 1:
                    # Store context with request_id
                    question_contexts[request_id] = args[0]
                    eval_logger.debug(f"Stored context for request {request_id}: {args[0][:100]}...")
        
        # Call the parent method to process the request
        return super().generate_until(requests)
    
    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """Parse generations from Nugen API responses and extract letter answers."""
        eval_logger.debug(f"Parsing generations")
        res = []
        full_responses = []
        request_id = kwargs.get('request_id', None)
        if request_id is None and 'args' in kwargs:
            request_id = kwargs['args'].get('request_id', None)
        
        eval_logger.debug(f"Request ID in parse_generations: {request_id}")
        
        direct_context = None
        if 'args' in kwargs and 'context' in kwargs['args']:
            direct_context = kwargs['args']['context']
            eval_logger.info(f"Using direct context from kwargs")
        # More robust context retrieval
        question_context = None
        # if request_id and request_id in question_contexts:
        #     question_context = question_contexts[request_id]
        #     eval_logger.debug(f"Found context for request {request_id}: {question_context[:100]}...")
        # else:
        #     eval_logger.warning(f"No context found for request_id: {request_id}")
        
        if question_context is None and direct_context is not None:
            question_context = direct_context
            eval_logger.info(f"Using direct context instead of stored context")
            
        if not isinstance(outputs, list):
            outputs = [outputs]
            
        for out in outputs:
            if 'choices' in out and isinstance(out['choices'], list):
                for choice in out['choices']:
                    if 'text' in choice:
                        # Store the full raw response
                        raw_text = choice['text'].strip()
                        full_responses.append(raw_text)
                        letter = extract_letter_answer(raw_text)

                        # Check for API failure messages first
                        if letter == "_RETRY_NEEDED_":
                            eval_logger.warning("API failure detected, flagging for retry")
                            res.append("")  # Empty string will be interpreted as a failure
                            kwargs['retry_needed'] = True  # Flag to trigger retry at a higher level
                            continue
                        
                        # First try direct pattern matching
                        
                        # If direct pattern matching fails, try Azure with complete context
                        if not letter and question_context:
                            eval_logger.info(f"Pattern matching failed, trying Azure extraction with context for: '{raw_text[:100]}...'")
                            letter = extract_answer_using_azure(raw_text, question_context)
                        
                        if letter:
                            eval_logger.info(f"Extracted answer: '{letter}' from '{raw_text}'")
                            res.append(letter)
                        else:
                            eval_logger.warning(f"Failed to extract letter from: '{raw_text[:100]}...'")
                            res.append("")
                    else:
                        eval_logger.warning("No 'text' field in response choice")
                        res.append("")
                        full_responses.append("")
            else:
                eval_logger.warning(f"Unexpected output format: {out}")
                res.append("")
                full_responses.append("")
        
        # Clean up the question context after use
        if request_id and request_id in question_contexts:
            del question_contexts[request_id]
        
        # Store the full responses in the kwargs for later access
        if 'args' in kwargs:
            kwargs['args']['full_responses'] = full_responses
        else:
            kwargs['full_responses'] = full_responses
                    
        return res