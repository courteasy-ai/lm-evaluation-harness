import time
import requests
import re

class LLMJudge:
    """
    Class that uses an LLM to compare and judge responses from different models.
    """
    
    def __init__(self, api_url, api_key, model="gpt-4o", debug=False):
        """
        Initialize the LLM Judge.
        
        Args:
            api_url (str): URL for the LLM API
            api_key (str): API key for authentication
            model (str): Model to use for judging
            debug (bool): Whether to print debug information
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.debug = debug
    
    def log(self, message):
        """Helper method to print debug messages"""
        if self.debug:
            print(f"[LLM Judge DEBUG] {message}")
    
    # need to change this based on the feedback
    def compare_responses(self, question, response_a, response_b, model_a_name="Model A", model_b_name="Model B"):
        """
        Compare two model responses and provide a judgment on which is better.
        
        Args:
            question (str): The original question
            response_a (str): Response from first model
            response_b (str): Response from second model
            model_a_name (str): Name of the first model
            model_b_name (str): Name of the second model
            
        Returns:
            dict: Judgment results containing winner, scores, and reasoning
        """
        # Define the prompt template for judgment
        prompt = f"""You are an impartial judge evaluating the quality of answers to legal questions. You will be given a question and two answers from different AI assistants. Your task is to decide which answer is better and provide a detailed explanation for your decision.

            QUESTION:
            {question}

            {model_a_name} ANSWER:
            {response_a}

            {model_b_name} ANSWER:
            {response_b}

            Please evaluate both answers based on the following criteria:
            1. Correctness and accuracy of the legal information
            2. Comprehensiveness of the answer
            3. Clarity and organization
            4. Relevance to the question asked

            For each answer, provide a score from 1-10 for each criterion, where 10 is the highest score.
            Then declare which answer is better overall and explain your reasoning.

            Your response should be structured as follows:
            SCORES FOR {model_a_name}:
            - Correctness: [score/10]
            - Comprehensiveness: [score/10]
            - Clarity: [score/10]
            - Relevance: [score/10]
            - TOTAL: [sum of scores/50]

            SCORES FOR {model_b_name}:
            - Correctness: [score/10]
            - Comprehensiveness: [score/10]
            - Clarity: [score/10]
            - Relevance: [score/10]
            - TOTAL: [sum of scores/50]

            WINNER: [{model_a_name} or {model_b_name}]

            REASONING:
            [Your detailed explanation for why one answer is better than the other]
        """
        
        self.log("Sending judgment request to LLM...")
        
        # Prepare request for the LLM API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert legal assistant that evaluates responses to legal questions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,  # Low temperature for more consistent evaluations
            "max_tokens": 1500
        }
        
        try:
            start_time = time.time()
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120  # Judging might take some time
            )
            response.raise_for_status()
            
            elapsed_time = time.time() - start_time
            self.log(f"Judgment received in {elapsed_time:.2f} seconds")
            
            # Parse the response to extract the judgment
            judgment_text = response.json()["choices"][0]["message"]["content"]
            
            # Parse the judgment to extract structured information
            result = self._parse_judgment(judgment_text, model_a_name, model_b_name)
            result["raw_judgment"] = judgment_text
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"LLM Judge API request error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response text: {e.response.text}")
            return {"error": str(e)}
        except Exception as e:
            print(f"Error processing judgment: {str(e)}")
            return {"error": str(e)}
    
    def _parse_judgment(self, judgment_text, model_a_name, model_b_name):
        """
        Parse the judgment text to extract structured information.
        
        Args:
            judgment_text (str): The raw judgment text from the LLM
            model_a_name (str): Name of the first model
            model_b_name (str): Name of the second model
            
        Returns:
            dict: Structured judgment information
        """
        result = {
            "winner": None,
            model_a_name: {"total": None, "scores": {}},
            model_b_name: {"total": None, "scores": {}},
            "reasoning": ""
        }
        
        # Try to extract the winner
        winner_match = re.search(r"WINNER:\s*(\w+)", judgment_text)
        if winner_match:
            winner = winner_match.group(1)
            if winner == model_a_name or model_a_name in winner:
                result["winner"] = model_a_name
            elif winner == model_b_name or model_b_name in winner:
                result["winner"] = model_b_name
        
        # Extract scores for model A
        try:
            # Extract totals
            total_a_match = re.search(r"TOTAL:\s*(\d+\.?\d*)/50", judgment_text.split(f"SCORES FOR {model_b_name}")[0])
            if total_a_match:
                result[model_a_name]["total"] = float(total_a_match.group(1))
            
            total_b_match = re.search(r"TOTAL:\s*(\d+\.?\d*)/50", judgment_text.split("WINNER:")[0])
            if total_b_match and "SCORES FOR" in judgment_text.split("WINNER:")[0]:
                result[model_b_name]["total"] = float(total_b_match.group(1))
            
            # Extract individual scores for model A
            for criterion in ["Correctness", "Comprehensiveness", "Clarity", "Relevance", "Citations"]:
                pattern = f"{criterion}:\s*(\d+\.?\d*)/10"
                match = re.search(pattern, judgment_text.split(f"SCORES FOR {model_b_name}")[0])
                if match:
                    result[model_a_name]["scores"][criterion.lower()] = float(match.group(1))
            
            # Extract individual scores for model B
            for criterion in ["Correctness", "Comprehensiveness", "Clarity", "Relevance", "Citations"]:
                pattern = f"{criterion}:\s*(\d+\.?\d*)/10"
                if f"SCORES FOR {model_b_name}" in judgment_text:
                    section = judgment_text.split(f"SCORES FOR {model_b_name}")[1]
                    section = section.split("WINNER:")[0] if "WINNER:" in section else section
                    match = re.search(pattern, section)
                    if match:
                        result[model_b_name]["scores"][criterion.lower()] = float(match.group(1))
        
        except Exception as e:
            self.log(f"Error parsing scores: {e}")
        
        # Extract reasoning
        if "REASONING:" in judgment_text:
            result["reasoning"] = judgment_text.split("REASONING:")[1].strip()
        
        return result