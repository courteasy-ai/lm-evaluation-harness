import os
from courteasy_client import CourtEasyClient
from nugen_client import NugenClient
from llm_judge import LLMJudge
from model_comparator import ModelComparator

def main():
    """
    Main function to run the model comparison with hardcoded settings
    using Azure OpenAI's GPT-4o mini as the judge.
    """
    # Set Azure OpenAI environment variables
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
    OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
    OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
    AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    
    # Configuration settings
    # File paths
    input_file = "generated_legal_queries_with_citations.xlsx"
    output_file = "model_comparison_results.xlsx"
       
    # CourtEasy settings
    courteasy_url = "https://api.dev-courteasy.in/"
    courteasy_username = "testuser01"
    courteasy_password = "testpassword123"
    
    # Nugen settings
    nugen_token = os.getenv("nugen_token")
    nugen_model = "llama-v3p1-405b-instruct"
    nugen_max_tokens = 400
    nugen_temperature = 0.0
    
    # LLM Judge settings - Using Azure OpenAI
    judge_url = f"{OPENAI_API_BASE}/openai/deployments/{OPENAI_DEPLOYMENT_NAME}/chat/completions?api-version={OPENAI_API_VERSION}"
    judge_key = AZURE_OPENAI_API_KEY
    judge_model = OPENAI_DEPLOYMENT_NAME  # For Azure, we use the deployment name
    
    # Processing settings
    num_questions = 5  # Change this value as needed
    start_idx = 0
    debug_mode = False  # Set to True for verbose logging
    
    print("=== Model Comparison with LLM Judge (Azure OpenAI) ===")
    
    # Initialize clients
    print("\n=== Initializing Clients ===")
    
    print(f"Initializing CourtEasy client with URL: {courteasy_url}")
    courteasy_client = CourtEasyClient(base_url=courteasy_url, debug=debug_mode)
    
    print(f"Authenticating with CourtEasy using username: {courteasy_username}")
    if not courteasy_client.authenticate(courteasy_username, courteasy_password):
        print("Failed to authenticate with CourtEasy. Exiting.")
        return
    
    print(f"Initializing Nugen client with model: {nugen_model}")
    nugen_client = NugenClient(api_token=nugen_token, debug=debug_mode)
    
    print(f"Initializing Azure OpenAI LLM Judge with deployment: {judge_model}")
    # We need to modify the LLMJudge class to work with Azure OpenAI
    azure_llm_judge = AzureLLMJudge(
        api_url=judge_url,
        api_key=judge_key,
        deployment_name=judge_model,
        debug=debug_mode
    )
    
    # Initialize model comparator
    print("Initializing Model Comparator...")
    comparator = ModelComparator(courteasy_client, nugen_client, azure_llm_judge)
    
    # Run the comparison
    print("\n=== Starting Model Comparison ===")
    print(f"Processing file: {input_file}")
    print(f"Saving results to: {output_file}")
    print(f"Number of questions: {num_questions or 'All'} starting from index {start_idx}")
    
    comparator.compare_models_on_dataset(
        input_file=input_file,
        output_file=output_file,
        num_questions=num_questions,
        start_idx=start_idx,
        nugen_model=nugen_model,
        max_tokens=nugen_max_tokens,
        temperature=nugen_temperature
    )
    
    print("\n=== Model Comparison Complete ===")


class AzureLLMJudge:
    """
    Class that uses Azure OpenAI to compare and judge responses from different models.
    Adapted from the LLMJudge class to work with Azure OpenAI.
    """
    
    def __init__(self, api_url, api_key, deployment_name, debug=False):
        """
        Initialize the Azure LLM Judge.
        
        Args:
            api_url (str): URL for the Azure OpenAI API endpoint
            api_key (str): API key for authentication
            deployment_name (str): The deployment name in Azure
            debug (bool): Whether to print debug information
        """
        self.api_url = api_url
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.debug = debug
    
    def log(self, message):
        """Helper method to print debug messages"""
        if self.debug:
            print(f"[Azure LLM Judge DEBUG] {message}")
    
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
5. Citation of relevant legal sources (if any)

For each answer, provide a score from 1-10 for each criterion, where 10 is the highest score.
Then declare which answer is better overall and explain your reasoning.

Your response should be structured as follows:
SCORES FOR {model_a_name}:
- Correctness: [score/10]
- Comprehensiveness: [score/10]
- Clarity: [score/10]
- Relevance: [score/10]
- Citations: [score/10]
- TOTAL: [sum of scores/50]

SCORES FOR {model_b_name}:
- Correctness: [score/10]
- Comprehensiveness: [score/10]
- Clarity: [score/10]
- Relevance: [score/10]
- Citations: [score/10]
- TOTAL: [sum of scores/50]

WINNER: [{model_a_name} or {model_b_name}]

REASONING:
[Your detailed explanation for why one answer is better than the other]
"""
        
        self.log("Sending judgment request to Azure OpenAI...")
        
        # Prepare request for the Azure OpenAI API
        headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are an expert legal assistant that evaluates responses to legal questions."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,  # Low temperature for more consistent evaluations
            "max_tokens": 1500
        }
        
        try:
            import time
            start_time = time.time()
            import requests
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
            print(f"Azure OpenAI API request error: {e}")
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
        import re
        winner_match = re.search(r"WINNER:\s*(\w+)", judgment_text)
        if winner_match:
            winner = winner_match.group(1)
            if winner == model_a_name or model_a_name in winner:
                result["winner"] = model_a_name
            elif winner == model_b_name or model_b_name in winner:
                result["winner"] = model_b_name
        
        # Extract scores
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


if __name__ == "__main__":
    main()