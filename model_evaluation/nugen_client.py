import time
import requests

class NugenClient:
    """
    Client for interacting with the Nugen API.
    """
    
    def __init__(self, api_token, debug=False):
        """
        Initialize the Nugen client.
        
        Args:
            api_token (str): API token for authentication
            debug (bool): Whether to print debug information
        """
        self.api_token = api_token
        self.debug = debug
    
    def log(self, message):
        """Helper method to print debug messages"""
        if self.debug:
            print(f"[Nugen DEBUG] {message}")
    
    def get_response(self, question, model="llama-v3p1-405b-instruct", max_tokens=400, temperature=0.0):
        """
        Query the Nugen API with the given question.
        
        Args:
            question (str): The question to ask the model
            model (str): The model to use
            max_tokens (int): Maximum number of tokens to generate
            temperature (float): Temperature parameter for generation
            
        Returns:
            str: The model's response or None if request failed
        """
        url = "https://api.nugen.in/inference/completions"
        
        payload = {
            "max_tokens": max_tokens,
            "model": model,
            "prompt": question,
            "temperature": temperature
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        self.log(f"Sending request to Nugen API with model: {model}")
        self.log(f"Question: {question}")
        
        try:
            start_time = time.time()
            response = requests.request("POST", url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            elapsed_time = time.time() - start_time
            self.log(f"Response received in {elapsed_time:.2f} seconds")
            
            result = response.json()["choices"][0]["text"]
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"Nugen API request error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response text: {e.response.text}")
            return None
        except (KeyError, IndexError) as e:
            print(f"Error parsing Nugen API response: {e}")
            if 'response' in locals():
                print(f"Response content: {response.text}")
            return None