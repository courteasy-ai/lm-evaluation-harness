import json
import time
import requests

class CourtEasyClient:
    """
    Client for interacting with the CourtEasy AI API with support for both standard and streaming responses.
    """
    
    def __init__(self, base_url="https://api.dev-courteasy.in/", debug=False):
        """
        Initialize the CourtEasy client.
        
        Args:
            base_url (str): Base URL for the API
            debug (bool): Whether to print debug information
        """
        self.base_url = base_url
        self.token = None
        self.debug = debug
    
    def log(self, message):
        """Helper method to print debug messages"""
        if self.debug:
            print(f"[CourtEasy DEBUG] {message}")
    
    def authenticate(self, username, password):
        """
        Authenticate with the CourtEasy API to obtain an access token.
        
        Args:
            username (str): User's username
            password (str): User's password
            
        Returns:
            bool: True if authentication was successful, False otherwise
        """
        token_url = f"{self.base_url}/token"
        self.log(f"Authenticating to {token_url}")
        
        # OAuth2 requires form data for authentication
        form_data = {
            'username': username,
            'password': password
        }
        
        try:
            response = requests.post(token_url, data=form_data)
            
            self.log(f"Auth response status: {response.status_code}")
            self.log(f"Raw auth response: {response.text}")
            
            if response.status_code != 200:
                print(f"Authentication failed with status code: {response.status_code}")
                try:
                    print(f"Error details: {response.json()}")
                except:
                    print(f"Response: {response.text}")
                return False
            
            data = response.json()
            self.token = data['access_token']
            print("CourtEasy authentication successful!")
            return True
        
        except requests.exceptions.RequestException as e:
            print(f"Authentication error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    print(f"API error: {error_data.get('detail', 'Unknown error')}")
                except ValueError:
                    print(f"Response text: {e.response.text}")
            return False
    
    def get_response(self, user_query, chat_history=None):
        """
        Send a question to the streaming response API and collect the full response.
        
        Args:
            user_query (str): The question to ask
            chat_history (list, optional): Previous chat history
            
        Returns:
            str: The complete response text, or None if the request failed
        """
        if not self.token:
            print("Not authenticated. Please authenticate first.")
            return None
        
        streaming_url = f"{self.base_url}/get_streaming_response"
        self.log(f"Sending question to streaming endpoint: {streaming_url}")
        
        # Prepare request headers with authorization
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}',
            'Accept': 'text/event-stream'
        }
        
        # Prepare request body
        request_body = {
            'user_query': user_query,
            'chat_history': chat_history or []
        }
        
        self.log(f"Request body: {json.dumps(request_body)}")
        
        try:
            # Make a streaming request
            start_time = time.time()
            response = requests.post(
                streaming_url, 
                headers=headers, 
                json=request_body, 
                stream=True,
                timeout=120  # Streaming might take longer
            )
            
            if response.status_code != 200:
                print(f"Request failed with status code: {response.status_code}")
                print(f"Error details: {response.text}")
                return None
            
            # Initialize the complete response
            complete_response = ""
            raw_json_chunks = []
            
            # Process the streaming response and collect it
            for line in response.iter_lines():
                if line:
                    # SSE format typically starts with "data: " prefix
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data: '):
                        # Extract the actual data portion
                        chunk = decoded_line[6:]  # Remove "data: " prefix
                        if chunk:
                            # Store the raw chunk for debugging
                            raw_json_chunks.append(chunk)
                            
                            # Try to parse the chunk as JSON
                            try:
                                json_chunk = json.loads(chunk)
                                # Extract the content from the delta
                                if 'choices' in json_chunk and len(json_chunk['choices']) > 0:
                                    if 'delta' in json_chunk['choices'][0] and 'content' in json_chunk['choices'][0]['delta']:
                                        content = json_chunk['choices'][0]['delta']['content']
                                        complete_response += content
                                        # Print progress only if debug is enabled
                                        if self.debug:
                                            print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                self.log(f"Could not parse chunk as JSON: {chunk}")
            
            elapsed_time = time.time() - start_time
            self.log(f"Response received in {elapsed_time:.2f} seconds")
            if self.debug:
                print()  # Add new line after streaming output
            
            # If response is empty but we have raw chunks, try another approach
            if not complete_response and raw_json_chunks:
                self.log("Attempting alternate parsing method...")
                
                # The chunks might be concatenated rather than separate JSON objects
                # Try to extract the content differently
                raw_response = "".join(raw_json_chunks)
                
                # Parse raw response as one string
                try:
                    # Create a regular expression to extract all content values
                    import re
                    content_matches = re.findall(r'"content":\s*"([^"]*)"', raw_response)
                    if content_matches:
                        complete_response = "".join(content_matches)
                        self.log(f"Alternate parsing successful. Extracted {len(complete_response)} characters.")
                except Exception as e:
                    self.log(f"Alternate parsing failed: {str(e)}")
            
            return complete_response
        
        except requests.exceptions.Timeout:
            print("Request timed out. The server may be taking too long to process your question.")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response text: {e.response.text}")
            return None
        except Exception as e:
            print(f"Error processing streaming response: {str(e)}")
            return None