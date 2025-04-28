import json
import time
import requests
import copy
import os
import pandas as pd
import openpyxl
from openpyxl.styles import Alignment

class CourtEasyClient:
    """
    Client for interacting with the CourtEasy AI API with support for both standard and streaming responses.
    """
    
    def __init__(self, base_url="http://localhost:8001", debug=True):
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
            print(f"[DEBUG] {message}")
    
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
            print("Authentication successful!")
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
    
    def get_streaming_response(self, user_query, chat_history=None):
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
                                        # Print progress as it comes in
                                        print(content, end='', flush=True)
                            except json.JSONDecodeError:
                                self.log(f"Could not parse chunk as JSON: {chunk}")
            
            elapsed_time = time.time() - start_time
            self.log(f"Response received in {elapsed_time:.2f} seconds")
            print()  # Add new line after streaming output
            
            # Print brief summary
            print(f"Received complete response from streaming endpoint ({len(complete_response)} characters)")
            
            # If response is empty but we have raw chunks, try another approach
            if not complete_response and raw_json_chunks:
                print("Attempting alternate parsing method...")
                
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
                        print(f"Alternate parsing successful. Extracted {len(complete_response)} characters.")
                except Exception as e:
                    print(f"Alternate parsing failed: {str(e)}")
            
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


def process_excel_file(input_file_path, output_file_path, api_url, username, password, limit=None):
    """
    Process an Excel file containing legal queries, get answers using the CourtEasy API,
    and create a new Excel file with the original questions, provided answers, and model answers.
    
    Args:
        input_file_path (str): Path to the input Excel file
        output_file_path (str): Path to the output Excel file
        api_url (str): URL of the CourtEasy API
        username (str): CourtEasy API username
        password (str): CourtEasy API password
        limit (int, optional): Limit the number of questions to process
    """
    print(f"Processing Excel file: {input_file_path}")
    
    # Read the input Excel file
    try:
        df = pd.read_excel(input_file_path)
        print(f"Successfully read Excel file with {len(df)} rows")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return
    
    # Limit the number of questions if specified
    if limit is not None and limit > 0:
        df = df.head(limit)
        print(f"Limiting to first {limit} questions")
    
    # Initialize the CourtEasy client
    client = CourtEasyClient(base_url=api_url, debug=True)
    
    # Authenticate
    if not client.authenticate(username, password):
        print("Authentication failed. Exiting.")
        return
    
    # Create a new column for model answers
    df['Model Answer'] = None
    
    # Process each question
    for index, row in df.iterrows():
        question = row['Enhanced Question']
        print(f"\nProcessing question {index+1}/{len(df)}: {question}")
        
        # Get the answer from the API
        model_answer = client.get_streaming_response(user_query=question)
        
        if model_answer:
            print(f"Successfully received answer for question {index+1}")
            # Store the model's answer
            df.at[index, 'Model Answer'] = model_answer
        else:
            print(f"Failed to get answer for question {index+1}")
            df.at[index, 'Model Answer'] = "Error: No response received from API"
        
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    # Save the results to a new Excel file
    try:
        # Save to Excel with better formatting
        with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Results')
            
            # Get the workbook and the worksheet
            workbook = writer.book
            worksheet = writer.sheets['Results']
            
            # Adjust column widths
            worksheet.column_dimensions['A'].width = 40  # Question
            worksheet.column_dimensions['B'].width = 40  # Enhanced Question
            worksheet.column_dimensions['C'].width = 80  # Answer
            worksheet.column_dimensions['D'].width = 20  # Domain
            worksheet.column_dimensions['E'].width = 20  # Persona
            worksheet.column_dimensions['F'].width = 40  # Citations
            worksheet.column_dimensions['G'].width = 80  # Model Answer
            
            # Enable wrap text for the answer columns
            for row in worksheet.iter_rows(min_row=2, max_row=len(df)+1, min_col=3, max_col=3):  # Original Answer
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
            
            for row in worksheet.iter_rows(min_row=2, max_row=len(df)+1, min_col=7, max_col=7):  # Model Answer
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical='top')
            
        print(f"Results saved to: {output_file_path}")
    except Exception as e:
        print(f"Error saving results to Excel file: {e}")


def main():
    # Settings
    input_file_path = "generated_legal_queries_with_citations.xlsx"
    output_file_path = "legal_queries_with_model_answers.xlsx"
    api_url = "https://api.dev-courteasy.in/"  # Update this URL as needed
    
    print(f"Using API URL: {api_url}")
    
    # Get user input for authentication
    username = input("Enter CourtEasy username: ")
    password = input("Enter CourtEasy password: ")
    
    # Ask if user wants to limit the number of questions
    limit_input = input("Enter number of questions to process (leave blank for all): ")
    limit = int(limit_input) if limit_input.strip() else None
    
    # Process the Excel file
    process_excel_file(
        input_file_path=input_file_path,
        output_file_path=output_file_path,
        api_url=api_url,
        username=username,
        password=password,
        limit=limit
    )


if __name__ == "__main__":
    main()