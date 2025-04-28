import pandas as pd
import requests
import os
import time
from tqdm import tqdm

def query_nugen_api(question, api_token, model="llama-v3p1-405b-instruct", max_tokens=400, temperature=0.0):
    """
    Query the Nugen API with the given question.
    
    Args:
        question (str): The question to ask the model
        api_token (str): Your Nugen API token
        model (str): The model to use
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature parameter for generation
        
    Returns:
        str: The model's response
    """
    url = "https://api.nugen.in/inference/completions"
    api_token = "nugen-CnStpNdbBczk3d8SZMhmnw"
    payload = {
        "max_tokens": max_tokens,
        "model": model,
        "prompt": question,
        "temperature": temperature
    }
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()["choices"][0]["text"]
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
        return f"Error: {str(e)}"
    except (KeyError, IndexError) as e:
        print(f"Error parsing API response: {e}")
        print(f"Response content: {response.text}")
        return f"Error parsing response: {str(e)}"

def process_excel_file(input_file, output_file, api_token, model="llama-v3p1-405b-instruct", max_tokens=400, temperature=0.0, num_questions=None, start_idx=0):
    """
    Process the Excel file, query the Nugen API for each question, and save results.
    
    Args:
        input_file (str): Path to the input Excel file
        output_file (str): Path to save the output Excel file
        api_token (str): Your Nugen API token
        model (str): The model to use
        max_tokens (int): Maximum number of tokens to generate
        temperature (float): Temperature parameter for generation
        num_questions (int, optional): Number of questions to process. If None, process all questions.
        start_idx (int, optional): Index to start processing from. Default is 0.
    """
    # Read the Excel file
    print(f"Reading input file: {input_file}")
    df = pd.read_excel(input_file)
    
    # Determine the number of questions to process
    total_questions = len(df)
    if num_questions is None:
        num_questions = total_questions - start_idx
    else:
        num_questions = min(num_questions, total_questions - start_idx)
    
    end_idx = start_idx + num_questions
    
    print(f"Processing {num_questions} questions (from index {start_idx} to {end_idx-1})...")
    
    # Create a new dataframe with only the rows we'll process
    output_df = df.iloc[start_idx:end_idx].copy()
    
    # Add a new column for model generated answers if it doesn't exist
    if 'Model Generated Answer' not in output_df.columns:
        output_df['Model Generated Answer'] = ""
    
    # Process specified rows
    for i, idx in enumerate(tqdm(range(start_idx, end_idx))):
        # Get the enhanced question
        question = df.loc[idx, 'Enhanced Question']
        
        # Remove any extra quotes if present
        if isinstance(question, str):
            question = question.strip('"')
        
        # Query the API
        model_answer = query_nugen_api(
            question=question,
            api_token=api_token,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        # Save the result to the output dataframe
        output_df.iloc[i, output_df.columns.get_loc('Model Generated Answer')] = model_answer
        
        # Add a small delay to avoid hitting rate limits
        time.sleep(0.5)
    
    # Rename columns for clarity if needed
    if 'Answer' in output_df.columns and 'Target Answer' not in output_df.columns:
        output_df = output_df.rename(columns={'Answer': 'Target Answer'})
    
    # Save to a new Excel file
    print(f"Saving results to: {output_file}")
    output_df.to_excel(output_file, index=False)
    print("Done!")
    print(f"Processed {num_questions} questions out of {total_questions} total questions.")

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "generated_legal_queries_with_citations.xlsx"
    OUTPUT_FILE = "nugen_evaluation_results.xlsx"
    MODEL = "llama-v3p1-405b-instruct"
    MAX_TOKENS = 400
    TEMPERATURE = 0.0
    
    # Number of questions to evaluate (None for all questions)
    NUM_QUESTIONS = 10  # Change this value to limit the number of questions
    START_IDX = 0  # Change this value to start from a specific question
    
    # Get API token from environment variable or use the default one
    api_token = os.environ.get("NUGEN_API_TOKEN", "nugen-CnStpNdbBczk3d8SZMhmnw")
    
    # Process the file
    process_excel_file(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        api_token=api_token,
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        num_questions=NUM_QUESTIONS,
        start_idx=START_IDX
    )