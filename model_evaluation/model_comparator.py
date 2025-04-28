import time
import pandas as pd
from tqdm import tqdm
from openpyxl.styles import Alignment

class ModelComparator:
    """
    Class for comparing responses from multiple models using an LLM as judge.
    """
    
    def __init__(self, courteasy_client, nugen_client, llm_judge):
        """
        Initialize the model comparator.
        
        Args:
            courteasy_client: Initialized CourtEasy client
            nugen_client: Initialized Nugen client
            llm_judge: Initialized LLM judge
        """
        self.courteasy_client = courteasy_client
        self.nugen_client = nugen_client
        self.llm_judge = llm_judge
    
    def compare_models_on_dataset(self, input_file, output_file, num_questions=None, start_idx=0, 
                                 nugen_model="llama-v3p1-405b-instruct", max_tokens=400, temperature=0.0):
        """
        Compare models on a dataset of questions.
        
        Args:
            input_file (str): Path to the input Excel file
            output_file (str): Path to save the output Excel file
            num_questions (int, optional): Number of questions to process. If None, process all questions.
            start_idx (int, optional): Index to start processing from. Default is 0.
            nugen_model (str): The Nugen model to use
            max_tokens (int): Maximum number of tokens to generate for Nugen
            temperature (float): Temperature parameter for generation for Nugen
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
        
        # Add new columns for results
        output_df['CourtEasy Answer'] = ""
        output_df['Nugen Answer'] = ""
        output_df['Winner'] = ""
        output_df['CourtEasy Score'] = ""
        output_df['Nugen Score'] = ""
        output_df['Judge Reasoning'] = ""
        
        # Process each question
        for i, idx in enumerate(tqdm(range(start_idx, end_idx), desc="Processing questions")):
            # Get the enhanced question
            question = df.loc[idx, 'Enhanced Question']
            
            # Remove any extra quotes if present
            if isinstance(question, str):
                question = question.strip('"')
            
            print(f"\nProcessing question {i+1}/{num_questions}: {question[:100]}...")
            
            # Get responses from both models
            print("Getting CourtEasy response...")
            courteasy_response = self.courteasy_client.get_response(question)
            
            print("Getting Nugen response...")
            nugen_response = self.nugen_client.get_response(
                question=question,
                model=nugen_model,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            # Save responses to dataframe
            output_df.iloc[i, output_df.columns.get_loc('CourtEasy Answer')] = courteasy_response or "Error: No response received"
            output_df.iloc[i, output_df.columns.get_loc('Nugen Answer')] = nugen_response or "Error: No response received"
            
            # Skip judgment if either response failed
            if not courteasy_response or not nugen_response:
                print("Skipping judgment due to missing response(s)")
                output_df.iloc[i, output_df.columns.get_loc('Winner')] = "N/A - Missing response(s)"
                continue
            
            # Get judgment
            print("Getting LLM judgment...")
            judgment = self.llm_judge.compare_responses(
                question=question,
                response_a=courteasy_response,
                response_b=nugen_response,
                model_a_name="CourtEasy",
                model_b_name="Nugen"
            )
            
            # Save judgment to dataframe
            if "error" not in judgment:
                output_df.iloc[i, output_df.columns.get_loc('Winner')] = judgment["winner"] or "Tie"
                output_df.iloc[i, output_df.columns.get_loc('CourtEasy Score')] = judgment["CourtEasy"]["total"] if judgment["CourtEasy"]["total"] is not None else "N/A"
                output_df.iloc[i, output_df.columns.get_loc('Nugen Score')] = judgment["Nugen"]["total"] if judgment["Nugen"]["total"] is not None else "N/A"
                output_df.iloc[i, output_df.columns.get_loc('Judge Reasoning')] = judgment["reasoning"]
            else:
                output_df.iloc[i, output_df.columns.get_loc('Winner')] = f"Error: {judgment['error']}"
            
            # Add a small delay between questions
            time.sleep(1)
            
            # Save intermediate results every 5 questions
            if (i + 1) % 5 == 0 or i == num_questions - 1:
                print(f"Saving intermediate results at question {i+1}...")
                intermediate_file = output_file.replace(".xlsx", f"_interim_{i+1}.xlsx")
                output_df.to_excel(intermediate_file, index=False)
        
        # Calculate statistics
        total_judgments = sum(1 for winner in output_df['Winner'] if winner not in ["N/A - Missing response(s)", ""] and not str(winner).startswith("Error"))
        courteasy_wins = sum(1 for winner in output_df['Winner'] if winner == "CourtEasy")
        nugen_wins = sum(1 for winner in output_df['Winner'] if winner == "Nugen")
        ties = sum(1 for winner in output_df['Winner'] if winner == "Tie")
        
        print("\n=== Comparison Results ===")
        print(f"Total valid judgments: {total_judgments}")
        print(f"CourtEasy wins: {courteasy_wins} ({courteasy_wins/total_judgments*100:.2f}% of valid judgments)" if total_judgments > 0 else "CourtEasy wins: 0 (0.00%)")
        print(f"Nugen wins: {nugen_wins} ({nugen_wins/total_judgments*100:.2f}% of valid judgments)" if total_judgments > 0 else "Nugen wins: 0 (0.00%)")
        print(f"Ties: {ties} ({ties/total_judgments*100:.2f}% of valid judgments)" if total_judgments > 0 else "Ties: 0 (0.00%)")
        
        # Save final results
        print(f"\nSaving final results to: {output_file}")
        
        # Save with better formatting
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            output_df.to_excel(writer, index=False, sheet_name='Results')
            
            # Get the workbook and the worksheet
            workbook = writer.book
            worksheet = writer.sheets['Results']
            
            # Adjust column widths for important columns
            worksheet.column_dimensions['B'].width = 40  # Enhanced Question
            worksheet.column_dimensions['G'].width = 80  # CourtEasy Answer
            worksheet.column_dimensions['H'].width = 80  # Nugen Answer
            worksheet.column_dimensions['I'].width = 15  # Winner
            worksheet.column_dimensions['J'].width = 15  # CourtEasy Score
            worksheet.column_dimensions['K'].width = 15  # Nugen Score
            worksheet.column_dimensions['L'].width = 80  # Judge Reasoning
            
            # Enable wrap text for text-heavy columns
            for col_letter in ['G', 'H', 'L']:  # CourtEasy Answer, Nugen Answer, Judge Reasoning
                for row in worksheet.iter_rows(min_row=2, max_row=len(output_df)+1, min_col=ord(col_letter)-64, max_col=ord(col_letter)-64):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical='top')
        
        print("Model comparison completed!")
        
        # Add a summary sheet
        self._add_summary_sheet(output_file, courteasy_wins, nugen_wins, ties, total_judgments)
    
    def _add_summary_sheet(self, output_file, courteasy_wins, nugen_wins, ties, total_judgments):
        """
        Add a summary sheet to the output Excel file.
        
        Args:
            output_file (str): Path to the output Excel file
            courteasy_wins (int): Number of CourtEasy wins
            nugen_wins (int): Number of Nugen wins
            ties (int): Number of ties
            total_judgments (int): Total number of valid judgments
        """
        try:
            # Read the existing Excel file
            book = pd.ExcelFile(output_file)
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a') as writer:
                # Create a new DataFrame for the summary
                summary_data = {
                    'Metric': ['Total Valid Judgments', 'CourtEasy Wins', 'Nugen Wins', 'Ties',
                              'CourtEasy Win Rate (%)', 'Nugen Win Rate (%)', 'Tie Rate (%)'],
                    'Value': [
                        total_judgments,
                        courteasy_wins,
                        nugen_wins,
                        ties,
                        f"{courteasy_wins/total_judgments*100:.2f}%" if total_judgments > 0 else "N/A",
                        f"{nugen_wins/total_judgments*100:.2f}%" if total_judgments > 0 else "N/A",
                        f"{ties/total_judgments*100:.2f}%" if total_judgments > 0 else "N/A"
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                
                # Write to a new sheet
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Format the summary sheet
                worksheet = writer.sheets['Summary']
                worksheet.column_dimensions['A'].width = 25
                worksheet.column_dimensions['B'].width = 15
                
                print("Added summary sheet to output file.")
        except Exception as e:
            print(f"Error adding summary sheet: {e}")