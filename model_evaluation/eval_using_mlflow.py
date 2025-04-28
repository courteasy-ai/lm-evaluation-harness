import os
import pandas as pd
import numpy as np
import mlflow
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from mlflow.metrics.genai import answer_correctness, answer_similarity, make_genai_metric

# Load environment variables from .env file
load_dotenv()

# Configure Azure OpenAI settings via environment variables
OPENAI_API_TYPE = os.getenv("OPENAI_API_TYPE")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "legal-answers-evaluation")

# Define custom metrics for legal answer evaluation
Completeness = make_genai_metric(
    name="Completeness",
    definition="Evaluates the completeness of an answer based on the target answer. Checks if the provided answer covers all the expected points or not",
    grading_prompt="""
    Evaluate the legal accuracy of the model-generated answer compared to the ground truth answer.
    
    Focus on:
    1. Accuracy of answer by comparing it with target answer. 
    2. Correctness of case law citations
    3. Completeness of an answer. Check if the provuided answer is complete or not. 
    
    Score as follows:
    - Score 1 : If the provided answer is totally incomplete and does not contain any significant information. 
    - Score 2 : If an answer contain very little content but miss out the majority of an answer.
    - Score 3-4: If an answer covers major aspects of question asked but misses out some additional context  
    - Score 5 : The answer is legally accurate and on par with the ground truth. And covers all the aspects of an answer.
    
    Provide a detailed explanation for your score.
    """,
    version="v1",
    model="openai:/gpt-4o-mini",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)

jurisdictional_correctness = make_genai_metric(
    name="jurisdictional_correctness",
    definition="Evaluates how correctly the answer applies law from the relevant jurisdiction and recognizes jurisdictional limitations",
    grading_prompt="""
    Evaluate how well the model-generated answer addresses jurisdictional aspects compared to the ground truth answer.
    
    Focus on:
    1. Correct identification of the applicable jurisdiction (state, federal, international)
    2. Proper application of laws specific to the identified jurisdiction
    3. Recognition of jurisdictional limitations or conflicts
    4. Awareness of variations in law across different jurisdictions when relevant
    5. Appropriate qualification of advice when jurisdictional information is unclear
    
    Score as follows:
    - Score 1: Serious jurisdictional errors that could lead to completely inapplicable legal advice
    - Score 2: Multiple jurisdictional issues or misapplications of law from incorrect jurisdictions
    - Score 3: Some jurisdictional awareness but minor errors or omissions in jurisdiction-specific application
    - Score 4: Good jurisdictional awareness with mostly correct application of jurisdiction-specific law
    - Score 5: Excellent jurisdictional precision matching the ground truth, with proper qualification when needed
    
    Provide a detailed explanation for your score, citing specific examples of correct or incorrect jurisdictional application.
    """,
    version="v1",
    model="openai:/gpt-4o-mini",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)

legal_reasoning_quality = make_genai_metric(
    name="legal_reasoning_quality",
    definition="Evaluates the logical structure, coherence, and soundness of legal arguments presented in the answer",
    grading_prompt="""
    Evaluate the quality of legal reasoning in the model-generated answer compared to the ground truth answer.
    
    Focus on:
    1. Logical structure of legal arguments (premises leading to valid conclusions)
    2. Proper application of legal principles and doctrines
    3. Sound legal analysis that connects facts to relevant law
    4. Identification of key legal issues without overlooking important considerations
    5. Appropriate use of legal reasoning methods (deductive, analogical, policy-based, etc.)
    
    Score as follows:
    - Score 1: Fundamentally flawed legal reasoning with invalid conclusions or logical fallacies
    - Score 2: Weak legal reasoning with significant gaps in analysis or tenuous connections between law and facts
    - Score 3: Adequate legal reasoning but lacking depth or sophistication compared to ground truth
    - Score 4: Good legal reasoning with mostly sound analysis and logical structure
    - Score 5: Excellent legal reasoning comparable to the ground truth, demonstrating sophisticated analysis
    
    Provide a detailed explanation for your score, analyzing specific examples of strong or weak reasoning in the answer.
    """,
    version="v1",
    model="openai:/gpt-4o-mini",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)

overall_quality = make_genai_metric(
    name="overall_quality",
    definition="Evaluates the overall quality of the legal answer considering all factors",
    grading_prompt="""
    Evaluate the overall quality of the model-generated answer compared to the ground truth.
    
    Consider all of these factors:
    1. Legal accuracy
    2. Clarity
    3. Practical usefulness
    4. Professional tone
    5. Explanation 
    
    Score as follows:
    - Score 1: The answer is significantly below the quality of the ground truth
    - Score 2: The answer is acceptable but notably inferior to the ground truth
    - Score 3-4: The answer is good quality, approaching ground truth standards
    - Score 5: The answer is excellent and comparable to the ground truth
    
    Provide a detailed explanation for your score and summarize the key strengths and weaknesses.
    """,
    version="v1",
    model="openai:/gpt-4o-mini",
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)

binary_correctness = make_genai_metric(
    name="binary_correctness",
    definition="This metric evaluates whether the model's answer is correct or incorrect based on the ground truth, providing a binary classification.",
    grading_prompt="""
    Evaluate whether the model's answer is correct or incorrect based on the ground truth.

    First, identify which option (A, B, C, or D) is indicated as correct in the ground truth.

    Then, carefully analyze the model's answer by checking:
    1. If the model explicitly selects an option by letter (A, B, C, or D)
    2. If the model provides content that substantively matches the content of any of the four options

    The model's answer should be considered correct if and only if:
    1. It clearly selects the same option letter (A, B, C, or D) as indicated in the ground truth, OR
    2. The content of the model's answer matches the content of the correct option as specified in the ground truth, even if it doesn't explicitly state the option letter.

    The model's answer should be considered incorrect if:
    1. It explicitly selects a different option letter than the one in the ground truth, OR
    2. The content of the model's answer matches an incorrect option rather than the correct one, OR
    3. It is ambiguous about which option it's selecting, OR
    4. It provides content that doesn't clearly match any of the given options.

    Score as follows:
    - Score 1: The answer is incorrect (does not correspond to the correct option)
    - Score 5: The answer is correct (corresponds to the correct option)

    No other scores should be used - this is a binary classification.
    """,
    version="v1",
    model="openai:/gpt-4o-mini",
    grading_context_columns=["targets"],
    parameters={"temperature": 0.0},
    aggregations=["mean", "variance", "p90"],
    greater_is_better=True,
)


# Function to convert any NumPy values to Python native types
def convert_numpy_types(obj):
    """Convert NumPy data types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Initialize MLflow
def init_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create experiment if it doesn't exist
    if mlflow.get_experiment_by_name(EXPERIMENT_NAME) is None:
        mlflow.create_experiment(EXPERIMENT_NAME)
    
    # Set the active experiment
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking initialized with experiment: {EXPERIMENT_NAME}")

# Load data from Excel file
def load_data(file_path):
    """Load data from the Excel file with model and ground truth answers."""
    try:
        df = pd.read_excel(file_path)
        print(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Prepare data for MLflow evaluation
def prepare_evaluation_data(df, limit=None):
    """Prepare the data in the format required by MLflow's evaluate function."""
    # Limit the number of questions if specified
    if limit is not None and limit > 0:
        df = df.head(limit)
        print(f"Limiting evaluation to first {limit} questions")
    
    # Create evaluation dataframe
    eval_data = pd.DataFrame()
    
    # Add required columns
    eval_data['inputs'] = df['Enhanced Question']
    eval_data['prediction'] = df['Model Answer']
    eval_data['ground_truth'] = df['Answer']
    eval_data['context'] = df['Citations']
    
    # Add metadata columns that might be useful for analysis
    eval_data['question_id'] = df.index
    eval_data['domain'] = df['Domain']
    eval_data['persona'] = df['Persona']
    
    return eval_data

def main():
    """Main function to run the evaluation."""
    print("Starting legal answer evaluation with MLflow and Azure OpenAI")
    
    # Initialize MLflow tracking
    init_mlflow()
    
    # Load the data
    input_file = "legal_queries_with_model_answers.xlsx"
    df = load_data(input_file)
    if df is None:
        return
    
    # Ask for confirmation
    num_questions = len(df)
    print(f"Found {num_questions} questions to evaluate")
    
    limit_input = input(f"How many questions do you want to evaluate? (1-{num_questions}, or 'all'): ")
    if limit_input.lower() != 'all':
        try:
            limit = int(limit_input)
            if 1 <= limit <= num_questions:
                limit = limit
                print(f"Limiting evaluation to first {limit} questions")
            else:
                print(f"Invalid number. Using all {num_questions} questions.")
                limit = None
        except ValueError:
            print(f"Invalid input. Using all {num_questions} questions.")
            limit = None
    else:
        limit = None
    
    # Prepare data for evaluation
    eval_data = prepare_evaluation_data(df, limit)
    print(f"Prepared evaluation data with {len(eval_data)} rows")
    
    # Start MLflow run for evaluation
    with mlflow.start_run(run_name="legal_answer_evaluation") as run:
        print(f"Started MLflow run: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_params({
            "model_name": "CourtEasy API",
            "judge_model": "gpt-4o-mini",
            "num_questions": len(eval_data)
        })
        
        # Run the evaluation using MLflow's evaluate function with custom metrics
        print("Starting evaluation with custom metrics...")
        try:
            eval_results = mlflow.evaluate(
                data=eval_data,
                model=None,  # No model provided since we already have predictions
                predictions="prediction",
                targets="ground_truth",
                evaluators="default",
                evaluator_config={
                    "col_mapping": {
                        "inputs": "inputs",
                        "context": "context"
                    }
                },
                extra_metrics=[
                    answer_correctness(model="openai:/gpt-4o-mini"),
                    answer_similarity(model="openai:/gpt-4o-mini"),
                    Completeness,
                    overall_quality,
                    jurisdictional_correctness,
                    legal_reasoning_quality,
                    binary_correctness
                ]
            )
            
            # Convert NumPy types for safe JSON serialization
            metrics_dict = convert_numpy_types(eval_results.metrics)
            
            # Log metrics summary
            for metric_name, metric_value in metrics_dict.items():
                print(f"{metric_name}: {metric_value}")
            
            # Save results to JSON
            results_json = json.dumps(metrics_dict, indent=2)
            with open("evaluation_metrics.json", "w") as f:
                f.write(results_json)
            mlflow.log_artifact("evaluation_metrics.json")
            
            print("\nEvaluation completed successfully!")
            print(f"Results logged to MLflow run: {run.info.run_id}")
            print(f"Results saved to evaluation_metrics.json")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
