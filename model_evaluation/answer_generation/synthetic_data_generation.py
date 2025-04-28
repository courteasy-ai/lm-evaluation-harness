import pandas as pd
import os
from openai import AzureOpenAI
import time
import random
from pydantic import BaseModel, Field
from typing import List, Optional
from pydantic import create_model
# Configure Azure OpenAI client
client = AzureOpenAI(
    api_key="05d8237b31e44ce9adebe8fd9fc1357e",
    api_version= "2024-12-01-preview",
    azure_endpoint="https://courteasy-azure-openai.openai.azure.com/"
)

# Define structured response models for legal answers with citations
class Citation(BaseModel):
    source: str = Field(..., description="Source of the citation (e.g., act name, section, case name)")
    text: str = Field(..., description="The quoted text from the source")
    reference: str = Field(..., description="Full reference including act, section, year, etc.")

class LegalAnswer(BaseModel):
    answer: str = Field(..., description="Complete answer to the legal query")
    citations: List[Citation] = Field(..., description="Legal citations supporting the answer")


def read_excel_file(file_path):
    """
    Read the Excel file containing existing user queries.
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Successfully read Excel file with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def extract_unique_questions(df, question_column="Question"):
    """
    Extract unique questions from the dataframe to avoid duplication.
    """
    if question_column not in df.columns:
        print(f"Column '{question_column}' not found. Available columns: {df.columns.tolist()}")
        return []
    
    # Extract unique questions and remove any empty ones
    questions = df[question_column].dropna().unique().tolist()
    questions = [q for q in questions if isinstance(q, str) and q.strip()]
    print(f"Extracted {len(questions)} unique questions")
    return questions

def generate_queries(existing_questions, domains=["Family Law"], personas=["Citizens"], questions_per_combination=5):
    """
    Generate new user queries for each combination of domain and persona.
    
    Args:
        existing_questions: List of existing questions to use as examples
        domains: List of legal domains (e.g., ["Contract Law", "Family Law"])
        personas: List of user personas (e.g., ["Citizens", "Supreme Court Lawyer"])
        questions_per_combination: Number of questions to generate per domain-persona combination
        
    Returns:
        Dictionary with domain-persona combinations as keys and lists of generated questions as values
    """
    # Select a random sample of existing questions to use as examples
    sample_size = min(15, len(existing_questions))
    sample_questions = random.sample(existing_questions, sample_size)
    examples = "\n".join([f"- {q}" for q in sample_questions])
    
    all_generated_queries = {}
    
    for domain in domains:
        for persona in personas:
            combo_key = f"{domain} - {persona}"
            print(f"Generating {questions_per_combination} queries for {combo_key}...")
            
            prompt = f"""
                # Persona 
                You are legal AI expert who is great at generating synthetic user queries. 

                # Task
                You are tasked with generating realistic user queries for testing a legal AI chatbot in India.

                # Inputs
                You will be provided a domain in the legal field, a specific user persona, a set of existing user queries, and the number of new queries to generate.

                1. Domain: {domain}
                2. User Persona: {persona}
                3. Here are some examples of existing user queries from our database:
                {examples}
                4. Please generate {questions_per_combination} new, diverse user queries that:

                # Constraints
                Make sure that you follow the following instructions strictly while generating the queries:
                1. The queries should be specifically related to {domain} in India
                2. The queries should be representative of questions that would be asked by a {persona} persona
                3. For "Citizens" persona: Use everyday language with common grammatical errors, typos, and abbreviations
                4. For "Supreme Court Lawyer" persona: Use more technical legal terminology and sophisticated questions
                5. Cover different aspects of {domain}
                6. Make sure the queries are diverse enough to capture all aspects of the domain from the perspective of the {persona}
                7. Return only the queries, one per line, with no numbering or additional text.

                # Example Queries by Persona
                ## For Citizens (Contract Law):
                1. Rights undr RERA if builder delays possession of flat?
                2. Can landlord increase rent without proper notice as per TPA?
                3. Is e-sign valid in ICA for property purchase agreement?
                4. How to file complaint if seller breached CPA during online shopping?
                5. Rights under COPRA if defective product delivered?

                ## For Supreme Court Lawyer (Family Law):
                1. What are the procedural requisites for filing a petition under Section 125 CrPC when the respondent resides outside territorial jurisdiction?
                2. Does the concept of constructive desertion apply in cases of restitution of conjugal rights under Hindu Marriage Act?
                3. Can a Court impose additional conditions beyond those stipulated in Section 13B of Hindu Marriage Act for mutual consent divorce?
                4. What is the evidentiary threshold required for proving mental cruelty under Section 13(1)(ia) of Hindu Marriage Act?
                5. Jurisprudential analysis of grandparental visitation rights vis-à-vis parental autonomy in Indian family law?

                ## For Supreme Court Lawyer (Contract Law):
                1. What is the judicial interpretation of Section 74 of Indian Contract Act regarding liquidated damages versus penalty clauses?
                2. Can electronic records satisfy the requirements of Section 10 of Specific Relief Act for specific performance of contracts?
                3. Applicability of doctrine of frustration under Section 56 in light of economic hardship caused by regulatory changes.
                4. Enforceability of non-compete clauses in employment contracts pursuant to Section 27 of Indian Contract Act.
                5. Jurisprudential analysis of quantum meruit claims in void contracts under Indian Contract Act.
                """
            
            try:
                response = client.chat.completions.create(
                    model="courteasy-ai-gpt-4o-mini",  # Use your specific Azure OpenAI model deployment name
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=512
                )
                
                # Extract and format the generated queries
                generated_text = response.choices[0].message.content
                new_queries = [q.strip() for q in generated_text.split('\n') if q.strip()]
                
                # Remove numbering if present
                new_queries = [q.split('. ', 1)[1] if '. ' in q and q[0].isdigit() else q for q in new_queries]
                new_queries = [q.strip('- ') for q in new_queries]  # Remove bullet points if present
                
                all_generated_queries[combo_key] = new_queries[:questions_per_combination]
                print(f"Generated {len(all_generated_queries[combo_key])} queries for {combo_key}")
                
                # Add a small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error generating queries for {combo_key}: {e}")
                all_generated_queries[combo_key] = []
    
    # Flatten the dictionary to a list if needed for the rest of the pipeline
    flattened_queries = []
    flattened_metadata = []  # To keep track of domain and persona for each query
    
    for combo_key, queries in all_generated_queries.items():
        domain, persona = combo_key.split(" - ")
        for query in queries:
            flattened_queries.append(query)
            flattened_metadata.append({"domain": domain, "persona": persona})
    
    print(f"Generated a total of {len(flattened_queries)} queries across all combinations")
    
    return flattened_queries, flattened_metadata
def enhance_queries(queries):
    """
    Create enhanced versions of queries with improved grammar and clarity.
    """
    enhanced_queries = []
    
    for i, query in enumerate(queries):
        prompt = f"""
            # Persona
            You are a legal language expert who specializes in refining user queries while maintaining their original intent, with deep knowledge of Indian legal terminology and abbreviations.

            # Task
            Improve the following user query by fixing grammar, spelling, and clarity issues while preserving the original meaning and intent.

            # Input
            Original query: "{query}"

            # Constraints
            1. Fix grammatical errors and typos while preserving the legal context
            2. Make the language more professional but still natural
            3. Maintain the original intent and meaning of the query
            4. Keep regional terminology related to Indian legal system intact
            5. Expand common legal abbreviations correctly while keeping them in the query
            6. Return only the enhanced query with no explanation or additional text

            # Abbreviation Guide
            When you encounter these common legal abbreviations in India, expand them as follows:
            - SC/ST Act = Scheduled Castes and the Scheduled Tribes (Prevention of Atrocities) Act, 1989
            - EC Act = Employees Compensation Act, 1923
            - SR Act = Specific Relief Act, 1963
            - PHRA = Protection of Human Rights Act, 1993
            - ITA = Income Tax Act, 1961
            - PMLA = Prevention of Money-Laundering Act, 2002
            - NDPS Act = Narcotic Drugs and Psychotropic Substances Act, 1985
            - ESI Act = Employees State Insurance Act, 1948
            - PCM Act = Prohibition of Child Marriage Act, 2006
            - TM Act = Trade Marks Act, 1999
            - EPF Act = Employees Provident Funds and Miscellaneous Provisions Act, 1952
            - TPA = Transfer of Property Act, 1882
            - CLPRA = Child and Adolescent Labour (Prohibition and Regulation) Act, 1986
            - ERA = Equal Remuneration Act, 1976
            - SRA = Special Relief Act, 1963
            - FERA = Foreign Exchange Regulation Act, 1973
            - FEMA = Foreign Exchange Management Act, 1999
            - CPC = Code of Civil Procedure
            - CrPC = Code of Criminal Procedure
            - IPC = Indian Penal Code
            - BNS = Bhartiya Nyay Sanhita
            - POSH = Prevention of Sexual Harassment

            # Examples
            Original: "wats d diff btwn BNS n IPC plz explain smpl terms"
            Enhanced: "What is the difference between Bhartiya Nyay Sanhita and Indian Penal Code? Please explain in simple terms."

            Original: "hw 2 file FIR in crPC section 154"
            Enhanced: "How to file an FIR under Code of Criminal Procedure Section 154?"

            Original: "wat r my rytes under POSH act if facng harassment"
            Enhanced: "What are my rights under the Prevention of Sexual Harassment Act if I am facing harassment?"

            Original: "hw 2 file CGST act refund fr wrong payment"
            Enhanced: "How to file for a refund under the CGST Act for wrong payment?"

            Original: "rights of child undr JJ act if in conflict with law"
            Enhanced: "What are the rights of a child under the Juvenile Justice Act if in conflict with law?"

            Original: "wat is d punishment 4 violating SC/ST act section 3?"
            Enhanced: "What is the punishment for violating Scheduled Castes and the Scheduled Tribes (Prevention of Atrocities) Act, 1989 Section 3?"

            Original: "filing case undr FEMA fr forex violation procedure"
            Enhanced: "What is the procedure for filing a case under Foreign Exchange Management Act, 1999 for foreign exchange violation?"
            """
        
        try:
            response = client.chat.completions.create(
                model="courteasy-ai-gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            enhanced_query = response.choices[0].message.content.strip()
            enhanced_queries.append(enhanced_query)
            
            print(f"Enhanced query {i+1}/{len(queries)}")
            
            # Add a small delay to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error enhancing query: {e}")
            enhanced_queries.append(query)  # Fall back to original if enhancement fails
    
    return enhanced_queries

def generate_answers(enhanced_queries):
    """
    Generate structured answers with citations for the enhanced queries.
    """
    structured_answers = []
    
    for i, query in enumerate(enhanced_queries):
        prompt = f"""
        # Persona
        You are an AI legal assistant specializing in Indian law with expertise in providing clear, accurate, and helpful responses to legal queries.

        # Task
        Provide a comprehensive yet concise response to the following legal query, including relevant citations from Indian law.

        # Input
        Query: {query}

        # Constraints
        1. Include relevant legal provisions, acts, or case law specific to Indian legal system where applicable
        2. Be concise yet comprehensive, prioritizing the most important information first
        3. Use simple language while maintaining legal accuracy and precision
        4. Provide practical guidance and next steps where appropriate
        5. Include specific citations to relevant legal authorities (acts, sections, case law)
        6. Consider regional variations in Indian law when applicable
        """
        
        try:
            # Using the beta.chat.completions.parse for structured output
            response = client.beta.chat.completions.parse(
                model="courteasy-ai-gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format=LegalAnswer,
                temperature=0.2,
                max_tokens=1600
            )
            
            # Check if there's a refusal
            if hasattr(response.choices[0].message, 'refusal') and response.choices[0].message.refusal:
                print(f"Model refused to generate answer for query {i+1}")
                # Create a placeholder answer with empty citations
                
                EmptyAnswer = create_model('EmptyAnswer', 
                    answer=(str, "Model refused to generate an answer for this query."),
                    citations=(List[Citation], [])
                )
                structured_answer = EmptyAnswer()
            else:
                # Extract the structured answer
                structured_answer = response.choices[0].message.parsed
            
            structured_answers.append(structured_answer)
            
            print(f"Generated answer {i+1}/{len(enhanced_queries)}")
            
            # Add a delay to avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            # Handle error cases by creating a dummy structured answer
            from pydantic import create_model
            ErrorAnswer = create_model('ErrorAnswer', 
                answer=(str, f"Error generating answer: {str(e)}"),
                citations=(List[Citation], [])
            )
            structured_answers.append(ErrorAnswer())
    
    return structured_answers

def save_results(original_queries, enhanced_queries, structured_answers, metadata, output_file):
    """
    Save all generated content to a new Excel file with just answers and citations.
    """
    # Basic DataFrame with text answers and metadata
    result_df = pd.DataFrame({
        "Question": original_queries,
        "Enhanced Question": enhanced_queries,
        "Answer": [s.answer if hasattr(s, 'answer') else "Error generating answer." for s in structured_answers],
        "Domain": [m["domain"] for m in metadata],
        "Persona": [m["persona"] for m in metadata]
    })
    
    # Add citations as a formatted string
    result_df["Citations"] = [
        ", ".join([f"{c.source} ({c.reference})" for c in s.citations]) 
        if hasattr(s, 'citations') and s.citations else ""
        for s in structured_answers
    ]
    
    try:
        result_df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")

def main():
    # Configuration
    input_file = "User Queries with updated response.xlsx" 
    output_file = "generated_legal_queries_with_citations.xlsx"
    domains = ["Criminal Law", "Civil Litigation", "Corporate/Business Law"]
    personas = ["Citizens", "Supreme Court Lawyer"]
    questions_per_combination = 5
    
    # Read existing queries
    df = read_excel_file(input_file)
    if df is None:
        return
    
    # Extract unique questions from the Question column
    existing_questions = extract_unique_questions(df, question_column="Question")
    
    if not existing_questions:
        print("No existing questions found in the 'Question' column")
        return
    
    # Generate new queries for each domain-persona combination
    print(f"Generating {questions_per_combination} queries for each combination of domains {domains} and personas {personas}...")
    new_queries, metadata = generate_queries(
        existing_questions, 
        domains=domains, 
        personas=personas, 
        questions_per_combination=questions_per_combination
    )
    
    if not new_queries:
        print("Failed to generate new queries.")
        return
    
    print(f"Successfully generated {len(new_queries)} new queries.")
    
    # Enhance queries
    print("Enhancing queries...")
    enhanced_queries = enhance_queries(new_queries)
    
    # Generate answers with citations
    print("Generating answers with citations...")
    structured_answers = generate_answers(enhanced_queries)
    
    # Save results with metadata
    result_df = pd.DataFrame({
        "Question": new_queries,
        "Enhanced Question": enhanced_queries,
        "Answer": [s.answer if hasattr(s, 'answer') else "Error generating answer." for s in structured_answers],
        "Domain": [m["domain"] for m in metadata],
        "Persona": [m["persona"] for m in metadata]
    })
    
    # Add citations as a formatted string
    result_df["Citations"] = [
        ", ".join([f"{c.source} ({c.reference})" for c in s.citations]) 
        if hasattr(s, 'citations') and s.citations else ""
        for s in structured_answers
    ]
    
    try:
        result_df.to_excel(output_file, index=False)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("Process completed!")

if __name__ == "__main__":
    main()