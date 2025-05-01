import pandas as pd
import re
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# preprocess text by converting to lowercase, removing non-alphanumeric characters, and collapsing whitespace
# args: text (str): input text to process
# returns: str: preprocessed text
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load sentence-BERT model for generating embeddings
# Return: SentenceTransformer: loaded model
def load_embedding_model() -> SentenceTransformer:
    return SentenceTransformer('all-MiniLM-L6-v2')

# Calculate cosine similarity between 2 texts using Sentence-BERT embeddings
# args: text1 (str): first text, text2 (str): second text, and loaded model
# returns: float (similarity score between 0 and 1)
def calculate_similarity(text1: str, text2: str, model: SentenceTransformer) -> float:
    # Generate embeddings
    embeddings = model.encode([text1, text2])
    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

# Extract keywords from text using RAKE
# args: text (str): input text
# returns: List[str]: List of extracted keywords
def extract_keywords(text: str) -> List[str]:
    r = Rake()
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

# Analyze skill gap between resume and job description
# Args: resume_text (str): resume text
# Returns: List[str]: list of missing skills
def analyze_skill_gap(resume_text: str, job_text: str) -> List[str]:
    resume_keywords = set(extract_keywords(resume_text))
    job_keywords = set(extract_keywords(job_text))
    return list(job_keywords - resume_keywords)

# Match resume with job description and return similarity score and missing skills
# args: resume_text(str), job_text (str), and loaded model
# returns: tuple[float, List[str]]: similarity score and list of missing skills
def match_resume_job(resume_text: str, job_text: str, model: SentenceTransformer) -> Tuple[float, List[str]]:
    # Preprocess texts
    resume_processed = preprocess_text(resume_text)
    job_processed = preprocess_text(job_text)
    
    # Calculate similarity
    similarity = calculate_similarity(resume_processed, job_processed, model)
    
    # Analyze skill gap
    missing_skills = analyze_skill_gap(resume_processed, job_processed)
    
    return similarity, missing_skills

# Process resume and job description files and print matching results
def main():
    try:
        # Load data
        print("Loading job descriptions...")
        jobs_df = pd.read_csv('data/jobs/job_title_des.csv')
        
        # Load model
        print("Loading Sentence-BERT model...")
        model = load_embedding_model()
        
        # Process each job description
        for _, job in jobs_df.iterrows():
            print(f"\nProcessing job: {job['Job Title']}")
            
            # Get job text
            job_text = f"{job['Job Title']} {job['Job Description']}"
            
            # For demonstration, let's use a sample resume
            sample_resume = """
            Experienced software engineer with 5 years in full-stack development.
            Proficient in Python, JavaScript, React, Node.js, and SQL.
            Led development of scalable web applications and microservices.
            Strong experience in cloud platforms (AWS, GCP) and DevOps practices.
            """
            
            # Match resume with job
            similarity, missing_skills = match_resume_job(sample_resume, job_text, model)
            
            # Print results
            print(f"Similarity Score: {similarity*100:.2f}%")
            print("Missing Skills:")
            for skill in missing_skills:
                print(f"- {skill}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 