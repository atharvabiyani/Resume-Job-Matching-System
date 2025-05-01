import pandas as pd
import time
from resume_job_matcher import (
    preprocess_text,
    load_embedding_model,
    calculate_similarity,
    extract_keywords,
    analyze_skill_gap,
    match_resume_job
)
# Run performance tests and return metrics
def run_performance_tests():
    results = {
        'processing_times': [],
        'similarity_scores': [],
        'skill_gaps': [],
        'missing_skills_details': []  # New field to store actual missing skills
    }
    
    # Load test data
    model = load_embedding_model()
    jobs_df = pd.read_csv('data/jobs/job_title_des.csv')
    
    # Test resumes with different profiles
    test_resumes = {
        'software_engineer': """
        Experienced software engineer with 8 years in full-stack development.
        Proficient in Python, JavaScript, React, Node.js, and SQL.
        Led development of scalable web applications and microservices.
        Strong experience in cloud platforms (AWS, GCP) and DevOps practices.
        """,
        
        'data_scientist': """
        Data scientist with 5 years experience in machine learning and analytics.
        Expert in Python, R, TensorFlow, and scikit-learn.
        Implemented predictive models and data pipelines.
        Published research in NLP and computer vision.
        """,
        
        'fresh_graduate': """
        Recent computer science graduate with internship experience.
        Knowledge of Python, Java, and web development.
        Completed projects in machine learning and database design.
        Eager to learn and contribute to innovative projects.
        """
    }
    
    print("\nRunning Performance Tests...")
    print("=" * 80)
    
    for profile, resume in test_resumes.items():
        print(f"\nTesting profile: {profile}")
        print("=" * 80)
        start_time = time.time()
        
        # Test with first 10 jobs for quick results
        for _, job in jobs_df.head(10).iterrows():
            job_text = f"{job['Job Title']} {job['Job Description']}"
            
            # Measure matching time
            match_start = time.time()
            similarity, missing_skills = match_resume_job(resume, job_text, model)
            match_time = time.time() - match_start
            
            results['processing_times'].append(match_time)
            results['similarity_scores'].append(similarity)
            results['skill_gaps'].append(len(missing_skills))
            results['missing_skills_details'].append(missing_skills)
            
            print(f"\nJob Title: {job['Job Title']}")
            print(f"Similarity Score: {similarity*100:.2f}%")
            print(f"Processing Time: {match_time:.3f}s")
            print("\nMissing Skills:")
            if missing_skills:
                for i, skill in enumerate(missing_skills, 1):
                    print(f"{i}. {skill}")
            else:
                print("None")
            print("-" * 80)
    
    # Calculate aggregate metrics
    avg_time = sum(results['processing_times']) / len(results['processing_times'])
    avg_similarity = sum(results['similarity_scores']) / len(results['similarity_scores'])
    avg_skill_gaps = sum(results['skill_gaps']) / len(results['skill_gaps'])
    
    print("\nOverall Results:")
    print("=" * 80)
    print(f"Average Processing Time: {avg_time:.3f}s")
    print(f"Average Similarity Score: {avg_similarity*100:.2f}%")
    print(f"Average Number of Missing Skills: {avg_skill_gaps:.1f}")
    
    # Analyze most common missing skills across all jobs
    all_missing_skills = [skill for skills in results['missing_skills_details'] for skill in skills]
    skill_frequency = pd.Series(all_missing_skills).value_counts()
    
    print("\nTop 10 Most Common Missing Skills:")
    print("=" * 80)
    for skill, count in skill_frequency.head(10).items():
        print(f"- {skill}: {count} occurrences")
    
    return results

if __name__ == "__main__":
    results = run_performance_tests() 