import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from resume_job_matcher import (
    preprocess_text,
    load_embedding_model,
    calculate_similarity,
    extract_keywords,
    analyze_skill_gap,
    match_resume_job
)
# Run tests to gather data for presentation slides
def run_presentation_tests():
    # Load model and data
    model = load_embedding_model()
    jobs_df = pd.read_csv('data/jobs/job_title_des.csv')
    
    # Select 5 representative jobs
    representative_jobs = [
        'Software Engineer',
        'Data Scientist',
        'Full Stack Developer',
        'DevOps Engineer',
        'Machine Learning Engineer'
    ]
    jobs_df = jobs_df[jobs_df['Job Title'].isin(representative_jobs)]
    
    # Test cases for different experience levels - Updated with more comprehensive skills
    test_cases = {
        'junior_developer': """
        Recent computer science graduate with internship experience.
        Proficient in Python, Java, JavaScript, HTML, CSS, and SQL.
        Experience with React, Node.js, and basic cloud computing.
        Strong understanding of data structures and algorithms.
        Completed projects in web development and database design.
        Familiar with Git, Docker, and basic CI/CD concepts.
        Strong problem-solving skills and eagerness to learn.
        """,
        
        'mid_level_developer': """
        Software engineer with 3 years of experience.
        Strong skills in Python, JavaScript, React, Node.js, and SQL.
        Experience with cloud platforms (AWS, GCP) and microservices.
        Proficient in Docker, Kubernetes, and CI/CD pipelines.
        Led small team projects and implemented automated testing.
        Experience with system design and architecture patterns.
        Strong debugging and performance optimization skills.
        """,
        
        'senior_developer': """
        Senior software engineer with 8+ years of experience.
        Expert in full-stack development and system architecture.
        Extensive experience with cloud platforms (AWS, GCP, Azure).
        Deep knowledge of microservices, Docker, and Kubernetes.
        Led multiple large-scale projects and mentored junior developers.
        Strong expertise in performance optimization and security.
        Experience with machine learning and data engineering.
        Proficient in multiple programming languages and frameworks.
        """
    }
    
    # Results storage
    results = {
        'job_titles': [],
        'experience_levels': [],
        'similarity_scores': [],
        'processing_times': [],
        'missing_skills_counts': []
    }
    
    print("\nRunning Presentation Tests...")
    print("=" * 80)
    
    # Test each resume against each job
    for exp_level, resume in test_cases.items():
        print(f"\nTesting {exp_level.replace('_', ' ').title()} Profile")
        print("=" * 80)
        
        for _, job in jobs_df.iterrows():
            job_text = f"{job['Job Title']} {job['Job Description']}"
            
            # Time the matching process
            import time
            start_time = time.time()
            similarity, missing_skills = match_resume_job(resume, job_text, model)
            processing_time = time.time() - start_time
            
            # Store results
            results['job_titles'].append(job['Job Title'])
            results['experience_levels'].append(exp_level)
            results['similarity_scores'].append(similarity)
            results['processing_times'].append(processing_time)
            results['missing_skills_counts'].append(len(missing_skills))
            
            print(f"Job: {job['Job Title']}")
            print(f"Similarity: {similarity*100:.2f}%")
            print(f"Processing Time: {processing_time:.3f}s")
            print(f"Missing Skills: {len(missing_skills)}")
            print("-" * 40)
    
    # Create visualizations
    plt.style.use('seaborn')
    
    # 1. Similarity Scores by Experience Level
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='experience_levels', y='similarity_scores', data=pd.DataFrame(results))
    plt.title('Similarity Scores by Experience Level')
    plt.xlabel('Experience Level')
    plt.ylabel('Similarity Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('similarity_by_experience.png')
    
    # 2. Processing Time Analysis
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='experience_levels', y='processing_times', data=pd.DataFrame(results))
    plt.title('Processing Time by Experience Level')
    plt.xlabel('Experience Level')
    plt.ylabel('Processing Time (seconds)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('processing_time.png')
    
    # 3. Missing Skills Analysis - Modified to show clearer differences
    plt.figure(figsize=(12, 8))
    df = pd.DataFrame(results)
    
    # Calculate average missing skills for each experience level
    avg_missing_skills = df.groupby('experience_levels')['missing_skills_counts'].mean()
    
    # Create bar plot
    plt.bar(avg_missing_skills.index, avg_missing_skills.values, 
            color=['#FF9999', '#66B2FF', '#99FF99'])
    
    # Add value labels on top of bars
    for i, v in enumerate(avg_missing_skills.values):
        plt.text(i, v + 1, f'{v:.1f}', ha='center', va='bottom')
    
    plt.title('Average Number of Missing Skills by Experience Level', fontsize=14)
    plt.xlabel('Experience Level', fontsize=12)
    plt.ylabel('Average Number of Missing Skills', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('missing_skills.png', dpi=300, bbox_inches='tight')
    
    # Save results to CSV for further analysis
    pd.DataFrame(results).to_csv('presentation_results.csv', index=False)
    
    print("\nTest results and visualizations have been saved.")
    return results

if __name__ == "__main__":
    results = run_presentation_tests() 