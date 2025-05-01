# Resume-Job-Matching-System
An intelligent system that automatically matches resumes with job descriptions using natural language processing and machine learning.

## 🚀 Features

- Text preprocessing and normalization  
- Semantic matching using Sentence-BERT embeddings  
- Cosine similarity scoring  
- Keyword extraction using RAKE  
- Skill gap analysis  
- Fast processing (~0.2s per match)

## 📦 Requirements

- Python 3.9+
- All dependencies listed in `requirements.txt`

---

## 📂 Dataset Sources

This project uses two Kaggle datasets for training and testing the resume-job matching system:

- **Resume Dataset**  
  - **Name:** Resume Dataset  
  - **Author:** [snehaanbhawal](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)  
  - **Description:** A CSV of real-world resume examples (skills, experience, education, objectives) scraped from LiveCareer.com

- **Job Description Dataset**  
  - **Name:** Jobs and Job Description  
  - **Author:** [kshitizregmi](https://www.kaggle.com/datasets/kshitizregmi/jobs-and-job-description)  
  - **Description:** A CSV of job titles paired with full job descriptions across industries

> Make sure to download and place the datasets into the appropriate `data/` directory for use.

