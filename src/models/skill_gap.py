"""Skill-Gap Report Module."""

import re
import pandas as pd
from src import config

# Curated list of technical and professional skills to extract as vocabulary
CURATED_SKILLS = {
    # Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "ruby", "php", "go", "rust", "scala", "kotlin", "swift", "sql", "r", "html", "css", "bash", "shell",
    # Frontend & Backend Web
    "react", "angular", "vue", "django", "flask", "fastapi", "spring", "spring boot", "laravel", "net core", "node.js", "express", "jquery", "ajax", "bootstrap",
    # Data Science, ML & Big Data
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision", "statistics", "data analysis", "tableau", "power bi", "excel", "data engineering", "etl", "data warehousing", "big data", "spark", "hadoop", "hive", "scikit-learn", "tensorflow", "pytorch", "keras", "opencv", "numpy", "pandas",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab", "ansible", "terraform", "ci/cd", "devops", "linux", "unix",
    # Databases
    "mysql", "postgresql", "mongodb", "sqlite", "redis", "cassandra", "oracle", "sql server", "nosql", "dynamodb",
    # Design & Business / PM
    "agile", "scrum", "project management", "pmp", "jira", "product management", "system design", "microservices", "rest api", "graphql", "ui/ux", "figma",
    # QA & Testing
    "testing", "automation testing", "selenium", "junit", "pytest", "cypress", "manual testing", "load testing",
    # Domain specific / Soft skills
    "communication", "leadership", "teamwork", "sales", "marketing", "recruiting", "talent acquisition", "hr", "finance", "accounting"
}

def clean_text_simple(text):
    if not isinstance(text, str):
        return ""
    return text.lower()

def extract_skills_from_text(text, vocabulary=CURATED_SKILLS):
    """Scan text for the presence of skill keywords."""
    text_lower = clean_text_simple(text)
    found_skills = set()
    for skill in vocabulary:
        # Use regex boundary matching to avoid partial matches (e.g. "go" in "good")
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.add(skill)
    return found_skills

def get_skill_gap_report(resume_text, target_role, jobs_df, role_col="category", top_n_skills=10):
    """Computes required skills for a target role, checks the resume, and returns the gap."""
    # 1. Extract candidate skills
    candidate_skills = extract_skills_from_text(resume_text)

    # 2. Filter jobs for the target role/category/cluster
    if role_col not in jobs_df.columns:
        role_jobs = jobs_df
    else:
        role_jobs = jobs_df[jobs_df[role_col].astype(str).str.lower() == str(target_role).lower()]
    
    if len(role_jobs) == 0:
        # Fallback to all jobs if no role-specific jobs found
        role_jobs = jobs_df
        
    # 3. Aggregate skill frequencies in these jobs
    skill_counts = {skill: 0 for skill in CURATED_SKILLS}
    total_role_jobs = len(role_jobs)
    
    for _, job in role_jobs.iterrows():
        # Combine title, skills field and description to find requirements
        job_text = " ".join([str(job.get("title", "")), str(job.get("skills", "")), str(job.get("description", ""))])
        job_skills = extract_skills_from_text(job_text)
        for skill in job_skills:
            skill_counts[skill] += 1
            
    # 4. Format and sort by importance/frequency
    sorted_skills = sorted(skill_counts.items(), key=lambda item: item[1], reverse=True)
    
    report = []
    for skill, count in sorted_skills[:top_n_skills]:
        # Importance percentage based on count in role jobs
        importance = int((count / total_role_jobs) * 100) if total_role_jobs > 0 else 0
        present = skill in candidate_skills
        report.append({
            "skill": skill,
            "present": present,
            "importance": importance
        })
        
    return report
