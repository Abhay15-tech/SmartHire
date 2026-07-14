"""Clean text and merge the job listings into one job corpus.

Run:  python -m src.data.preprocess

Outputs:
    data/interim/job_corpus.csv      merged job table (common schema)
    data/processed/jobs_clean.csv    + a cleaned `text` column for TF-IDF
    data/processed/resumes_clean.csv cleaned resume text + Category
"""

import re
import pandas as pd
from src import config
from src.data import load_data


# ----------------------------------------------------------------------------
# Text cleaning
# ----------------------------------------------------------------------------
def clean_text(text):
    """Lowercase, strip URLs/punctuation/extra spaces. Safe on NaN/non-strings."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # drop URLs
    text = re.sub(r"[^a-z0-9+#. ]", " ", text)       # keep words, digits, c++/c#/.net
    text = re.sub(r"\s+", " ", text).strip()          # collapse whitespace
    return text


# ----------------------------------------------------------------------------
# Map each source to the common schema
# ----------------------------------------------------------------------------
def _map_naukri(df):
    """Naukri columns -> title, company, location, skills, description, experience, category."""
    out = df.rename(columns={
        "jobtitle": "title",
        "company": "company",
        "joblocation_address": "location",
        "skills": "skills",
        "jobdescription": "description",
        "experience": "experience",
        "industry": "category",
    })
    out["source"] = "naukri"
    if "category" not in out.columns:
        out["category"] = ""
    return out[config.COMMON_JOB_COLUMNS + ["source"]]


def _map_linkedin(df):
    """LinkedIn columns -> the same common schema."""
    out = df.rename(columns={
        "title": "title",
        "company_name": "company",
        "location": "location",
        "skills": "skills",
        "description": "description",
        "formatted_experience_level": "experience",
    })
    out["category"] = ""
    out["source"] = "linkedin"
    return out[config.COMMON_JOB_COLUMNS + ["source"]]


# ----------------------------------------------------------------------------
# Re-derive categories using a baseline classifier
# ----------------------------------------------------------------------------
def re_derive_categories(corpus):
    """Trains a baseline resume classifier on the fly to predict / correct categories in job corpus."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    print("Loading resumes for training category classifier...")
    resumes_df = load_data.load_resumes()
    resumes_df["cleaned_text"] = resumes_df["Resume"].map(clean_text)

    # Train classifier
    print("Training TF-IDF + Logistic Regression on resume categories...")
    vec = TfidfVectorizer(max_features=3000, stop_words="english")
    X = vec.fit_transform(resumes_df["cleaned_text"])
    y = resumes_df["Category"]

    clf = LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)
    clf.fit(X, y)
    print("Baseline resume classifier successfully trained.")

    # Predict categories for job corpus
    print("Predicting job categories...")
    job_texts = (corpus["title"] + " " + corpus["skills"] + " " + corpus["description"]).map(clean_text)
    X_jobs = vec.transform(job_texts)
    predictions = clf.predict(X_jobs)

    corrected_count = 0
    new_categories = []

    tech_keywords = {
        "engineer", "developer", "programmer", "software", "analyst", "data", 
        "science", "testing", "qa", "java", "python", "devops", "hadoop", 
        "blockchain", "sap", "database", "net", "web", "frontend", "backend"
    }

    for i, (orig, pred, title) in enumerate(zip(corpus["category"], predictions, corpus["title"])):
        title_lower = str(title).lower()
        orig_str = str(orig).strip()
        orig_lower = orig_str.lower()

        # Check if title has tech keywords but original is something completely non-tech
        is_tech_role = any(kw in title_lower for kw in tech_keywords)
        is_mislabeled = False

        mislabeled_sectors = ["accounting", "finance", "audit", "hr", "recruitment", "retail", "advertising", "media"]
        if is_tech_role and any(sector in orig_lower for sector in mislabeled_sectors):
            is_mislabeled = True

        # If original is empty, Not Specified, mislabeled tech, or generic "IT-Software", re-derive
        generic_industries = ["it-software", "software services", "internet", "other"]
        if (
            orig_str == "" or 
            orig_str.lower() == "nan" or 
            orig_str == "Not specified" or 
            is_mislabeled or 
            any(gen in orig_lower for gen in generic_industries)
        ):
            new_categories.append(pred)
            corrected_count += 1
        else:
            new_categories.append(orig_str)

    corpus["category"] = new_categories
    print(f"Re-derived/corrected category for {corrected_count:,} out of {len(corpus):,} jobs.")
    return corpus


# ----------------------------------------------------------------------------
# Build the merged job corpus
# ----------------------------------------------------------------------------
def build_job_corpus():
    """Merge all available job sources into one clean corpus and save it."""
    frames = [_map_naukri(load_data.load_naukri())]

    if load_data.linkedin_available():
        print("LinkedIn found -> merging it into the corpus.")
        frames.append(_map_linkedin(load_data.load_linkedin()))
    else:
        print("LinkedIn not found -> building the corpus from Naukri only.")

    corpus = pd.concat(frames, ignore_index=True)

    # Audit NaNs in company, location, and experience columns before filling
    print("\n=== Auditing NaNs in Merged Job Corpus ===")
    total_rows = len(corpus)
    nan_company = corpus["company"].isna().sum()
    nan_location = corpus["location"].isna().sum()
    nan_experience = corpus["experience"].isna().sum()
    print(f"Total Rows: {total_rows:,}")
    print(f"NaNs in 'company': {nan_company:,} ({nan_company / total_rows * 100:.2f}%)")
    print(f"NaNs in 'location': {nan_location:,} ({nan_location / total_rows * 100:.2f}%)")
    print(f"NaNs in 'experience': {nan_experience:,} ({nan_experience / total_rows * 100:.2f}%)")

    # Impute defaults for NaN values as dropping would lose too much data (especially experience)
    corpus["company"] = corpus["company"].fillna("Not specified")
    corpus["location"] = corpus["location"].fillna("Not specified")
    corpus["experience"] = corpus["experience"].fillna("Not specified")
    corpus["skills"] = corpus["skills"].fillna("")
    corpus["description"] = corpus["description"].fillna("")
    corpus["title"] = corpus["title"].fillna("")
    corpus["category"] = corpus["category"].fillna("")

    # Drop rows with no title and no description
    corpus = corpus[(corpus["title"].str.strip() != "") |
                    (corpus["description"].str.strip() != "")]

    # Drop duplicate postings
    before_dedup = len(corpus)
    corpus = corpus.drop_duplicates(subset=["title", "company", "location"])
    corpus = corpus.reset_index(drop=True)
    after_dedup = len(corpus)
    print(f"Duplicates removed: {before_dedup - after_dedup:,} rows (Before: {before_dedup:,}, After: {after_dedup:,})")

    # Fix Category Mislabeling & Impute categories via trained Resume Classifier
    print("\n=== Fixing/Re-deriving Job Categories ===")
    corpus = re_derive_categories(corpus)

    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    corpus.to_csv(config.JOB_CORPUS_CSV, index=False)
    print(f"Saved {len(corpus):,} jobs -> {config.JOB_CORPUS_CSV}")

    # Model-ready version with one cleaned text column
    processed = corpus.copy()
    processed["text"] = (
        processed["title"] + " " + processed["skills"] + " " + processed["description"]
    ).map(clean_text)
    processed = processed[processed["text"].str.strip() != ""]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed.to_csv(config.JOBS_CLEAN_CSV, index=False)
    print(f"Saved {len(processed):,} model-ready jobs -> {config.JOBS_CLEAN_CSV}")
    return processed


# ----------------------------------------------------------------------------
# Build the cleaned resume table
# ----------------------------------------------------------------------------
def build_clean_resumes():
    """Clean the resume text and save Category + cleaned text."""
    df = load_data.load_resumes()
    df["text"] = df["Resume"].map(clean_text)
    df = df[["Category", "text"]]
    df = df[df["text"].str.strip() != ""]

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.RESUMES_CLEAN_CSV, index=False)
    print(f"Saved {len(df):,} resumes -> {config.RESUMES_CLEAN_CSV}")
    return df


def main():
    print("=== Building resume table ===")
    build_clean_resumes()
    print("\n=== Building job corpus ===")
    build_job_corpus()
    print("\nDone.")


if __name__ == "__main__":
    main()
