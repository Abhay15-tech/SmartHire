"""Read the raw datasets. Returns each source with its original column names;
renaming and merging happen in preprocess.py.
"""

import pandas as pd

from src import config


def _read_csv(path, usecols=None):
    """Read a CSV, falling back to latin-1 for files that aren't valid UTF-8."""
    try:
        return pd.read_csv(path, usecols=usecols, low_memory=False, on_bad_lines="skip")
    except UnicodeDecodeError:
        return pd.read_csv(
            path, usecols=usecols, low_memory=False, on_bad_lines="skip",
            encoding="latin-1",
        )


def load_resumes():
    """Resume dataset -> columns: Category, Resume."""
    return _read_csv(config.RESUME_CSV)


def load_naukri():
    """Naukri jobs -> only the columns we map to the common schema."""
    cols = ["jobtitle", "company", "joblocation_address",
            "skills", "jobdescription", "experience", "industry"]
    return _read_csv(config.NAUKRI_CSV, usecols=cols)


def linkedin_available():
    """True only if the optional LinkedIn files are present in data/raw/linkedin/."""
    return (
        config.LINKEDIN_POSTINGS_CSV.exists()
        and config.LINKEDIN_JOB_SKILLS_CSV.exists()
        and config.LINKEDIN_SKILLS_MAP_CSV.exists()
    )


def _linkedin_skills_by_job():
    """One skills string per job_id.

    job_skills.csv holds skill codes (job_id, skill_abr); mappings/skills.csv maps
    those codes to names. Join them, then collapse to one string per job.
    """
    job_skills = _read_csv(config.LINKEDIN_JOB_SKILLS_CSV, usecols=["job_id", "skill_abr"])
    skill_map = _read_csv(config.LINKEDIN_SKILLS_MAP_CSV, usecols=["skill_abr", "skill_name"])

    merged = job_skills.merge(skill_map, on="skill_abr", how="left")
    merged = merged.dropna(subset=["skill_name"])

    grouped = (
        merged.groupby("job_id")["skill_name"]
        .apply(lambda names: ", ".join(sorted(set(names))))
        .reset_index()
        .rename(columns={"skill_name": "skills"})
    )
    return grouped


def load_linkedin():
    """LinkedIn postings with a `skills` column joined on from the skill files."""
    cols = ["job_id", "title", "company_name", "location",
            "description", "formatted_experience_level"]
    postings = _read_csv(config.LINKEDIN_POSTINGS_CSV, usecols=cols)

    skills = _linkedin_skills_by_job()
    postings = postings.merge(skills, on="job_id", how="left")
    return postings
