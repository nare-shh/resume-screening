"""
generate_data.py
-----------------
Generates synthetic labeled resume-job description pairs for training.
Labels: 1 = Good Match (Shortlist), 0 = Poor Match (Reject)
"""

import pandas as pd
import random
import os

random.seed(42)

# ─── Skills pools ────────────────────────────────────────────────────────────

DATA_SCIENCE_SKILLS = [
    "Python", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch",
    "Scikit-learn", "Pandas", "NumPy", "SQL", "Data Analysis", "Statistics",
    "NLP", "Computer Vision", "MLflow", "Docker", "Jupyter", "Matplotlib",
    "Seaborn", "XGBoost", "LightGBM", "Feature Engineering", "A/B Testing",
    "Tableau", "Power BI", "Apache Spark", "Hadoop", "AWS", "GCP",
]

SOFTWARE_SKILLS = [
    "Java", "Python", "JavaScript", "TypeScript", "React", "Node.js",
    "Spring Boot", "REST API", "Microservices", "Docker", "Kubernetes",
    "CI/CD", "Jenkins", "Git", "AWS", "Azure", "SQL", "PostgreSQL",
    "MongoDB", "Redis", "GraphQL", "Agile", "Scrum", "Linux", "C++",
    "Go", "Rust", "System Design", "DSA",
]

HR_SKILLS = [
    "Recruitment", "Talent Acquisition", "Employee Relations", "HRMS",
    "Onboarding", "Performance Management", "Payroll", "Labor Law",
    "Conflict Resolution", "Training & Development", "Culture Building",
    "Compensation & Benefits", "Succession Planning", "SAP HR", "Workday",
    "Job Portals", "LinkedIn Recruiter", "Communication", "Leadership",
]

MARKETING_SKILLS = [
    "Digital Marketing", "SEO", "SEM", "Google Analytics", "Facebook Ads",
    "Content Strategy", "Copywriting", "Email Marketing", "CRM", "HubSpot",
    "Social Media Marketing", "Brand Management", "Market Research",
    "A/B Testing", "Conversion Optimization", "Product Marketing",
    "Campaign Management", "Data-Driven Marketing",
]

FINANCE_SKILLS = [
    "Financial Analysis", "Budgeting", "Forecasting", "Excel", "SAP",
    "GAAP", "IFRS", "Risk Management", "Investment Analysis", "Valuation",
    "Financial Modeling", "Accounting", "Audit", "Tax", "Bloomberg",
    "Python", "SQL", "Tableau", "Portfolio Management",
]

ROLES = {
    "Data Scientist": {"skills": DATA_SCIENCE_SKILLS, "required": 8},
    "Software Engineer": {"skills": SOFTWARE_SKILLS, "required": 8},
    "HR Manager": {"skills": HR_SKILLS, "required": 6},
    "Marketing Manager": {"skills": MARKETING_SKILLS, "required": 6},
    "Finance Analyst": {"skills": FINANCE_SKILLS, "required": 6},
}

EDUCATION_LEVELS = [
    "B.Tech in Computer Science", "M.Tech in AI/ML", "MBA", "B.E in Electronics",
    "B.Sc in Statistics", "M.Sc in Data Science", "B.Com", "CA", "PhD in Mathematics",
    "BBA", "M.S in Computer Science",
]

COMPANIES = [
    "Google", "Microsoft", "Amazon", "Infosys", "TCS", "Wipro",
    "Accenture", "IBM", "Deloitte", "McKinsey", "Goldman Sachs",
    "JP Morgan", "Flipkart", "Swiggy", "Zomato", "Paytm",
]


def generate_resume(role_name: str, match: bool) -> str:
    """Generate a synthetic resume text."""
    role_info = ROLES[role_name]
    all_skills = role_info["skills"]
    req_count = role_info["required"]

    if match:
        # Pick mostly matching skills
        skills_chosen = random.sample(all_skills, min(req_count + random.randint(0, 4), len(all_skills)))
        years_exp = random.randint(2, 10)
    else:
        # Pick a different role's skills + a few matching ones
        other_roles = [r for r in ROLES if r != role_name]
        other_role = random.choice(other_roles)
        other_skills = ROLES[other_role]["skills"]
        skills_chosen = random.sample(other_skills, min(req_count, len(other_skills)))
        # Add only 1-2 relevant skills
        skills_chosen += random.sample(all_skills, min(2, len(all_skills)))
        years_exp = random.randint(1, 5)

    random.shuffle(skills_chosen)
    education = random.choice(EDUCATION_LEVELS)
    company = random.choice(COMPANIES)
    prev_company = random.choice(COMPANIES)

    resume = f"""
PROFESSIONAL SUMMARY
Experienced professional with {years_exp} years of experience in {role_name if match else random.choice(list(ROLES.keys()))} domain.
Proven track record at {prev_company} and {company}.

EDUCATION
{education} | Graduated {random.randint(2010, 2022)}

TECHNICAL SKILLS
{', '.join(skills_chosen)}

WORK EXPERIENCE
{company} | {random.randint(1, years_exp)} years
- Led multiple projects and delivered results in cross-functional teams
- Collaborated with stakeholders to align technical solutions with business goals
- Worked on end-to-end pipelines and system design

{prev_company} | {years_exp - random.randint(0, min(2, years_exp))} years
- Contributed to core product development
- Participated in Agile/Scrum processes
- Mentored junior team members

CERTIFICATIONS
{random.choice(['AWS Certified', 'Google Cloud Professional', 'PMP', 'CFA Level 1', 'Certified Scrum Master', 'SHRM-CP'])}
    """.strip()
    return resume


def generate_job_description(role_name: str) -> str:
    """Generate a job description for a given role."""
    role_info = ROLES[role_name]
    all_skills = role_info["skills"]
    req_skills = random.sample(all_skills, min(role_info["required"], len(all_skills)))
    years = random.randint(2, 7)

    jd = f"""
JOB TITLE: {role_name}

ABOUT THE ROLE
We are looking for a talented {role_name} to join our growing team.
The ideal candidate will have {years}+ years of experience.

KEY RESPONSIBILITIES
- Design and implement solutions to complex business problems
- Collaborate with cross-functional teams
- Drive innovation and best practices
- Mentor junior team members and contribute to knowledge sharing

REQUIRED SKILLS
{', '.join(req_skills)}

QUALIFICATIONS
- {years}+ years of relevant experience
- Strong analytical and problem-solving skills
- Excellent communication and team collaboration skills

NICE TO HAVE
{', '.join(random.sample(all_skills, min(3, len(all_skills))))}
    """.strip()
    return jd


def generate_dataset(n_samples: int = 400) -> pd.DataFrame:
    """Generate a balanced dataset of resume-JD pairs."""
    records = []
    roles = list(ROLES.keys())

    for _ in range(n_samples):
        role = random.choice(roles)
        match = random.random() > 0.45  # ~55% match, 45% no-match for slight imbalance

        resume_text = generate_resume(role, match)
        job_desc = generate_job_description(role)
        label = 1 if match else 0

        records.append({
            "resume_text": resume_text,
            "job_desc": job_desc,
            "role": role,
            "label": label,
        })

    df = pd.DataFrame(records)
    print(f"Dataset generated: {len(df)} samples | Match: {df['label'].sum()} | No-Match: {(df['label'] == 0).sum()}")
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(400)
    df.to_csv("data/resume_dataset.csv", index=False)
    print("Saved to data/resume_dataset.csv")
