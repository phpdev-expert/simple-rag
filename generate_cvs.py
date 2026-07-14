"""Generate 10 sample CV PDF files under data/cvs/.

Run: python generate_cvs.py
"""
from __future__ import annotations

import os

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "cvs")

# 10 fictional candidates with varied backgrounds so screening has signal.
CANDIDATES = [
    {
        "name": "Ava Chen",
        "title": "Senior Backend Engineer",
        "email": "ava.chen@example.com",
        "location": "San Francisco, CA",
        "years": 8,
        "skills": ["Python", "Django", "PostgreSQL", "AWS", "Docker", "Kubernetes", "REST APIs"],
        "summary": "Backend engineer with 8 years building scalable Python services and distributed systems.",
        "experience": [
            ("Staff Engineer, Stripe (2021-Present)", "Led payments platform team; scaled services to 50k RPS."),
            ("Senior Engineer, Dropbox (2017-2021)", "Owned file-sync backend, cut latency 40%."),
        ],
        "education": "B.S. Computer Science, UC Berkeley",
    },
    {
        "name": "Marcus Boateng",
        "title": "Machine Learning Engineer",
        "email": "m.boateng@example.com",
        "location": "London, UK",
        "years": 5,
        "skills": ["Python", "PyTorch", "TensorFlow", "LangChain", "MLOps", "GCP", "SQL"],
        "summary": "ML engineer focused on NLP and LLM application development.",
        "experience": [
            ("ML Engineer, DeepMind (2020-Present)", "Built retrieval-augmented generation pipelines."),
            ("Data Scientist, Revolut (2018-2020)", "Fraud detection models, 25% fewer false positives."),
        ],
        "education": "M.S. Machine Learning, Imperial College London",
    },
    {
        "name": "Priya Nair",
        "title": "Full-Stack Developer",
        "email": "priya.nair@example.com",
        "location": "Bangalore, India",
        "years": 4,
        "skills": ["JavaScript", "React", "Node.js", "TypeScript", "MongoDB", "GraphQL"],
        "summary": "Full-stack developer building web apps with React and Node.",
        "experience": [
            ("Full-Stack Dev, Flipkart (2021-Present)", "Owned checkout UI serving 10M users."),
            ("Frontend Dev, Freshworks (2019-2021)", "Built reusable component library."),
        ],
        "education": "B.Tech Information Technology, VIT Vellore",
    },
    {
        "name": "Diego Fernandez",
        "title": "DevOps Engineer",
        "email": "diego.f@example.com",
        "location": "Madrid, Spain",
        "years": 6,
        "skills": ["AWS", "Terraform", "Kubernetes", "Docker", "CI/CD", "Python", "Bash"],
        "summary": "DevOps engineer automating cloud infrastructure at scale.",
        "experience": [
            ("Senior DevOps, Cabify (2019-Present)", "Migrated 200+ services to Kubernetes."),
            ("SRE, Telefonica (2016-2019)", "Built IaC pipelines with Terraform."),
        ],
        "education": "B.S. Software Engineering, Universidad Politecnica de Madrid",
    },
    {
        "name": "Sarah Kim",
        "title": "Data Scientist",
        "email": "sarah.kim@example.com",
        "location": "Seoul, South Korea",
        "years": 7,
        "skills": ["Python", "R", "SQL", "scikit-learn", "Pandas", "Spark", "Tableau"],
        "summary": "Data scientist with strong statistics and experimentation background.",
        "experience": [
            ("Lead Data Scientist, Coupang (2020-Present)", "A/B testing platform and demand forecasting."),
            ("Data Scientist, Naver (2016-2020)", "Recommendation systems for search."),
        ],
        "education": "Ph.D. Statistics, Seoul National University",
    },
    {
        "name": "Tom Whitfield",
        "title": "Junior Software Engineer",
        "email": "tom.w@example.com",
        "location": "Austin, TX",
        "years": 1,
        "skills": ["Java", "Spring Boot", "MySQL", "Git"],
        "summary": "Recent grad eager to grow as a backend engineer.",
        "experience": [
            ("Software Engineer Intern, IBM (2023)", "Built internal reporting microservice."),
        ],
        "education": "B.S. Computer Science, University of Texas at Austin",
    },
    {
        "name": "Fatima Al-Sayed",
        "title": "Cloud Solutions Architect",
        "email": "fatima.as@example.com",
        "location": "Dubai, UAE",
        "years": 10,
        "skills": ["AWS", "Azure", "Kubernetes", "Python", "Terraform", "Microservices", "System Design"],
        "summary": "Cloud architect with a decade designing enterprise-grade systems.",
        "experience": [
            ("Principal Architect, Emirates (2018-Present)", "Led cloud migration for airline booking platform."),
            ("Solutions Architect, SAP (2014-2018)", "Designed multi-region deployments."),
        ],
        "education": "M.S. Computer Engineering, American University of Sharjah",
    },
    {
        "name": "Liam O'Brien",
        "title": "Mobile Engineer",
        "email": "liam.ob@example.com",
        "location": "Dublin, Ireland",
        "years": 5,
        "skills": ["Swift", "Kotlin", "iOS", "Android", "Firebase", "REST APIs"],
        "summary": "Mobile engineer shipping native iOS and Android apps.",
        "experience": [
            ("Senior Mobile Engineer, Intercom (2020-Present)", "Rebuilt messenger SDK, 4.8 star rating."),
            ("iOS Developer, Bank of Ireland (2018-2020)", "Mobile banking app features."),
        ],
        "education": "B.S. Computer Science, Trinity College Dublin",
    },
    {
        "name": "Nadia Petrova",
        "title": "AI Research Engineer",
        "email": "nadia.p@example.com",
        "location": "Berlin, Germany",
        "years": 6,
        "skills": ["Python", "PyTorch", "LangChain", "LangGraph", "Transformers", "CUDA", "MLOps"],
        "summary": "Research engineer building LLM agents and multi-agent systems.",
        "experience": [
            ("AI Engineer, Hugging Face (2021-Present)", "Contributed to open-source agent frameworks."),
            ("Research Engineer, Zalando (2018-2021)", "Product-recommendation deep learning."),
        ],
        "education": "M.S. Artificial Intelligence, TU Berlin",
    },
    {
        "name": "Carlos Mendoza",
        "title": "Engineering Manager",
        "email": "carlos.m@example.com",
        "location": "Mexico City, Mexico",
        "years": 12,
        "skills": ["Python", "Go", "Leadership", "System Design", "AWS", "Agile", "Mentoring"],
        "summary": "Engineering manager leading distributed backend teams.",
        "experience": [
            ("Engineering Manager, MercadoLibre (2017-Present)", "Managed 3 teams, 20 engineers."),
            ("Tech Lead, Nubank (2013-2017)", "Built core ledger service in Go."),
        ],
        "education": "B.S. Computer Science, UNAM",
    },
]


def build_pdf(candidate: dict, path: str) -> None:
    doc = SimpleDocTemplate(
        path, pagesize=LETTER,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
    )
    styles = getSampleStyleSheet()
    name_style = ParagraphStyle("Name", parent=styles["Title"], fontSize=22, spaceAfter=2)
    title_style = ParagraphStyle("JobTitle", parent=styles["Normal"], fontSize=12, textColor="#555555", spaceAfter=10)
    heading = ParagraphStyle("Heading", parent=styles["Heading2"], fontSize=13, spaceBefore=12, spaceAfter=4)

    flow = []
    flow.append(Paragraph(candidate["name"], name_style))
    flow.append(Paragraph(candidate["title"], title_style))
    flow.append(Paragraph(
        f'{candidate["email"]} &nbsp;|&nbsp; {candidate["location"]} &nbsp;|&nbsp; '
        f'{candidate["years"]} years experience',
        styles["Normal"],
    ))

    flow.append(Paragraph("Summary", heading))
    flow.append(Paragraph(candidate["summary"], styles["Normal"]))

    flow.append(Paragraph("Skills", heading))
    flow.append(Paragraph(", ".join(candidate["skills"]), styles["Normal"]))

    flow.append(Paragraph("Experience", heading))
    for role, detail in candidate["experience"]:
        flow.append(Paragraph(f"<b>{role}</b>", styles["Normal"]))
        flow.append(Paragraph(detail, styles["Normal"]))
        flow.append(Spacer(1, 4))

    flow.append(Paragraph("Education", heading))
    flow.append(Paragraph(candidate["education"], styles["Normal"]))

    doc.build(flow)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for i, candidate in enumerate(CANDIDATES, start=1):
        slug = candidate["name"].lower().replace(" ", "_").replace("'", "")
        path = os.path.join(OUT_DIR, f"{i:02d}_{slug}.pdf")
        build_pdf(candidate, path)
        print(f"  wrote {path}")
    print(f"\nGenerated {len(CANDIDATES)} CV PDFs in {OUT_DIR}")


if __name__ == "__main__":
    main()
