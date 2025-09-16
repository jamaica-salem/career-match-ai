import re
import time
import string
from typing import List, Set, Tuple
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gradio as gr

# ---------------------------
# Technical and Soft Skills
# ---------------------------
TECHNICAL_SKILLS = [
    # Programming Languages
    "python", "java", "c", "c++", "c#", "go", "golang", "rust", "javascript", "typescript",
    "scala", "kotlin", "swift", "objective-c", "ruby", "php", "perl", "r", "matlab",

    # Frontend Development
    "html", "css", "sass", "less", "jquery", "bootstrap",
    "react", "react.js", "reactjs", "vue", "vue.js", "angular", "svelte",
    "next.js", "next", "nuxt", "ember", "backbone",

    # Backend Development & Architectures
    "node.js", "node", "express", "django", "flask", "spring", "spring boot",
    "laravel", "asp.net", "ruby on rails", "rails", "fastapi", "gin", "phoenix",
    "graphql", "rest api", "microservices", "monolith", "serverless", "event-driven",

    # Databases & Data Engineering
    "sql", "mysql", "postgres", "postgresql", "mariadb", "sqlite", "oracle",
    "mongodb", "cassandra", "redis", "memcached", "dynamodb", "cockroachdb",
    "timescale", "neo4j", "firebase", "fauna",
    "data warehouse", "data lake", "big data", "hadoop", "spark", "apache spark",
    "hive", "pig", "etl", "airflow", "beam", "kafka", "kinesis", "databricks",
    "data analysis", "data engineering", "data modelling", "bigquery", "redshift",
    "snowflake", "dbt",

    # Machine Learning / AI
    "machine learning", "ml", "deep learning", "neural networks",
    "tensorflow", "keras", "pytorch", "scikit-learn", "xgboost", "lightgbm",
    "catboost", "nlp", "computer vision", "predictive modeling",
    "mlflow", "onnx", "tensorflow serving", "kubeflow",

    # Data Visualization & BI
    "pandas", "numpy", "matplotlib", "seaborn", "power bi", "tableau",
    "looker", "qlik", "superset", "d3.js",

    # Cloud & DevOps / Infrastructure
    "aws", "amazon web services", "azure", "microsoft azure", "gcp",
    "google cloud", "google cloud platform", "cloud", "cloud computing",
    "ec2", "s3", "lambda", "cloud security", "apigateway", "iam",
    "cloudformation", "terraform", "ansible", "chef", "puppet",
    "docker", "kubernetes", "k8s", "helm", "istio", "containerd",
    "ci/cd", "jenkins", "github actions", "gitlab ci", "circleci",
    "bamboo", "teamcity", "buildkite",

    # Security & Networking
    "cybersecurity", "security", "application security", "devsecops",
    "owasp", "penetration testing", "vulnerability assessment", "siem",
    "splunk", "sast", "dast", "burpsuite", "wireshark",
    "firewalls", "networking", "network protocol", "dns", "http",
    "https", "tls", "ssl", "tcp/ip", "vpn", "intrusion detection",
    "endpoint protection",

    # Monitoring & Observability
    "prometheus", "grafana", "datadog", "new relic", "elk", "elasticsearch",
    "logstash", "kibana", "opentelemetry", "monitoring",

    # Version Control & Project Tools
    "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
    "jira", "confluence", "notion", "trello", "asana", "clickup",
    "monday.com", "slack", "teams",

    # Operating Systems & Tools
    "linux", "unix", "bash", "powershell", "windows server", "macos",
    "vscode", "visual studio", "eclipse", "intellij",
    "shell scripting", "virtualization", "container", "server management",

    # Testing & Quality Assurance
    "testing", "test automation", "unit testing", "integration testing",
    "selenium", "cypress", "playwright", "junit", "pytest", "mocha",
    "jest", "karma", "robot framework", "qtp",
    "performance testing", "load testing", "profiling", "benchmarking",

    # Mobile Development
    "android", "ios", "react native", "react-native", "flutter", "xamarin", "ionic",

    # Design & UX (Graphic Designers)
    "ux", "ui", "accessibility", "a11y",
    "figma", "adobe xd", "sketch", "photoshop", "illustrator", "indesign",
    "lightroom", "after effects", "premiere pro", "xd", "canva",
    "coreldraw", "affinity designer", "procreate", "blender", "cinema 4d",
    "graphic design", "design systems", "ui design", "ux research",

    # Content Creation (Design + Marketing)
    "video editing", "motion graphics", "3d modeling", "wireframing",
    "prototyping", "branding", "typography", "color theory",

    # Virtual Assistant & Productivity Tools
    "microsoft office", "excel", "word", "powerpoint", "outlook",
    "google workspace", "google sheets", "google docs", "google slides",
    "calendar management", "crm tools", "salesforce", "hubspot", "zoho crm",
    "zapier", "ifttt", "quickbooks", "xero",

    # Compliance & Data
    "gdpr", "hipaa", "pci-dss", "data governance", "data privacy",
    "business intelligence", "analytics",

    # Programming Concepts & Software Practices
    "api", "sdk", "ide", "oop", "ddd", "tdd", "bdd", "design patterns",
    "software architecture", "system design", "distributed systems",
    "container orchestration", "real time systems", "message queue",
    "rabbitmq", "activemq", "sqs", "grpc",
    "agile", "scrum", "kanban", "product management",
    "documentation", "technical writing"
]


SOFT_SKILLS = [
    # Communication
    "communication", "verbal communication", "written communication",
    "presentation skills", "public speaking", "listening", "storytelling",
    "cross-cultural communication", "client communication", "email etiquette",

    # Team & Leadership
    "teamwork", "collaboration", "working in cross-functional teams",
    "leadership", "mentoring", "coaching", "stakeholder management",
    "delegation", "people management", "relationship building",

    # Thinking & Problem Solving
    "problem solving", "analytical thinking", "critical thinking",
    "decision making", "strategic thinking", "logical reasoning",

    # Adaptability & Work Style
    "adaptability", "resilience", "flexibility", "time management",
    "organization", "prioritization", "multitasking",
    "stress management", "adaptation to change", "reliability",
    "self motivation", "initiative", "proactivity",

    # Creativity & Design
    "creativity", "innovation", "visual storytelling",
    "aesthetic sense", "user-centric thinking", "design thinking",
    "branding awareness", "conceptual thinking",

    # Emotional & Social
    "empathy", "emotional intelligence", "conflict resolution",
    "negotiation", "patience", "diplomacy", "cultural sensitivity",

    # Professional & Work Ethic
    "attention to detail", "quality conscious", "accountability",
    "dependability", "work ethic", "professionalism", "confidentiality",
    "discretion", "integrity",

    # Customer & Client Orientation
    "customer focus", "client relationship management",
    "user empathy", "service orientation", "feedback reception",
    "active listening", "problem ownership",

    # Continuous Growth
    "curiosity", "continuous learning", "knowledge sharing",
    "coaching others", "mentoring", "growth mindset",

    # Virtual Assistant & Remote Work Specific
    "calendar management", "email management", "task coordination",
    "remote collaboration", "independence", "virtual meeting etiquette",
    "documentation", "follow-through", "organizational support",

    # Graphic Design Specific
    "visual communication", "storyboarding", "concept presentation",
    "collaboration with clients", "brand consistency",
    "receiving and applying feedback", "creative brainstorming"
]

# ---------------------------
# Mega Skill Synonyms Dictionary
# ---------------------------
SKILL_SYNONYMS = {
    # Programming Languages
    "py": "python",
    "python3": "python",
    "java script": "javascript",
    "js": "javascript",
    "ts": "typescript",
    "csharp": "c#",
    "golang": "go",
    "objc": "objective-c",
    "r lang": "r",
    "mat lab": "matlab",
    "c plus plus": "c++",

    # Frontend
    "reactjs": "react",
    "react.js": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "angularjs": "angular",
    "nextjs": "next.js",
    "nuxtjs": "nuxt",
    "bs": "bootstrap",
    "jq": "jquery",
    "sass/scss": "sass",
    "less css": "less",

    # Backend
    "node": "node.js",
    "expressjs": "express",
    "rails": "ruby on rails",
    "ror": "ruby on rails",
    "fast api": "fastapi",
    "rest": "rest api",
    "springboot": "spring boot",
    "asp net": "asp.net",
    "phoenix framework": "phoenix",

    # Databases / Data
    "pg": "postgresql",
    "postgres": "postgresql",
    "sqlite3": "sqlite",
    "dw": "data warehouse",
    "dl": "data lake",
    "bi": "business intelligence",
    "etl": "etl",
    "dbt": "dbt",
    "big data": "big data",
    "data analytics": "data analysis",
    "data eng": "data engineering",
    "data modeling": "data modelling",
    "bq": "bigquery",
    "rs": "redshift",
    "sf": "snowflake",

    # ML / AI
    "ml": "machine learning",
    "dl": "deep learning",
    "nn": "neural networks",
    "tf": "tensorflow",
    "torch": "pytorch",
    "sklearn": "scikit-learn",
    "xgb": "xgboost",
    "lgbm": "lightgbm",
    "cb": "catboost",
    "cv": "computer vision",
    "nlp": "nlp",
    "predictive modeling": "predictive modeling",
    "onnx": "onnx",
    "kubeflow": "kubeflow",
    "mlflow": "mlflow",
    "tf serving": "tensorflow serving",

    # Data Viz / BI
    "pp": "power bi",
    "powerbi": "power bi",
    "ms excel": "excel",
    "ms word": "word",
    "ms powerpoint": "powerpoint",
    "pyplot": "matplotlib",
    "d3": "d3.js",
    "tbl": "tableau",

    # Cloud / DevOps
    "aws": "amazon web services",
    "azure": "microsoft azure",
    "gcp": "google cloud platform",
    "google cloud": "google cloud platform",
    "ec2": "ec2",
    "s3": "s3",
    "lambda": "lambda",
    "iam": "iam",
    "apigw": "apigateway",
    "ci/cd": "ci/cd",
    "jenkins ci": "jenkins",
    "docker container": "docker",
    "k8s": "kubernetes",
    "helm chart": "helm",
    "istio service mesh": "istio",

    # Security / Networking
    "sec": "security",
    "cyber sec": "cybersecurity",
    "app sec": "application security",
    "devsec": "devsecops",
    "vpn": "vpn",
    "tls": "tls",
    "ssl": "ssl",
    "tcp": "tcp/ip",
    "ids": "intrusion detection",
    "firewall": "firewalls",

    # Monitoring / Observability
    "elk": "elk",
    "es": "elasticsearch",
    "log": "logstash",
    "graf": "grafana",
    "dd": "datadog",
    "newrelic": "new relic",

    # Version Control / Project Tools
    "gh": "github",
    "gl": "gitlab",
    "bb": "bitbucket",
    "svn": "svn",
    "hg": "mercurial",
    "conf": "confluence",
    "asana": "asana",
    "trello": "trello",
    "notion": "notion",
    "clickup": "clickup",
    "slack": "slack",
    "ms teams": "teams",

    # OS / Tools
    "win": "windows server",
    "osx": "macos",
    "vs code": "vscode",
    "intellij idea": "intellij",
    "bash scripting": "bash",
    "shell": "shell scripting",

    # Testing / QA
    "unit test": "unit testing",
    "integration test": "integration testing",
    "e2e": "test automation",
    "perf testing": "performance testing",
    "load test": "load testing",
    "qa": "testing",

    # Mobile
    "rn": "react native",
    "flutter sdk": "flutter",
    "xam": "xamarin",

    # Design / UX
    "uiux": "ui",
    "adobe xd": "xd",
    "ps": "photoshop",
    "ai": "illustrator",
    "lr": "lightroom",
    "ae": "after effects",
    "pr": "premiere pro",
    "cinema4d": "cinema 4d",

    # Productivity / VA
    "gsheets": "google sheets",
    "gdocs": "google docs",
    "gslides": "google slides",
    "crm": "crm tools",
    "zap": "zapier",
    "ifttt": "ifttt",

    # Soft Skills / Shorthand
    "comm": "communication",
    "verbal comm": "verbal communication",
    "written comm": "written communication",
    "lead": "leadership",
    "tm": "teamwork",
    "ps": "problem solving",
    "tmgmt": "time management",
    "adapt": "adaptability",
    "resil": "resilience",
    "flex": "flexibility",
    "org": "organization",
    "priorit": "prioritization",
    "multi task": "multitasking",
    "creat": "creativity",
    "emot intel": "emotional intelligence",
    "acct": "accountability",
    "growth": "growth mindset",
    "curio": "curiosity",
    "coaching": "mentoring",
    "negotiat": "negotiation",
    "problem own": "problem ownership"
}

# ---------------------------
# Load Sentence-Transformer Model (CPU-friendly)
# ---------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# ---------------------------
# Helper Functions
# ---------------------------
@lru_cache(maxsize=128)
def get_embedding(text: str) -> np.ndarray:
    if not text:
        return np.zeros((model.get_sentence_embedding_dimension(),), dtype=float)
    return model.encode([text], convert_to_numpy=True)[0]

def clean_text(text: str) -> str:
    return (text or "").strip()

def preprocess_text_for_skills(text: str) -> str:
    if not text:
        return ""
    text_low = text.lower()
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text_clean = text_low.translate(translator)
    return " ".join(text_clean.split())

def extract_skills(text: str, skills_list: List[str]) -> Set[str]:
    text_clean = preprocess_text_for_skills(text)
    found_skills = set()
    skill_map = {**{s.lower(): s for s in skills_list},
                 **{syn.lower(): real for syn, real in SKILL_SYNONYMS.items() if real in skills_list}}
    for key, actual_skill in skill_map.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, text_clean):
            found_skills.add(actual_skill)
    return found_skills

def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    try:
        return float(np.clip(cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0], 0.0, 1.0))
    except Exception:
        return 0.0

def format_score_pct(sim: float) -> str:
    return f"{int(round(sim * 100))}%"

def top_keywords_from_text(source: str, top_n: int = 8) -> List[str]:
    if not source:
        return []
    tokens = re.findall(r"[a-zA-Z0-9+#+\.\-]+", source.lower())
    stopwords = {"the","and","for","with","to","a","an","of","in","on","by","is","are","be","as","or"}
    freq = {}
    for t in tokens:
        if len(t) <= 2 or t in stopwords:
            continue
        freq[t] = freq.get(t, 0) + 1
    return sorted(freq, key=lambda k: (-freq[k], k))[:top_n]

def generate_suggestions(missing_skills: Set[str], jd_top_keywords: List[str], pct: int) -> str:
    bullets = []
    missing_list = sorted(list(missing_skills))[:5]
    if pct < 50:
        bullets.append("**Add key missing skills:** " + ", ".join(missing_list))
        bullets.append("**Highlight transferable skills:** Projects, coursework, or volunteering.")
        bullets.append("**Entry-level guidance:** Consider certifications or online courses.")
        bullets.append("**Use simple phrasing:** Short sentences and bullet points.")
        bullets.append("**Next steps:** Tailor resume and increase match percentage.")
    elif 50 <= pct <= 80:
        bullets.append("**Close minor gaps:** Add missing skills: " + ", ".join(missing_list))
        bullets.append("**Optimize phrasing:** Use action verbs and quantify achievements.")
        bullets.append("**Highlight experience:** Ensure keywords and relevant experience visible.")
        bullets.append("**Refine formatting:** Consistent bullets, fonts, and sections.")
        bullets.append("**Next steps:** Re-run analysis after updates.")
    else:
        bullets.append("**Fine-tune your resume:** Clean formatting and concise bullet points.")
        bullets.append("**Emphasize accomplishments:** Measurable impact and leadership.")
        bullets.append("**Include optional skills:** " + ", ".join(missing_list))
        bullets.append("**Refine keywords:** Integrate top JD keywords naturally.")
        bullets.append("**Advanced tips:** Tailor each section for ATS and relevance.")
    return "\n\n".join(bullets)

def render_donut_svg(pct: int, size: int = 180) -> str:
    uid = str(int(time.time() * 1000))[-6:]
    pct = max(0, min(100, pct))
    radius = (size-32)/2
    cx = cy = size/2
    circumference = 2*np.pi*radius
    dash = circumference*pct/100
    gap = circumference - dash
    stroke_color = "#ef4444" if pct<50 else "#facc15" if pct<=80 else "#22c55e"
    return f'''
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
  <circle cx="{cx}" cy="{cy}" r="{radius}" stroke="#334155" stroke-width="14" fill="none"/>
  <circle cx="{cx}" cy="{cy}" r="{radius}" stroke="{stroke_color}" stroke-width="14" fill="none" stroke-linecap="round" stroke-dasharray="{dash} {gap}" transform="rotate(-90 {cx} {cy})"/>
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" style="font-size:20px;font-weight:700;fill:#ffffff;">{pct}%</text>
</svg>
'''

# ---------------------------
# Main Analysis
# ---------------------------
def analyze(jd_text: str, resume_text: str) -> Tuple[str, str, str, str, str, int]:
    jd, resume = clean_text(jd_text), clean_text(resume_text)
    pct = int(round(cosine_score(get_embedding(jd), get_embedding(resume))*100))
    donut_svg = render_donut_svg(pct)

    jd_tech = extract_skills(jd, TECHNICAL_SKILLS)
    resume_tech = extract_skills(resume, TECHNICAL_SKILLS)
    jd_soft = extract_skills(jd, SOFT_SKILLS)
    resume_soft = extract_skills(resume, SOFT_SKILLS)

    missing_skills = (jd_tech | jd_soft) - (resume_tech | resume_soft)
    missing_text_tech = "\n".join(sorted(jd_tech - resume_tech)) if jd_tech - resume_tech else "None"
    missing_text_soft = "\n".join(sorted(jd_soft - resume_soft)) if jd_soft - resume_soft else "None"

    jd_keywords = top_keywords_from_text(jd, 8)
    suggestions = generate_suggestions(missing_skills, jd_keywords, pct)

    explanation = f"**Match Score:** {pct}%\n\nThis is cosine similarity between JD and resume embeddings.\n\n"
    score_md = explanation + f"**Top JD Keywords:** {', '.join(jd_keywords)}<br>{render_keyword_heatmap(jd_keywords, jd)}"

    return donut_svg, score_md, missing_text_tech, missing_text_soft, suggestions, pct

def format_skill_chips(skills: str, skill_type: str) -> str:
    if not skills or skills.lower()=="none":
        return "<span class='small-note'>None</span>"
    colors = {"tech":["#3b82f6","#06b6d4","#0ea5e9","#2563eb"], "soft":["#a78bfa","#c084fc","#d8b4fe","#e9d5ff"]}
    skill_items = [s.strip("- ").title() for s in skills.splitlines()]
    return "".join(f'<span style="display:inline-block;margin:4px;padding:6px 10px;border-radius:12px;background:{colors.get(skill_type,["#94a3b8"])[i%4]};font-weight:600;font-size:13px;">{s}</span>' for i,s in enumerate(skill_items))

# ---------------------------
# Gradio UI
# ---------------------------
def build_ui():
    with gr.Blocks() as demo:
        gr.HTML("<h2>Resume — Job Match Scorer</h2>")
        jd_input = gr.Textbox(lines=14, placeholder="Job Description")
        resume_input = gr.Textbox(lines=14, placeholder="Resume / CV")
        run_btn = gr.Button("Analyze")

        out_chart = gr.HTML(render_donut_svg(0))
        out_score = gr.Markdown("Score will appear here.")
        out_missing_tech = gr.Markdown("Missing technical skills here.")
        out_missing_soft = gr.Markdown("Missing soft skills here.")
        out_suggestions = gr.Markdown("Suggestions here.")

        def _on_click(jd, resume):
            donut, score_md, missing_tech, missing_soft, suggestions, pct = analyze(jd, resume)
            missing_tech_chips = format_skill_chips(missing_tech, "tech")
            missing_soft_chips = format_skill_chips(missing_soft, "soft")
            return donut, score_md, missing_tech_chips, missing_soft_chips, suggestions

        run_btn.click(
            _on_click, 
            inputs=[jd_input, resume_input],
            outputs=[out_chart, out_score, out_missing_tech, out_missing_soft, out_suggestions]
        )

    return demo

if __name__ == "__main__":
    build_ui().launch(share=True)
