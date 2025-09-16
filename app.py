#app.py

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
# Extensive curated IT skills list
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
# Load small sentence-transformers model once
# ---------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# ---------------------------
# Helper functions
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

    # Lowercase
    text_low = text.lower()

    # Normalize common word separators to space
    separators = ["-", "_", "/"]
    for sep in separators:
        text_low = text_low.replace(sep, " ")

    # Replace remaining punctuation with spaces
    translator = str.maketrans(string.punctuation, " " * len(string.punctuation))
    text_clean = text_low.translate(translator)

    # Collapse multiple spaces
    text_clean = " ".join(text_clean.split())

    return text_clean

def extract_skills(text: str, skills_list: List[str]) -> Set[str]:
    """
    Extract exact and synonym-mapped skills using word boundaries to reduce false positives.
    """
    text_clean = preprocess_text_for_skills(text)
    found_skills = set()

    # Merge skill synonyms into a lookup
    skill_map = {**{s.lower(): s for s in skills_list}, **{syn.lower(): real for syn, real in SKILL_SYNONYMS.items() if real in skills_list}}

    for key, actual_skill in skill_map.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        if re.search(pattern, text_clean):
            found_skills.add(actual_skill)

    return found_skills

def cosine_score(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    try:
        sim = cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]
    except Exception:
        sim = 0.0
    return float(np.clip(sim, 0.0, 1.0))

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
    keywords = sorted(freq, key=lambda k: (-freq[k], k))
    return keywords[:top_n]

def format_matched_skills(jd_skills: Set[str], resume_skills: Set[str]) -> str:
    matched = jd_skills & resume_skills
    if not matched:
        return "<span class='small-note'>No matched skills detected.</span>"
    skill_items = sorted(matched)
    colors = ["#22c55e", "#16a34a", "#4ade80", "#86efac"]  # green tones
    chips_html = ""
    for i, skill in enumerate(skill_items):
        color = colors[i % len(colors)]
        chips_html += f'<span style="display:inline-block;margin:4px;padding:6px 10px;border-radius:12px;background:{color};font-weight:600;font-size:13px;animation:chip-fade 1s ease;">{skill.title()}</span>'
    return chips_html

def render_keyword_heatmap(jd_keywords: List[str], jd_text: str) -> str:
    if not jd_keywords:
        return "<i>No keywords detected for wordmap.</i>"

    # Count frequencies
    tokens = re.findall(r"[a-zA-Z0-9+#+\.\-]+", jd_text.lower())
    freq = {k: tokens.count(k) for k in jd_keywords}
    max_freq = max(freq.values()) if freq else 1

    # Render as inline colored spans
    heatmap_html = ""
    for kw in jd_keywords:
        weight = int(14 + 16 * (freq.get(kw, 0) / max_freq))  # font size 14-30px
        opacity = 0.4 + 0.6 * (freq.get(kw, 0) / max_freq)      # opacity 0.4-1
        heatmap_html += f'<span style="font-size:{weight}px; opacity:{opacity}; margin:4px; display:inline-block;">{kw}</span>'
    return heatmap_html

def generate_suggestions(missing_skills: Set[str], jd_top_keywords: List[str], resume: str, pct: int) -> str:
    """
    Generate 5 dynamic improvement suggestions based on missing skills and match percentage.
    """
    bullets = []

    # Low match (<50%) → focus on missing key skills and entry-level guidance
    if pct < 50:
        bullets.append("**Add key missing skills:** Include these essential skills from the job description:\n" +
                       "\n".join(f"- {s.title()}" for s in sorted(list(missing_skills)[:5])))
        bullets.append("**Highlight transferable skills:** Show relevant projects, coursework, or volunteer work.")
        bullets.append("**Entry-level guidance:** Consider certifications or online courses for core missing skills.")
        bullets.append("**Use simple, clear phrasing:** Short sentences, bullet points, and consistent tense help readability.")
        bullets.append("**Next steps:** Tailor your resume for this role and aim to increase the match percentage before applying.")

    # Medium match (50–80%) → optimize phrasing, highlight experience, minor gaps
    elif 50 <= pct <= 80:
        bullets.append("**Close minor gaps:** Add these missing skills if relevant:\n" +
                       "\n".join(f"- {s.title()}" for s in sorted(list(missing_skills)[:5])))
        bullets.append("**Optimize phrasing:** Use action verbs and quantify achievements where possible.")
        bullets.append("**Highlight experience:** Move relevant experience to the top and ensure keywords are present.")
        bullets.append("**Refine formatting:** Ensure consistent fonts, bullet points, and clear sections.")
        bullets.append("**Next steps:** Re-run the analysis after updates to reach a high match score (>80%).")

    # High match (>80%) → fine-tuning, emphasizing achievements, advanced tips
    else:
        bullets.append("**Fine-tune your resume:** Ensure formatting is clean and bullet points are concise.")
        bullets.append("**Emphasize accomplishments:** Highlight measurable impact and leadership experiences.")
        bullets.append("**Include optional skills:** Add these missing skills to further strengthen your profile:\n" +
                       "\n".join(f"- {s.title()}" for s in sorted(list(missing_skills)[:5])))
        bullets.append("**Refine keywords:** Ensure top job description keywords are integrated naturally.")
        bullets.append("**Advanced tips:** Tailor each section for maximum relevance to this role and ATS-friendly wording.")

    # Always include JD keyword suggestion if space allows
    if jd_top_keywords and len(bullets) < 5:
        bullets.append(f"**Match keywords from the job description:** Include relevant terms like {', '.join(jd_top_keywords[:5])}.")

    return "\n\n".join(bullets)


# ---------------------------
# Main analysis
# ---------------------------

def analyze(job_description: str, resume_text: str) -> Tuple[str, str, str, str, str, int]:
    jd, resume = clean_text(job_description), clean_text(resume_text)
    sim = cosine_score(get_embedding(jd), get_embedding(resume))
    pct = int(round(sim * 100))

    # Donut SVG
    donut_svg = render_donut_svg(pct)

    # Detect missing technical skills
    jd_tech = extract_skills(jd, TECHNICAL_SKILLS)
    resume_tech = extract_skills(resume, TECHNICAL_SKILLS)
    missing_tech = jd_tech - resume_tech

    # Detect missing soft skills
    jd_soft = extract_skills(jd, SOFT_SKILLS)
    resume_soft = extract_skills(resume, SOFT_SKILLS)
    missing_soft = jd_soft - resume_soft

    missing_tech_text = "\n".join(f"- {s}" for s in sorted(missing_tech)) if missing_tech else "None"
    missing_soft_text = "\n".join(f"- {s}" for s in sorted(missing_soft)) if missing_soft else "None"

    jd_keywords = top_keywords_from_text(jd, top_n=8)
    suggestions_md = generate_suggestions(missing_tech | missing_soft, jd_keywords, resume, pct)

    explanation = (
        f"**Match Score:** {format_score_pct(sim)}\n\n"
        "This score is the cosine similarity between the job description and resume embeddings. "
        "It's a heuristic indicator — useful for tailoring but not a perfect predictor.\n\n"
    )
    score_md = explanation + f"**Top keywords from job description:** {', '.join(jd_keywords)}\n\n"

    return donut_svg, score_md, missing_tech_text, missing_soft_text, suggestions_md, pct


# ---------------------------
# Donut SVG
# ---------------------------

def render_donut_svg(pct: int, size: int = 180) -> str:
    uid = str(int(time.time() * 1000))[-6:]
    pct = max(0, min(100, int(pct)))

    radius = (size - 32) / 2
    cx = cy = size / 2
    circumference = 2 * np.pi * radius
    dash = circumference * pct / 100
    gap = circumference - dash

    # Determine color based on pct
    if pct < 50:
        stroke_color = "#ef4444"  # red
    elif pct <= 80:
        stroke_color = "#facc15"  # yellow
    else:
        stroke_color = "#22c55e"  # green

    return f'''
<div style="display:flex;align-items:center;justify-content:center;padding:6px;">
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
  <!-- Background circle -->
  <circle cx="{cx}" cy="{cy}" r="{radius}" stroke="#334155" stroke-width="14" fill="none"/>
  <!-- Progress arc -->
  <circle 
    cx="{cx}" cy="{cy}" r="{radius}" 
    stroke="{stroke_color}" 
    stroke-width="14" 
    fill="none" 
    stroke-linecap="round"
    stroke-dasharray="{dash} {gap}" 
    stroke-dashoffset="0"
    transform="rotate(-90 {cx} {cy})"
    style="transition: stroke-dasharray 1s ease-out;" />
  <!-- Center text -->
  <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
        style="font-size:20px;font-weight:700;fill:#ffffff;font-family:'Poppins';">
    {pct}%
  </text>
</svg>
</div>
'''

# ---------------------------
# Format missing skills as animated chips
# ---------------------------
def format_skill_chips(skills: str, skill_type: str) -> str:
    if not skills or skills.lower() == "none":
        return "<span class='small-note'>None</span>"
    colors = {
        "tech": ["#3b82f6", "#06b6d4", "#0ea5e9", "#2563eb"],
        "soft": ["#a78bfa", "#c084fc", "#d8b4fe", "#e9d5ff"]
    }
    color_list = colors.get(skill_type, ["#94a3b8"])
    skill_items = [s.strip("- ").title() for s in skills.splitlines()]
    chips_html = ""
    for i, skill in enumerate(skill_items):
        color = color_list[i % len(color_list)]
        chips_html += f'<span style="display:inline-block;margin:4px;padding:6px 10px;border-radius:12px;background:{color};font-weight:600;font-size:13px;animation:chip-fade 1s ease;">{skill}</span>'
    return chips_html

# ---------------------------
# Gradio UI
# ---------------------------

def build_ui():
    style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    * {
        font-family: 'Poppins' !important;
    }

    body { 
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%); 
        color: #f1f5f9;
    }
    .card { 
        background: #1e293b; 
        border-radius: 12px; 
        padding: 18px; 
        box-shadow: 0 6px 20px rgba(0,0,0,0.4); 
        color: #f1f5f9;
    }
    .muted { color: #94a3b8; }
    .header h2, .header div { color: #f8fafc; }
    .small-note { font-size:13px; color:#cbd5e1; }
    textarea, input[type=text] { 
        border-radius:8px; 
        border:1px solid #334155; 
        padding:10px; 
        background:#0f172a; 
        color:#f1f5f9; 
    }
    #analyze-btn { 
        background: linear-gradient(90deg,#4f46e5,#06b6d4); 
        color: white; 
        border: none; 
        padding: 10px 18px; 
        border-radius: 10px; 
        font-weight: 700; 
        box-shadow: 0 8px 20px rgba(0,0,0,0.4); 
        cursor: pointer; 
    }
    #analyze-btn:hover { 
        transform: translateY(-3px); 
        box-shadow: 0 12px 30px rgba(0,0,0,0.6); 
    }
    .logo { 
        width:56px; 
        height:56px; 
        border-radius:12px; 
        background: linear-gradient(135deg, #4f46e5, #06b6d4); 
        display:flex; align-items:center; 
        justify-content:center; 
        color:white; 
        font-weight:700; 
        font-size:18px; 
        font-family: 'Poppins', system-ui, sans-serif; 
        animation: logo-swish 1.6s ease infinite; 
    }
    @keyframes chip-fade {
      0% {opacity:0; transform: translateY(-6px);}
      100% {opacity:1; transform: translateY(0);}
    }
    </style>
    """

    with gr.Blocks(css="") as demo:
        gr.HTML(style)
        with gr.Column():
            with gr.Row():
                gr.HTML('<div class="header"><div class="logo">RJ</div><div><h2>Resume — Job Match Scorer for IT-related Jobs</h2><div class="muted">Paste a job description and a resume/CV. The app returns a match score, missing skills, and suggestions to improve your resume for this job.</div></div></div>')
            with gr.Row():
                with gr.Column(scale=7):
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("#### Job description")
                        jd_input = gr.Textbox(lines=14, placeholder="Paste the full job description here...")
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("#### Resume / CV")
                        resume_input = gr.Textbox(lines=14, placeholder="Paste the complete resume...")
                    run_btn = gr.Button("Analyze", elem_id="analyze-btn")
                with gr.Column(scale=5):
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("#### Match Overview")
                        out_chart = gr.HTML(render_donut_svg(0))
                        out_score = gr.Markdown("**Match score and explanation will appear here.**")
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("#### Matched Skills")
                        out_matched_skills = gr.HTML("Matched skills will appear here.")
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("#### Missing Technical Skills")
                        out_missing_tech = gr.Markdown("No missing technical skills detected yet.")
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("#### Missing Soft Skills")
                        out_missing_soft = gr.Markdown("No missing soft skills detected yet.")
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("#### Suggested Improvements")
                        out_suggestions = gr.Markdown("Suggested improvements will appear here.")
        def _on_click(jd, resume):
            donut_svg, score_md, missing_tech_text, missing_soft_text, suggestions, pct = analyze(jd, resume)
            
            # Format missing skill chips
            missing_tech_chips = format_skill_chips(missing_tech_text, "tech")
            missing_soft_chips = format_skill_chips(missing_soft_text, "soft")
        
            # Combine all skills for matched computation
            all_skills = TECHNICAL_SKILLS + SOFT_SKILLS
            
            # Extract skills from JD and resume
            jd_skills = extract_skills(jd, all_skills)
            resume_skills = extract_skills(resume, all_skills)
            
            # Compute matched skills (intersection)
            matched_chips_html = format_matched_skills(jd_skills, resume_skills)
            
            # Heatmap
            heatmap_html = render_keyword_heatmap(top_keywords_from_text(jd, 8), jd)
            score_md_with_heatmap = score_md + "<br><b>JD Keyword Density Word Cloud:</b><br>" + heatmap_html
        
            return donut_svg, score_md_with_heatmap, missing_tech_chips, missing_soft_chips, matched_chips_html, suggestions


        run_btn.click(
            _on_click, 
            inputs=[jd_input, resume_input], 
            outputs=[out_chart, out_score, out_missing_tech, out_missing_soft, out_matched_skills, out_suggestions]
        )
    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=True)