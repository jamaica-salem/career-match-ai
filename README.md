# 🤖 Resume — Job Match Scorer for IT Jobs

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Gradio](https://img.shields.io/badge/Gradio-3.40-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A **web-based AI tool** to analyze the match between a **resume/CV** and a **job description**, specifically for IT-related roles. This app evaluates technical and soft skills, calculates a match score using **semantic similarity**, highlights missing skills, and provides actionable suggestions to improve your resume.

---

## 🌟 Features

- **Resume-to-Job Match Score:** Uses `SentenceTransformer` embeddings and cosine similarity.  
- **Technical & Soft Skill Extraction:** Detects over **500 IT and soft skills** with synonym mapping.  
- **Missing Skills Analysis:** Highlights skills required by the job but missing from the resume.  
- **Matched Skills Visualization:** Displays matched skills as color-coded chips.  
- **Keyword Heatmap:** Highlights the most important keywords from the job description.  
- **Actionable Suggestions:** Provides tips to improve your resume based on gaps and match score.  
- **Fast & Interactive UI:** Powered by **Gradio**; average processing time **< 3 seconds**.

---

## ⚡ Technologies Used

- **Python 3.9+**  
- **Gradio** – Web interface for interactive demo  
- **Sentence Transformers** – Semantic embeddings for job/resume similarity  
- **Scikit-learn** – Cosine similarity computations  
- **NumPy** – Numerical operations  

---

## 🛠️ How It Works

1. **Input:** Paste the job description and your resume/CV.  
2. **Skill Extraction:** Detects technical and soft skills from both inputs.  
3. **Similarity Analysis:** Computes semantic similarity to generate a match score.  
4. **Visual Output:**
   - Donut chart showing match percentage  
   - Matched skills displayed as color-coded chips  
   - Missing technical and soft skills listed  
   - Job description keyword heatmap  
5. **Suggestions:** Actionable tips to improve your resume based on match score.

---

## 🎨 User Interface

- Modern **dark-mode Gradio UI**  
- Gradient cards for clear sections  
- Real-time analysis and updates  
- Color-coded skill chips for easy reading  
- Keyword heatmap shows density of important JD terms  

---

## 📊 Output Example

- **Match Score:** 85% ✅
- **Matched Skills:** Python, React, Machine Learning, Communication, Leadership
- **Missing Technical Skills:** Docker, Kubernetes, GraphQL
- **Missing Soft Skills:** Strategic Thinking, Time Management
- **Suggestions:**
  1. Add top missing skills: Docker, Kubernetes, GraphQL
  2. Highlight transferable skills and relevant projects
  3. Optimize phrasing and quantify achievements

## 📝 Notes

- Uses pre-trained MiniLM model: `all-MiniLM-L6-v2`
- Synonyms and alternate names are mapped for accurate skill detection
- Supports IT-related roles across Frontend, Backend, Cloud, DevOps, ML/AI, Security, and more

```bash
pip install -r requirements.txt
