import streamlit as st

# --- MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Automated Resume Relevance Checker", layout="wide")

import fitz  # PyMuPDF
import docx
import spacy
from sentence_transformers import SentenceTransformer, util
import google.generativeai as genai
import sqlite3
import pandas as pd
from datetime import datetime
import io
import os

# --- Database Setup ---
def init_db():
    conn = sqlite3.connect('resume_analysis.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY,
            timestamp DATETIME,
            resume_name TEXT,
            jd_name TEXT,
            score INTEGER,
            verdict TEXT,
            feedback TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_result(resume_name, jd_name, score, verdict, feedback):
    conn = sqlite3.connect('resume_analysis.db')
    c = conn.cursor()
    timestamp = datetime.now()
    c.execute("INSERT INTO results (timestamp, resume_name, jd_name, score, verdict, feedback) VALUES (?, ?, ?, ?, ?, ?)",
              (timestamp, resume_name, jd_name, score, verdict, feedback))
    conn.commit()
    conn.close()

def get_all_results():
    conn = sqlite3.connect('resume_analysis.db')
    df = pd.read_sql_query("SELECT timestamp, resume_name, jd_name, score, verdict FROM results ORDER BY timestamp DESC", conn)
    conn.close()
    return df

init_db()

# --- Model Loading ---
@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        st.error("Spacy model not found. Please wait while we download it...")
        os.system("python -m spacy download en_core_web_sm")
        return spacy.load("en_core_web_sm")

@st.cache_resource
def load_sentence_transformer_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

nlp = load_spacy_model()
st_model = load_sentence_transformer_model()

# --- Helper Functions ---
def extract_text_from_file(file):
    if file.type == "application/pdf":
        text = ""
        with fitz.open(stream=file.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    return ""

def extract_skills(text, skill_keywords):
    doc = nlp(text.lower())
    found_skills = set()
    for skill in skill_keywords:
        if skill in text.lower():
            found_skills.add(skill)
    return list(found_skills)

def calculate_semantic_similarity(text1, text2):
    embedding1 = st_model.encode(text1, convert_to_tensor=True)
    embedding2 = st_model.encode(text2, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embedding1, embedding2)
    return cosine_scores.item() * 100

def generate_feedback(jd_text, missing_skills):
    """Generates personalized feedback using Google Gemini."""
    try:
        # Check for API key in environment variables
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return "Google API key not found. Please set it in the Space settings."
            
        # Configure the Google AI client
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = f"""
        You are an expert career coach providing feedback to a student applying for a job.
        Job Description Summary: {jd_text[:2000]}
        The student's resume is missing the following key skills: {', '.join(missing_skills)}.
        Please provide a short, encouraging paragraph of personalized feedback.
        Suggest 1-2 practical ways the student could gain experience in these missing areas (e.g., online courses, personal projects).
        Keep the tone positive and constructive.
        """
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Could not generate feedback due to an error: {e}"

# --- Streamlit App Interface ---
st.title("🤖 Automated Resume Relevance Check System")
st.markdown("*Powered by Google Gemini and Hugging Face*")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Inputs")
    jd_file = st.file_uploader("Upload Job Description (PDF/DOCX)", type=["pdf", "docx"])
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
    analyze_button = st.button("Analyze Resume ✨", type="primary")

SKILL_KEYWORDS = [
    'python', 'java', 'c++', 'sql', 'javascript', 'react', 'vue', 'angular',
    'machine learning', 'data analysis', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'pytorch',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'jira', 'agile', 'scrum'
]

if analyze_button and jd_file and resume_file:
    jd_text = extract_text_from_file(jd_file)
    resume_text = extract_text_from_file(resume_file)
    
    with col2:
        st.header("Analysis Results")
        with st.spinner("Analyzing... This may take a moment."):
            jd_skills = extract_skills(jd_text, SKILL_KEYWORDS)
            resume_skills = extract_skills(resume_text, SKILL_KEYWORDS)
            common_skills = list(set(jd_skills) & set(resume_skills))
            missing_skills = list(set(jd_skills) - set(resume_skills))
            hard_match_score = (len(common_skills) / len(jd_skills) * 100) if jd_skills else 0
            semantic_score = calculate_semantic_similarity(jd_text, resume_text)
            final_score = int((hard_match_score * 0.4) + (semantic_score * 0.6))
            
            if final_score >= 75: 
                verdict = "High Suitability"
                st.success(f"Verdict: **{verdict}**")
            elif final_score >= 50: 
                verdict = "Medium Suitability"
                st.warning(f"Verdict: **{verdict}**")
            else: 
                verdict = "Low Suitability"
                st.error(f"Verdict: **{verdict}**")

            st.subheader(f"Final Relevance Score: {final_score}%")
            st.progress(final_score / 100)
            
            st.subheader("Personalized Feedback")
            if missing_skills:
                feedback = generate_feedback(jd_text, missing_skills)
            else:
                feedback = "Excellent! The resume appears to contain all the key skills from the job description."
            st.markdown(feedback)
            
            save_result(resume_file.name, jd_file.name, final_score, verdict, feedback)

            with st.expander("Show Detailed Analysis"):
                st.markdown(f"**✅ Common Skills ({len(common_skills)}):** `{', '.join(common_skills) if common_skills else 'None'}`")
                st.markdown(f"**❌ Missing Skills ({len(missing_skills)}):** `{', '.join(missing_skills) if missing_skills else 'None'}`")
                st.markdown(f"**Keyword Match Score:** `{int(hard_match_score)}%`")
                st.markdown(f"**Contextual Match Score:** `{int(semantic_score)}%`")
else:
    with col2:
        st.info("Upload a job description and a resume, then click 'Analyze'.")

st.header("Submission History")
if st.button("Refresh History"):
    st.rerun()

try:
    df = get_all_results()
    if not df.empty:
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No analysis history yet. Upload some files to get started!")
except Exception as e:
    st.info("No analysis history yet. Upload some files to get started!")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and hosted on 🤗 Hugging Face Spaces")