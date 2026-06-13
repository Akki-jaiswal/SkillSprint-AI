import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px

# 1. PAGE SETUP & CONFIGURATION
st.set_page_config(page_title="SkillSprint AI", layout="wide", page_icon="🚀")

@st.cache_resource
def load_model():
    # Caching the model prevents long reload times during the live judge demo
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Skill Complexity Weights for realistic timeline generation
SKILL_COMPLEXITY = {
    'arm cortex': 4, 'rtos': 4, 'hardware debugging': 3, 'systemverilog': 4,
    'asic design flow': 4, 'sta basics': 3, 'aws iot': 3, 'pytorch': 4,
    'scikit-learn': 3, 'nlp basics': 3, 'typescript': 2, 'mongodb': 2,
    'rest apis': 2, 'node.js': 2, 'react': 2, 'esp32/raspberry pi': 3,
    'mqtt/coap': 2, 'linux': 2, 'tcl scripting': 2, 'pandas': 2, 'python': 2
}

# 2. MOCK DATA GENERATION (Re-engineered to model hard constraints)
@st.cache_data
def load_synthetic_data():
    students = pd.DataFrame([
        {"student_id": "STU003", "branch": "ECE", "semester": 4, "cgpa": 7.80, "current_skills": "C, Python, MATLAB, Microprocessors, Embedded C", "completed_projects": "Temperature Monitoring System with ESP8266", "target_domain": "Embedded Systems"},
        {"student_id": "STU012", "branch": "CSE", "semester": 6, "cgpa": 6.96, "current_skills": "Git, C++, SQL, Python, HTML/CSS, React (Basic)", "completed_projects": "E-commerce landing page clone using React", "target_domain": "Web Development"},
        {"student_id": "STU039", "branch": "CSE", "semester": 5, "cgpa": 6.96, "current_skills": "SQL, C, React (Basic)", "completed_projects": "Basic Portfolio Website using HTML, CSS", "target_domain": "AI-ML"}
    ])
    
    jobs = pd.DataFrame([
        {"job_id": "JOB002", "company_type": "Mid-size Product", "domain": "Embedded Systems", "required_skills": "Hardware Debugging, ARM Cortex, Embedded C", "min_experience_level": "6 Months Project Experience", "ideal_prep_time_weeks": 14},
        {"job_id": "JOB010", "company_type": "Core Engineering Firm", "domain": "Web Development", "required_skills": "Node.js, REST APIs, React", "min_experience_level": "Internship", "ideal_prep_time_weeks": 14},
        {"job_id": "JOB013", "company_type": "Core Engineering Firm", "domain": "AI-ML", "required_skills": "SQL, Scikit-Learn, NLP basics, PyTorch", "min_experience_level": "6 Months Project Experience", "ideal_prep_time_weeks": 14}
    ])
    return students, jobs

df_students, df_jobroles = load_synthetic_data()

# 3. SIDEBAR: USER CONDITIONS & RESOURCES (The Constraints Engine)
st.sidebar.header("👤 Student Profile & Constraints")
selected_stu_id = st.sidebar.selectbox("Select Student ID for Demo:", df_students["student_id"])

st.sidebar.subheader("⏳ Weekly Commitment")
hours_per_week = st.sidebar.slider("Available hours per week:", 5, 40, 15)
academic_load = st.sidebar.selectbox("Current Academic Load:", ["Light Workload", "Moderate (Regular Classes)", "Heavy (Lab Practicals & Exams)"])

st.sidebar.subheader("🛠️ Available Physical Resources")
has_esp32 = st.sidebar.checkbox("ESP32 / Arduino Development Board", value=True)
has_sensors = st.sidebar.checkbox("Basic Sensor Kits / Logic Analyzers", value=False)
has_gpu = st.sidebar.checkbox("Cloud Credit / Dedicated GPU Access", value=False)

# Fetch active student data
student_profile = df_students[df_students["student_id"] == selected_stu_id].iloc[0]

# 4. BACKEND PIPELINE: HYBRID SEARCH & SEMANTIC MATCHING
def calculate_hybrid_match(student, jobs_df):
    results = []
    student_skills_list = [s.strip().lower() for s in student["current_skills"].split(",")]
    
    for _, job in jobs_df.iterrows():
        # Hard Constraint 1: Domain Alignment Filter
        domain_multiplier = 1.0 if job["domain"].lower() == student["target_domain"].lower() else 0.6
        
        # Semantic Skill Matching using Vector Embeddings
        job_skills_list = [s.strip().lower() for s in job["required_skills"].split(",")]
        matched_skills = []
        missing_skills = []
        
        if job_skills_list:
            # Embed individual skills to check cross-semantic matches (e.g., "React (Basic)" -> "React")
            s_embs = model.encode(student_skills_list)
            j_embs = model.encode(job_skills_list)
            sim_matrix = cosine_similarity(j_embs, s_embs)
            
            for idx, job_skill in enumerate(job_skills_list):
                best_match_idx = np.argmax(sim_matrix[idx])
                # Responsibility Guardrail: Similarity threshold set to 0.82
                if sim_matrix[idx][best_match_idx] >= 0.82:
                    matched_skills.append(job_skill)
                else:
                    missing_skills.append(job_skill)
        
        # Hard Skill Score calculation
        skill_score = len(matched_skills) / len(job_skills_list) if job_skills_list else 0
        
        # Contextual Description Embedding Match
        s_text = f"{student['completed_projects']}. Target: {student['target_domain']}"
        j_text = f"{job['company_type']} seeking {job['domain']} specialist skilled in {job['required_skills']}"
        semantic_score = float(cosine_similarity(model.encode([s_text]), model.encode([j_text]))[0][0])
        
        # Final Architecture Scoring Equation
        final_score = ((0.5 * semantic_score) + (0.5 * skill_score)) * domain_multiplier
        
        results.append({
            "job_id": job["job_id"],
            "domain": job["domain"],
            "company": job["company_type"],
            "score": min(final_score, 1.0),
            "missing": missing_skills,
            "matched": matched_skills,
            "total_weeks": job["ideal_prep_time_weeks"]
        })
    return sorted(results, key=lambda x: x["score"], reverse=True)

match_results = calculate_hybrid_match(student_profile, df_jobroles)
best_match = match_results[0]

# 5. UI LAYOUT & GRAPHICS
st.title("🎯 SkillSprint AI")
st.subheader("Resource-Aware Semantic Matching & Curriculum Engine")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown(f"### Current Profile: **{student_profile['student_id']}**")
    st.info(f"**Branch/Sem:** {student_profile['branch']} - Sem {student_profile['semester']}  \n"
            f"**Current CGPA:** {student_profile['cgpa']}  \n"
            f"**Skills:** {student_profile['current_skills']}")
    
    st.markdown("### 🏆 Top Recommended Match")
    score_pct = int(best_match['score'] * 100)
    
    # Responsible AI Guardrail check for low scores
    if score_pct < 40:
        st.warning("⚠️ **Guardrail Notice:** No high-confidence job placements match your current core domain. The profile mismatch exceeds safe parameters.")
    else:
        st.metric(label=f"{best_match['company']} ({best_match['domain']} Role)", value=f"{score_pct}% Match Affinity")

with col2:
    st.markdown("### 📊 Skill Gap Map")
    chart_data = pd.DataFrame({
        'Skill Status': ['Matched Matrix', 'Gaps Identified'],
        'Count': [len(best_match['matched']), len(best_match['missing'])]
    })
    fig = px.pie(chart_data, values='Count', names='Skill Status', color_discrete_sequence=['#2ecc71', '#e74c3c'], hole=0.4)
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=220)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("### 📅 Personalized 14-Week Sprint Schedule")

# 6. COMPLEXITY-WEIGHTED TIMELINE ENGINE WITH RESOURCE REASONING
total_sprint_weeks = 14
missing_skills = best_match['missing']

# Generate context-aware architectural reasoning
reasoning_pieces = []
if academic_load == "Heavy (Lab Practicals & Exams)":
    reasoning_pieces.append(f"Because you are under a heavy academic load, weekly pacing is distributed into shorter macro-sessions to protect your {student_profile['cgpa']} CGPA.")
if student_profile["branch"] == "ECE" and best_match["domain"] == "Embedded Systems" and has_esp32:
    reasoning_pieces.append("Leveraging your physical ESP32 architecture hardware allows us to eliminate abstract software emulation, shifting technical training straight to bare-metal compilation.")
elif has_gpu and best_match["domain"] == "AI-ML":
    reasoning_pieces.append("With explicit cloud GPU access enabled, your project modules focus on accelerated model training cycles rather than localized API constraints.")

st.markdown(f"> **🤖 AI Engine Context Reasoning:** {' '.join(reasoning_pieces) if reasoning_pieces else 'Sprint optimized for standard learning paths.'}")

if not missing_skills:
    st.success("🎉 Full skill parity achieved! Devote the full 14 weeks to production portfolio development, mock technical screens, and pipeline deployment.")
else:
    # Compute relative task complexity allocations
    total_weight = sum(SKILL_COMPLEXITY.get(skill, 2) for skill in missing_skills)
    current_week = 1
    
    for idx, skill in enumerate(missing_skills):
        skill_weight = SKILL_COMPLEXITY.get(skill, 2)
        allocated_weeks = max(1, round((skill_weight / total_weight) * total_sprint_weeks))
        
        # Ensure exact bounding to the 14-week limit
        if idx == len(missing_skills) - 1 or current_week + allocated_weeks > total_sprint_weeks:
            end_week = total_sprint_weeks
        else:
            end_week = current_week + allocated_weeks - 1
            
        if current_week <= total_sprint_weeks:
            with st.expander(f"📆 Weeks {current_week} - {end_week}: Target Skill Development -> **{skill.upper()}**"):
                st.write(f"**Focus Area:** Deep dive processing and practical implementation blocks for `{skill}`.")
                if student_profile["branch"] == "ECE" and has_esp32 and skill in ['hardware debugging', 'arm cortex', 'rtos']:
                    st.write("🔬 *Resource Alignment:* Flash your scripts directly to your physical ESP32 infrastructure to observe functional hardware register execution shifts in real-time.")
                else:
                    st.write("💻 *Resource Alignment:* Utilize community documentation resources and modular local compiler building environments.")
            current_week = end_week + 1