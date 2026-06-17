import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 1. PAGE CONFIGURATION & AESTHETICS
st.set_page_config(
    page_title="SkillSprint AI - Career & Life Decision Simulator",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern premium look
st.markdown("""
    <style>
    .main {
        background-color: #fbfbfd;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #eaeaea;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #1d1d1f;
    }
    .metric-label {
        font-size: 14px;
        color: #86868b;
        margin-top: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. CACHED MODEL LOAD
@st.cache_resource(show_spinner="Loading Semantic Vector Model...")
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_embedder()

# Skill Complexity mapping for prep-time heuristics
SKILL_COMPLEXITY = {
    'arm cortex': 4, 'rtos': 4, 'hardware debugging': 3, 'systemverilog': 4,
    'asic design flow': 4, 'sta basics': 3, 'aws iot': 3, 'pytorch': 4,
    'scikit-learn': 3, 'nlp basics': 3, 'typescript': 2, 'mongodb': 2,
    'rest apis': 2, 'node.js': 2, 'react': 2, 'esp32/raspberry pi': 3,
    'mqtt/coap': 2, 'linux': 2, 'tcl scripting': 2, 'pandas': 2, 'python': 2
}

# 3. DATA LOADERS
@st.cache_data
def load_data():
    try:
        students = pd.read_csv("student_profiles.csv")
        jobs = pd.read_csv("job_requirements.csv")
    except Exception as e:
        # Fallback to initial mock datasets if files are missing/corrupted
        students = pd.DataFrame([
            {"student_id": "STU003", "branch": "ECE", "semester": 4, "cgpa": 7.80, "current_skills": "C, Python, MATLAB, Microprocessors, Embedded C", "completed_projects": "Temperature Monitoring System with ESP8266 and ThingSpeak", "target_domain": "Embedded Systems"},
            {"student_id": "STU012", "branch": "CSE", "semester": 6, "cgpa": 6.96, "current_skills": "Git, C++, SQL, Python, HTML/CSS, React (Basic)", "completed_projects": "E-commerce landing page clone using React and Firebase", "target_domain": "Web Development"},
            {"student_id": "STU039", "branch": "CSE", "semester": 5, "cgpa": 6.96, "current_skills": "SQL, C, React (Basic)", "completed_projects": "Basic Portfolio Website using HTML, CSS, and JS", "target_domain": "AI-ML"}
        ])
        jobs = pd.DataFrame([
            {"job_id": "JOB002", "company_type": "Mid-size Product", "domain": "Embedded Systems", "required_skills": "Hardware Debugging, ARM Cortex, Embedded C", "min_experience_level": "6 Months Project Experience", "ideal_prep_time_weeks": 14},
            {"job_id": "JOB010", "company_type": "Core Engineering Firm", "domain": "Web Development", "required_skills": "Node.js, REST APIs, React", "min_experience_level": "Internship", "ideal_prep_time_weeks": 14},
            {"job_id": "JOB013", "company_type": "Core Engineering Firm", "domain": "AI-ML", "required_skills": "SQL, Scikit-Learn, NLP basics, PyTorch", "min_experience_level": "6 Months Project Experience", "ideal_prep_time_weeks": 14}
        ])
    return students, jobs

df_students, df_jobs = load_data()

# 4. SIDEBAR - PROFILE LOADING & CUSTOMIZATION
st.sidebar.title("👤 Input & Constraints")

profile_source = st.sidebar.radio("Profile Source:", ["Choose Existing Profile", "Create Custom Profile"])

if profile_source == "Choose Existing Profile":
    selected_stu_id = st.sidebar.selectbox("Select Student ID:", df_students["student_id"])
    active_profile = df_students[df_students["student_id"] == selected_stu_id].iloc[0].to_dict()
else:
    # Custom profile creation for testing
    active_profile = {}
    active_profile["student_id"] = "CUSTOM"
    active_profile["branch"] = st.sidebar.selectbox("Branch:", ["CSE", "ECE", "EEE", "Mech", "Civil"])
    active_profile["semester"] = st.sidebar.slider("Current Semester:", 1, 8, 5)
    active_profile["cgpa"] = st.sidebar.slider("Current CGPA:", 4.0, 10.0, 7.5, 0.1)
    active_profile["current_skills"] = st.sidebar.text_input("Skills (comma separated):", "Python, SQL, HTML/CSS")
    active_profile["completed_projects"] = st.sidebar.text_area("Completed Projects:", "Personal Portfolio Page")
    active_profile["target_domain"] = st.sidebar.selectbox("Target Domain:", ["Web Development", "Embedded Systems", "VLSI", "AI-ML", "IoT"])

# 5. SIDEBAR - CONSTRAINTS & PERSONAL UTILITY
st.sidebar.markdown("---")
st.sidebar.subheader("⏳ Weekly Commitment")
hours_per_week = st.sidebar.slider("Available study hours/week:", 5, 40, 15)
academic_load = st.sidebar.selectbox("Current Academic Load:", ["Light Workload", "Moderate (Regular Classes)", "Heavy (Lab Practicals & Exams)"])

st.sidebar.subheader("🛠️ Available Physical Resources")
has_esp32 = st.sidebar.checkbox("ESP32 / Microcontrollers", value=False)
has_gpu = st.sidebar.checkbox("Cloud Credits / GPU Access", value=False)
has_sensors = st.sidebar.checkbox("Basic Sensor Kits / Breadboards", value=False)

st.sidebar.subheader("🎲 Risk Profile")
risk_profile = st.sidebar.select_slider("Risk Tolerance:", options=["Low Risk", "Moderate Risk", "High Risk"], value="Moderate Risk")

# 6. HEURISTIC & SEMANTIC COMPUTING ENGINES
def evaluate_paths(student, jobs_df):
    results = {}
    
    # --- PATH A: INDUSTRY PLACEMENT MATCHING ---
    # Find matching jobs by domain
    domain_jobs = jobs_df[jobs_df["domain"].str.lower() == student["target_domain"].lower()]
    if domain_jobs.empty:
        domain_jobs = jobs_df # Fallback if target domain isn't in CSV
        domain_mismatch = True
    else:
        domain_mismatch = False
        
    student_skills = [s.strip().lower() for s in student["current_skills"].split(",")]
    
    best_job = None
    best_score = -1.0
    matched_skills = []
    missing_skills = []
    
    # Semantic Matching using Cosine Similarity on Sentence Embeddings
    for _, job in domain_jobs.iterrows():
        job_skills = [s.strip().lower() for s in job["required_skills"].split(",")]
        temp_matched = []
        temp_missing = []
        
        if job_skills and student_skills:
            student_embs = model.encode(student_skills)
            job_embs = model.encode(job_skills)
            sim_matrix = cosine_similarity(job_embs, student_embs)
            
            for idx, job_skill in enumerate(job_skills):
                best_match_idx = np.argmax(sim_matrix[idx])
                # Confidence threshold set to 0.80
                if sim_matrix[idx][best_match_idx] >= 0.80:
                    temp_matched.append(job_skill)
                else:
                    temp_missing.append(job_skill)
        else:
            temp_missing = job_skills
            
        skill_score = len(temp_matched) / len(job_skills) if job_skills else 0.0
        
        # Project semantic context comparison
        s_text = f"Projects: {student['completed_projects']}. Target: {student['target_domain']}"
        j_text = f"Seeking {job['domain']} developer. Required: {job['required_skills']}"
        project_sim = float(cosine_similarity(model.encode([s_text]), model.encode([j_text]))[0][0])
        
        combined_score = (0.6 * skill_score) + (0.4 * project_sim)
        if domain_mismatch:
            combined_score *= 0.6 # Penalty for out of domain
            
        if combined_score > best_score:
            best_score = combined_score
            best_job = job
            matched_skills = temp_matched
            missing_skills = temp_missing

    # Heuristic Prep Time for Job
    base_weeks = best_job["ideal_prep_time_weeks"] if best_job is not None else 12
    gap_complexity_weeks = sum(SKILL_COMPLEXITY.get(s, 2.5) for s in missing_skills)
    
    # Scale based on constraints
    time_multiplier = 1.0
    if hours_per_week < 15:
        time_multiplier += 0.4
    elif hours_per_week > 25:
        time_multiplier -= 0.15
        
    if academic_load == "Heavy (Lab Practicals & Exams)":
        time_multiplier += 0.3
    elif academic_load == "Light Workload":
        time_multiplier -= 0.1
        
    # Resource accelerations
    resource_savings = 0
    if student["target_domain"] == "Embedded Systems" or student["target_domain"] == "IoT":
        if has_esp32:
            resource_savings += 2
        if has_sensors:
            resource_savings += 1
    elif student["target_domain"] == "AI-ML" and has_gpu:
        resource_savings += 2
        
    job_prep_weeks = max(4, round((base_weeks + gap_complexity_weeks) * time_multiplier - resource_savings))
    
    # Success Probability (representation of uncertainty)
    job_success_base = best_score * 100
    # Adjustment based on hours
    if hours_per_week < 12:
        job_success_base -= 12
    # Adjust for CGPA
    if student["cgpa"] < 6.5:
        job_success_base -= 10
    
    job_success = max(15, min(95, round(job_success_base)))
    # Uncertainty interval decreases with more preparation time/commitment
    job_uncertainty = max(4, round(15 - (hours_per_week * 0.25)))
    
    # Stress Index (1-10)
    job_stress = 3
    if academic_load == "Heavy (Lab Practicals & Exams)":
        job_stress += 4
    elif academic_load == "Moderate (Regular Classes)":
        job_stress += 2
    if hours_per_week > 30:
        job_stress += 2 # Overwork risk
    if len(missing_skills) > 3:
        job_stress += 1
    job_stress = min(10, job_stress)
    
    results["Industry"] = {
        "title": f"Industry placement ({best_job['company_type']} - {best_job['domain']})" if best_job is not None else "Industry Placement",
        "score": best_score,
        "success": job_success,
        "uncertainty": job_uncertainty,
        "prep_weeks": job_prep_weeks,
        "stress": job_stress,
        "matched": matched_skills,
        "missing": missing_skills,
        "growth": 7.5 if best_job is not None and best_job["company_type"] == "Early-stage Startup" else 6.5
    }
    
    # --- PATH B: GRADUATE SCHOOL (M.Tech / MS) ---
    grad_success_base = 50.0
    # CGPA is key constraint
    if student["cgpa"] >= 8.5:
        grad_success_base = 85.0
    elif student["cgpa"] >= 7.5:
        grad_success_base = 70.0
    elif student["cgpa"] >= 6.5:
        grad_success_base = 45.0
    else:
        grad_success_base = 20.0
        
    # Adjustment for study hours
    if hours_per_week >= 20:
        grad_success_base += 10
    else:
        grad_success_base -= 15
        
    grad_success = max(10, min(95, round(grad_success_base)))
    grad_uncertainty = max(5, round(18 - (student["cgpa"] * 1.5)))
    
    grad_prep_weeks = max(8, round(24 * (18 / hours_per_week)))
    if academic_load == "Heavy (Lab Practicals & Exams)":
        grad_prep_weeks = round(grad_prep_weeks * 1.25)
        
    grad_stress = 4
    if student["cgpa"] < 7.5:
        grad_stress += 3 # pressure to score high on entrance exams
    if academic_load == "Heavy (Lab Practicals & Exams)":
        grad_stress += 3
    grad_stress = min(10, grad_stress)
    
    results["GraduateSchool"] = {
        "title": "Graduate School (M.Tech / MS / MBA)",
        "success": grad_success,
        "uncertainty": grad_uncertainty,
        "prep_weeks": grad_prep_weeks,
        "stress": grad_stress,
        "growth": 8.5 # High specialization floor
    }
    
    # --- PATH C: SELF-TAUGHT / INDIE CREATOR ROUTE ---
    indie_success_base = 45.0
    # Driven heavily by hours/week and project history
    if hours_per_week >= 25:
        indie_success_base += 15
    elif hours_per_week < 12:
        indie_success_base -= 20
        
    if len(student["completed_projects"]) > 20: # has active project history
        indie_success_base += 10
        
    indie_success = max(10, min(90, round(indie_success_base)))
    # Self-taught path has high inherent uncertainty
    indie_uncertainty = max(8, round(22 - (hours_per_week * 0.3)))
    
    indie_prep_weeks = max(6, round(16 * (15 / hours_per_week)))
    
    indie_stress = 2
    if hours_per_week > 30:
        indie_stress += 3 # high burnout risk when self-studying alone
    if academic_load == "Heavy (Lab Practicals & Exams)":
        indie_stress += 2
    indie_stress = min(10, indie_stress)
    
    results["SelfTaught"] = {
        "title": "Self-Taught / Indie Creator Route",
        "success": indie_success,
        "uncertainty": indie_uncertainty,
        "prep_weeks": indie_prep_weeks,
        "stress": indie_stress,
        "growth": 7.0
    }
    
    return results

eval_results = evaluate_paths(active_profile, df_jobs)

# 7. MAIN UI LAYOUT
st.title("🎯 SkillSprint AI: Career & Life Decision Simulator")
st.subheader("Undergraduate Track | AI-Powered Framework for Structured Decision Making")
st.write("An interactive planning assistant that models tradeoffs, surfaces hidden considerations, and frames market uncertainty honestly to support high-stakes life choices.")
st.markdown("---")

# Display current profile metrics
p_col1, p_col2, p_col3, p_col4 = st.columns(4)
with p_col1:
    st.markdown(f"**ID:** `{active_profile['student_id']}`  \n**Branch:** {active_profile['branch']}")
with p_col2:
    st.markdown(f"**Current CGPA:** `{active_profile['cgpa']}`  \n**Semester:** {active_profile['semester']}")
with p_col3:
    st.markdown(f"**Target Domain:** {active_profile['target_domain']}")
with p_col4:
    st.markdown(f"**Skills:** {active_profile['current_skills']}")

st.markdown("---")

# 8. VISUAL DASHBOARD & PATH COMPARISONS
chart_col, details_col = st.columns([1.1, 0.9])

with chart_col:
    st.markdown("### 📊 Path Metrics Comparison")
    
    paths_list = ["Industry Placement", "Graduate School", "Self-Taught / Creator"]
    success_chances = [eval_results["Industry"]["success"], eval_results["GraduateSchool"]["success"], eval_results["SelfTaught"]["success"]]
    uncertainty_vals = [eval_results["Industry"]["uncertainty"], eval_results["GraduateSchool"]["uncertainty"], eval_results["SelfTaught"]["uncertainty"]]
    prep_times = [eval_results["Industry"]["prep_weeks"], eval_results["GraduateSchool"]["prep_weeks"], eval_results["SelfTaught"]["prep_weeks"]]
    stresses = [eval_results["Industry"]["stress"], eval_results["GraduateSchool"]["stress"], eval_results["SelfTaught"]["stress"]]

    fig = go.Figure()

    # Success Chance with error bars (uncertainty)
    fig.add_trace(go.Bar(
        name="Success Chance (%)",
        x=paths_list,
        y=success_chances,
        error_y=dict(type='data', array=uncertainty_vals, visible=True),
        marker_color='#3498db',
        text=[f"{s}% (±{u}%)" for s, u in zip(success_chances, uncertainty_vals)],
        textposition='auto',
    ))

    # Prep Time
    fig.add_trace(go.Bar(
        name="Prep Duration (Weeks)",
        x=paths_list,
        y=prep_times,
        marker_color='#2ecc71',
        text=[f"{w} wks" for w in prep_times],
        textposition='auto',
    ))

    # Stress Index
    fig.add_trace(go.Bar(
        name="Stress Index (1-10)",
        x=paths_list,
        y=[s * 10 for s in stresses], # Scaled by 10 to fit visual range
        marker_color='#e74c3c',
        text=[f"Stress: {s}/10" for s in stresses],
        textposition='auto',
    ))

    fig.update_layout(
        barmode='group',
        height=380,
        margin=dict(t=20, b=20, l=40, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='#eaeaea')
    )
    
    st.plotly_chart(fig, use_container_width=True)

with details_col:
    st.markdown("### 🤖 Scenario-Aware AI Diagnostics")
    
    # Generate dynamic, personalized reasoning paragraphs
    diagnostics = []
    
    # 1. Academic Load Analysis
    if academic_load == "Heavy (Lab Practicals & Exams)":
        diagnostics.append(
            f"**Academic Pressure:** Your current heavy academic load acts as a critical constraint. "
            f"Preparation velocity is reduced by ~30% to protect your current CGPA of {active_profile['cgpa']}. "
            f"Graduate exams prep or startup skills acquisition will feel highly compressed."
        )
    elif academic_load == "Light Workload":
        diagnostics.append(
            "**Academic Pressure:** A lighter academic load gives you a significant time advantage, "
            "allowing you to accelerate project development and reduce total preparation weeks by 10-15%."
        )
        
    # 2. Resource Synergy
    resource_synergies = []
    if active_profile["target_domain"] == "Embedded Systems" and has_esp32:
        resource_synergies.append("Your physical ESP32 board enables bare-metal deployment modules, bridging conceptual gap to physical register level.")
    if active_profile["target_domain"] == "AI-ML" and has_gpu:
        resource_synergies.append("Cloud GPU access allows you to run local model training, bypassing typical notebook hardware constraints.")
    if resource_synergies:
        diagnostics.append(f"**Resource Optimization:** { ' '.join(resource_synergies) }")
    else:
        diagnostics.append(
            "**Resource Constraint:** No specialized hardware resources matching your domain were selected. "
            "You will rely primarily on software simulations and free computing tiers."
        )
        
    # 3. Placement Warning Safeguard
    if eval_results["Industry"]["success"] < 45:
        diagnostics.append(
            "⚠️ **Responsible AI Notice:** Your current skill alignment with industry target roles falls below the safe confidence threshold. "
            "Direct placement is highly risky right now. We suggest starting with a Self-Taught bridging path to establish fundamentals."
        )

    for item in diagnostics:
        st.info(item)

st.markdown("---")

# 9. DETAILED TRADEOFF ANALYSIS (COLUMNS)
st.markdown("### ⚖️ Multi-Path Tradeoff Matrix")

col_ind, col_grad, col_indie = st.columns(3)

with col_ind:
    st.subheader("💼 Path A: Industry Job")
    st.write(f"**Recommended Target:** `{eval_results['Industry']['title']}`")
    
    # Skill Match analysis
    ind_data = eval_results["Industry"]
    if "matched" in ind_data and ind_data["matched"]:
        st.write(f"✅ **Semantic Matches:** {', '.join(ind_data['matched'])}")
    if "missing" in ind_data and ind_data["missing"]:
        st.write(f"⚠️ **Skill Gaps:** {', '.join(ind_data['missing'])}")
    else:
        st.write("🎉 **Parity:** No missing technical skills identified!")
        
    st.markdown("""
    * **Growth Potential:** Structured salary increments. High initial learning if matched to startup; slower corporate path in MNCs.
    * **Hidden Considerations:** Potential relocation costs, commute times, and company culture alignment.
    * **Risk Mitigation:** If placement fails, you still retain marketable skills for other roles.
    """)

with col_grad:
    st.subheader("🎓 Path B: Graduate School")
    st.write("**Recommended Target:** `M.Tech / MS / MBA Entrance`")
    
    grad_data = eval_results["GraduateSchool"]
    if active_profile["cgpa"] < 7.5:
        st.markdown("> ⚠️ **CGPA Alert:** Your CGPA is below the typical 7.5 cutoff for premier institutes, making exam scores extremely critical.")
        
    st.markdown(f"""
    * **Eligibility Chance:** High dependency on exam ranking. Admission probability modeled around **{grad_data['success']}% ± {grad_data['uncertainty']}%**.
    * **Financial Impact:** Requires upfront tuition fees; defers active industry income by 2 full years.
    * **Career Ceiling:** Greatly elevates long-term career ceiling, opening doors to R&D, advanced research, and global opportunities.
    """)

with col_indie:
    st.subheader("🌱 Path C: Self-Taught & Indie")
    st.write("**Recommended Target:** `Project Portfolio & Freelance`")
    
    st.markdown("""
    * **Financial Impact:** Minimal financial overhead. No tuition fees, using open-source resources.
    * **Flexibility & Control:** 100% control over target stack, zero commute, and self-paced execution.
    * **Core Risks:** High risk of isolation, lack of structured feedback, and struggles to pass standard HR/ATS automated filters. Requires strong self-discipline.
    """)

st.markdown("---")

# 10. RESPONSIBLE AI & JUDGE'S LENS DETAILS
with st.expander("🧠 Behind the AI: Architecture, Methodology & Guardrails"):
    st.markdown("""
    ### 🔬 Justification of AI Approach (Judge's Lens)
    - **Why not a Rules Engine?**
      - Career paths and resumes are highly unstructured. A rules engine would require exact keyword matching (failing to connect "C++" to "C/C++" or "React (Basic)" to "React").
      - Our pipeline uses semantic vector embeddings via the cached **SentenceTransformer (`all-MiniLM-L6-v2`)** to evaluate skill alignment. This calculates cosine similarity between the student's project descriptions and target job descriptions, capturing contextual meaning rather than rigid spelling.
    
    ### 🛡️ Responsible AI Guardrails & Mitigations
    1. **Realistic Risk (Over-reliance):**
       - Users might take the AI success probabilities as deterministic predictions, leading to false confidence or demotivation.
    2. **Concrete Mitigation (Uncertainty Framing):**
       - Success metrics are framed with statistical uncertainty intervals (e.g. `± 8%`), reminding users that outcomes depend on variable market cycles and exam conditions.
    3. **Human-in-the-Loop Design:**
       - The AI is structured strictly as a *decision support input*, not a decision maker. The user must manually lock in their choice, providing personal rationale that the AI cannot model (interest, family obligations, passion).
    """)

st.markdown("---")

# 11. HUMAN-IN-THE-LOOP DECISION PANEL
st.subheader("✍️ Human-in-the-Loop: Lock In Your Choice")
st.write("Review the tradeoffs above. The AI does not make this decision for you. Select your final direction and provide your personal reasoning.")

decision_col1, decision_col2 = st.columns([1, 2])

with decision_col1:
    user_choice = st.selectbox("Your Final Choice:", ["Select Choice...", "Path A: Industry Placement", "Path B: Graduate School", "Path C: Self-Taught / Indie Creator"])
    
with decision_col2:
    user_rationale = st.text_area("Your Decision Rationale:", placeholder="Describe why you chose this path over the others, factoring in the tradeoffs and constraints modeled above...")

lock_btn = st.button("Lock In Decision & Generate Action Plan")

# 12. MILESTONE PLAN GENERATOR
if lock_btn:
    if user_choice == "Select Choice...":
        st.error("Please select a path to lock in your decision.")
    elif not user_rationale.strip():
        st.warning("Please provide your rationale to proceed.")
    else:
        st.success(f"🔒 **Decision Locked:** {user_choice}")
        st.markdown(f"**Your Justification:** *{user_rationale}*")
        
        st.markdown("### 📅 Your Personalized Milestone & Action Plan")
        
        step_col, milestone_col = st.columns([1, 2])
        
        with step_col:
            st.info("⚡ **First Real Step (Next 24 Hours)**\n\nTo overcome starting inertia, take this immediate action:")
            if user_choice == "Path A: Industry Placement":
                missing = eval_results["Industry"]["missing"]
                first_skill = missing[0].upper() if missing else "SYSTEMS DESIGN"
                st.markdown(f"**Action:** Create a GitHub repository named `learning-{first_skill.lower()}`. Write a 1-page README outlining three mini-projects you will build to master `{first_skill}`.")
            elif user_choice == "Path B: Graduate School":
                st.markdown("**Action:** Download the official syllabus for your target exam (GATE/GRE) and schedule a mock diagnostic test for this weekend to baseline your score.")
            else:
                st.markdown("**Action:** List three portfolio project ideas matching your niche and write down the user stories or mockups for the first project.")
                
        with milestone_col:
            st.markdown("#### 🚀 30/60/90-Day Milestones")
            if user_choice == "Path A: Industry Placement":
                missing = eval_results["Industry"]["missing"]
                st.markdown(f"""
                - **Day 30 (Skill Bridge):** Complete intensive study modules for `{', '.join(missing[:2]) if missing else 'missing domain skills'}`. Allocate {hours_per_week} hours weekly.
                - **Day 60 (Project Build):** Develop a custom project addressing `{active_profile['target_domain']}` utilizing available resources.
                - **Day 90 (Apply & Network):** Refine resume highlighting the new project. Submit applications targeting the best-matched roles.
                """)
            elif user_choice == "Path B: Graduate School":
                st.markdown(f"""
                - **Day 30 (Fundamentals):** Complete primary conceptual blocks of core subjects (representing ~30% of target exam syllabus).
                - **Day 60 (Problem Solving):** Focus on previous years' exam question papers under timed environments (10-15 hours/week).
                - **Day 90 (Mock Exams):** Transition fully to weekly mock exams, target error analysis, and application submission to target universities.
                """)
            else:
                st.markdown(f"""
                - **Day 30 (Niche Definition):** Build the core repository, setup infrastructure, and complete initial project prototype.
                - **Day 60 (Portfolio Development):** Release the first public version (beta) of your tool/portfolio and solicit feedback from dev communities.
                - **Day 90 (Launch & Monetize/Apply):** Complete second major project. Package your portfolio site and begin pitching to freelance clients or contract opportunities.
                """)