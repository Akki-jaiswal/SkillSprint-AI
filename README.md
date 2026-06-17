# 🎯 SkillSprint AI: Life & Career Decision Simulator
**Undergraduate Track | AI for Life, Learning & Work**
*Built for the USAII Global AI Hackathon 2026*

SkillSprint AI is an intelligent, constraint-aware decision support system designed as a "Second Brain" to help students navigate high-stakes life choices: choosing between an **Industry Placement**, pursuing **Graduate School**, or going down a **Self-Taught / Indie Creator** path.

By modeling a user's academic constraints, weekly study hours, risk profile, and physical hardware resources, the simulator visualizes tradeoffs, represents market uncertainty honestly, and guides the user toward a committed decision and actionable roadmap.

---

## 📌 The Problem
Engineering students face overloaded choices and high-stakes decisions under conditions of incomplete information. Most career tools either oversimplify choices into static pros/cons lists, or overwhelm students with massive databases of jobs. Standard search algorithms use rigid keyword match filters that ignore:
1. **Semantic Equivalence:** A student's project on a "Temperature Monitoring System with ESP8266" is contextually equivalent to "embedded microcontrollers," but traditional keyword engines mark this as a total mismatch.
2. **Real-Life Constraints:** Recommending an intensive coding sprint ignores academic workloads, exam seasons, and lack of essential resources (such as not owning microcontrollers or not having cloud GPU access).

---

## 🧠 AI Architecture & Decision Pipeline

The application models decisions using a structured **Data → AI Reasoning → Interactive Tradeoff → Action** pipeline:

```
[Input: Student Profile & Constraints] 
      │
      ▼
[Hugging Face Encoder (all-MiniLM-L6-v2)] ──► Computes Semantic Skill Similarities
      │
      ▼
[Multi-Path Constraint Engine] ──────────► Propagates study hours, academic load, & hardware resources
      │
      ▼
[Plotly Comparison Dashboard] ───────────► Visualizes Success Chance (± Range), Prep Time, & Stress Index
      │
      ▼
[Human-in-the-Loop Lock] ────────────────► User logs final choice and rationale
      │
      ▼
[30/60/90-Day Execution Plan] ───────────► Breaking inertia with "First Step" within 24 Hours
```

### 🔬 Why AI/NLP over a Rules Engine? (Judge's Lens)
- **Unstructured Matching:** Resumes and project descriptions contain rich, free-form text. A rules engine fails on spelling differences, acronyms, and synonyms. 
- **Semantic Overlaps:** Our pipeline embeds skill sets into a vector space using a cached **SentenceTransformer (`all-MiniLM-L6-v2`)** model. Cosine similarity is calculated to match skills at a `0.80` confidence threshold, automatically mapping "React (Basic)" to a "React" requirement, and "C" / "Embedded C" to microprocessor domains.
- **Probabilistic Timelines:** Rules engines are binary. Our system models preparation timelines and success chances as continuous variables scaled dynamically by active resource constraints.

---

## 🛡️ Responsible AI Guardrails

1. **Realistic Risk (Over-Reliance):** 
   - A major risk of decision-support AI is that users treat success probabilities as definitive prophecies, creating false confidence or unnecessary defeatism.
2. **Concrete Mitigation (Uncertainty Framing):**
   - Success percentages are modeled and displayed with statistical uncertainty bounds (e.g., `78% ± 6%`), indicating that external variables (economic shifts, admission curves) are outside the AI's modeling parameters.
3. **Domain Parity Safeguard:**
   - If a student's profile exhibits less than a `45%` match affinity with target industry roles, the system triggers a warning block advising foundational bridging before applying to jobs.
4. **Human-in-the-Loop Design:**
   - The AI does *not* make the final career decision. The system remains strictly an analysis input. The student must manually select their final path, write a personalized rationale (weighting qualitative interests, family circumstances, and passions), and lock it in to generate the execution plan.

---

## 🛠️ Tech Stack & Dependencies
* **UI Framework:** Streamlit (Vanilla style)
* **NLP & Vector Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`)
* **Vector Math:** Scikit-Learn (Cosine Similarity)
* **Data Processing:** Pandas, NumPy
* **Interactive Charting:** Plotly Express / Graph Objects

*Note: In compliance with student data privacy and PII protection standard regulations, all student profiles and job requirement datasets utilized are synthetic, loaded locally from `student_profiles.csv` and `job_requirements.csv`.*

---

## ⚙️ Running Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```
3. The app will launch in your default web browser (typically at `http://localhost:8501`).
