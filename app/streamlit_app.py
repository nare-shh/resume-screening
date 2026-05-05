"""
streamlit_app.py
-----------------
Streamlit web UI for Resume Screening.
Supports PDF/DOCX/TXT upload or paste resume text.
Shows side-by-side comparison of both ML models.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Resume Screener AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
        font-size: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-card h2 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    .metric-card p {
        margin: 0;
        opacity: 0.9;
        font-size: 0.9rem;
    }
    .shortlisted {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    }
    .rejected {
        background: linear-gradient(135deg, #fc5c7d 0%, #6a3093 100%) !important;
    }
    .verdict-box {
        padding: 1rem 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    .model-card {
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        background: #fafafa;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    .info-badge {
        background: #e8f4fd;
        color: #1565c0;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Caching ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ML models...")
def load_models():
    """Load both trained models (cached across sessions)."""
    from models.model1_tfidf_rf import TFIDFRandomForestScreener
    from models.model2_bert import BERTSimilarityScreener

    model1 = TFIDFRandomForestScreener()
    model2 = BERTSimilarityScreener()

    model1_path = "saved_models/model1_tfidf_rf.pkl"
    model2_path = "saved_models/model2_bert.pkl"

    if os.path.exists(model1_path):
        model1.load(model1_path)
        m1_loaded = True
    else:
        m1_loaded = False

    if os.path.exists(model2_path):
        model2.load(model2_path)
        m2_loaded = True
    else:
        m2_loaded = False

    return model1, model2, m1_loaded, m2_loaded


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded PDF, DOCX, or TXT file."""
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = " ".join(page.extract_text() or "" for page in reader.pages)
            return text.strip()
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""

    elif filename.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(uploaded_file.read()))
            return " ".join(p.text for p in doc.paragraphs).strip()
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""

    elif filename.endswith(".txt"):
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()

    else:
        st.warning("Unsupported file type. Attempting as plain text.")
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()


def make_gauge_chart(score: float, title: str, color: str = "#667eea") -> go.Figure:
    """Create a gauge chart for a model score."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 16}},
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 40], "color": "#ffe0e0"},
                {"range": [40, 60], "color": "#fff8e1"},
                {"range": [60, 100], "color": "#e8f5e9"},
            ],
            "threshold": {
                "line": {"color": "red", "width": 3},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_comparison_bar(results: list) -> go.Figure:
    """Create a bar chart comparing both models."""
    model_names = [r["model_name"].replace(" + ", "\n+\n") for r in results]
    scores = [r["score"] for r in results]
    colors = ["#667eea" if r["shortlisted"] else "#fc5c7d" for r in results]

    fig = go.Figure(go.Bar(
        x=model_names,
        y=scores,
        marker_color=colors,
        text=[f"{s:.1f}%" for s in scores],
        textposition="outside",
        width=0.4,
    ))
    fig.add_hline(
        y=50, line_dash="dash",
        line_color="gray",
        annotation_text="Threshold (50%)",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Model Comparison",
        yaxis=dict(range=[0, 110], title="Match Score (%)"),
        xaxis_title="Model",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


# ─── App Layout ───────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<div class="main-header">📄 Resume Screener AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by TF-IDF + Random Forest & BERT Similarity | MLflow Tracked</div>', unsafe_allow_html=True)

    # Load models
    model1, model2, m1_loaded, m2_loaded = load_models()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        threshold = st.slider(
            "Shortlist Threshold (%)",
            min_value=30, max_value=80, value=50, step=5,
            help="Resumes scoring above this threshold are shortlisted"
        )

        st.divider()
        st.header("📊 Model Status")

        if m1_loaded:
            st.success("✅ Model 1: TF-IDF + RF")
        else:
            st.error("❌ Model 1 not found — run train.py")

        if m2_loaded:
            st.success("✅ Model 2: BERT Similarity")
        else:
            st.error("❌ Model 2 not found — run train.py")

        st.divider()
        st.header("🔗 MLflow")
        st.info("To view experiment tracking:\n\n`mlflow ui --port 5000`\n\nthen open http://localhost:5000")

        st.divider()
        st.caption("Built with ❤️ | TF-IDF + BERT + MLflow + Jenkins")

    if not (m1_loaded and m2_loaded):
        st.error("⚠️ One or more models are not trained yet. Please run `python train.py` first.")
        st.code("python train.py", language="bash")
        return

    # Update thresholds dynamically
    model1.threshold = threshold / 100
    model2.threshold = threshold / 100

    # ─── Input Tabs ───────────────────────────────────────────────────────────

    tab1, tab2 = st.tabs(["📁 Upload Files", "✏️ Paste Text"])

    resume_text = ""
    job_desc = ""

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 Resume")
            resume_file = st.file_uploader(
                "Upload resume",
                type=["pdf", "docx", "txt"],
                key="resume_file",
            )
            if resume_file:
                resume_text = extract_text_from_file(resume_file)
                if resume_text:
                    with st.expander("Preview extracted text"):
                        st.text(resume_text[:800] + ("..." if len(resume_text) > 800 else ""))

        with col2:
            st.subheader("📋 Job Description")
            jd_file = st.file_uploader(
                "Upload job description",
                type=["pdf", "docx", "txt"],
                key="jd_file",
            )
            if jd_file:
                job_desc = extract_text_from_file(jd_file)
                if job_desc:
                    with st.expander("Preview extracted text"):
                        st.text(job_desc[:800] + ("..." if len(job_desc) > 800 else ""))

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📄 Resume Text")
            resume_text_input = st.text_area(
                "Paste resume here",
                height=300,
                placeholder="Paste the full resume text here...\n\nExample:\nJohn Doe | Python, Machine Learning, TensorFlow\n5 years experience at Google...",
                key="resume_text_input",
            )
            if resume_text_input.strip():
                resume_text = resume_text_input

        with col2:
            st.subheader("📋 Job Description Text")
            jd_text_input = st.text_area(
                "Paste job description here",
                height=300,
                placeholder="Paste the job description here...\n\nExample:\nWe are looking for a Data Scientist with 3+ years of experience in Python, ML...",
                key="jd_text_input",
            )
            if jd_text_input.strip():
                job_desc = jd_text_input

    # ─── Sample Data Button ───────────────────────────────────────────────────

    with st.expander("🧪 Try with sample data"):
        if st.button("Load Sample Resume & JD", type="secondary"):
            st.session_state["sample_resume"] = """
John Smith | Data Scientist
Email: john@email.com | LinkedIn: linkedin.com/in/johnsmith

EDUCATION
M.Sc in Data Science, IIT Bombay | 2021

SKILLS
Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, Scikit-learn,
Pandas, NumPy, SQL, NLP, MLflow, Docker, AWS, Apache Spark, Tableau,
Feature Engineering, A/B Testing, Statistical Analysis

EXPERIENCE
Senior Data Scientist | Google | 3 years
- Built ML models for recommendation systems (CTR improved by 15%)
- Led NLP project for sentiment analysis on 10M+ customer reviews
- Deployed models using Docker + Kubernetes on GCP

Data Analyst | Amazon | 2 years
- Analyzed large datasets using Python and SQL
- Created dashboards in Tableau for business insights
- Implemented A/B tests for product features

CERTIFICATIONS
AWS Certified Machine Learning Specialist
            """.strip()
            st.session_state["sample_jd"] = """
JOB TITLE: Senior Data Scientist

ABOUT THE ROLE
We are looking for an experienced Data Scientist to join our AI team.

REQUIRED SKILLS
Python, Machine Learning, Deep Learning, TensorFlow or PyTorch,
Scikit-learn, Pandas, NumPy, SQL, NLP, Docker, Cloud platforms (AWS/GCP)

RESPONSIBILITIES
- Build and deploy ML models at scale
- Work with large datasets and extract insights
- Collaborate with engineering teams for model deployment
- A/B testing and statistical analysis

QUALIFICATIONS
- 3+ years in Data Science or ML engineering
- Strong Python and SQL skills
- Experience with MLOps tools (MLflow, Docker, Kubernetes)
            """.strip()
            st.rerun()

    # Prefill from session state if sample loaded
    if "sample_resume" in st.session_state and not resume_text:
        resume_text = st.session_state["sample_resume"]
        st.info("📌 Sample resume loaded! Switch to 'Paste Text' tab to edit or use directly.")
    if "sample_jd" in st.session_state and not job_desc:
        job_desc = st.session_state["sample_jd"]

    # ─── Screen Button ────────────────────────────────────────────────────────

    st.divider()
    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])
    with col_btn2:
        screen_clicked = st.button(
            "🔍 Screen Resume",
            type="primary",
            use_container_width=True,
            disabled=not (resume_text.strip() and job_desc.strip()),
        )

    if not resume_text.strip() or not job_desc.strip():
        st.info("👆 Upload files or paste text above, then click **Screen Resume**.")
        return

    # ─── Results ──────────────────────────────────────────────────────────────

    if screen_clicked or ("last_results" in st.session_state):
        if screen_clicked:
            with st.spinner("🧠 Running both ML models..."):
                result1 = model1.predict(resume_text, job_desc)
                result2 = model2.predict(resume_text, job_desc)
                combined_score = (result1["score"] + result2["score"]) / 2
                combined_shortlisted = combined_score >= threshold
                st.session_state["last_results"] = (result1, result2, combined_score, combined_shortlisted)

        result1, result2, combined_score, combined_shortlisted = st.session_state["last_results"]

        st.divider()
        st.subheader("📊 Screening Results")

        # Combined verdict banner
        verdict_class = "shortlisted" if combined_shortlisted else "rejected"
        verdict_text = "✅ SHORTLISTED" if combined_shortlisted else "❌ REJECTED"
        combined_bg = "#11998e" if combined_shortlisted else "#fc5c7d"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {combined_bg} 0%, {'#38ef7d' if combined_shortlisted else '#6a3093'} 100%);
                    padding: 1.5rem; border-radius: 12px; text-align: center; color: white; margin-bottom: 1rem;">
            <h1 style="margin:0; font-size: 2.5rem;">{verdict_text}</h1>
            <p style="margin:0; font-size: 1.2rem; opacity: 0.9;">Combined Score: {combined_score:.1f}% (Threshold: {threshold}%)</p>
        </div>
        """, unsafe_allow_html=True)

        # Model gauges
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Model 1: TF-IDF + RF")
            st.plotly_chart(
                make_gauge_chart(result1["score"], "TF-IDF + Random Forest", "#667eea"),
                use_container_width=True,
            )
            verdict_color = "success" if result1["shortlisted"] else "error"
            if result1["shortlisted"]:
                st.success(result1["verdict"])
            else:
                st.error(result1["verdict"])

        with col2:
            st.markdown("### Model 2: BERT")
            st.plotly_chart(
                make_gauge_chart(result2["score"], "BERT Similarity", "#11998e"),
                use_container_width=True,
            )
            if result2["shortlisted"]:
                st.success(result2["verdict"])
            else:
                st.error(result2["verdict"])
            # BERT-specific metrics
            if "cosine_similarity" in result2:
                st.caption(f"🧠 Semantic Similarity: {result2['cosine_similarity']:.1f}%")
                st.caption(f"🔑 Keyword Overlap: {result2['keyword_overlap']:.1f}%")

        with col3:
            st.markdown("### Comparison")
            st.plotly_chart(
                make_comparison_bar([result1, result2]),
                use_container_width=True,
            )

        # Detailed metrics table
        st.divider()
        st.subheader("📋 Detailed Metrics")

        metrics_data = {
            "Metric": ["Match Score", "Probability", "Verdict", "Threshold"],
            "TF-IDF + Random Forest": [
                f"{result1['score']:.1f}%",
                f"{result1['probability']:.4f}",
                result1["verdict"],
                f"{result1['threshold_used']*100:.0f}%",
            ],
            "BERT Similarity": [
                f"{result2['score']:.1f}%",
                f"{result2['probability']:.4f}",
                result2["verdict"],
                f"{result2['threshold_used']*100:.0f}%",
            ],
        }

        if "cosine_similarity" in result2:
            metrics_data["Metric"].extend(["Semantic Similarity", "Keyword Overlap"])
            metrics_data["TF-IDF + Random Forest"].extend(["N/A", "N/A"])
            metrics_data["BERT Similarity"].extend([
                f"{result2['cosine_similarity']:.1f}%",
                f"{result2['keyword_overlap']:.1f}%",
            ])

        st.dataframe(
            pd.DataFrame(metrics_data),
            use_container_width=True,
            hide_index=True,
        )

        # Action buttons
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Screen Another Resume"):
                for key in ["last_results", "sample_resume", "sample_jd"]:
                    st.session_state.pop(key, None)
                st.rerun()
        with col2:
            # Export results as CSV
            results_df = pd.DataFrame([{
                "Model": "TF-IDF + RF",
                "Score": result1["score"],
                "Verdict": result1["verdict"],
            }, {
                "Model": "BERT Similarity",
                "Score": result2["score"],
                "Verdict": result2["verdict"],
                "Semantic Similarity": result2.get("cosine_similarity", "N/A"),
                "Keyword Overlap": result2.get("keyword_overlap", "N/A"),
            }, {
                "Model": "Combined",
                "Score": combined_score,
                "Verdict": verdict_text,
            }])
            st.download_button(
                "⬇️ Export Results (CSV)",
                results_df.to_csv(index=False),
                file_name="screening_result.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
