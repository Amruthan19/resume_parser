# app1.py
import streamlit as st
import pandas as pd
import os
import sys
import time
import io
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import ResumeRanker

load_dotenv()


def main():
    st.set_page_config(
        page_title="AI Resume Ranker",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main-header { font-size: 2.5rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; border-radius: 10px; text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">🎯 AI Resume Ranker</h1>', unsafe_allow_html=True)
    st.markdown("### Rank resumes (PDF / DOCX / TXT) against a job description")

    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'job_desc_used' not in st.session_state:
        st.session_state.job_desc_used = ""

    with st.sidebar:
        st.header("🔑 API Configuration")
        cohere_key = st.text_input(
            "Cohere API Key", type="password",
            placeholder="Enter Cohere API key", value=os.getenv("COHERE_API_KEY")
        )
        groq_key = st.text_input(
            "Groq API Key", type="password",
            placeholder="Enter Groq API key", value=os.getenv("GROQ_API_KEY")
        )

        st.markdown("---")
        st.header("⚙️ Settings")
        top_k = st.slider("Number of top matches to show", 3, 20, 10)

        st.markdown("**Score weighting**")
        similarity_weight_pct = st.slider(
            "NLP similarity weight (%)", 0, 100, 50,
            help="The rest is weighted toward the LLM's qualitative assessment."
        )
        similarity_weight = similarity_weight_pct / 100
        llm_weight = 1 - similarity_weight

        st.markdown("---")
        st.header("ℹ️ How It Works")
        st.markdown("""
        1. **Enter Job Description**
        2. **Upload Resumes** — PDF, DOCX, or TXT
        3. **Retrieval** — top-matching chunks per resume are found via
           embedding cosine similarity (NLP similarity score)
        4. **LLM Analysis** — skills, experience, education are extracted
           and a qualitative match score is produced
        5. **Final Score** = blend of similarity + LLM scores
        """)
        with st.expander("📄 Scoring Methodology"):
            st.markdown(_scoring_method_text())

    tab1, tab2 = st.tabs(["📋 Job & Resumes", "📊 Results"])

    with tab1:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🎯 Job Description")
            job_desc = st.text_area(
                "Enter the job description:",
                height=300,
                placeholder="Paste the complete job description here...",
                help="Be specific about required skills, experience, and qualifications for better matching"
            )
            if job_desc:
                st.info(f"📝 Job description length: {len(job_desc)} characters")

        with col2:
            st.subheader("📁 Upload Resumes")
            uploaded_files = st.file_uploader(
                "Choose resume files",
                type=["pdf", "docx", "txt"],
                accept_multiple_files=True,
                help="Upload 10+ resumes as PDF, DOCX, or TXT"
            )
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} resume(s) uploaded")
                with st.expander("📄 View Uploaded Resumes"):
                    for i, file in enumerate(uploaded_files):
                        st.write(f"{i+1}. **{file.name}** ({file.size // 1024} KB)")

        if job_desc and uploaded_files and cohere_key and groq_key:
            if st.button("🚀 Rank Resumes", type="primary", use_container_width=True):
                process_resumes(job_desc, uploaded_files, cohere_key, groq_key, top_k,
                                 similarity_weight, llm_weight)
        else:
            missing = []
            if not job_desc: missing.append("job description")
            if not uploaded_files: missing.append("resume files")
            if not cohere_key: missing.append("Cohere API key")
            if not groq_key: missing.append("Groq API key")
            if missing:
                st.warning(f"⚠️ Please provide: {', '.join(missing)}")

    with tab2:
        if st.session_state.processing_complete and st.session_state.processed_results is not None:
            display_results(st.session_state.processed_results, st.session_state.job_desc_used,
                             similarity_weight, llm_weight)
        else:
            st.info("👆 Go to 'Job & Resumes' tab to upload and rank resumes")


def process_resumes(job_desc, uploaded_files, cohere_key, groq_key, top_k,
                     similarity_weight, llm_weight):
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔄 Initializing AI models...")
        ranker = ResumeRanker(similarity_weight=similarity_weight, llm_weight=llm_weight)
        progress_bar.progress(10)

        status_text.text("📁 Preparing resume files...")
        resume_files_data = []
        for uploaded_file in uploaded_files:
            resume_files_data.append({
                'name': uploaded_file.name,
                'content': uploaded_file.getvalue()
            })
        progress_bar.progress(20)

        status_text.text("🔍 Analyzing resumes against job description...")
        results_df = ranker.rank_resumes(
            job_desc=job_desc,
            resume_files_data=resume_files_data,
            cohere_key=cohere_key,
            groq_key=groq_key,
            k=top_k
        )
        progress_bar.progress(80)

        if not results_df.empty:
            def get_match_level(score):
                if score >= 90: return "🎯 Excellent Match"
                elif score >= 80: return "⭐ Very Good Match"
                elif score >= 70: return "✓ Good Match"
                elif score >= 60: return "↻ Partial Match"
                else: return "⚠️ Needs Review"

            results_df['Match_Level'] = results_df['Score'].apply(get_match_level)

            st.session_state.processed_results = results_df
            st.session_state.job_desc_used = job_desc
            st.session_state.processing_complete = True

            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            st.rerun()
        else:
            st.error("❌ No results generated. Please check your files and try again.")

    except Exception as e:
        st.error(f"❌ Error processing resumes: {str(e)}")
        status_text.empty()
        progress_bar.empty()


def display_results(results_df, job_desc, similarity_weight, llm_weight):
    st.success(f"✅ Successfully analyzed {len(results_df)} resumes!")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Average Score</h3><h2>{results_df["Score"].mean():.1f}/100</h2></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Top Score</h3><h2>{results_df["Score"].max()}/100</h2></div>', unsafe_allow_html=True)
    with col3:
        qualified = len(results_df[results_df['Score'] >= 70])
        st.markdown(f'<div class="metric-card"><h3>Qualified</h3><h2>{qualified}</h2></div>', unsafe_allow_html=True)
    with col4:
        strong = len(results_df[results_df['Score'] >= 85])
        st.markdown(f'<div class="metric-card"><h3>Strong Matches</h3><h2>{strong}</h2></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("👑 Top Candidate Spotlight")
    top = results_df.iloc[0]
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Overall Score", f"{top['Score']}/100")
        st.caption(f"NLP similarity: {top['NLP_Similarity_Score']} | LLM assessment: {top['LLM_Assessment_Score']}")
        st.write(f"**Match Level:** {top['Match_Level']}")
    with col2:
        st.markdown(f"**Resume:** `{top['Resume']}`")
        t1, t2, t3, t4 = st.tabs(["✅ Strengths", "🧩 Skills", "🎓 Education/Exp", "💡 Gaps"])
        with t1: st.info(top['Strengths'])
        with t2: st.success(top['Skills'])
        with t3:
            st.write(f"**Education:** {top['Education']}")
            st.write(f"**Experience:** {top['Experience']}")
        with t4: st.warning(top['Gaps'])

    st.markdown("---")
    st.subheader("📊 Resume Rankings")

    display_columns = ['Rank', 'Resume', 'Score', 'NLP_Similarity_Score', 'LLM_Assessment_Score',
                        'Match_Level', 'Skills', 'Education', 'Strengths', 'Gaps']
    display_columns = [c for c in display_columns if c in results_df.columns]
    styled_df = results_df[display_columns].copy()

    def color_score(val):
        if isinstance(val, (int, float)):
            if val >= 90: return 'color: green; font-weight: bold'
            elif val >= 80: return 'color: blue; font-weight: bold'
            elif val >= 70: return 'color: orange'
            else: return 'color: red'
        return ''

    styled_display = styled_df.style.map(color_score, subset=['Score'])
    st.dataframe(styled_display, use_container_width=True, height=400)

    st.subheader("🔍 Detailed Analysis")
    for idx, row in results_df.iterrows():
        score_color = "🟢" if row['Score'] >= 90 else "🔵" if row['Score'] >= 80 else "🟠" if row['Score'] >= 70 else "🔴"
        with st.expander(f"{score_color} #{row['Rank']} {row['Resume']} - Score: {row['Score']}/100 - {row['Match_Level']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**✅ Strengths & Match:**")
                st.info(row['Strengths'])
                st.write("**💼 Experience:**")
                st.text(row['Experience'])
                st.write("**🎓 Education:**")
                st.text(row['Education'])
            with col2:
                st.write("**📈 Areas for Improvement:**")
                st.warning(row['Gaps'])
                st.write("**🔧 Skills Identified:**")
                st.success(row['Skills'])
                st.caption(f"NLP similarity score: {row['NLP_Similarity_Score']} | LLM score: {row['LLM_Assessment_Score']}")

    st.markdown("---")
    st.subheader("💾 Download Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        csv_data = results_df.to_csv(index=False)
        st.download_button("📥 Download CSV", data=csv_data, file_name="resume_ranking_results.csv",
                            mime="text/csv", use_container_width=True)

    with col2:
        ranker = ResumeRanker(similarity_weight=similarity_weight, llm_weight=llm_weight)
        json_data = ranker.to_json(results_df, job_desc=job_desc)
        st.download_button("🧾 Download JSON", data=json_data, file_name="resume_ranking_results.json",
                            mime="application/json", use_container_width=True)

    with col3:
        summary_text = generate_summary_report(results_df)
        st.download_button("📄 Download Summary", data=summary_text, file_name="ranking_summary.txt",
                            mime="text/plain", use_container_width=True)

    with col4:
        @st.cache_data
        def convert_to_excel(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Ranking Results')
            return output.getvalue()

        excel_data = convert_to_excel(results_df)
        st.download_button("📊 Download Excel", data=excel_data, file_name="resume_ranking_results.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True)


def generate_summary_report(results_df):
    top = results_df.iloc[0]
    report = f"""RESUME RANKING SUMMARY REPORT
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Total Resumes Analyzed: {len(results_df)}

TOP CANDIDATE
-------------
Name: {top['Resume']}
Score: {top['Score']}/100  (NLP similarity: {top['NLP_Similarity_Score']}, LLM: {top['LLM_Assessment_Score']})
Match Level: {top['Match_Level']}
Skills: {top['Skills']}
Education: {top['Education']}
Strengths: {top['Strengths']}
Gaps: {top['Gaps']}

OVERALL RANKINGS
----------------
"""
    for idx, row in results_df.iterrows():
        report += f"\n{row['Rank']}. {row['Resume']} - {row['Score']}/100 - {row['Match_Level']}"

    report += f"""

SUMMARY STATISTICS
-------------------
Average Score: {results_df['Score'].mean():.1f}/100
Top Score: {results_df['Score'].max()}/100
Qualified Candidates (70+): {len(results_df[results_df['Score'] >= 70])}
Strong Matches (85+): {len(results_df[results_df['Score'] >= 85])}
"""
    return report


def _scoring_method_text():
    return """
**Score = (similarity_weight × NLP Similarity Score) + (llm_weight × LLM Assessment Score)**

- **NLP Similarity Score**: average cosine similarity (via Cohere embeddings,
  retrieved from Qdrant) between the job description and the resume's most
  relevant chunks, scaled to 0–100.
- **LLM Assessment Score**: a Llama-3.3-70B (via Groq) qualitative score (0–100)
  based on skills, experience, and education alignment, along with written
  strengths/gaps reasoning.

Full details in `SCORING_METHOD.md`.
"""


if __name__ == "__main__":
    main()
