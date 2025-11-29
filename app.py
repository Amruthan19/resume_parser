# app1.py (updated)
import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
import sys
import time
import io
from dotenv import load_dotenv
# Add the current directory to path to import your module
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
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .score-high { background-color: #d4edda; padding: 10px; border-radius: 5px; }
        .score-medium { background-color: #fff3cd; padding: 10px; border-radius: 5px; }
        .score-low { background-color: #f8d7da; padding: 10px; border-radius: 5px; }
        .resume-card { 
            border: 1px solid #ddd; 
            border-radius: 10px; 
            padding: 15px; 
            margin: 10px 0;
            background-color: #f9f9f9;
        }
        .top-candidate {
            border: 2px solid #28a745;
            background-color: #f0fff4;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">🎯 AI Resume Ranker</h1>', unsafe_allow_html=True)
    st.markdown("### Rank resumes based on job description relevance")
    
    # Initialize session state
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    
    # Sidebar for API keys
    with st.sidebar:
        st.header("🔑 API Configuration")
        
        cohere_key = st.text_input(
            "Cohere API Key",
            type="password",
            help="Enter your Cohere API key for embeddings",
            placeholder="Enter Cohere API key",
            value=os.getenv("COHERE_API_KEY")
        )
        
        groq_key = st.text_input(
            "Groq API Key", 
            type="password",
            help="Enter your Groq API key for LLM",
            placeholder="Enter Groq API key",
            value=os.getenv("GROQ_API_KEY")
        )
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        top_k = st.slider(
            "Number of top matches to show",
            min_value=3,
            max_value=20,
            value=10,
            help="How many top matching resumes to display"
        )
        
        st.markdown("---")
        st.header("ℹ️ How It Works")
        st.markdown("""
        1. **Enter Job Description** - What skills/experience you need
        2. **Upload Resumes** - PDF format
        3. **AI Analysis** - Compares resumes to job requirements
        4. **Get Rankings** - Scores based on relevance match
        
        **Scoring Criteria:**
        - 🎯 Skills match
        - 💼 Experience relevance  
        - 📚 Education alignment
        - 🔍 Keyword relevance
        """)
    
    # Main content area
    tab1, tab2 = st.tabs(["📋 Job & Resumes", "📊 Results"])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Job Description")
            job_desc = st.text_area(
                "Enter the job description:",
                height=300,
                placeholder="Paste the complete job description here...\n\nInclude:\n- Required skills\n- Experience level\n- Qualifications\n- Responsibilities\n- Technologies needed\n\nExample:\n'We are looking for a Senior Python Developer with 5+ years of experience in web development. Must have expertise in Django, FastAPI, PostgreSQL, and AWS. Experience with machine learning and Docker is a plus. The candidate should have strong problem-solving skills and experience leading technical teams.'",
                help="Be specific about required skills, experience, and qualifications for better matching"
            )
            
            if job_desc:
                st.info(f"📝 Job description length: {len(job_desc)} characters")
        
        with col2:
            st.subheader("📁 Upload Resumes")
            uploaded_files = st.file_uploader(
                "Choose resume PDF files",
                type="pdf",
                accept_multiple_files=True,
                help="Upload multiple resume PDFs for ranking"
            )
            
            if uploaded_files:
                st.success(f"✅ {len(uploaded_files)} resume(s) uploaded")
                
                # Show file details
                with st.expander("📄 View Uploaded Resumes"):
                    for i, file in enumerate(uploaded_files):
                        st.write(f"{i+1}. **{file.name}** ({file.size // 1024} KB)")
    
        # Process button
        if job_desc and uploaded_files and cohere_key and groq_key:
            if st.button("🚀 Rank Resumes", type="primary", use_container_width=True):
                process_resumes(job_desc, uploaded_files, cohere_key, groq_key, top_k)
        else:
            # Show what's missing
            missing = []
            if not job_desc:
                missing.append("job description")
            if not uploaded_files:
                missing.append("resume PDFs")
            if not cohere_key:
                missing.append("Cohere API key")
            if not groq_key:
                missing.append("Groq API key")
                
            if missing:
                st.warning(f"⚠️ Please provide: {', '.join(missing)}")
    
    with tab2:
        if st.session_state.processing_complete and st.session_state.processed_results is not None:
            display_results(st.session_state.processed_results)
        else:
            st.info("👆 Go to 'Job & Resumes' tab to upload and rank resumes")
            
            # Show sample results for demo
            with st.expander("📸 See Sample Output"):
                sample_data = {
                    'Resume': ['John_Doe_Resume.pdf', 'Jane_Smith_CV.pdf', 'Mike_Johnson_Resume.pdf'],
                    'Score': [92, 85, 78],
                    'Match_Level': ['Excellent', 'Very Good', 'Good'],
                    'Key_Skills': ['Python, Django, AWS, PostgreSQL', 'Python, FastAPI, Docker, ML', 'JavaScript, React, Node.js'],
                    'Experience_Match': ['5+ years relevant', '4 years relevant', '3 years partial'],
                    'Strengths': ['Perfect skill match, leadership experience', 'Strong technical skills, ML background', 'Good fundamentals, fast learner'],
                    'Gaps': ['None significant', 'Less cloud experience', 'Missing some required technologies']
                }
                sample_df = pd.DataFrame(sample_data)
                st.dataframe(sample_df, use_container_width=True)

def process_resumes(job_desc, uploaded_files, cohere_key, groq_key, top_k):
    """Process resumes and generate rankings"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Initialize the ranker
        status_text.text("🔄 Initializing AI models...")
        ranker = ResumeRanker()
        progress_bar.progress(10)
        
        # Prepare file data for processing
        status_text.text("📁 Preparing resume files...")
        pdf_files_data = []
        for uploaded_file in uploaded_files:
            pdf_files_data.append({
                'name': uploaded_file.name,
                'content': uploaded_file.getvalue()
            })
        progress_bar.progress(20)
        
        # Run the ranking
        status_text.text("🔍 Analyzing resumes against job description...")
        results_df = ranker.rank_resumes(
            job_desc=job_desc,
            pdf_files_data=pdf_files_data,
            cohere_key=cohere_key,
            groq_key=groq_key,
            k=top_k
        )
        progress_bar.progress(80)
        
        if not results_df.empty:
            # Add match level based on score
            def get_match_level(score):
                if score >= 90:
                    return "🎯 Excellent Match"
                elif score >= 80:
                    return "⭐ Very Good Match"
                elif score >= 70:
                    return "✓ Good Match"
                elif score >= 60:
                    return "↻ Partial Match"
                else:
                    return "⚠️ Needs Review"
            
            results_df['Match_Level'] = results_df['Score'].apply(get_match_level)
            
            # Store results in session state
            st.session_state.processed_results = results_df
            st.session_state.processing_complete = True
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            time.sleep(1)
            status_text.empty()
            progress_bar.empty()
            
            # Switch to results tab
            st.rerun()
            
        else:
            st.error("❌ No results generated. Please check your files and try again.")
            
    except Exception as e:
        st.error(f"❌ Error processing resumes: {str(e)}")
        status_text.empty()
        progress_bar.empty()

def display_results(results_df):
    """Display ranking results in an organized way"""
    
    st.success(f"✅ Successfully analyzed {len(results_df)} resumes!")
    
    # Summary metrics with better styling
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_score = results_df['Score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Average Score</h3>
            <h2>{avg_score:.1f}/100</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        top_score = results_df['Score'].max()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Top Score</h3>
            <h2>{top_score}/100</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        qualified = len(results_df[results_df['Score'] >= 70])
        st.markdown(f"""
        <div class="metric-card">
            <h3>Qualified</h3>
            <h2>{qualified}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        strong_matches = len(results_df[results_df['Score'] >= 85])
        st.markdown(f"""
        <div class="metric-card">
            <h3>Strong Matches</h3>
            <h2>{strong_matches}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Top candidate spotlight
    st.subheader("👑 Top Candidate Spotlight")
    top_candidate = results_df.iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        score = top_candidate['Score']
        score_class = "score-high" if score >= 90 else "score-medium" if score >= 70 else "score-low"
        
        st.markdown(f'<div class="{score_class}">', unsafe_allow_html=True)
        st.metric("Overall Match Score", f"{score}/100")
        st.write(f"**Match Level:** {top_candidate['Match_Level']}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"**Resume:** `{top_candidate['Resume']}`")
        
        tab1, tab2, tab3 = st.tabs(["✅ Strengths", "📈 Key Skills", "💡 Suggestions"])
        
        with tab1:
            st.info(top_candidate['Strengths'])
        
        with tab2:
            if 'Key_Skills' in top_candidate:
                st.success(top_candidate['Key_Skills'])
            else:
                st.info("Skills analysis available in detailed view")
        
        with tab3:
            st.warning(top_candidate['Gaps'])
    
    st.markdown("---")
    
    # Main results table - FIXED VERSION
    st.subheader("📊 Resume Rankings")
    
    # Enhanced dataframe display
    display_columns = ['Resume', 'Score', 'Match_Level']
    if 'Key_Skills' in results_df.columns:
        display_columns.append('Key_Skills')
    display_columns.extend(['Strengths', 'Gaps'])
    
    styled_df = results_df[display_columns].copy()
    
    # FIX: Use Styler.map instead of Styler.applymap
    def color_score(val):
        if isinstance(val, (int, float)):
            if val >= 90:
                return 'color: green; font-weight: bold'
            elif val >= 80:
                return 'color: blue; font-weight: bold'
            elif val >= 70:
                return 'color: orange'
            else:
                return 'color: red'
        return ''
    
    # Apply the styling
    styled_display = styled_df.style.map(color_score, subset=['Score'])
    
    st.dataframe(
        styled_display,
        use_container_width=True,
        height=400
    )
    
    # Detailed analysis for each candidate
    st.subheader("🔍 Detailed Analysis")
    
    for idx, row in results_df.iterrows():
        # Create a colored badge based on score
        score_color = "🟢" if row['Score'] >= 90 else "🔵" if row['Score'] >= 80 else "🟠" if row['Score'] >= 70 else "🔴"
        
        with st.expander(f"{score_color} {row['Resume']} - Score: {row['Score']}/100 - {row['Match_Level']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**✅ Strengths & Match:**")
                st.info(row['Strengths'])
                
                if 'Experience_Match' in row:
                    st.write("**💼 Experience Match:**")
                    st.text(row['Experience_Match'])
            
            with col2:
                st.write("**📈 Areas for Improvement:**")
                st.warning(row['Gaps'])
                
                if 'Key_Skills' in row:
                    st.write("**🔧 Key Skills Identified:**")
                    st.success(row['Key_Skills'])
    
    # Download section
    st.markdown("---")
    st.subheader("💾 Download Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV download
        csv_data = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name="resume_ranking_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Summary report
        summary_text = generate_summary_report(results_df)
        st.download_button(
            label="📄 Download Summary Report",
            data=summary_text,
            file_name="ranking_summary.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col3:
        # Excel download
        @st.cache_data
        def convert_to_excel(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Ranking Results')
            return output.getvalue()
        
        excel_data = convert_to_excel(results_df)
        st.download_button(
            label="📊 Download Excel",
            data=excel_data,
            file_name="resume_ranking_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

def generate_summary_report(results_df):
    """Generate a text summary report"""
    top_candidate = results_df.iloc[0]
    
    report = f"""
RESUME RANKING SUMMARY REPORT
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
Total Resumes Analyzed: {len(results_df)}

TOP CANDIDATE:
---------------
Name: {top_candidate['Resume']}
Score: {top_candidate['Score']}/100
Match Level: {top_candidate['Match_Level']}

Strengths: {top_candidate['Strengths']}
Areas for Improvement: {top_candidate['Gaps']}

OVERALL RANKINGS:
-----------------
"""
    
    for idx, row in results_df.iterrows():
        report += f"\n{idx+1}. {row['Resume']} - {row['Score']}/100 - {row['Match_Level']}"
    
    report += f"""

SUMMARY STATISTICS:
-------------------
Average Score: {results_df['Score'].mean():.1f}/100
Top Score: {results_df['Score'].max()}/100
Qualified Candidates (70+): {len(results_df[results_df['Score'] >= 70])}
Strong Matches (85+): {len(results_df[results_df['Score'] >= 85])}

RECOMMENDATIONS:
----------------
"""
    
    # Add recommendations based on results
    if len(results_df[results_df['Score'] >= 80]) >= 3:
        report += "- Strong candidate pool available. Consider interviewing top 3-5 candidates.\n"
    elif len(results_df[results_df['Score'] >= 70]) >= 2:
        report += "- Moderate candidate pool. Interview top 2-3 candidates.\n"
    else:
        report += "- Limited qualified candidates. Consider reposting the job or expanding search criteria.\n"
    
    return report

if __name__ == "__main__":
    main()