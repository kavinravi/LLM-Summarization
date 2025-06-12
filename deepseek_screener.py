#!/usr/bin/env python
"""
Streamlit Web App for Renewable Energy Project Document Screening

This app provides a user-friendly interface for:
1. Uploading and screening documents against renewable energy criteria
2. Managing and customizing screening criteria
3. Generating marketing blurbs from screening results
4. Viewing and downloading results
"""

import streamlit as st
import json
import tempfile
import os
from datetime import datetime
import pandas as pd

# Import our existing utility functions
from deepseek_util import extract_text, deepseek_screen, deepseek_blurb
import textwrap

# Page configuration
st.set_page_config(
    page_title="Renewable Energy Project Screener",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CHUNK_SIZE_CHARS = 15000

def load_default_criteria():
    """Load the default criteria from criteria.json"""
    try:
        with open("criteria.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error("Default criteria file not found. Please ensure criteria.json exists.")
        return []

def process_document(uploaded_file, criteria, progress_bar=None):
    """Process an uploaded document against the given criteria"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        # Extract text from document
        if progress_bar:
            progress_bar.progress(20, "Extracting text from document...")
        
        raw_text = extract_text(tmp_file_path)
        
        if progress_bar:
            progress_bar.progress(40, "Chunking document...")
        
        # Split into chunks
        chunks = textwrap.wrap(raw_text, CHUNK_SIZE_CHARS)
        
        if progress_bar:
            progress_bar.progress(60, "Screening against criteria...")
        
        # Screen each chunk
        verdicts = {}
        for i, chunk in enumerate(chunks):
            chunk_name = f"chunk_{i+1}"
            verdicts[chunk_name] = deepseek_screen(chunk, criteria)
            
            if progress_bar:
                progress = 60 + (30 * (i + 1) / len(chunks))
                progress_bar.progress(int(progress), f"Processing chunk {i+1} of {len(chunks)}...")
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        if progress_bar:
            progress_bar.progress(100, "Complete!")
        
        return verdicts, raw_text
        
    except Exception as e:
        # Clean up temporary file in case of error
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise e

def format_verdict_results(verdicts):
    """Format verdict results for display, consolidating duplicate short-name keys."""
    all_results = {}
    
    # Build a helper mapping from prefix to full criterion string (if available)
    original_criteria = st.session_state.get('used_criteria', [])
    prefix_map = {}
    for crit in original_criteria:
        prefix = crit.split('–')[0].strip()  # en dash or hyphen
        prefix_map[prefix.lower()] = crit
    
    def _map_name(name: str) -> str:
        if name in original_criteria:
            return name
        pref = name.split('–')[0].strip().lower()
        return prefix_map.get(pref, name)
    
    # Aggregate results from all chunks
    for chunk_name, chunk_results in verdicts.items():
        for criterion, result in chunk_results.items():
            criterion = _map_name(criterion)
            if criterion not in all_results:
                all_results[criterion] = {
                    'yes_count': 0,
                    'no_count': 0,
                    'reasons': []
                }
            
            if isinstance(result, dict):
                verdict = result.get('verdict', 'unknown').lower()
                reason = result.get('reason', 'No reason provided')
            else:
                verdict = str(result).lower()
                reason = 'No reason provided'
            
            if verdict == 'yes':
                all_results[criterion]['yes_count'] += 1
            elif verdict == 'no':
                all_results[criterion]['no_count'] += 1
            else:
                all_results[criterion]['no_count'] += 1
            
            all_results[criterion]['reasons'].append(f"{chunk_name}: {reason}")
    
    return all_results

def create_results_dataframe(formatted_results):
    """Create a pandas DataFrame from formatted results"""
    data = []
    for criterion, result in formatted_results.items():
        total_chunks = result['yes_count'] + result['no_count']
        pass_rate = (result['yes_count'] / total_chunks * 100) if total_chunks > 0 else 0
        
        overall_verdict = "PASS" if result['yes_count'] > result['no_count'] else "FAIL"
        
        data.append({
            'Criterion': criterion,
            'Overall Verdict': overall_verdict,
            'Pass Rate': f"{pass_rate:.1f}%",
            'Passes': result['yes_count'],
            'Fails': result['no_count'],
            'Total Chunks': total_chunks
        })
    
    return pd.DataFrame(data)

def consolidate_chunk_results(verdicts, criteria):
    """Consolidate results from all chunks into one final verdict per criterion."""
    consolidated = {}
    
    for criterion in criteria:
        # Collect all verdicts and reasons for this criterion across chunks
        all_verdicts = []
        all_reasons = []
        
        for chunk_name, chunk_results in verdicts.items():
            if criterion in chunk_results:
                result = chunk_results[criterion]
                verdict = result.get('verdict', 'unknown')
                reason = result.get('reason', '')
                all_verdicts.append(verdict)
                all_reasons.append(reason)
        
        # Determine final verdict (if any chunk says 'yes', it's a pass)
        yes_count = all_verdicts.count('yes')
        no_count = all_verdicts.count('no')
        
        if yes_count > 0:
            final_verdict = 'PASS'
            # Find the best 'yes' reason
            positive_reasons = [r for i, r in enumerate(all_reasons) if all_verdicts[i] == 'yes']
            final_reason = positive_reasons[0] if positive_reasons else 'Criterion met in document'
        elif no_count > 0:
            final_verdict = 'FAIL'
            # Combine the 'no' reasons
            negative_reasons = [r for i, r in enumerate(all_reasons) if all_verdicts[i] == 'no']
            final_reason = negative_reasons[0] if negative_reasons else 'Criterion not met in document'
        else:
            final_verdict = 'UNKNOWN'
            final_reason = 'Insufficient information to determine compliance'
        
        consolidated[criterion] = {
            'verdict': final_verdict,
            'reason': final_reason
        }
    
    return consolidated

# Main app
def main():
    # Header
    st.title("🌱 Renewable Energy Project Document Screener")
    st.markdown("Upload documents and screen them against renewable energy project criteria using AI analysis.")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📄 Document Screening", 
        "📊 Results Analysis",
        "✍️ Marketing Blurb Generator"
    ])
    
    if page == "📄 Document Screening":
        document_screening_page()
    elif page == "📊 Results Analysis":
        results_analysis_page()
    elif page == "✍️ Marketing Blurb Generator":
        marketing_blurb_page()

def document_screening_page():
    st.header("Document Screening")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload documents to screen",
        type=['pdf', 'docx', 'doc', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Supported formats: PDF, DOCX, DOC, XLSX, XLS"
    )
    
    # Criteria selection
    st.subheader("Screening Criteria")
    
    criteria_source = st.radio(
        "Choose criteria source:",
        ["Use default criteria", "Upload custom criteria", "Enter custom criteria"]
    )
    
    criteria = []
    
    if criteria_source == "Use default criteria":
        criteria = load_default_criteria()
        st.success(f"Loaded {len(criteria)} default criteria")
        
        with st.expander("View default criteria"):
            for i, criterion in enumerate(criteria, 1):
                st.write(f"{i}. {criterion}")
    
    elif criteria_source == "Upload custom criteria":
        criteria_file = st.file_uploader(
            "Upload criteria file",
            type=['json', 'yml', 'yaml'],
            help="Upload a JSON or YAML file containing criteria"
        )
        
        if criteria_file:
            try:
                if criteria_file.name.endswith('.json'):
                    criteria = json.load(criteria_file)
                else:
                    import yaml
                    criteria = yaml.safe_load(criteria_file)
                
                st.success(f"Loaded {len(criteria)} criteria from file")
                
                with st.expander("View uploaded criteria"):
                    for i, criterion in enumerate(criteria, 1):
                        st.write(f"{i}. {criterion}")
            except Exception as e:
                st.error(f"Error loading criteria file: {str(e)}")
    
    elif criteria_source == "Enter custom criteria":
        st.write("Enter criteria (one per line):")
        criteria_text = st.text_area(
            "Criteria",
            height=200,
            placeholder="Enter each criterion on a new line...",
            label_visibility="collapsed"
        )
        
        if criteria_text.strip():
            criteria = [line.strip() for line in criteria_text.split('\n') if line.strip()]
            st.success(f"Added {len(criteria)} criteria")
    
    # Process documents
    if uploaded_files and criteria:
        if st.button("🔍 Start Screening", type="primary"):
            results = {}
            
            for uploaded_file in uploaded_files:
                st.subheader(f"Processing: {uploaded_file.name}")
                
                progress_bar = st.progress(0, f"Processing {uploaded_file.name}...")
                
                try:
                    verdicts, raw_text = process_document(uploaded_file, criteria, progress_bar)
                    results[uploaded_file.name] = {
                        'verdicts': verdicts,
                        'raw_text': raw_text,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.success(f"✅ Completed screening of {uploaded_file.name}")
                    
                except Exception as e:
                    st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            
            # Store results in session state
            st.session_state['screening_results'] = results
            st.session_state['used_criteria'] = criteria
            
            # Display summary
            if results:
                st.success(f"✅ Successfully screened {len(results)} document(s)")
                st.info("💡 Go to the 'Results Analysis' page to view detailed results.")

def results_analysis_page():
    st.header("Results Analysis")
    
    if 'screening_results' not in st.session_state:
        st.info("No screening results available. Please run document screening first.")
        return
    
    results = st.session_state['screening_results']
    criteria = st.session_state.get('used_criteria', [])
    
    # File selection
    selected_file = st.selectbox(
        "Select file to analyze:",
        list(results.keys())
    )
    
    if selected_file:
        result = results[selected_file]
        
        # Consolidate all chunk results into final verdicts
        consolidated = consolidate_chunk_results(result['verdicts'], criteria)
        
        st.subheader("📋 Final Analysis Results")
        
        # Display each criterion with its final verdict
        for criterion, analysis in consolidated.items():
            verdict = analysis['verdict']
            reason = analysis['reason']
            
            # Color code the verdict
            if verdict == 'PASS':
                verdict_display = "✅ **PASS**"
                color = "#039125"
            elif verdict == 'FAIL':
                verdict_display = "❌ **FAIL**"
                color = "#8f010e"
            else:
                verdict_display = "❓ **UNKNOWN**"
                color = "#b08704"
            
            # Display in a nice card format
            st.markdown(
                f"""
                <div style="
                    background-color: {color};
                    border-radius: 5px;
                    padding: 15px;
                    margin: 10px 0;
                    border-left: 4px solid #6c757d;
                ">
                <strong>{criterion}</strong><br/>
                {verdict_display}: {reason}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Download options
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download consolidated results as JSON
            consolidated_json = json.dumps(consolidated, indent=2)
            st.download_button(
                "📄 Download Final Results (JSON)",
                data=consolidated_json,
                file_name=f"final_analysis_{selected_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Download as readable summary
            summary_text = f"Final Analysis Results for {selected_file}\n"
            summary_text += f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            
            for criterion, analysis in consolidated.items():
                summary_text += f"{criterion}:\n"
                summary_text += f"  Verdict: {analysis['verdict']}\n"
                summary_text += f"  Reasoning: {analysis['reason']}\n\n"
            
            st.download_button(
                "📊 Download Summary (TXT)",
                data=summary_text,
                file_name=f"final_summary_{selected_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

def marketing_blurb_page():
    st.header("Marketing Blurb Generator")
    
    if 'screening_results' not in st.session_state:
        st.info("No screening results available. Please run document screening first.")
        return
    
    results = st.session_state['screening_results']
    
    # File selection
    selected_file = st.selectbox(
        "Select file to generate blurb for:",
        list(results.keys()),
        key="blurb_file_select"
    )
    
    if selected_file:
        result = results[selected_file]
        
        # Fixed temperature
        temperature = 0.6
        
        # Generate blurb
        if st.button("✍️ Generate Marketing Blurb", type="primary"):
            with st.spinner("Generating marketing blurb..."):
                try:
                    blurb = deepseek_blurb(result['verdicts'], temperature)
                    
                    # Store blurb in session state
                    st.session_state['generated_blurb'] = blurb
                    st.session_state['blurb_filename'] = selected_file
                    
                    st.subheader("📝 Generated Marketing Blurb")
                    
                    # Display blurb in a nice box
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f0f7ff;
                            border-left: 4px solid #1f77b4;
                            padding: 20px;
                            margin: 20px 0;
                            border-radius: 5px;
                            font-style: italic;
                        ">
                        {blurb}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating blurb: {str(e)}")
        
        # Show download button if blurb exists
        if 'generated_blurb' in st.session_state and st.session_state.get('blurb_filename') == selected_file:
            blurb_content = st.session_state['generated_blurb']
            if blurb_content and blurb_content.strip():
                st.download_button(
                    "📥 Download Blurb",
                    data=blurb_content,
                    file_name=f"marketing_blurb_{selected_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
            else:
                st.warning("⚠️ Generated blurb is empty. Try generating again.")

if __name__ == "__main__":
    main() 