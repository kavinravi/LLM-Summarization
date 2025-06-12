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

# Main app
def main():
    # Header
    st.title("🌱 Renewable Energy Project Document Screener")
    st.markdown("Upload documents and screen them against renewable energy project criteria using AI analysis.")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📄 Document Screening", 
        "📝 Manage Criteria", 
        "📊 Results Analysis",
        "✍️ Marketing Blurb Generator"
    ])
    
    if page == "📄 Document Screening":
        document_screening_page()
    elif page == "📝 Manage Criteria":
        manage_criteria_page()
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
                st.subheader("📊 Screening Summary")
                
                for filename, result in results.items():
                    with st.expander(f"Results for {filename}"):
                        formatted_results = format_verdict_results(result['verdicts'])
                        df = create_results_dataframe(formatted_results)
                        st.dataframe(df, use_container_width=True)
                
                st.info("💡 Go to the 'Results Analysis' page for detailed results and download options.")

def manage_criteria_page():
    st.header("Manage Screening Criteria")
    
    # Load current criteria
    default_criteria = load_default_criteria()
    
    st.subheader("Current Default Criteria")
    
    # Display current criteria with edit capability
    edited_criteria = []
    
    for i, criterion in enumerate(default_criteria):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            edited_criterion = st.text_input(
                f"Criterion {i+1}",
                value=criterion,
                key=f"criterion_{i}",
                label_visibility="collapsed"
            )
            edited_criteria.append(edited_criterion)
        
        with col2:
            if st.button("🗑️", key=f"delete_{i}", help="Delete this criterion"):
                st.session_state[f'delete_{i}'] = True
    
    # Remove deleted criteria
    edited_criteria = [
        criterion for i, criterion in enumerate(edited_criteria)
        if not st.session_state.get(f'delete_{i}', False)
    ]
    
    # Add new criterion
    st.subheader("Add New Criterion")
    new_criterion = st.text_input("Enter new criterion:")
    
    if st.button("➕ Add Criterion") and new_criterion.strip():
        edited_criteria.append(new_criterion.strip())
        st.success("Criterion added!")
    
    # Save changes
    if st.button("💾 Save Changes", type="primary"):
        try:
            with open("criteria.json", "w") as f:
                json.dump(edited_criteria, f, indent=2)
            st.success("✅ Criteria saved successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error saving criteria: {str(e)}")

def results_analysis_page():
    st.header("Results Analysis")
    
    if 'screening_results' not in st.session_state:
        st.info("No screening results available. Please run document screening first.")
        return
    
    results = st.session_state['screening_results']
    
    # File selection
    selected_file = st.selectbox(
        "Select file to analyze:",
        list(results.keys())
    )
    
    if selected_file:
        result = results[selected_file]
        
        # Overview metrics
        st.subheader("📊 Overview")
        
        formatted_results = format_verdict_results(result['verdicts'])
        df = create_results_dataframe(formatted_results)
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_criteria = len(formatted_results)
        passed_criteria = len(df[df['Overall Verdict'] == 'PASS'])
        pass_rate = (passed_criteria / total_criteria * 100) if total_criteria > 0 else 0
        
        col1.metric("Total Criteria", total_criteria)
        col2.metric("Passed Criteria", passed_criteria)
        col3.metric("Overall Pass Rate", f"{pass_rate:.1f}%")
        col4.metric("Total Chunks", len(result['verdicts']))
        
        # Detailed results table
        st.subheader("📋 Detailed Results")
        st.dataframe(df, use_container_width=True)
        
        # Download options
        st.subheader("📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as JSON
            results_json = json.dumps(result['verdicts'], indent=2)
            st.download_button(
                "📄 Download Full Results (JSON)",
                data=results_json,
                file_name=f"screening_results_{selected_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Download summary as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                "📊 Download Summary (CSV)",
                data=csv,
                file_name=f"screening_summary_{selected_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
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
                    
                    # Download option
                    st.download_button(
                        "📥 Download Blurb",
                        data=blurb,
                        file_name=f"marketing_blurb_{selected_file}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error generating blurb: {str(e)}")

if __name__ == "__main__":
    main() 