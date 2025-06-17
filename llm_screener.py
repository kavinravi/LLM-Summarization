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
from llm_util import extract_text, llm_screen, llm_blurb
import textwrap

# Page configuration
st.set_page_config(
    page_title="Project Document Screener",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
CHUNK_SIZE_CHARS = 150000  # Gemini 2.5 flash supports much larger context

# Cache directory for persistent storage
CACHE_DIR = "screening_cache"

def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def save_results_to_cache(results, criteria):
    """Save screening results to disk cache."""
    ensure_cache_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cache_data = {
        'results': results,
        'criteria': criteria,
        'timestamp': datetime.now().isoformat(),
        'files': list(results.keys())
    }
    
    cache_file = os.path.join(CACHE_DIR, f"screening_{timestamp}.json")
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)
    
    return cache_file

def load_cached_results():
    """Load all cached results from disk."""
    ensure_cache_dir()
    cached_sessions = []
    
    for filename in os.listdir(CACHE_DIR):
        if filename.startswith("screening_") and filename.endswith(".json"):
            filepath = os.path.join(CACHE_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    cached_sessions.append({
                        'filename': filename,
                        'filepath': filepath,
                        'timestamp': data.get('timestamp', ''),
                        'files': data.get('files', []),
                        'data': data
                    })
            except Exception as e:
                st.error(f"Error loading cached file {filename}: {e}")
    
    # Sort by timestamp (newest first)
    cached_sessions.sort(key=lambda x: x['timestamp'], reverse=True)
    return cached_sessions

def load_session_from_cache(cache_data):
    """Load a specific cached session into current session state."""
    st.session_state['screening_results'] = cache_data['results']
    st.session_state['used_criteria'] = cache_data['criteria']

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
        
        # Screen each chunk with error handling
        verdicts = {}
        for i, chunk in enumerate(chunks):
            chunk_name = f"chunk_{i+1}"
            try:
                chunk_result = llm_screen(chunk, criteria)
                
                # Display debug info if available
                if '_debug' in chunk_result:
                    with st.expander(f"🔍 Debug Info for {chunk_name}", expanded=False):
                        for debug_msg in chunk_result['_debug']:
                            st.text(debug_msg)
                    # Remove debug from actual results
                    del chunk_result['_debug']
                
                verdicts[chunk_name] = chunk_result
                
                if progress_bar:
                    progress = 60 + (30 * (i + 1) / len(chunks))
                    progress_bar.progress(int(progress), f"Processing chunk {i+1} of {len(chunks)}...")
                    
                # Save progress every 5 chunks to prevent loss
                if (i + 1) % 5 == 0:
                    st.session_state[f'temp_verdicts_{uploaded_file.name}'] = verdicts
                    
            except Exception as e:
                st.error(f"Error processing chunk {i+1}: {str(e)}")
                # Continue with remaining chunks
                verdicts[chunk_name] = {criterion: {"verdict": "unknown", "reason": f"Processing error: {str(e)}"} for criterion in criteria}
        
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
    
    # Debug: Log what chunks we're processing
    print(f"DEBUG: format_verdict_results processing {len(verdicts)} chunks:")
    for chunk_name in verdicts.keys():
        chunk_data = verdicts[chunk_name]
        print(f"  - {chunk_name}: {len(chunk_data)} criteria")
    
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
        print(f"DEBUG: Processing {chunk_name} with {len(chunk_results)} results")
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
            
            print(f"DEBUG:   {criterion}: {verdict}")
            if verdict == 'yes':
                all_results[criterion]['yes_count'] += 1
            elif verdict == 'no':
                all_results[criterion]['no_count'] += 1
            else:
                all_results[criterion]['no_count'] += 1
            
            all_results[criterion]['reasons'].append(f"{chunk_name}: {reason}")
    
    # Debug: Show final counts
    print("DEBUG: Final aggregated results:")
    for criterion, result in all_results.items():
        total = result['yes_count'] + result['no_count']
        print(f"  {criterion}: {result['yes_count']} yes, {result['no_count']} no (total: {total})")
    
    return all_results

def create_results_dataframe(formatted_results):
    """Create a pandas DataFrame from formatted results"""
    data = []
    for criterion, result in formatted_results.items():
        total_chunks = result['yes_count'] + result['no_count']
        pass_rate = (result['yes_count'] / total_chunks * 100) if total_chunks > 0 else 0
        
        overall_verdict = "PASS" if result['yes_count'] > 0 else "FAIL"
        
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
    st.title("Project Document Screener")
    st.markdown("Upload documents and screen them against custom criteria using AI analysis.")
    
    # Load cached results on startup
    if 'cached_sessions' not in st.session_state:
        st.session_state['cached_sessions'] = load_cached_results()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📄 Document Screening", 
        "📊 Results Analysis",
        "✍️ Marketing Blurb Generator",
        "💾 Cached Results"
    ])
    
    if page == "📄 Document Screening":
        document_screening_page()
    elif page == "📊 Results Analysis":
        results_analysis_page()
    elif page == "✍️ Marketing Blurb Generator":
        marketing_blurb_page()
    elif page == "💾 Cached Results":
        cached_results_page()

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
    st.subheader("📋 Screening Criteria")
    st.markdown("---")  # Visual separator
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Load default criteria
        default_criteria = load_default_criteria()
        
        # Format criteria with numbers for better readability
        formatted_criteria = []
        for i, criterion in enumerate(default_criteria, 1):
            formatted_criteria.append(f"{i}. {criterion}")
        
        # Use auto-formatted content if available, otherwise use default
        display_value = st.session_state.get('auto_format_criteria', "\n\n".join(formatted_criteria))
        
        # Allow users to modify criteria with auto-numbering
        criteria_text = st.text_area(
            "Edit criteria (one per line):",
            value=display_value,  # Use either auto-formatted or default
            height=400,  # Increased height to accommodate better formatting
            help="Each line represents one screening criterion. Add new criteria on new lines, then click 'Auto-Number' to format!",
            key="criteria_text_area"
        )
        
        # Clear the auto-format state after using it
        if 'auto_format_criteria' in st.session_state:
            del st.session_state['auto_format_criteria']
        
        # Parse criteria from text area and auto-renumber
        import re
        raw_lines = [line.strip() for line in criteria_text.split('\n') if line.strip()]
        criteria = []
        
        # Extract actual criteria content (remove any existing numbering)
        for line in raw_lines:
            cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line)
            if cleaned_line:
                criteria.append(cleaned_line)
        
        # Show auto-numbered preview of what they're editing
        if criteria:
            st.markdown("**✨ Auto-numbered preview:**")
            preview_container = st.container()
            with preview_container:
                cols = st.columns(2)
                for i, criterion in enumerate(criteria):
                    col_idx = i % 2
                    with cols[col_idx]:
                        st.markdown(f"**{i+1}.** {criterion[:80]}{'...' if len(criterion) > 80 else ''}")
            st.markdown("---")
    
    with col2:
        st.markdown("**Criteria Management**")
        
        # Auto-format button
        if st.button("🔢 Auto-Number", help="Automatically format and number all criteria"):
            if criteria:
                # Re-format with proper numbering and spacing
                formatted_criteria_new = []
                for i, criterion in enumerate(criteria, 1):
                    formatted_criteria_new.append(f"{i}. {criterion}")
                # Update session state to trigger re-render
                st.session_state['auto_format_criteria'] = "\n\n".join(formatted_criteria_new)
                st.success("✨ Criteria auto-numbered!")
                st.rerun()
        
        if st.button("📄 Load Default"):
            st.rerun()
        
        if st.button("💾 Save Custom"):
            with open("custom_criteria.json", "w", encoding="utf-8") as f:
                json.dump(criteria, f, indent=2)
            st.success("Custom criteria saved!")
        
        if st.button("📂 Load Custom"):
            try:
                with open("custom_criteria.json", "r", encoding="utf-8") as f:
                    custom_criteria = json.load(f)
                st.session_state['custom_criteria'] = custom_criteria
                st.success("Custom criteria loaded!")
                st.rerun()
            except FileNotFoundError:
                st.error("No custom criteria file found!")
    
    # Check if custom criteria should be loaded
    if 'custom_criteria' in st.session_state:
        criteria_text = st.text_area(
            "Edit criteria (one per line):",
            value="\n".join(st.session_state['custom_criteria']),
            height=300,
            key="custom_criteria_area"
        )
        criteria = [line.strip() for line in criteria_text.split('\n') if line.strip()]
        del st.session_state['custom_criteria']
    
    # Display number of criteria and preview
    st.info(f"📋 {len(criteria)} criteria loaded")
    
    # Show a nice preview of the criteria
    if criteria:
        with st.expander("👀 Preview Current Criteria", expanded=False):
            for i, criterion in enumerate(criteria, 1):
                st.markdown(f"**{i}.** {criterion}")
            st.caption("These are the criteria that will be used for screening.")
    
    # Screening section
    if uploaded_files and criteria:
        st.subheader("Run Screening")
        
        if st.button("🔍 Start Screening Process", type="primary"):
            st.session_state['screening_results'] = {}
            st.session_state['used_criteria'] = criteria
            
            # Progress tracking
            progress_container = st.container()
            
            with progress_container:
                st.write("Processing documents...")
                overall_progress = st.progress(0)
                status_text = st.empty()
                
                for file_idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name} ({file_idx + 1}/{len(uploaded_files)})")
                    
                    # Individual file progress
                    file_progress = st.progress(0)
                    
                    try:
                        verdicts, raw_text = process_document(uploaded_file, criteria, file_progress)
                        
                        st.session_state['screening_results'][uploaded_file.name] = {
                            'verdicts': verdicts,
                            'raw_text': raw_text,
                            'criteria': criteria
                        }
                        
                        st.success(f"✅ Completed: {uploaded_file.name}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
                    # Update overall progress
                    overall_progress.progress((file_idx + 1) / len(uploaded_files))
                
                status_text.text("Screening complete!")
                
                # Save results to cache
                if st.session_state.get('screening_results'):
                    cache_file = save_results_to_cache(
                        st.session_state['screening_results'],
                        st.session_state['used_criteria']
                    )
                    st.info(f"💾 Results cached to: {os.path.basename(cache_file)}")
                    
                    # Refresh cached sessions
                    st.session_state['cached_sessions'] = load_cached_results()
        
        # Display results if available
        if st.session_state.get('screening_results'):
            st.subheader("📋 Screening Results Summary")
            
            for filename, file_data in st.session_state['screening_results'].items():
                with st.expander(f"📄 {filename}", expanded=True):
                    verdicts = file_data['verdicts']
                    
                    # Format and display results
                    formatted_results = format_verdict_results(verdicts)
                    df = create_results_dataframe(formatted_results)
                    
                    # Color-code the dataframe
                    def highlight_verdict(val):
                        if val == 'PASS':
                            return 'background-color: #d4edda; color: #155724'
                        elif val == 'FAIL':
                            return 'background-color: #f8d7da; color: #721c24'
                        else:
                            return 'background-color: #fff3cd; color: #856404'
                    
                    styled_df = df.style.map(highlight_verdict, subset=['Overall Verdict'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Download button for individual file results
                    json_str = json.dumps(file_data, indent=2)
                    st.download_button(
                        label=f"📥 Download {filename} Results (JSON)",
                        data=json_str,
                        file_name=f"{filename}_screening_results.json",
                        mime="application/json"
                    )

def results_analysis_page():
    st.header("📊 Results Analysis")
    
    if not st.session_state.get('screening_results'):
        st.warning("⚠️ No screening results available. Please run document screening first.")
        return
    
    st.subheader("Detailed Analysis")
    
    # File selector
    filenames = list(st.session_state['screening_results'].keys())
    selected_file = st.selectbox("Select file to analyze:", filenames)
    
    if selected_file:
        file_data = st.session_state['screening_results'][selected_file]
        
        # Detailed results view
        st.subheader(f"Analysis for: {selected_file}")
        
        formatted_results = format_verdict_results(file_data['verdicts'])
        
        # Summary statistics
        col1, col2, col3 = st.columns(3)
        
        total_criteria = len(formatted_results)
        passed_criteria = sum(1 for r in formatted_results.values() if r['yes_count'] > 0)
        failed_criteria = total_criteria - passed_criteria
        
        with col1:
            st.metric("Total Criteria", total_criteria)
        with col2:
            st.metric("Passed", passed_criteria, delta=f"{passed_criteria/total_criteria*100:.1f}%")
        with col3:
            st.metric("Failed", failed_criteria, delta=f"-{failed_criteria/total_criteria*100:.1f}%")
        
        # Detailed breakdown
        for criterion, result in formatted_results.items():
            with st.expander(f"📋 {criterion}"):
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    total_chunks = result['yes_count'] + result['no_count']
                    pass_rate = (result['yes_count'] / total_chunks * 100) if total_chunks > 0 else 0
                    
                    st.metric("Pass Rate", f"{pass_rate:.1f}%")
                    st.write(f"✅ Passes: {result['yes_count']}")
                    st.write(f"❌ Fails: {result['no_count']}")
                
                with col2:
                    st.write("**Detailed Reasons:**")
                    for reason in result['reasons']:
                        st.write(f"• {reason}")

def marketing_blurb_page():
    st.header("✍️ Marketing Blurb Generator")
    
    if not st.session_state.get('screening_results'):
        st.warning("⚠️ No screening results available. Please run document screening first.")
        return
    
    # File selector
    filenames = list(st.session_state['screening_results'].keys())
    selected_file = st.selectbox("Select file for blurb generation:", filenames)
    
    if selected_file:
        file_data = st.session_state['screening_results'][selected_file]
        
        # Consolidate results for blurb generation
        consolidated = consolidate_chunk_results(file_data['verdicts'], file_data['criteria'])
        
        st.subheader("Consolidated Results")
        
        # Show consolidated results
        for criterion, result in consolidated.items():
            status_emoji = "✅" if result['verdict'] == 'PASS' else "❌" if result['verdict'] == 'FAIL' else "❓"
            st.write(f"{status_emoji} **{criterion}**: {result['verdict']}")
            st.write(f"   *{result['reason']}*")
        
        # Generate blurb
        if st.button("📝 Generate Marketing Blurb", type="primary"):
            try:
                with st.spinner("Generating marketing blurb..."):
                    blurb = llm_blurb(consolidated)
                
                st.subheader("Generated Marketing Blurb")
                st.write(blurb)
                
                # Copy button
                st.code(blurb, language=None)
                
            except Exception as e:
                st.error(f"Error generating blurb: {str(e)}")

def cached_results_page():
    st.header("💾 Cached Results")
    
    cached_sessions = st.session_state.get('cached_sessions', [])
    
    if not cached_sessions:
        st.info("No cached results found.")
        return
    
    st.write(f"Found {len(cached_sessions)} cached screening sessions:")
    
    for session in cached_sessions:
        with st.expander(f"📅 {session['timestamp']} - {len(session['files'])} files"):
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write("**Files processed:**")
                for filename in session['files']:
                    st.write(f"• {filename}")
                
                st.write(f"**Timestamp:** {session['timestamp']}")
            
            with col2:
                if st.button("📂 Load Session", key=f"load_{session['filename']}"):
                    load_session_from_cache(session['data'])
                    st.success("Session loaded!")
                    st.rerun()
                
                # Download button
                json_str = json.dumps(session['data'], indent=2)
                st.download_button(
                    label="💾 Download",
                    data=json_str,
                    file_name=session['filename'],
                    mime="application/json",
                    key=f"download_{session['filename']}"
                )

if __name__ == "__main__":
    main() 