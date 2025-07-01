#!/usr/bin/env python
"""
Document Analysis Tool - Modular Interface

A clean, intuitive tool for document analysis with progressive disclosure:
1. Upload your documents
2. Choose what you want to do with them
3. Configure the specific tool
4. View results
"""

import streamlit as st
import json
import tempfile
import os
from datetime import datetime
from typing import List, Dict, Any

# Import our existing utility functions
from utils.llm_util import extract_text, llm_screen, llm_blurb, llm_summarize

# Tool modules (we'll create these)
from tools.document_screener import DocumentScreener
from tools.document_summarizer import DocumentSummarizer
from tools.marketing_blurb import MarketingBlurbGenerator
from tools.content_analyzer import ContentAnalyzer

# Page configuration
st.set_page_config(
    page_title="Document Analysis Tool",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .tool-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .tool-card:hover {
        border-color: #ff6b6b;
        box-shadow: 0 4px 12px rgba(255,107,107,0.2);
        transform: translateY(-2px);
    }
    .tool-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }

    .step-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Available tools configuration
AVAILABLE_TOOLS = {
    "screener": {
        "name": "Document Screener",
        "icon": "📋",
        "description": "Screen documents against specific criteria with yes/no verdicts",
        "class": DocumentScreener
    },
    "summarizer": {
        "name": "Document Summarizer", 
        "icon": "📄",
        "description": "Generate focused summaries based on your areas of interest",
        "class": DocumentSummarizer
    },
    "blurb": {
        "name": "Marketing Blurb Generator",
        "icon": "🎯", 
        "description": "Create marketing content from document analysis results",
        "class": MarketingBlurbGenerator
    },
    "analyzer": {
        "name": "Content Analyzer",
        "icon": "🔍",
        "description": "Deep analysis of document content and structure",
        "class": ContentAnalyzer
    }
}

def initialize_session_state():
    """Initialize session state variables"""
    if 'stage' not in st.session_state:
        st.session_state.stage = 'upload'  # upload, tool_selection, configure, results
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'selected_tool' not in st.session_state:
        st.session_state.selected_tool = None
    if 'tool_config' not in st.session_state:
        st.session_state.tool_config = {}
    if 'results' not in st.session_state:
        st.session_state.results = {}

def show_upload_stage():
    """Stage 1: Document Upload"""
    st.markdown('<div class="step-header">📁 Upload Your Documents</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 📄 Ready to analyze your documents?")
        st.markdown("Upload PDFs, Word docs, Excel files, or CSV files to get started")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'doc', 'xlsx', 'xls', 'csv'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            
            # Show uploaded files
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            
            with st.expander("📋 Uploaded Files", expanded=True):
                for file in uploaded_files:
                    file_size = len(file.getbuffer()) / 1024  # KB
                    st.write(f"• **{file.name}** ({file_size:.1f} KB)")
            
            # Next button
            if st.button("➡️ Choose What To Do With These Documents", type="primary", use_container_width=True):
                st.session_state.stage = 'tool_selection'
                st.rerun()

def show_tool_selection_stage():
    """Stage 2: Tool Selection"""
    st.markdown('<div class="step-header">🛠️ What would you like to do with your documents?</div>', unsafe_allow_html=True)
    
    # Show uploaded files summary
    with st.expander("📁 Your Documents", expanded=False):
        for file in st.session_state.uploaded_files:
            st.write(f"• {file.name}")
    
    st.markdown("### Choose an analysis tool:")
    
    # Create tool cards in a grid
    cols = st.columns(2)
    
    for i, (tool_id, tool_info) in enumerate(AVAILABLE_TOOLS.items()):
        with cols[i % 2]:
            if st.button(
                f"{tool_info['icon']} {tool_info['name']}\n\n{tool_info['description']}", 
                key=f"tool_{tool_id}",
                use_container_width=True,
                help=f"Click to configure {tool_info['name']}"
            ):
                st.session_state.selected_tool = tool_id
                st.session_state.stage = 'configure'
                st.rerun()
    
    # Back button
    if st.button("⬅️ Upload Different Documents", use_container_width=True):
        st.session_state.stage = 'upload'
        st.session_state.uploaded_files = []
        st.rerun()

def show_configure_stage():
    """Stage 3: Tool Configuration"""
    tool_info = AVAILABLE_TOOLS[st.session_state.selected_tool]
    
    st.markdown(f'<div class="step-header">{tool_info["icon"]} Configure {tool_info["name"]}</div>', unsafe_allow_html=True)
    
    # Show context
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Tool:** {tool_info['name']}")
        st.write(f"**Files:** {len(st.session_state.uploaded_files)} document(s)")
    with col2:
        if st.button("🔄 Choose Different Tool"):
            st.session_state.stage = 'tool_selection'
            st.rerun()
    
    # Initialize the tool and show its configuration interface
    tool_instance = tool_info['class']()
    config = tool_instance.show_configuration_ui()
    
    if config is not None:  # User completed configuration
        st.session_state.tool_config = config
        
        # Run analysis button
        if st.button(f"🚀 Run {tool_info['name']}", type="primary", use_container_width=True):
            with st.spinner(f"Running {tool_info['name']}..."):
                try:
                    results = tool_instance.run_analysis(
                        st.session_state.uploaded_files, 
                        config
                    )
                    st.session_state.results = results
                    st.session_state.stage = 'results'
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

def show_results_stage():
    """Stage 4: Results Display"""
    tool_info = AVAILABLE_TOOLS[st.session_state.selected_tool]
    
    st.markdown(f'<div class="step-header">✅ {tool_info["name"]} Results</div>', unsafe_allow_html=True)
    
    # Action buttons at the top
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Run Another Analysis", type="primary"):
            st.session_state.stage = 'tool_selection'
            st.rerun()
    with col2:
        if st.button("📁 Upload New Documents"):
            st.session_state.stage = 'upload'
            st.session_state.uploaded_files = []
            st.rerun()
    with col3:
        if st.button("⚙️ Adjust Configuration"):
            st.session_state.stage = 'configure'
            st.rerun()
    
    # Show results using the tool's display method
    tool_instance = tool_info['class']()
    tool_instance.display_results(st.session_state.results)

def main():
    """Main application flow"""
    initialize_session_state()
    
    # App header
    st.title("📄 Document Analysis Tool")
    st.markdown("*A simple, powerful way to analyze your documents*")
    
    # Progress indicator
    stages = ['Upload', 'Select Tool', 'Configure', 'Results']
    current_stage_idx = ['upload', 'tool_selection', 'configure', 'results'].index(st.session_state.stage)
    
    progress_cols = st.columns(4)
    for i, stage_name in enumerate(stages):
        with progress_cols[i]:
            if i <= current_stage_idx:
                st.markdown(f"**✅ {stage_name}**")
            else:
                st.markdown(f"⭕ {stage_name}")
    
    st.markdown("---")
    
    # Route to appropriate stage
    if st.session_state.stage == 'upload':
        show_upload_stage()
    elif st.session_state.stage == 'tool_selection':
        show_tool_selection_stage()
    elif st.session_state.stage == 'configure':
        show_configure_stage()
    elif st.session_state.stage == 'results':
        show_results_stage()

if __name__ == "__main__":
    main() 