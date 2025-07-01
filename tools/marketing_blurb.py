"""
Marketing Blurb Generator Tool

Creates marketing content from document analysis results.
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Any

from .base_tool import BaseTool
from utils.llm_util import llm_blurb


class MarketingBlurbGenerator(BaseTool):
    """Tool for generating marketing blurbs from document analysis"""
    
    def __init__(self):
        super().__init__()
        self.name = "Marketing Blurb Generator"
        self.description = "Create marketing content from document analysis results"
        self.icon = "🎯"
    
    def show_configuration_ui(self) -> Dict[str, Any]:
        """Show configuration UI for marketing blurb generation"""
        st.markdown("### 🎯 Configure Marketing Blurb Generator")
        st.markdown("Generate compelling marketing content based on your documents.")
        
        # Check if we have any screening results to work with
        use_existing = False
        if 'screening_results' in st.session_state and st.session_state.screening_results:
            st.success("✅ Found previous screening results that can be used for blurb generation")
            use_existing = st.checkbox(
                "Use existing screening results from Document Screener", 
                value=True,
                help="This will use the criteria analysis you already ran to generate more targeted marketing content"
            )
            
            if use_existing:
                # Show preview of what will be used
                screening_data = st.session_state.screening_results
                criteria_count = len(screening_data.get('criteria', []))
                files_count = len(screening_data.get('individual_results', {}))
                
                with st.expander("📋 Preview of Screening Data to Use", expanded=False):
                    st.write(f"• **Criteria analyzed:** {criteria_count}")
                    st.write(f"• **Files analyzed:** {files_count}")
                    st.write(f"• **Analysis timestamp:** {screening_data.get('timestamp', 'Unknown')}")
                    
                    if screening_data.get('consolidated_results'):
                        passed_criteria = sum(
                            1 for result in screening_data['consolidated_results'].values() 
                            if result.get('verdict') == 'yes'
                        )
                        st.write(f"• **Criteria passed:** {passed_criteria}/{criteria_count}")
        else:
            st.info("💡 This tool works best with screening results from the Document Screener. You can also analyze documents directly.")
        
        # Blurb style options
        blurb_style = st.selectbox(
            "Marketing Style",
            [
                "Professional & Trustworthy",
                "Dynamic & Exciting", 
                "Technical & Detailed",
                "Casual & Approachable"
            ],
            help="The tone and style of your marketing content"
        )
        
        # Target audience
        target_audience = st.selectbox(
            "Target Audience",
            [
                "General Public",
                "Business Professionals",
                "Technical Experts",
                "Investors"
            ],
            help="Who is your primary audience?"
        )
        
        # Content focus
        focus_areas = st.multiselect(
            "Key Focus Areas",
            [
                "Key Benefits",
                "Unique Features", 
                "Competitive Advantages",
                "Technical Specifications"
            ],
            default=["Key Benefits", "Unique Features"],
            help="Select the main points to highlight"
        )
        
        temperature = st.slider(
            "Creativity Level", 
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Higher values make content more creative"
        )
        
        if not focus_areas:
            st.warning("Please select at least one focus area")
            return None
        
        return {
            'use_existing_results': use_existing,
            'blurb_style': blurb_style,
            'target_audience': target_audience,
            'focus_areas': focus_areas,
            'temperature': temperature
        }
    
    def run_analysis(self, uploaded_files: List, config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate marketing blurb"""
        
        progress_bar = st.progress(0, "Starting blurb generation...")
        
        if config['use_existing_results'] and 'screening_results' in st.session_state:
            # Use existing screening results
            progress_bar.progress(30, "Using existing screening results...")
            screening_data = st.session_state.screening_results
            
            # Convert screening results to verdict format for blurb generation
            verdict_data = screening_data.get('individual_results', {})
        else:
            # Analyze documents first (simplified screening)
            progress_bar.progress(20, "Analyzing documents...")
            verdict_data = self._analyze_documents_for_blurb(uploaded_files, progress_bar)
        
        progress_bar.progress(70, "Generating marketing content...")
        
        try:
            blurb_text = llm_blurb(verdict_data, temperature=config.get('temperature', 0.6))
            
            progress_bar.progress(100, "Marketing blurb complete!")
            
            return {
                'blurb_text': blurb_text,
                'config': config,
                'timestamp': datetime.now().isoformat(),
                'filenames': [f.name for f in uploaded_files]
            }
            
        except Exception as e:
            return {'error': str(e), 'config': config}
    
    def _analyze_documents_for_blurb(self, uploaded_files, progress_bar):
        """Quick analysis of documents for blurb generation"""
        from utils.llm_util import llm_screen
        import textwrap
        
        basic_criteria = [
            "Key benefits or advantages mentioned",
            "Unique features or capabilities", 
            "Performance metrics or specifications"
        ]
        
        all_verdicts = {}
        
        for file_idx, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress(
                30 + int((file_idx / len(uploaded_files)) * 30),
                f"Analyzing {uploaded_file.name}"
            )
            
            text = self._extract_text_from_file(uploaded_file)
            chunks = textwrap.wrap(text, 10000)
            
            file_verdicts = {}
            for chunk_idx, chunk in enumerate(chunks[:2]):  # First 2 chunks only
                try:
                    chunk_result = llm_screen(chunk, basic_criteria, temperature=0.1)
                    file_verdicts[f"chunk_{chunk_idx + 1}"] = chunk_result
                except Exception as e:
                    file_verdicts[f"chunk_{chunk_idx + 1}"] = {
                        criterion: {"verdict": "unknown", "reason": f"Error: {str(e)}"}
                        for criterion in basic_criteria
                    }
            
            all_verdicts[uploaded_file.name] = file_verdicts
        
        return all_verdicts
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display marketing blurb results"""
        if 'error' in results:
            st.error(f"Blurb generation failed: {results['error']}")
            return
        
        blurb_text = results.get('blurb_text', '')
        filenames = results.get('filenames', [])
        
        st.markdown("### 🎯 Your Marketing Blurb")
        
        if filenames:
            st.markdown(f"*Generated from: {', '.join(filenames)}*")
        
        if blurb_text:
            st.markdown(f"""
            <div style="
                background-color: #f8f9fa;
                color: #212529;
                border-left: 4px solid #007bff;
                padding: 20px;
                margin: 20px 0;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.6;
            ">
                {blurb_text.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
            
            word_count = len(blurb_text.split())
            st.caption(f"Word count: {word_count}")
            
            # Download option
            st.download_button(
                label="💾 Download as Text",
                data=blurb_text,
                file_name=f"marketing_blurb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.warning("No marketing content was generated.") 