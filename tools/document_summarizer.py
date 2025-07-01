"""
Document Summarizer Tool

Generates focused summaries based on areas of interest.
"""

import streamlit as st
import json
import textwrap
from datetime import datetime
from typing import Dict, List, Any

from .base_tool import BaseTool
from utils.llm_util import llm_summarize


class DocumentSummarizer(BaseTool):
    """Tool for generating focused document summaries"""
    
    def __init__(self):
        super().__init__()
        self.name = "Document Summarizer"
        self.description = "Generate focused summaries based on your areas of interest"
        self.icon = "📄"
        self.chunk_size_chars = 15000
    
    def show_configuration_ui(self) -> Dict[str, Any]:
        """Show configuration UI for document summarization"""
        st.markdown("### 📄 Configure Document Summarizer")
        st.markdown("Set up focus areas to guide the summarization of your documents.")
        
        # Summary type selection
        summary_type = st.radio(
            "What type of summary do you want?",
            ["General summary", "Focused summary", "Comparative analysis"],
            help="Choose the type of summary to generate"
        )
        
        focus_areas = []
        
        if summary_type == "General summary":
            st.info("📋 This will create a general overview of your documents without specific focus areas.")
            focus_areas = ["key findings", "main topics", "important conclusions"]
            
        elif summary_type == "Focused summary":
            st.markdown("**Enter your focus areas (one per line):**")
            focus_text = st.text_area(
                "Focus Areas",
                height=150,
                placeholder="Example:\nFinancial performance\nRisk factors\nMarket opportunities\nTechnical specifications",
                help="Each line will be treated as a separate focus area for the summary"
            )
            
            if focus_text.strip():
                focus_areas = [line.strip() for line in focus_text.split('\n') if line.strip()]
                st.info(f"📝 {len(focus_areas)} focus areas entered")
            else:
                st.warning("Please enter at least one focus area")
                return None
                
        elif summary_type == "Comparative analysis":
            st.markdown("**This will compare and contrast information across your documents.**")
            focus_areas = ["similarities", "differences", "trends", "contradictions", "key themes"]
            st.info("📊 Will analyze: " + ", ".join(focus_areas))
        
        # Advanced options
        with st.expander("⚙️ Advanced Options", expanded=False):
            temperature = st.slider(
                "Creativity Level",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1,
                help="Higher values make summaries more creative, lower values more factual"
            )
            
            summary_length = st.selectbox(
                "Summary Length",
                ["Brief", "Standard", "Detailed"],
                index=1,
                help="How detailed should the summary be?"
            )
            
            include_quotes = st.checkbox(
                "Include Direct Quotes",
                value=True,
                help="Include relevant quotes from the documents in the summary"
            )
        
        return {
            'summary_type': summary_type,
            'focus_areas': focus_areas,
            'temperature': temperature,
            'summary_length': summary_length,
            'include_quotes': include_quotes
        }
    
    def run_analysis(self, uploaded_files: List, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run document summarization analysis"""
        focus_areas = config['focus_areas']
        temperature = config.get('temperature', 0.3)
        
        # Create progress bar
        progress_bar = st.progress(0, "Starting summarization...")
        
        # Extract text from all files
        all_texts = []
        for file_idx, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress(
                int((file_idx / len(uploaded_files)) * 50),
                f"Processing {uploaded_file.name}"
            )
            
            text = self._extract_text_from_file(uploaded_file)
            all_texts.append({
                'filename': uploaded_file.name,
                'text': text
            })
        
        # Combine all text if multiple files
        if len(uploaded_files) > 1:
            combined_text = "\n\n--- DOCUMENT SEPARATOR ---\n\n".join([
                f"DOCUMENT: {doc['filename']}\n{doc['text']}" 
                for doc in all_texts
            ])
        else:
            combined_text = all_texts[0]['text']
        
        progress_bar.progress(60, "Generating summary...")
        
        # Split into chunks and summarize
        chunks = textwrap.wrap(combined_text, self.chunk_size_chars)
        chunk_summaries = []
        
        for chunk_idx, chunk in enumerate(chunks):
            try:
                summary = llm_summarize(
                    chunk,
                    focus_areas=focus_areas,
                    temperature=temperature
                )
                chunk_summaries.append(summary)
            except Exception as e:
                st.error(f"Error summarizing chunk {chunk_idx + 1}: {str(e)}")
                chunk_summaries.append({
                    'error': str(e),
                    'focus_area_summaries': {}
                })
        
        progress_bar.progress(90, "Consolidating results...")
        
        # Consolidate summaries
        consolidated = self._consolidate_summaries(chunk_summaries, focus_areas)
        
        progress_bar.progress(100, "Summary complete!")
        
        return {
            'type': 'multi_document' if len(uploaded_files) > 1 else 'single_document',
            'filenames': [f.name for f in uploaded_files],
            'chunk_summaries': chunk_summaries,
            'consolidated_summary': consolidated,
            'focus_areas': focus_areas,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
    
    def _consolidate_summaries(self, chunk_summaries, focus_areas):
        """Consolidate multiple chunk summaries into a final summary"""
        consolidated = {}
        
        for focus_area in focus_areas:
            all_findings = []
            all_citations = []
            
            # Collect all findings for this focus area
            for chunk_summary in chunk_summaries:
                if 'error' not in chunk_summary:
                    # Look for focus_area_insights (the actual key used)
                    area_insights = chunk_summary.get('focus_area_insights', {}).get(focus_area, {})
                    
                    # Get findings text
                    findings = area_insights.get('findings', '')
                    if findings and len(findings.strip()) > 20:
                        all_findings.append(findings.strip())
                    
                    # Get citations
                    citations = area_insights.get('citations', [])
                    if citations:
                        all_citations.extend(citations)
            
            # Create consolidated summary
            if all_findings:
                # Combine findings and remove duplicates
                unique_findings = []
                seen = set()
                for finding in all_findings:
                    finding_lower = finding.lower()
                    if finding_lower not in seen:
                        unique_findings.append(finding)
                        seen.add(finding_lower)
                
                # Create summary from combined findings
                combined_summary = " ".join(unique_findings)
                
                # Extract key points from findings
                key_points = []
                for finding in unique_findings[:3]:  # Top 3 findings
                    # Split by sentences and take first substantial one
                    sentences = finding.split('. ')
                    if sentences:
                        key_points.append(sentences[0] + ('.' if not sentences[0].endswith('.') else ''))
                
                consolidated[focus_area] = {
                    'key_points': key_points,
                    'summary': combined_summary,
                    'citations': list(set(all_citations))[:5]  # Top 5 unique citations
                }
            else:
                consolidated[focus_area] = {
                    'key_points': [],
                    'summary': f"No specific information found for {focus_area}",
                    'citations': []
                }
        
        return consolidated
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display summarization results"""
        if 'error' in results:
            st.error(f"Summarization failed: {results['error']}")
            return
        
        consolidated = results['consolidated_summary']
        focus_areas = results['focus_areas']
        filenames = results.get('filenames', [])
        
        # Header
        if len(filenames) == 1:
            st.markdown(f"### 📄 Summary of {filenames[0]}")
        else:
            st.markdown(f"### 📚 Summary of {len(filenames)} Documents")
            with st.expander("📁 Documents Analyzed", expanded=False):
                for filename in filenames:
                    st.write(f"• {filename}")
        
        # Summary by focus area
        st.markdown("### 📋 Summary by Focus Area")
        
        for focus_area in focus_areas:
            area_data = consolidated.get(focus_area, {})
            summary_text = area_data.get('summary', '')
            key_points = area_data.get('key_points', [])
            citations = area_data.get('citations', [])
            
            with st.expander(f"🎯 {focus_area.title()}", expanded=True):
                if summary_text and not summary_text.startswith("No specific information"):
                    st.markdown("**Summary:**")
                    st.write(summary_text)
                    
                    if key_points:
                        st.markdown("**Key Points:**")
                        for i, point in enumerate(key_points, 1):
                            st.markdown(f"{i}. {point}")
                    
                    if citations:
                        st.markdown("**📚 Supporting Citations:**")
                        with st.container():
                            for i, citation in enumerate(citations, 1):
                                st.markdown(f"{i}. *\"{citation}\"*")
                else:
                    st.info(f"No specific information found for {focus_area}")
        
        # Export options
        st.markdown("### 💾 Export Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export summary as text
            summary_text = self._format_summary_for_export(consolidated, focus_areas)
            st.download_button(
                label="📄 Download Summary (TXT)",
                data=summary_text,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        with col2:
            # Export full results as JSON
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📋 Download Full Results (JSON)",
                data=json_data,
                file_name=f"summary_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def _format_summary_for_export(self, consolidated, focus_areas):
        """Format the summary for text export"""
        lines = []
        lines.append("DOCUMENT SUMMARY")
        lines.append("=" * 50)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        for focus_area in focus_areas:
            area_data = consolidated.get(focus_area, {})
            summary_text = area_data.get('summary', '')
            key_points = area_data.get('key_points', [])
            citations = area_data.get('citations', [])
            
            lines.append(f"{focus_area.upper()}")
            lines.append("-" * len(focus_area))
            
            if summary_text and not summary_text.startswith("No specific information"):
                lines.append("SUMMARY:")
                lines.append(summary_text)
                lines.append("")
                
                if key_points:
                    lines.append("KEY POINTS:")
                    for i, point in enumerate(key_points, 1):
                        lines.append(f"{i}. {point}")
                    lines.append("")
                
                if citations:
                    lines.append("SUPPORTING CITATIONS:")
                    for i, citation in enumerate(citations, 1):
                        lines.append(f"{i}. \"{citation}\"")
            else:
                lines.append("No specific information found.")
            
            lines.append("")
        
        return "\n".join(lines) 