"""
Content Analyzer Tool

Deep analysis of document content and structure.
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Any
import re

from .base_tool import BaseTool


class ContentAnalyzer(BaseTool):
    """Tool for analyzing document content and structure"""
    
    def __init__(self):
        super().__init__()
        self.name = "Content Analyzer"
        self.description = "Deep analysis of document content and structure"
        self.icon = "🔍"
    
    def show_configuration_ui(self) -> Dict[str, Any]:
        """Show configuration UI for content analysis"""
        st.markdown("### 🔍 Configure Content Analyzer")
        st.markdown("Analyze document structure, readability, and key patterns.")
        
        analysis_types = st.multiselect(
            "Select Analysis Types",
            [
                "Basic Statistics",
                "Readability Analysis",
                "Keyword Extraction",
                "Document Structure",
                "Data Extraction"
            ],
            default=["Basic Statistics", "Keyword Extraction"],
            help="Choose which types of analysis to perform"
        )
        
        if "Keyword Extraction" in analysis_types:
            keyword_count = st.number_input(
                "Number of Keywords to Extract",
                min_value=5,
                max_value=50,
                value=15,
                help="How many top keywords to identify"
            )
        else:
            keyword_count = 15
        
        if not analysis_types:
            st.warning("Please select at least one analysis type")
            return None
        
        return {
            'analysis_types': analysis_types,
            'keyword_count': keyword_count
        }
    
    def run_analysis(self, uploaded_files: List, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run content analysis"""
        
        progress_bar = st.progress(0, "Starting content analysis...")
        
        results = {
            'individual_files': {},
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process each file
        for file_idx, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress(
                int((file_idx / len(uploaded_files)) * 100),
                f"Analyzing {uploaded_file.name}"
            )
            
            text = self._extract_text_from_file(uploaded_file)
            file_analysis = {}
            
            if "Basic Statistics" in config['analysis_types']:
                file_analysis['basic_stats'] = self._analyze_basic_stats(text)
            
            if "Readability Analysis" in config['analysis_types']:
                file_analysis['readability'] = self._analyze_readability(text)
            
            if "Keyword Extraction" in config['analysis_types']:
                file_analysis['keywords'] = self._extract_keywords(text, config['keyword_count'])
            
            if "Document Structure" in config['analysis_types']:
                file_analysis['structure'] = self._analyze_structure(text)
            
            if "Data Extraction" in config['analysis_types']:
                file_analysis['extracted_data'] = self._extract_data_patterns(text)
            
            results['individual_files'][uploaded_file.name] = file_analysis
        
        progress_bar.progress(100, "Content analysis complete!")
        
        return results
    
    def _analyze_basic_stats(self, text):
        """Analyze basic text statistics"""
        words = text.split()
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0
        }
    
    def _analyze_readability(self, text):
        """Analyze text readability (simplified)"""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return {'grade_level': 'Unknown'}
        
        avg_sentence_length = len(words) / len(sentences)
        
        if avg_sentence_length <= 10:
            grade_level = "Easy"
        elif avg_sentence_length <= 15:
            grade_level = "Standard"
        elif avg_sentence_length <= 20:
            grade_level = "Fairly Difficult"
        else:
            grade_level = "Difficult"
        
        return {
            'grade_level': grade_level,
            'avg_sentence_length': round(avg_sentence_length, 1)
        }
    
    def _extract_keywords(self, text, count):
        """Extract top keywords (simplified)"""
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Common stop words to filter out
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Filter and count
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:count]
        
        return [{'word': word, 'frequency': freq} for word, freq in top_keywords]
    
    def _analyze_structure(self, text):
        """Analyze document structure"""
        lines = text.split('\n')
        
        headers = []
        for line in lines:
            line = line.strip()
            if line and (line.isupper() or line.endswith(':') or len(line.split()) <= 8):
                headers.append(line)
        
        return {
            'total_lines': len(lines),
            'blank_lines': sum(1 for line in lines if not line.strip()),
            'potential_headers': headers[:10]
        }
    
    def _extract_data_patterns(self, text):
        """Extract specific data patterns"""
        extracted = {}
        
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        extracted['emails'] = list(set(emails))
        
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        extracted['phones'] = list(set(phones))
        
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
        extracted['urls'] = list(set(urls))
        
        money = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d{2})?', text)
        extracted['monetary_values'] = list(set(money))
        
        return extracted
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display content analysis results"""
        if 'error' in results:
            st.error(f"Content analysis failed: {results['error']}")
            return
        
        individual_files = results['individual_files']
        config = results['config']
        
        st.markdown("### 🔍 Content Analysis Results")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 Files Analyzed", len(individual_files))
        
        with col2:
            total_words = sum(
                file_data.get('basic_stats', {}).get('word_count', 0) 
                for file_data in individual_files.values()
            )
            st.metric("📝 Total Words", f"{total_words:,}")
        
        with col3:
            analysis_count = len(config.get('analysis_types', []))
            st.metric("🔍 Analysis Types", analysis_count)
        
        # Individual file results
        st.markdown("### 📁 Individual File Analysis")
        
        for filename, file_data in individual_files.items():
            with st.expander(f"📄 {filename}", expanded=False):
                
                if 'basic_stats' in file_data:
                    st.markdown("**Basic Statistics:**")
                    stats = file_data['basic_stats']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Words", f"{stats['word_count']:,}")
                        st.metric("Characters", f"{stats['character_count']:,}")
                    with col2:
                        st.metric("Sentences", stats['sentence_count'])
                        st.metric("Paragraphs", stats['paragraph_count'])
                
                if 'readability' in file_data:
                    st.markdown("**Readability:**")
                    readability = file_data['readability']
                    st.write(f"• **Grade Level:** {readability['grade_level']}")
                    st.write(f"• **Average Sentence Length:** {readability['avg_sentence_length']} words")
                
                if 'keywords' in file_data:
                    st.markdown("**Top Keywords:**")
                    keywords = file_data['keywords'][:10]
                    keyword_text = ", ".join([f"{kw['word']} ({kw['frequency']})" for kw in keywords])
                    st.write(keyword_text)
                
                if 'structure' in file_data:
                    st.markdown("**Document Structure:**")
                    structure = file_data['structure']
                    st.write(f"• **Total Lines:** {structure['total_lines']}")
                    st.write(f"• **Blank Lines:** {structure['blank_lines']}")
                    if structure['potential_headers']:
                        st.write("• **Potential Headers:**")
                        for header in structure['potential_headers'][:5]:
                            st.write(f"  - {header}")
                
                if 'extracted_data' in file_data:
                    st.markdown("**Extracted Data:**")
                    data = file_data['extracted_data']
                    for data_type, items in data.items():
                        if items:
                            st.write(f"• **{data_type.title()}:** {len(items)} found")
                            for item in items[:3]:  # Show first 3
                                st.write(f"  - {item}")
        
        # Export options
        st.markdown("### 💾 Export Analysis")
        
        json_data = json.dumps(results, indent=2, default=str)
        st.download_button(
            label="📋 Download Full Analysis (JSON)",
            data=json_data,
            file_name=f"content_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        ) 