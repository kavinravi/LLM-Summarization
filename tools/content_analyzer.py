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
            options=[
                "Document Insights",
                "Financial Analysis", 
                "Keyword Extraction",
                "Document Structure",
                "Data Extraction",
                "Entity Recognition"
            ],
            default=["Document Insights", "Keyword Extraction"],
            help="Choose which types of analysis to perform",
            placeholder="Click to see all available analysis options..."
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
            
            if "Document Insights" in config['analysis_types']:
                file_analysis['document_insights'] = self._analyze_document_insights(text)
            
            if "Financial Analysis" in config['analysis_types']:
                file_analysis['financial_analysis'] = self._analyze_financial_data(text)
            
            if "Entity Recognition" in config['analysis_types']:
                file_analysis['entities'] = self._extract_entities(text)
            
            if "Keyword Extraction" in config['analysis_types']:
                file_analysis['keywords'] = self._extract_keywords(text, config['keyword_count'])
            
            if "Document Structure" in config['analysis_types']:
                file_analysis['structure'] = self._analyze_structure(text)
            
            if "Data Extraction" in config['analysis_types']:
                file_analysis['extracted_data'] = self._extract_data_patterns(text)
            
            results['individual_files'][uploaded_file.name] = file_analysis
        
        progress_bar.progress(100, "Content analysis complete!")
        
        return results
    
    def _analyze_document_insights(self, text):
        """Analyze meaningful document insights"""
        insights = {}
        
        # Document type detection
        doc_indicators = {
            'contract': ['agreement', 'contract', 'terms', 'conditions', 'party', 'obligations'],
            'financial_report': ['revenue', 'profit', 'loss', 'balance sheet', 'income statement'],
            'technical_spec': ['specification', 'requirements', 'technical', 'performance', 'standards'],
            'business_plan': ['market', 'strategy', 'growth', 'opportunity', 'business model'],
            'legal_document': ['whereas', 'hereby', 'pursuant', 'compliance', 'jurisdiction'],
            'research_paper': ['abstract', 'methodology', 'results', 'conclusion', 'references']
        }
        
        doc_type_scores = {}
        text_lower = text.lower()
        for doc_type, keywords in doc_indicators.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                doc_type_scores[doc_type] = score
        
        likely_type = max(doc_type_scores.items(), key=lambda x: x[1])[0] if doc_type_scores else "general_document"
        insights['document_type'] = likely_type
        insights['type_confidence'] = doc_type_scores.get(likely_type, 0) if doc_type_scores else 0
        
        # Time references
        import re
        dates = re.findall(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}|\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\b\d{4}[-/]\d{1,2}[-/]\d{1,2}', text)
        years = re.findall(r'\b(20\d{2}|19\d{2})\b', text)
        
        insights['date_references'] = len(set(dates))
        insights['year_references'] = len(set(years))
        insights['latest_year'] = max(years) if years else None
        
        # Key metrics detection
        percentages = re.findall(r'\b\d+(?:\.\d+)?%', text)
        insights['percentage_mentions'] = len(percentages)
        
        # Urgency indicators
        urgency_words = ['urgent', 'immediate', 'asap', 'deadline', 'critical', 'priority']
        urgency_count = sum(1 for word in urgency_words if word in text_lower)
        insights['urgency_level'] = 'high' if urgency_count >= 3 else 'medium' if urgency_count >= 1 else 'low'
        
        # Document length assessment
        word_count = len(text.split())
        if word_count < 500:
            length_category = 'brief'
        elif word_count < 2000:
            length_category = 'standard'
        elif word_count < 5000:
            length_category = 'detailed'
        else:
            length_category = 'comprehensive'
        
        insights['length_category'] = length_category
        insights['word_count'] = word_count
        
        return insights
    
    def _analyze_financial_data(self, text):
        """Extract and analyze financial information"""
        import re
        financial_data = {}
        
        # Currency amounts
        currency_patterns = [
            r'\$\d+(?:,\d{3})*(?:\.\d{2})?(?:\s*(?:million|billion|trillion|k|M|B|T))?',
            r'\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|dollars?)',
            r'(?:million|billion|trillion)\s*dollars?'
        ]
        
        all_amounts = []
        for pattern in currency_patterns:
            amounts = re.findall(pattern, text, re.IGNORECASE)
            all_amounts.extend(amounts)
        
        financial_data['monetary_references'] = len(all_amounts)
        financial_data['sample_amounts'] = all_amounts[:5]  # First 5 examples
        
        # Financial terms
        financial_terms = {
            'revenue_terms': ['revenue', 'income', 'sales', 'earnings', 'turnover'],
            'cost_terms': ['cost', 'expense', 'expenditure', 'spending', 'outlay'],
            'profit_terms': ['profit', 'margin', 'earnings', 'net income', 'operating income'],
            'investment_terms': ['investment', 'capital', 'funding', 'financing', 'equity'],
            'performance_terms': ['roi', 'return on investment', 'growth', 'yield', 'performance']
        }
        
        text_lower = text.lower()
        for category, terms in financial_terms.items():
            count = sum(1 for term in terms if term in text_lower)
            financial_data[category] = count
        
        # Financial ratios and percentages
        ratios = re.findall(r'\d+(?:\.\d+)?%', text)
        financial_data['percentage_metrics'] = len(ratios)
        
        # Time periods (fiscal, quarterly, etc.)
        time_periods = re.findall(r'\b(?:fiscal|quarter|quarterly|annual|yearly|monthly)\b', text, re.IGNORECASE)
        financial_data['time_period_references'] = len(time_periods)
        
        return financial_data
    
    def _extract_entities(self, text):
        """Extract key entities from the document"""
        import re
        entities = {}
        
        # Company names (simple heuristic)
        # Look for capitalized words followed by Corp, Inc, LLC, etc.
        companies = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\s+(?:Corp|Inc|LLC|Ltd|Company|Co\.)', text)
        entities['companies'] = list(set(companies))
        
        # Locations (cities, states, countries)
        location_indicators = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z]{2}\b|\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,\s*[A-Z][a-z]+\b', text)
        entities['locations'] = list(set(location_indicators))
        
        # People names (simple heuristic - capitalized first and last names)
        names = re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)
        # Filter out common false positives
        common_false_positives = ['United States', 'New York', 'Los Angeles', 'San Francisco', 'Las Vegas']
        names = [name for name in names if name not in common_false_positives]
        entities['people'] = list(set(names))[:10]  # Limit to 10
        
        # Product/service names (capitalized terms that appear multiple times)
        capitalized_terms = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', text)
        term_frequency = {}
        for term in capitalized_terms:
            if len(term.split()) <= 3:  # Only short phrases
                term_frequency[term] = term_frequency.get(term, 0) + 1
        
        # Get terms that appear more than once
        frequent_terms = {term: count for term, count in term_frequency.items() if count > 1}
        entities['frequent_terms'] = dict(sorted(frequent_terms.items(), key=lambda x: x[1], reverse=True)[:10])
        
        return entities
    
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
                file_data.get('document_insights', {}).get('word_count', 0) 
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
                
                if 'document_insights' in file_data:
                    st.markdown("**📊 Document Insights:**")
                    insights = file_data['document_insights']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"• **Document Type:** {insights['document_type'].replace('_', ' ').title()}")
                        st.write(f"• **Length Category:** {insights['length_category'].title()}")
                        st.write(f"• **Word Count:** {insights['word_count']:,}")
                    with col2:
                        st.write(f"• **Date References:** {insights['date_references']}")
                        st.write(f"• **Latest Year:** {insights['latest_year'] or 'None'}")
                        st.write(f"• **Urgency Level:** {insights['urgency_level'].title()}")
                
                if 'financial_analysis' in file_data:
                    st.markdown("**💰 Financial Analysis:**")
                    financial = file_data['financial_analysis']
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"• **Monetary References:** {financial['monetary_references']}")
                        st.write(f"• **Revenue Terms:** {financial['revenue_terms']}")
                        st.write(f"• **Cost Terms:** {financial['cost_terms']}")
                    with col2:
                        st.write(f"• **Investment Terms:** {financial['investment_terms']}")
                        st.write(f"• **Percentage Metrics:** {financial['percentage_metrics']}")
                        if financial['sample_amounts']:
                            st.write(f"• **Sample Amounts:** {', '.join(financial['sample_amounts'][:3])}")
                
                if 'entities' in file_data:
                    st.markdown("**👤 Entity Recognition:**")
                    entities = file_data['entities']
                    if entities['companies']:
                        st.write(f"• **Companies:** {', '.join(entities['companies'][:3])}")
                    if entities['locations']:
                        st.write(f"• **Locations:** {', '.join(entities['locations'][:3])}")
                    if entities['people']:
                        st.write(f"• **People:** {', '.join(entities['people'][:3])}")
                    if entities['frequent_terms']:
                        top_terms = list(entities['frequent_terms'].keys())[:3]
                        st.write(f"• **Key Terms:** {', '.join(top_terms)}")
                
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