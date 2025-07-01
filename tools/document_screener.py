"""
Document Screener Tool

Screens documents against specific criteria with yes/no verdicts.
"""

import streamlit as st
import json
import textwrap
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

from .base_tool import BaseTool
from utils.llm_util import llm_screen


class DocumentScreener(BaseTool):
    """Tool for screening documents against yes/no criteria"""
    
    def __init__(self):
        super().__init__()
        self.name = "Document Screener"
        self.description = "Screen documents against specific criteria with yes/no verdicts"
        self.icon = "📋"
        self.chunk_size_chars = 15000
    
    def show_configuration_ui(self) -> Dict[str, Any]:
        """Show configuration UI for document screening"""
        st.markdown("### 📋 Configure Document Screening")
        st.markdown("Set up criteria to screen your documents against. Each criterion will get a yes/no verdict.")
        
        # Criteria source selection
        criteria_source = st.radio(
            "How would you like to set your criteria?",
            ["Use default criteria", "Enter custom criteria", "Upload criteria file"],
            help="Choose how to define the screening criteria"
        )
        
        criteria = []
        
        if criteria_source == "Use default criteria":
            criteria = self._load_default_criteria()
            if criteria:
                st.success(f"✅ Loaded {len(criteria)} default criteria")
                with st.expander("📋 Default Criteria", expanded=False):
                    for i, criterion in enumerate(criteria, 1):
                        st.write(f"{i}. {criterion}")
            else:
                st.error("Could not load default criteria")
                return None
                
        elif criteria_source == "Enter custom criteria":
            st.markdown("**Enter your criteria (one per line):**")
            criteria_text = st.text_area(
                "Criteria",
                height=200,
                placeholder="Example:\nRevenue > $1M annually\nLocated in California\nMore than 50 employees",
                help="Each line will be treated as a separate criterion"
            )
            
            if criteria_text.strip():
                criteria = [line.strip() for line in criteria_text.split('\n') if line.strip()]
                st.info(f"📝 {len(criteria)} criteria entered")
            else:
                st.warning("Please enter at least one criterion")
                return None
                
        elif criteria_source == "Upload criteria file":
            uploaded_file = st.file_uploader(
                "Upload criteria JSON file",
                type=['json'],
                help="Upload a JSON file containing an array of criteria strings"
            )
            
            if uploaded_file:
                try:
                    criteria = json.loads(uploaded_file.read().decode('utf-8'))
                    if isinstance(criteria, list):
                        st.success(f"✅ Loaded {len(criteria)} criteria from file")
                        with st.expander("📋 Uploaded Criteria", expanded=False):
                            for i, criterion in enumerate(criteria, 1):
                                st.write(f"{i}. {criterion}")
                    else:
                        st.error("File must contain a JSON array of criteria strings")
                        return None
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
                    return None
            else:
                st.warning("Please upload a criteria file")
                return None
        
        if not criteria:
            return None
        
        # Validate criteria (safety guardrail)
        validation_issues = self._validate_criteria(criteria)
        if validation_issues:
            st.warning("⚠️ Potential issues with your criteria:")
            for issue in validation_issues:
                st.write(f"• {issue}")
            
            if st.checkbox("Proceed anyway", help="Check this to continue despite validation warnings"):
                pass
            else:
                return None
        
        # Advanced options
        with st.expander("⚙️ Advanced Options", expanded=False):
            temperature = st.slider(
                "Analysis Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Higher values make the analysis more creative, lower values more consistent"
            )
            
            # Custom prompt with templates
            prompt_option = st.radio(
                "System Prompt",
                ["Use default prompt", "Choose from templates", "Write custom prompt"],
                help="Select how to configure the analysis prompt"
            )
            
            custom_prompt = None
            
            if prompt_option == "Choose from templates":
                template_choice = st.selectbox(
                    "Select Prompt Template",
                    [
                        "Strict Compliance Focus",
                        "Investment Analysis Focus", 
                        "Technical Specification Focus",
                        "Risk Assessment Focus",
                        "Environmental Impact Focus"
                    ]
                )
                
                templates = {
                    "Strict Compliance Focus": """You are a strict compliance analyst. Your task is to determine if documents meet specific criteria with high precision.

For each criterion, you must:
1. **Find Direct Evidence:** Look for explicit statements that directly address the criterion
2. **Apply Strict Standards:** Only return 'yes' if there is clear, unambiguous evidence
3. **Be Conservative:** When in doubt, prefer 'no' or 'unknown' over 'yes'
4. **Quote Precisely:** Always include exact quotes from the document

Return ONLY JSON in this format:
{"criterion name": {"verdict": "yes|no|unknown", "reason": "Found: [exact quote]. [explanation]"}}""",

                    "Investment Analysis Focus": """You are an investment analyst evaluating documents for financial opportunities. Focus on quantitative data and financial metrics.

For each criterion:
1. **Look for Numbers:** Prioritize concrete financial data, percentages, costs, revenues
2. **Assess Viability:** Consider the business/investment potential
3. **Use Web Search:** When you find project names or locations, search for missing financial data
4. **Be Optimistic but Realistic:** Look for potential even in incomplete information

Return ONLY JSON in this format:
{"criterion name": {"verdict": "yes|no|unknown", "reason": "Found: [data/quote]. [financial analysis]"}}""",

                    "Technical Specification Focus": """You are a technical expert analyzing documents for engineering and technical compliance.

For each criterion:
1. **Find Technical Data:** Look for specifications, measurements, technical parameters
2. **Verify Standards:** Check if technical requirements meet industry standards
3. **Search for Missing Specs:** Use web search to find technical details for equipment/systems mentioned
4. **Consider Technical Feasibility:** Assess if proposed solutions are technically sound

Return ONLY JSON in this format:
{"criterion name": {"verdict": "yes|no|unknown", "reason": "Found: [technical data]. [technical assessment]"}}""",

                    "Risk Assessment Focus": """You are a risk management specialist evaluating potential risks and mitigation strategies.

For each criterion:
1. **Identify Risk Factors:** Look for potential problems, challenges, or failure points
2. **Assess Mitigation:** Evaluate proposed risk management strategies
3. **Consider Uncertainty:** Pay attention to unknowns and potential issues
4. **Search for Risk Data:** Look up historical data about similar projects/locations

Return ONLY JSON in this format:
{"criterion name": {"verdict": "yes|no|unknown", "reason": "Found: [risk evidence]. [risk analysis]"}}""",

                    "Environmental Impact Focus": """You are an environmental analyst assessing ecological and regulatory compliance.

For each criterion:
1. **Environmental Evidence:** Look for environmental studies, impact assessments, permits
2. **Regulatory Compliance:** Check for mentions of environmental regulations and compliance
3. **Search Environmental Data:** Look up environmental conditions, protected areas, regulations for locations mentioned
4. **Sustainability Focus:** Consider long-term environmental sustainability

Return ONLY JSON in this format:
{"criterion name": {"verdict": "yes|no|unknown", "reason": "Found: [environmental evidence]. [environmental analysis]"}}"""
                }
                
                custom_prompt = templates[template_choice]
                
                with st.expander("📋 Preview Selected Template", expanded=False):
                    st.code(custom_prompt, language="text")
                    
            elif prompt_option == "Write custom prompt":
                custom_prompt = st.text_area(
                    "Custom System Prompt",
                    height=200,
                    placeholder="""Example:
You are a [role] analyzing documents for [purpose].

For each criterion:
1. [instruction 1]
2. [instruction 2]
3. [instruction 3]

Return ONLY JSON in this format:
{"criterion name": {"verdict": "yes|no|unknown", "reason": "[explanation]"}}""",
                    help="Write your own custom system prompt for specialized analysis"
                )
                
                if custom_prompt:
                    st.info("💡 Tip: Include clear instructions for verdict determination and JSON format requirements")
            
            chunk_size = st.number_input(
                "Chunk Size (characters)",
                min_value=5000,
                max_value=50000,
                value=self.chunk_size_chars,
                step=5000,
                help="Size of text chunks for processing large documents"
            )
        
        return {
            'criteria': criteria,
            'temperature': temperature,
            'custom_prompt': custom_prompt if custom_prompt and custom_prompt.strip() else None,
            'chunk_size': chunk_size
        }
    
    def run_analysis(self, uploaded_files: List, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run document screening analysis"""
        criteria = config['criteria']
        temperature = config.get('temperature', 0.0)
        custom_prompt = config.get('custom_prompt')
        chunk_size = config.get('chunk_size', self.chunk_size_chars)
        
        results = {}
        
        # Create progress bar
        progress_bar = st.progress(0, "Starting analysis...")
        
        for file_idx, uploaded_file in enumerate(uploaded_files):
            file_progress_start = int((file_idx / len(uploaded_files)) * 100)
            file_progress_end = int(((file_idx + 1) / len(uploaded_files)) * 100)
            
            progress_bar.progress(
                file_progress_start, 
                f"Processing {uploaded_file.name} ({file_idx + 1}/{len(uploaded_files)})"
            )
            
            try:
                # Extract text
                raw_text = self._extract_text_from_file(uploaded_file)
                
                # Split into chunks
                chunks = textwrap.wrap(raw_text, chunk_size)
                
                # Screen each chunk
                file_verdicts = {}
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_name = f"chunk_{chunk_idx + 1}"
                    
                    chunk_progress = file_progress_start + int(
                        ((chunk_idx + 1) / len(chunks)) * (file_progress_end - file_progress_start)
                    )
                    progress_bar.progress(
                        chunk_progress,
                        f"Screening {uploaded_file.name} - chunk {chunk_idx + 1}/{len(chunks)}"
                    )
                    
                    try:
                        chunk_result = llm_screen(
                            chunk, 
                            criteria,
                            temperature=temperature,
                            custom_system_prompt=custom_prompt
                        )
                        
                        # Remove debug info if present
                        if '_debug' in chunk_result:
                            del chunk_result['_debug']
                            
                        file_verdicts[chunk_name] = chunk_result
                        
                    except Exception as e:
                        st.error(f"Error processing chunk {chunk_idx + 1} of {uploaded_file.name}: {str(e)}")
                        # Create error verdicts
                        file_verdicts[chunk_name] = {
                            criterion: {
                                "verdict": "error", 
                                "reason": f"Processing error: {str(e)}"
                            } for criterion in criteria
                        }
                
                results[uploaded_file.name] = {
                    'verdicts': file_verdicts,
                    'raw_text': raw_text,
                    'chunk_count': len(chunks)
                }
                
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                results[uploaded_file.name] = {
                    'error': str(e),
                    'verdicts': {},
                    'raw_text': '',
                    'chunk_count': 0
                }
        
        progress_bar.progress(100, "Analysis complete!")
        
        # Consolidate results
        consolidated = self._consolidate_results(results, criteria)
        
        final_results = {
            'individual_results': results,
            'consolidated_results': consolidated,
            'criteria': criteria,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to cache for future reference
        self._save_results_to_cache(final_results)
        
        return final_results
    
    def display_results(self, results: Dict[str, Any]) -> None:
        """Display screening results"""
        if 'error' in results:
            st.error(f"Analysis failed: {results['error']}")
            return
        
        consolidated = results['consolidated_results']
        individual = results['individual_results']
        criteria = results['criteria']
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Files Analyzed", len(individual))
        
        with col2:
            total_chunks = sum(file_data.get('chunk_count', 0) for file_data in individual.values())
            st.metric("📝 Text Chunks", total_chunks)
        
        with col3:
            st.metric("📋 Criteria", len(criteria))
        
        with col4:
            yes_count = sum(1 for verdict in consolidated.values() if verdict.get('verdict') == 'yes')
            st.metric("✅ Passed Criteria", f"{yes_count}/{len(criteria)}")
        
        # Consolidated results table
        st.markdown("### 📊 Consolidated Results")
        st.markdown("*Overall verdict across all documents and chunks*")
        
        df_data = []
        for criterion, result in consolidated.items():
            df_data.append({
                'Criterion': criterion,
                'Verdict': result['verdict'],
                'Confidence': result.get('confidence', 'N/A'),
                'Supporting Evidence': result.get('reason', '')[:100] + "..." if len(result.get('reason', '')) > 100 else result.get('reason', '')
            })
        
        df = pd.DataFrame(df_data)
        
        # Color code the results
        def highlight_verdict(val):
            if val == 'yes':
                return 'background-color: #d4edda; color: #155724'
            elif val == 'no':
                return 'background-color: #f8d7da; color: #721c24'
            elif val == 'unknown':
                return 'background-color: #fff3cd; color: #856404'
            else:
                return 'background-color: #f8f9fa; color: #495057'
        
        styled_df = df.style.applymap(highlight_verdict, subset=['Verdict'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Individual file results
        st.markdown("### 📁 Individual File Results")
        
        for filename, file_data in individual.items():
            if 'error' in file_data:
                st.error(f"❌ {filename}: {file_data['error']}")
                continue
                
            with st.expander(f"📄 {filename} ({file_data['chunk_count']} chunks)", expanded=False):
                # File-level summary
                file_verdicts = file_data['verdicts']
                
                if file_verdicts:
                    # Create summary for this file
                    file_summary = {}
                    for chunk_name, chunk_results in file_verdicts.items():
                        for criterion, result in chunk_results.items():
                            if criterion not in file_summary:
                                file_summary[criterion] = {'yes': 0, 'no': 0, 'unknown': 0, 'error': 0}
                            verdict = result.get('verdict', 'unknown')
                            file_summary[criterion][verdict] = file_summary[criterion].get(verdict, 0) + 1
                    
                    # Display file summary
                    file_df_data = []
                    for criterion, counts in file_summary.items():
                        total = sum(counts.values())
                        majority_verdict = max(counts.items(), key=lambda x: x[1])[0]
                        file_df_data.append({
                            'Criterion': criterion,
                            'Majority Verdict': majority_verdict,
                            'Yes': counts.get('yes', 0),
                            'No': counts.get('no', 0),
                            'Unknown': counts.get('unknown', 0),
                            'Total Chunks': total
                        })
                    
                    file_df = pd.DataFrame(file_df_data)
                    st.dataframe(file_df, use_container_width=True)
                    
                    # Option to see detailed chunk results
                    if st.checkbox(f"Show detailed chunk results for {filename}", key=f"details_{filename}"):
                        for chunk_name, chunk_results in file_verdicts.items():
                            st.markdown(f"**{chunk_name}:**")
                            for criterion, result in chunk_results.items():
                                verdict = result.get('verdict', 'unknown')
                                reason = result.get('reason', 'No reason provided')
                                
                                if verdict == 'yes':
                                    st.success(f"✅ {criterion}: {reason[:200]}...")
                                elif verdict == 'no':
                                    st.error(f"❌ {criterion}: {reason[:200]}...")
                                elif verdict == 'unknown':
                                    st.warning(f"❓ {criterion}: {reason[:200]}...")
                                else:
                                    st.info(f"ℹ️ {criterion}: {reason[:200]}...")
                else:
                    st.warning("No screening results available for this file")
        
        # Export options
        st.markdown("### 💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export consolidated results as CSV
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 Download Consolidated Results (CSV)",
                data=csv_data,
                file_name=f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export full results as JSON
            json_data = json.dumps(results, indent=2, default=str)
            st.download_button(
                label="📋 Download Full Results (JSON)",
                data=json_data,
                file_name=f"screening_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def _load_default_criteria(self) -> List[str]:
        """Load default criteria from config/criteria.json"""
        try:
            with open("config/criteria.json", "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            st.error("Default criteria file not found. Please ensure config/criteria.json exists.")
            return []
        except Exception as e:
            st.error(f"Error loading default criteria: {str(e)}")
            return []
    
    def _consolidate_results(self, results: Dict, criteria: List[str]) -> Dict:
        """Consolidate chunk results into overall verdicts per criterion"""
        consolidated = {}
        
        for criterion in criteria:
            all_verdicts = []
            all_reasons = []
            
            # Collect all verdicts for this criterion across all files and chunks
            for file_data in results.values():
                if 'verdicts' in file_data:
                    for chunk_results in file_data['verdicts'].values():
                        if criterion in chunk_results:
                            verdict_data = chunk_results[criterion]
                            verdict = verdict_data.get('verdict', 'unknown')
                            reason = verdict_data.get('reason', '')
                            
                            all_verdicts.append(verdict)
                            if reason:
                                all_reasons.append(reason)
            
            if not all_verdicts:
                consolidated[criterion] = {
                    'verdict': 'unknown',
                    'reason': 'No analysis results available',
                    'confidence': 'low'
                }
                continue
            
            # Determine overall verdict using majority rule with priority
            verdict_counts = {}
            for verdict in all_verdicts:
                verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            
            # Priority: yes > no > unknown > error
            if verdict_counts.get('yes', 0) > 0:
                overall_verdict = 'yes'
                confidence = 'high' if verdict_counts['yes'] > len(all_verdicts) * 0.7 else 'medium'
            elif verdict_counts.get('no', 0) > len(all_verdicts) * 0.5:
                overall_verdict = 'no'
                confidence = 'high' if verdict_counts['no'] > len(all_verdicts) * 0.7 else 'medium'
            else:
                overall_verdict = 'unknown'
                confidence = 'low'
            
            # Combine reasons
            unique_reasons = list(set(all_reasons))[:3]  # Top 3 unique reasons
            combined_reason = "; ".join(unique_reasons) if unique_reasons else "No specific evidence found"
            
            consolidated[criterion] = {
                'verdict': overall_verdict,
                'reason': combined_reason,
                'confidence': confidence,
                'verdict_counts': verdict_counts
            }
        
        return consolidated
    
    def _save_results_to_cache(self, results):
        """Save screening results to disk cache for future reference"""
        import os
        
        cache_dir = "cache/screening_cache"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_file = os.path.join(cache_dir, f"screening_{timestamp}.json")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Store in session state for immediate access
            if 'screening_results' not in st.session_state:
                st.session_state.screening_results = {}
            st.session_state.screening_results = results
            
        except Exception as e:
            st.warning(f"Could not save results to cache: {str(e)}")
    
    def _load_cached_results():
        """Load cached screening results (static method for potential future use)"""
        import os
        
        cache_dir = "cache/screening_cache"
        if not os.path.exists(cache_dir):
            return []
        
        cached_sessions = []
        for filename in os.listdir(cache_dir):
            if filename.startswith("screening_") and filename.endswith(".json"):
                filepath = os.path.join(cache_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        cached_sessions.append({
                            'filename': filename,
                            'timestamp': data.get('timestamp', ''),
                            'criteria_count': len(data.get('criteria', [])),
                            'files_count': len(data.get('individual_results', {})),
                            'data': data
                        })
                except Exception as e:
                    continue
        
        # Sort by timestamp (newest first)
        cached_sessions.sort(key=lambda x: x['timestamp'], reverse=True)
        return cached_sessions
    
    def _validate_criteria(self, criteria):
        """Validate criteria for common issues (safety guardrail)"""
        issues = []
        
        for i, criterion in enumerate(criteria, 1):
            # Check for very short criteria
            if len(criterion.strip()) < 10:
                issues.append(f"Criterion {i} is very short: '{criterion}' - consider adding more detail")
            
            # Check for vague language
            vague_words = ['good', 'bad', 'nice', 'adequate', 'sufficient', 'reasonable']
            if any(word in criterion.lower() for word in vague_words):
                issues.append(f"Criterion {i} contains vague language - consider using specific metrics")
            
            # Check for missing units in quantitative criteria
            if any(char.isdigit() for char in criterion):
                units = ['%', '$', 'km', 'meters', 'miles', 'MW', 'kW', 'acres', 'years', 'months', 'days']
                if not any(unit in criterion for unit in units):
                    issues.append(f"Criterion {i} has numbers but unclear units - specify measurement units")
            
            # Check for question marks (criteria should be statements)
            if '?' in criterion:
                issues.append(f"Criterion {i} appears to be a question - rephrase as a statement to evaluate")
        
        # Check for too many criteria
        if len(criteria) > 15:
            issues.append(f"You have {len(criteria)} criteria - consider grouping similar ones for better performance")
        
        return issues 