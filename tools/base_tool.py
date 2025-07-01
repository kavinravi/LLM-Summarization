"""
Base Tool Class

Defines the common interface that all analysis tools must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import streamlit as st


class BaseTool(ABC):
    """Base class for all document analysis tools"""
    
    def __init__(self):
        self.name = "Base Tool"
        self.description = "Base tool description"
        self.icon = "🔧"
    
    @abstractmethod
    def show_configuration_ui(self) -> Dict[str, Any]:
        """
        Show the configuration UI for this tool.
        
        Returns:
            Dict containing the tool configuration, or None if configuration is incomplete
        """
        pass
    
    @abstractmethod
    def run_analysis(self, uploaded_files: List, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the analysis with the given files and configuration.
        
        Args:
            uploaded_files: List of uploaded file objects
            config: Configuration dictionary from show_configuration_ui()
            
        Returns:
            Dictionary containing analysis results
        """
        pass
    
    @abstractmethod
    def display_results(self, results: Dict[str, Any]) -> None:
        """
        Display the results in the Streamlit UI.
        
        Args:
            results: Results dictionary from run_analysis()
        """
        pass
    
    def _extract_text_from_file(self, uploaded_file) -> str:
        """Helper method to extract text from an uploaded file"""
        import tempfile
        import os
        from utils.llm_util import extract_text
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        
        try:
            # Extract text from document
            text = extract_text(tmp_file_path)
            return text
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    def _show_progress(self, progress: int, message: str):
        """Helper method to show progress"""
        if hasattr(st.session_state, 'progress_bar') and st.session_state.progress_bar:
            st.session_state.progress_bar.progress(progress, message) 