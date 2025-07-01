#!/usr/bin/env python
# llm_util.py
"""
Utility for:
1.  Screening PDFs/DOCX/XLSX against yes/no criteria (temperature = 0).
2.  Generating a marketing blurb from the screening verdict (temperature ~0.6).

Usage examples
--------------
# Compliance screen:
python llm_util.py screen my_memo.pdf --criteria_file config/criteria.json

# Marketing blurb (uses prior JSON verdict):
python llm_util.py blurb verdict.json --temp 0.7
"""
# Python <3.10 compatibility for type-hints
from __future__ import annotations

import os, json, textwrap, argparse
import pdfplumber, docx, pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
# Import web_search locally when needed to avoid import issues
from typing import Optional  # For Python <3.10 union
import re

# ---------- ENVIRONMENT ----------
load_dotenv()  # Load environment variables from .env file

def get_env_var(key: str, default: str = None) -> str:
    """Get environment variable from Streamlit secrets or .env file."""
    try:
        import streamlit as st
        return st.secrets[key]
    except:
        return os.getenv(key, default)

# Single model configuration
MODEL = get_env_var("GEMINI_MODEL", "gemini-2.0-flash")
print(f"DEBUG: Loaded MODEL = {MODEL}")
print(f"DEBUG: GEMINI_MODEL env var = {os.getenv('GEMINI_MODEL', 'NOT_SET')}")
print(f"DEBUG: .env loaded = {os.path.exists('.env')}")

# Configure Gemini
genai.configure(api_key=get_env_var("GOOGLE_API_KEY"))

MAX_INPUT_TOKENS = 2000000  # Gemini 1.5 Flash has 2M token context window
# 150,000 chars ≈ 100k tokens – much larger chunks possible with Gemini's 1.5M window
CHUNK_SIZE_CHARS = 15000


# ---------- FILE I/O ----------
def _clean_extracted_text(text: str) -> str:
    """Clean up common text extraction issues from PDFs"""
    import re
    
    # Fix common number formatting issues (e.g., "333.3billionincastandcashequivalents")
    # Add space before "billion", "million", "thousand", etc.
    text = re.sub(r'(\d+\.?\d*)(billion|million|thousand|trillion)', r'\1 \2', text, flags=re.IGNORECASE)
    
    # Add space before "in" when it follows a number/word without space
    text = re.sub(r'([a-zA-Z])in([A-Z])', r'\1 in \2', text)
    
    # Fix "andcash" type issues - add space before common words that get concatenated
    text = re.sub(r'([a-z])(and|cash|equivalents|the|of|in|for|with|from|to|by)', r'\1 \2', text)
    
    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    raw_text = ""
    
    if ext == ".pdf":
        with pdfplumber.open(path) as pdf:
            # Using layout=True can help preserve spacing in complex documents
            raw_text = "\n".join(p.extract_text(layout=True) or "" for p in pdf.pages)
    elif ext in (".docx", ".doc"):
        raw_text = "\n".join(p.text for p in docx.Document(path).paragraphs)
    elif ext in (".xlsx", ".xls"):
        dfs = pd.read_excel(path, sheet_name=None)
        raw_text = "\n".join(df.to_csv(index=False) for df in dfs.values())
    elif ext == ".csv":
        df = pd.read_csv(path)
        raw_text = df.to_csv(index=False)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    
    # Clean the extracted text
    return _clean_extracted_text(raw_text)


# ---------- LLM CALLS ----------
def chat_complete(prompt: str, temperature: float, max_tokens: int = 8192, json_mode: bool = False, model: Optional[str] = None):
    """Chat completion using Gemini API."""
    model_name = model or MODEL
    
    # Configure generation parameters
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    
    # Add JSON mode if requested
    if json_mode:
        generation_config["response_mime_type"] = "application/json"
    
    # Create model instance
    gemini_model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config
    )
    
    # Generate response
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


def _web_search_tool_spec():
    """Creates the web search tool specification for Gemini."""
    return genai.protos.FunctionDeclaration(
        name="web_search",
        description="Search the web for missing quantitative data before marking criteria as 'unknown'. Use this when you find partial information (like a substation name) but need specific numbers (like capacity, voltage, distance, cost). REQUIRED before returning 'unknown' for any quantitative criterion.",
        parameters=genai.protos.Schema(
            type=genai.protos.Type.OBJECT,
            properties={
                "query": genai.protos.Schema(
                    type=genai.protos.Type.STRING,
                    description="Specific search query for missing data (e.g., 'Harry Allen Substation capacity MW', 'BLM land lease cost Nevada per acre')"
                )
            },
            required=["query"]
        )
    )


# ---------------- LLM SCREENING -----------------

def _screen_with_model(chunk: str, criteria: list[str], model: str, temperature: float, max_rounds: int, custom_system_prompt: str = None) -> dict:
    """Internal helper that runs the full tool-calling loop with a given model."""

    # Use custom system prompt if provided, otherwise use default
    if custom_system_prompt:
        system_prompt = custom_system_prompt
    else:
        system_prompt = (
            "You are a diligent project analyst. Your task is to analyze a document and determine if it meets a list of criteria.\\n"
            "\\n"
            "For each criterion, you must perform the following steps:\\n"
            "1.  **Find Evidence:** Scour the document for any text relevant to the criterion. You MUST quote the best snippet you find.\\n"
            "2.  **Analyze Evidence:** Look at the evidence you found.\\n"
            "3.  **Make a Verdict:**\\n"
            "    -   If the evidence directly confirms the criterion, the verdict is 'yes'.\\n"
            "    -   If the evidence provides strong contextual clues that logically imply the criterion is met, the verdict is 'yes'. This requires you to connect different pieces of information to reach a conclusion.\\n"
            "    -   If the evidence contradicts the criterion, the verdict is 'no'.\\n"
            "    -   If there is no evidence, or the evidence is insufficient to make a logical conclusion, the verdict is 'unknown'.\\n"
            "\\n"
            "**Web Search (when needed):** You have access to web search for gathering additional data. Use it strategically when:\\n"
            "- You find strong evidence but need one specific missing piece of quantitative data\\n"
            "- The document clearly implies something but lacks the exact numbers needed\\n"
            "- You find entity/location names that could yield specific measurements\\n"
            "\\n"
            "**Search Strategy:** Use broad, simple queries that are likely to find data. Accept any numerical data from search results, even if approximate. For locations, try searching for the city/region name plus the data type (e.g., 'Las Vegas solar irradiance', 'Nevada transmission lines').\\n"
            "\\n"
            "Do NOT return 'unknown' for quantitative criteria without first attempting web search. If search returns any relevant numbers, use them.\\n"
            "\\n"
            "Return ONLY JSON in this format:\\n"
            "{\\n  \\\"criterion name\\\": {\\\"verdict\\\": \\\"yes|no|unknown\\\", \\\"reason\\\": \\\"Found: [quoted text]. [explanation]\\\"},\\n  ...\\n}\\n"
        )

    user_prompt = (
        f"Criteria:\n{json.dumps(criteria, indent=2)}\n\n"
        f"Document chunk:\n{chunk}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Configure generation parameters including temperature and max tokens
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": 8192,  # Increased from default to handle complex analysis
    }
    
    web_search_tool_spec = _web_search_tool_spec()
    gemini_model = genai.GenerativeModel(
        model_name=model,
        tools=[web_search_tool_spec],
        system_instruction=system_prompt,
        generation_config=generation_config
    )
    debug_messages = []
    debug_messages.append(f"DEBUG: Starting screening with {len(criteria)} criteria")
    debug_messages.append(f"DEBUG: Criteria: {criteria}")
    debug_messages.append(f"DEBUG: Chunk length: {len(chunk)} chars")
    debug_messages.append(f"DEBUG: Chunk preview (first 500 chars): {repr(chunk[:500])}")
    debug_messages.append(f"DEBUG: Model: {model}, Temperature: {temperature}, Max rounds: {max_rounds}")
    debug_messages.append("DEBUG: Created model WITH web_search tool enabled")
    
    # Web search function for Gemini tool-calling
    def web_search_func(query: str):
        """Web search function for Gemini tool-calling."""
        debug_messages.append(f"DEBUG: Web search called with query: {query}")
        try:
            from .search_tool import web_search
            results = web_search(query)
            debug_messages.append(f"DEBUG: Search results length: {len(results)}")
            return results
        except ImportError as e:
            debug_messages.append(f"DEBUG: Could not import web_search: {e}")
            return f"Web search unavailable for query: {query}"
    
    # Create chat session (tool execution handled manually in loop)
    chat = gemini_model.start_chat()
    
    for round_num in range(max_rounds):
        debug_msg = f"DEBUG: Round {round_num + 1}/{max_rounds}"
        debug_messages.append(debug_msg)
        
        try:
            # Send just the user prompt since system prompt is already set
            response = chat.send_message(user_prompt)
            
            # Check if response has valid content
            response_text = ""
            finish_reason = None
            
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)
                
                # Log finish reason for debugging
                if finish_reason is not None:
                    if hasattr(finish_reason, 'name'):
                        finish_str = finish_reason.name
                    else:
                        finish_str = str(finish_reason)
                    debug_messages.append(f"DEBUG: Finish reason: {finish_str}")
                
                # Handle function calls in a loop (model might make multiple calls)
                function_call_count = 0
                max_function_calls = 5  # Prevent infinite loops
                
                while function_call_count < max_function_calls:
                    has_function_call = False
                    
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'function_call'):
                                has_function_call = True
                                function_call_count += 1
                                debug_messages.append(f"DEBUG: Found function_call #{function_call_count} in response parts")
                                func_call = part.function_call
                                if func_call.name == 'web_search':
                                    query = func_call.args.get('query', '')
                                    debug_messages.append(f"DEBUG: Executing web_search with query: {query}")
                                    search_result = web_search_func(query)
                                    debug_messages.append(f"DEBUG: Web search result length: {len(search_result)}")
                                    
                                    # Send search result back to model
                                    function_response = genai.protos.Part(
                                        function_response=genai.protos.FunctionResponse(
                                            name='web_search',
                                            response={'result': search_result}
                                        )
                                    )
                                    new_response = chat.send_message(function_response)
                                    
                                    # Process the new response - update the response variable
                                    response = new_response
                                    if hasattr(new_response, 'candidates') and new_response.candidates:
                                        candidate = new_response.candidates[0]
                                        finish_reason = getattr(candidate, 'finish_reason', None)
                                    
                                    debug_messages.append(f"DEBUG: Updated response after web search #{function_call_count}")
                                    break  # Exit the parts loop after handling function call
                    
                    if not has_function_call:
                        debug_messages.append(f"DEBUG: No more function calls found after {function_call_count} calls")
                        break
                
                if function_call_count >= max_function_calls:
                    debug_messages.append(f"DEBUG: Hit max function calls limit ({max_function_calls})")
                
                # Now try to get response text safely
                try:
                    response_text = response.text or ""
                    debug_messages.append(f"DEBUG: Successfully got response.text, length: {len(response_text)}")
                except (ValueError, AttributeError) as e:
                    debug_messages.append(f"DEBUG: Could not access response.text: {e}")
                    # Try to get text from parts directly
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                                debug_messages.append(f"DEBUG: Got text from part, length: {len(part.text)}")
            
            debug_messages.append(f"DEBUG: Got response, text length: {len(response_text)}")
            
            # Check for safety blocks or other issues
            if not response_text and finish_reason:
                if hasattr(finish_reason, 'name'):
                    reason_str = finish_reason.name
                else:
                    reason_str = str(finish_reason)
                
                if reason_str in ['SAFETY', 'BLOCKED_REASON_SAFETY']:
                    debug_messages.append(f"DEBUG: Response blocked for safety reasons: {reason_str}")
                    return {"_debug": debug_messages}
                elif reason_str in ['MAX_TOKENS', 'FINISH_REASON_MAX_TOKENS']:
                    debug_messages.append(f"DEBUG: Hit max token limit - response was truncated: {reason_str}")
                    debug_messages.append("DEBUG: Trying to continue with partial response or fallback")
                    # Don't return immediately - try to get partial response
                elif reason_str == 'FUNCTION_CALL':
                    debug_messages.append("DEBUG: Function call handled above, continuing...")
                else:
                    if not response_text:  # Only return error if we truly have no content
                        debug_messages.append(f"DEBUG: No content and finish reason: {reason_str}")
                        return {"_debug": debug_messages}
            
            # Try to parse JSON response
            if response_text:
                debug_messages.append("DEBUG: Attempting to parse JSON response")
                debug_messages.append(f"DEBUG: Full response content: {repr(response_text[:1000])}")
                
                # Try to find JSON in response (with or without markdown code blocks)
                json_str = ""
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # Try to find JSON without code blocks
                    json_match = re.search(r'(\{.*?\})', response_text, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        json_str = response_text.strip()
                
                if json_str:
                    try:
                        result = json.loads(json_str)
                        debug_messages.append(f"DEBUG: JSON parsing successful, type: {type(result)}")
                        
                        if isinstance(result, dict):
                            debug_messages.append(f"DEBUG: Successfully parsed JSON with {len(result)} items")
                            debug_messages.append(f"DEBUG: JSON keys: {list(result.keys())}")
                            
                            # Check if this is a financial analysis response (has 'results' and 'ranking' keys)
                            if 'results' in result and 'ranking' in result:
                                debug_messages.append("DEBUG: Detected financial analysis format")
                                # Convert financial format to standard format
                                converted_result = {}
                                if result['results'] and len(result['results']) > 0:
                                    ticker_data = result['results'][0]  # Use first ticker's data
                                    for key, value in ticker_data.items():
                                        if key != 'ticker':
                                            # Map financial fields to criteria
                                            matching_criterion = None
                                            for crit in criteria:
                                                if key in crit.lower() or key.replace('_', '-') in crit.lower():
                                                    matching_criterion = crit
                                                    break
                                            
                                            if matching_criterion:
                                                if isinstance(value, (int, float)):
                                                    converted_result[matching_criterion] = {
                                                        "verdict": "calculated", 
                                                        "reason": f"Calculated value: {value}"
                                                    }
                                                elif value in ['yes', 'no', 'unknown']:
                                                    converted_result[matching_criterion] = {
                                                        "verdict": value, 
                                                        "reason": f"Analysis result: {value}"
                                                    }
                                                else:
                                                    converted_result[matching_criterion] = {
                                                        "verdict": str(value), 
                                                        "reason": f"Value: {value}"
                                                    }
                                
                                # Add the original financial data for reference
                                converted_result['_financial_data'] = result
                                result = converted_result
                                debug_messages.append(f"DEBUG: Converted to standard format with {len(result)} criteria")
                            
                            # Check if any keys match our criteria (standard format)
                            matching_criteria = [crit for crit in criteria if crit in result]
                            debug_messages.append(f"DEBUG: Matching criteria found: {len(matching_criteria)}")
                            
                            # Store debug info for potential display
                            result['_debug'] = debug_messages
                            
                            # Ensure every criterion is present; if missing, mark unknown
                            for crit in criteria:
                                if crit not in result:
                                    debug_messages.append(f"DEBUG: Missing criterion '{crit}', adding as unknown")
                                    result[crit] = {"verdict": "unknown", "reason": "Not mentioned"}
                            
                            return result
                        else:
                            debug_messages.append(f"DEBUG: JSON result is not a dict: {type(result)}")
                    except json.JSONDecodeError as e:
                        debug_messages.append(f"DEBUG: JSON parse error: {e}")
                        debug_messages.append(f"DEBUG: Problematic content: {repr(response_text[:500])}")
                        # fall through to retry
                        pass
            else:
                debug_messages.append(f"DEBUG: No content in response")

            # Continue to next round if no valid JSON yet
            
        except Exception as e:
            debug_messages.append(f"DEBUG: API call error in round {round_num + 1}: {type(e).__name__}: {e}")
            import traceback
            debug_messages.append(f"DEBUG: Traceback: {traceback.format_exc()}")
            break

    # Fallback if all rounds exhausted
    debug_messages.append(f"DEBUG: All rounds exhausted, returning fallback")
    debug_messages.append(f"DEBUG: Final message count: {len(messages)}")
    debug_messages.append(f"DEBUG: Message roles: {[m.get('role', 'unknown') for m in messages]}")
    if messages:
        last_msg = messages[-1]
        debug_messages.append(f"DEBUG: Last message role: {last_msg.get('role')}")
        debug_messages.append(f"DEBUG: Last message content preview: {str(last_msg.get('content', ''))[:200]}")
    
    fallback_result = {c: {"verdict": "unknown", "reason": "Could not evaluate"} for c in criteria}
    fallback_result['_debug'] = debug_messages
    return fallback_result


def llm_screen(chunk: str, criteria: list[str], temperature: float = 0.0, max_rounds: int = 3, custom_system_prompt: str = None) -> dict:
    """Single-model screening with web search capability."""
    
    # Use web search version directly to test the fixes
    return _screen_with_model(chunk, criteria, MODEL, temperature, max_rounds, custom_system_prompt)


def llm_blurb(verdict_json: dict, temperature: float = 0.6) -> str:
    """Generate marketing blurb using Gemini with proper response handling."""
    
    system_prompt = (
        "You are a copywriter for a renewable-energy SaaS startup. "
        "Write concise, optimistic marketing copy that highlights project strengths."
    )
    
    user_prompt = (
        "Write a concise (≤ 120 words) marketing blurb that:\n"
        "• Highlights the top PASS items from the screening result.\n"
        "• Soft-pedals any FAIL items as future mitigation steps.\n"
        "• Uses an optimistic, professional tone.\n\n"
        f"Screening result:\n{json.dumps(verdict_json, indent=2)}"
    )
    
    # Configure generation parameters
    generation_config = {
        "temperature": temperature,
    }
    
    # Create model instance without tools (no web search needed for marketing copy)
    gemini_model = genai.GenerativeModel(
        model_name=MODEL,
        generation_config=generation_config,
        system_instruction=system_prompt
    )
    
    try:
        # Generate response
        response = gemini_model.generate_content(user_prompt)
        
        # Try to get response text safely
        response_text = ""
        finish_reason = None
        
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            
            # Log finish reason for debugging
            if finish_reason is not None:
                if hasattr(finish_reason, 'name'):
                    finish_str = finish_reason.name
                else:
                    finish_str = str(finish_reason)
                print(f"DEBUG: Marketing blurb finish reason: {finish_str}")
            
            try:
                response_text = response.text or ""
                print(f"DEBUG: Marketing blurb response length: {len(response_text)}")
            except (ValueError, AttributeError) as e:
                print(f"DEBUG: Could not access response.text: {e}")
                # Try to get text from parts directly
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text
                            print(f"DEBUG: Got text from part, length: {len(part.text)}")
        else:
            print("DEBUG: No candidates in response")
        
        if response_text:
            return response_text.strip()
        else:
            error_msg = f"Unable to generate marketing blurb. Finish reason: {finish_reason}"
            print(f"DEBUG: {error_msg}")
            return error_msg
            
    except Exception as e:
        error_msg = f"Error generating blurb: {str(e)}"
        print(f"DEBUG: {error_msg}")
        return error_msg


def llm_summarize(chunk: str, focus_areas: list[str] = None, temperature: float = 0.3) -> dict:
    """Summarize document chunk with optional focus areas, providing structured output with highlights and citations."""
    
    # Build focus areas text
    focus_text = "Provide a general summary of the document."
    if focus_areas:
        focus_list = "\n".join([f'- "{area}"' for area in focus_areas])
        focus_text = (
            "Pay special attention to the following topics. For each topic, provide a detailed analysis and supporting quotes "
            f"under the 'focus_area_insights' key in the JSON output. The topics are:\n{focus_list}"
        )

    system_prompt = (
        "You are an expert document analyst. Your task is to create a concise but structured summary of a given document chunk. "
        "You MUST respond with ONLY a single, valid JSON object that adheres to the specified format. Do not add any text before or after the JSON object. "
        "Do not use markdown `json` block, just output the raw JSON."
        "\n\n"
        "The JSON object must have the following structure. Keep all summaries and lists concise.\n"
        "{\n"
        '  "executive_summary": "Brief 2-4 sentence overview of the entire document chunk.",\n'
        '  "key_highlights": ["List of the top 5 most critical points as strings."],\n'
        '  "main_sections": {\n'
        '    "Section Title From Document": {\n'
        '      "summary": "Concise 3-5 sentence summary of this specific section.",\n'
        '      "key_points": ["List of the top 3-5 key takeaways from this section."],\n'
        '      "citations": ["Provide 1-2 brief, relevant quotes from the document that support the summary."]\n'
        '    }\n'
        '  },\n'
        '  "focus_area_insights": {\n'
        '    "Name of Focus Area": {\n'
        '      "findings": "Concise 3-5 sentence analysis of what the document says about this focus area.",\n'
        '      "citations": ["Provide 1-2 brief, relevant quotes from the document related to this focus area."]\n'
        '    }\n'
        '  },\n'
        '  "data_points": ["List of the top 5-7 key metrics, numbers, or quantitative information found."],\n'
        '  "action_items": ["List of the top 3-5 potential next steps or recommendations based on the content."]\n'
        "}\n"
    )
    
    user_prompt = (
        f"Document Chunk:\n```\n{chunk}\n```\n\n"
        f"Your Task:\n{focus_text}\n\n"
        "Please generate the JSON summary now."
    )
    
    # Configure generation parameters
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json"
    }
    
    # Create model instance
    gemini_model = genai.GenerativeModel(
        model_name=MODEL,
        generation_config=generation_config,
        system_instruction=system_prompt,
        # Add safety settings to be less restrictive, if needed
        safety_settings={
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
        }
    )
    
    try:
        # Generate response
        response = gemini_model.generate_content(user_prompt)
        
        # Check for blocking reasons or other issues
        if not response.candidates:
            block_reason = "Unknown"
            if response.prompt_feedback and hasattr(response.prompt_feedback, 'block_reason'):
                block_reason = response.prompt_feedback.block_reason.name
            print(f"DEBUG: Prompt was blocked or failed. Reason: {block_reason}")
            return {
                "executive_summary": f"Content generation blocked. Reason: {block_reason}.",
                "key_highlights": ["The AI's safety filters may have been triggered, or there was another issue with the request."],
                "main_sections": {}, "focus_area_insights": {}, "data_points": [], "action_items": []
            }

        response_text = ""
        candidate = response.candidates[0]
        
        # Check if generation finished for a non-standard reason
        finish_reason = getattr(candidate, 'finish_reason', "UNKNOWN").name
        if finish_reason != "STOP":
            print(f"DEBUG: Generation finished for a reason other than STOP: {finish_reason}")
            # Even if it stopped for another reason, there might still be partial content
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))
            
            if not response_text:
                return {
                    "executive_summary": "Summary generation stopped prematurely.",
                    "key_highlights": [f"Reason: {finish_reason}. This can happen with very long documents or due to safety filters."],
                    "main_sections": {}, "focus_area_insights": {}, "data_points": [], "action_items": []
                }
        else:
             if hasattr(candidate.content, 'parts') and candidate.content.parts:
                response_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text'))

        if response_text:
            try:
                # The response is now expected to be a clean JSON string
                summary_result = json.loads(response_text)
                return summary_result
            except json.JSONDecodeError as e:
                print(f"DEBUG: JSON parse error in summarization: {e}")
                return {
                    "executive_summary": "Error parsing summary response from AI.",
                    "key_highlights": ["The AI returned a summary, but it was not in a valid JSON format."],
                    "main_sections": {}, "focus_area_insights": {}, "data_points": [], "action_items": [],
                    "_raw_response": response_text[:1000]
                }
        else:
            return {
                "executive_summary": "No response was generated by the AI.",
                "key_highlights": ["This could be due to a network issue, an invalid API key, or a problem with the AI service."],
                "main_sections": {}, "focus_area_insights": {}, "data_points": [], "action_items": []
            }
            
    except Exception as e:
        print(f"DEBUG: An unexpected error occurred in summarization: {e}")
        import traceback
        print(traceback.format_exc())
        return {
            "executive_summary": f"An unexpected error occurred: {str(e)}",
            "key_highlights": ["Summary generation failed due to a system error."],
            "main_sections": {}, "focus_area_insights": {}, "data_points": [], "action_items": []
        }


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="LLM screening + marketing helper")
    sub = p.add_subparsers(dest="mode", required=True)

    # Screen sub-command
    ps = sub.add_parser("screen", help="Compliance screening")
    ps.add_argument("file", help="PDF/DOCX/XLSX to screen")
    ps.add_argument("--criteria_file", default="config/criteria.json",
                    help="YAML or JSON file containing list of criteria")
    ps.add_argument("--out", default="verdict.json", help="Output JSON")

    # Blurb sub-command
    pb = sub.add_parser("blurb", help="Generate marketing blurb")
    pb.add_argument("verdict_json", help="JSON file from the screen step")
    pb.add_argument("--temp", type=float, default=0.6,
                    help="Temperature for creativity (0.5-0.7 recommended)")

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "screen":
        # Load criteria
        with open(args.criteria_file, "r", encoding="utf-8") as f:
            if args.criteria_file.endswith((".yml", ".yaml")):
                import yaml
                criteria = yaml.safe_load(f)
            else:
                criteria = json.load(f)

        # Extract and chunk text
        raw = extract_text(args.file)
        chunks = textwrap.wrap(raw, CHUNK_SIZE_CHARS)
        verdicts = {f"chunk_{i}": llm_screen(c, criteria) for i, c in enumerate(chunks, 1)}

        # Save
        with open(args.out, "w") as f:
            json.dump(verdicts, f, indent=2)
        print(f"Saved screening verdict → {args.out}")

    elif args.mode == "blurb":
        with open(args.verdict_json, encoding="utf-8") as f:
            verdict = json.load(f)
        blurb = llm_blurb(verdict, temperature=args.temp)
        print("\n" + blurb + "\n")


if __name__ == "__main__":
    main() 