#!/usr/bin/env python
# llm_util.py
"""
Utility for:
1.  Screening PDFs/DOCX/XLSX against yes/no criteria (temperature = 0).
2.  Generating a marketing blurb from the screening verdict (temperature ~0.6).

Usage examples
--------------
# Compliance screen:
python llm_util.py screen my_memo.pdf --criteria_file criteria.json

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
MODEL = get_env_var("GEMINI_MODEL", "gemini-2.5-flash")
print(f"DEBUG: Loaded MODEL = {MODEL}")
print(f"DEBUG: GEMINI_MODEL env var = {os.getenv('GEMINI_MODEL', 'NOT_SET')}")
print(f"DEBUG: .env loaded = {os.path.exists('.env')}")

# Configure Gemini
genai.configure(api_key=get_env_var("GOOGLE_API_KEY"))

MAX_INPUT_TOKENS = 2000000  # Gemini 2.0 Flash has 2M token context window
# 150,000 chars ≈ 100k tokens – much larger chunks possible with Gemini's 2M window
CHUNK_SIZE_CHARS = 150000


# ---------- FILE I/O ----------
def extract_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    if ext in (".docx", ".doc"):
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    if ext in (".xlsx", ".xls"):
        dfs = pd.read_excel(path, sheet_name=None)
        return "\n".join(df.to_csv(index=False) for df in dfs.values())
    raise ValueError(f"Unsupported file type: {ext}")


# ---------- LLM CALLS ----------
def chat_complete(prompt: str, temperature: float, max_tokens: int = 2048, json_mode: bool = False, model: Optional[str] = None):
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

def _screen_with_model(chunk: str, criteria: list[str], model: str, temperature: float, max_rounds: int) -> dict:
    """Internal helper that runs the full tool-calling loop with a given model."""

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
        "**IMPORTANT - Use Web Search:** When you find evidence that partially meets a criterion but is missing specific quantitative data, you MUST call the web_search function to find the missing information. Examples:\\n"
        "- If you find a facility name but not its capacity/size, search for: '[facility name] capacity' or '[facility name] specifications'\\n"
        "- If you find a measurement in one unit but need another, search for: '[value] [unit1] to [unit2] conversion'\\n"
        "- If you find a location but need specific data about it, search for: '[location] [specific data needed]'\\n"
        "- If you find an entity but need missing quantitative details, search for: '[entity name] [missing detail]'\\n"
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

    # Temporarily disable web search tool to debug safety issues
    web_search_tool_spec = _web_search_tool_spec()
    gemini_model = genai.GenerativeModel(
        model_name=model,
        tools=[web_search_tool_spec],
        system_instruction=system_prompt
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
            from search_tool import web_search
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
            # Send prompt (combine system and user message for Gemini)
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = chat.send_message(full_prompt)
            
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
                
                # Try to get response text safely
                try:
                    response_text = response.text or ""
                except (ValueError, AttributeError) as e:
                    debug_messages.append(f"DEBUG: Could not access response.text: {e}")
                    # Try to get text from parts directly
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                            # Check for function calls in parts
                            elif hasattr(part, 'function_call'):
                                debug_messages.append("DEBUG: Found function_call in response parts")
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
                                    
                                    # Process the new response
                                    try:
                                        response_text = new_response.text or ""
                                        debug_messages.append(f"DEBUG: Got response after web search, text length: {len(response_text)}")
                                    except (ValueError, AttributeError) as e:
                                        debug_messages.append(f"DEBUG: Could not access response.text after web search: {e}")
                                        response_text = ""
                                    break  # Exit the parts loop after handling function call
            
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
                elif reason_str == 'FUNCTION_CALL':
                    debug_messages.append("DEBUG: Model invoked web_search function")
                    # Handle function calling
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'function_call'):
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
                                        response = chat.send_message(function_response)
                                        
                                        # Process the new response
                                        try:
                                            response_text = response.text or ""
                                            debug_messages.append(f"DEBUG: Got response after web search, text length: {len(response_text)}")
                                        except (ValueError, AttributeError) as e:
                                            debug_messages.append(f"DEBUG: Could not access response.text after web search: {e}")
                                            response_text = ""
                else:
                    if not response_text:  # Only return error if we truly have no content
                        debug_messages.append(f"DEBUG: No content and finish reason: {reason_str}")
                        return {"_debug": debug_messages}
            
            # Try to parse JSON response
            if response_text:
                debug_messages.append("DEBUG: Attempting to parse JSON response")
                debug_messages.append(f"DEBUG: Full response content: {repr(response_text[:1000])}")
                
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    try:
                        result = json.loads(json_str)
                        debug_messages.append(f"DEBUG: JSON parsing successful, type: {type(result)}")
                        if isinstance(result, dict):
                            debug_messages.append(f"DEBUG: Successfully parsed JSON with {len(result)} items")
                            debug_messages.append(f"DEBUG: JSON keys: {list(result.keys())}")
                            
                            # Check if any keys match our criteria
                            matching_criteria = [crit for crit in criteria if crit in result]
                            debug_messages.append(f"DEBUG: Matching criteria found: {len(matching_criteria)}")
                            
                            # Check for quantitative criteria that are unknown (might need web search)
                            quantitative_keywords = ['≥', '≤', '$', 'km', 'MW', 'kWh', '%', 'months']
                            unknown_quantitative = []
                            for crit, verdict_data in result.items():
                                if isinstance(verdict_data, dict) and verdict_data.get('verdict', '').lower() == 'unknown':
                                    if any(keyword in crit for keyword in quantitative_keywords):
                                        unknown_quantitative.append(crit)
                            
                            if unknown_quantitative:
                                debug_messages.append(f"DEBUG: Found {len(unknown_quantitative)} unknown quantitative criteria that could benefit from web search:")
                                for crit in unknown_quantitative[:3]:  # Show first 3
                                    debug_messages.append(f"DEBUG:   - {crit[:80]}...")
                            
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


def llm_screen(chunk: str, criteria: list[str], temperature: float = 0.0, max_rounds: int = 3) -> dict:
    """Single-model screening with web search capability."""
    
    # Use web search version directly to test the fixes
    return _screen_with_model(chunk, criteria, MODEL, temperature, max_rounds)


def llm_screen_no_web(chunk: str, criteria: list[str], temperature: float = 0.0) -> dict:
    """Document-only screening without web search capability."""
    
    system_prompt = (
        "You are a project-diligence analyst.\n"
        "You will be given a document chunk and a list of renewable-energy siting criteria.\n"
        "CRITICAL: Analyze the document chunk thoroughly for ALL relevant information.\n"
        "The document is an Environmental Assessment containing project-specific details, locations, measurements, environmental data, etc.\n"
        "Make reasonable inferences from the available information. If specific data is not explicitly stated but can be reasonably inferred, make that inference.\n"
        "When you have analyzed the document, return ONLY JSON with this exact structure (no markdown, no extra text):\n"
        "{\n  \"criterion name\": {\"verdict\": \"yes|no|unknown\", \"reason\": \"short explanation\"},\n  ...one object per criterion...\n}\n"
        "Every criterion listed below MUST appear once in the JSON, and the JSON **key must be copied verbatim from the list** (no re-phrasing or truncation).\n"
    )

    user_prompt = (
        f"Criteria:\n{json.dumps(criteria, indent=2)}\n\n"
        f"Document chunk:\n{chunk}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    try:
        # Use Gemini instead
        gemini_model = genai.GenerativeModel(
            model_name=MODEL,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 4096,
                "response_mime_type": "application/json"
            }
        )
        
        # Combine prompts for Gemini
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        response = gemini_model.generate_content(full_prompt)
        
        if response.text:
            # Try to clean up the response if it has markdown formatting
            content = response.text.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                result = json.loads(content)
                if isinstance(result, dict):
                    # Ensure every criterion is present; if missing, mark unknown
                    for crit in criteria:
                        if crit not in result:
                            result[crit] = {"verdict": "unknown", "reason": "Not mentioned"}
                    return result
            except json.JSONDecodeError:
                pass

    except Exception:
        pass

    # Fallback
    return {c: {"verdict": "unknown", "reason": "Could not evaluate"} for c in criteria}





def llm_blurb(verdict_json: dict, temperature: float = 0.6) -> str:
    prompt = (
        "You are a copywriter for a renewable-energy SaaS startup.\n"
        "Write a concise (≤ 120 words) marketing blurb that:\n"
        "• Highlights the top PASS items from the screening result.\n"
        "• Soft-pedals any FAIL items as future mitigation steps.\n"
        "• Uses an optimistic, professional tone.\n\n"
        f"Screening result:\n{json.dumps(verdict_json, indent=2)}"
    )
    return chat_complete(prompt, temperature, max_tokens=256)


# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser(description="LLM screening + marketing helper")
    sub = p.add_subparsers(dest="mode", required=True)

    # Screen sub-command
    ps = sub.add_parser("screen", help="Compliance screening")
    ps.add_argument("file", help="PDF/DOCX/XLSX to screen")
    ps.add_argument("--criteria_file", default="criteria.json",
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