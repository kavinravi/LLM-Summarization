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
MODEL = get_env_var("GEMINI_MODEL", "gemini-2.0-flash-exp")

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
    """Return Gemini function spec for web_search."""
    return {
        "name": "web_search", 
        "description": "Search the public web ONLY when absolutely critical information is completely missing from the document. Use sparingly.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web."
                }
            },
            "required": ["query"]
        }
    }


# ---------------- LLM SCREENING -----------------

def _screen_with_model(chunk: str, criteria: list[str], model: str, temperature: float, max_rounds: int) -> dict:
    """Internal helper that runs the full tool-calling loop with a given model."""

    system_prompt = (
        "You are a project-diligence analyst.\n"
        "You will be given a document chunk and a list of renewable-energy siting criteria.\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. FIRST: Thoroughly analyze the document chunk for ALL relevant information - project locations, distances, measurements, environmental data, etc.\n"
        "2. Look for indirect clues and inferences you can make from the document content.\n"
        "3. ONLY use web_search as a last resort if absolutely critical information is completely missing from the document.\n"
        "4. The document is an Environmental Assessment and likely contains most technical details needed for evaluation.\n"
        "5. If you can make a reasonable determination from the document content, do NOT search the web.\n"
        "When you have enough info, return ONLY JSON with this exact structure (no markdown, no extra text):\n"
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

    tools = _web_search_tool_spec()

    # For Streamlit debugging
    debug_messages = []
    debug_messages.append(f"DEBUG: Starting screening with {len(criteria)} criteria")
    debug_messages.append(f"DEBUG: Criteria: {criteria}")
    debug_messages.append(f"DEBUG: Chunk length: {len(chunk)} chars")
    debug_messages.append(f"DEBUG: Chunk preview (first 500 chars): {repr(chunk[:500])}")
    debug_messages.append(f"DEBUG: Model: {model}, Temperature: {temperature}, Max rounds: {max_rounds}")
    
    for round_num in range(max_rounds):
        debug_msg = f"DEBUG: Round {round_num + 1}/{max_rounds}"
        debug_messages.append(debug_msg)
        
        try:
            # For now, simplify to basic Gemini call without tools
            # Create model instance  
            gemini_model = genai.GenerativeModel(
                model_name=model,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": 4096,
                }
            )
            
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = gemini_model.generate_content(full_prompt)
            
            # Create a mock msg object for compatibility
            class MockMessage:
                def __init__(self, content):
                    self.content = content
            
            msg = MockMessage(response.text)
            debug_messages.append(f"DEBUG: Got response, content length: {len(msg.content or '')}")
            debug_messages.append(f"DEBUG: Has tool calls: {bool(getattr(msg, 'tool_calls', None))}")

            # If the model wants to call the web_search tool
            if getattr(msg, "tool_calls", None):
                debug_messages.append(f"DEBUG: Processing {len(msg.tool_calls)} tool calls")
                messages.append({
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                })

                for tc in msg.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                        query = arguments["query"]
                        debug_messages.append(f"DEBUG: Web search query: {query}")
                    except Exception as e:
                        debug_messages.append(f"DEBUG: Error parsing arguments: {e}")
                        query = tc.function.arguments if isinstance(tc.function.arguments, str) else str(tc.function.arguments)

                    # Import web_search locally to avoid import issues
                    try:
                        from search_tool import web_search
                        results_text = web_search(query)
                    except ImportError as e:
                        debug_messages.append(f"DEBUG: Could not import web_search: {e}")
                        results_text = f"Web search unavailable for query: {query}"
                    
                    debug_messages.append(f"DEBUG: Search results length: {len(results_text)}")
                    debug_messages.append(f"DEBUG: Search results preview: {repr(results_text[:200])}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": results_text,
                    })
                # go to next round
                continue

            # Otherwise, try to parse the assistant content as JSON answer
            if msg.content:
                debug_messages.append(f"DEBUG: Attempting to parse JSON response")
                debug_messages.append(f"DEBUG: Full response content: {repr(msg.content)}")
                
                # Try to clean up the response if it has markdown formatting
                content = msg.content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                debug_messages.append(f"DEBUG: Cleaned content: {repr(content)}")
                
                try:
                    result = json.loads(content)
                    debug_messages.append(f"DEBUG: JSON parsing successful, type: {type(result)}")
                    if isinstance(result, dict):
                        debug_messages.append(f"DEBUG: Successfully parsed JSON with {len(result)} items")
                        debug_messages.append(f"DEBUG: JSON keys: {list(result.keys())}")
                        
                        # Check if any keys match our criteria
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
                    debug_messages.append(f"DEBUG: Problematic content: {repr(content[:500])}")
                    # fall through to retry
                    pass
            else:
                debug_messages.append(f"DEBUG: No content in message")

            # Append the assistant content and retry (may help the model)
            messages.append({"role": "assistant", "content": msg.content or ""})
            
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