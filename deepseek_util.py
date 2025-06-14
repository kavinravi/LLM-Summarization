#!/usr/bin/env python
# deepseek_util.py
"""
Utility for:
1.  Screening PDFs/DOCX/XLSX against yes/no criteria (temperature = 0).
2.  Generating a marketing blurb from the screening verdict (temperature ~0.6).

Usage examples
--------------
# Compliance screen:
python deepseek_util.py screen my_memo.pdf --criteria_file criteria.json

# Marketing blurb (uses prior JSON verdict):
python deepseek_util.py blurb verdict.json --temp 0.7
"""
# Python <3.10 compatibility for type-hints
from __future__ import annotations

import os, json, textwrap, argparse
import pdfplumber, docx, pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from search_tool import web_search  # NEW IMPORT
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

# Allow model names to be overridden via env for easy experimentation
FAST_MODEL = get_env_var("DEEPSEEK_FAST_MODEL", "deepseek-chat")      # low-latency model (8k context)
SLOW_MODEL = get_env_var("DEEPSEEK_SLOW_MODEL", "deepseek-reasoner")  # high-accuracy model (64k context)

client = OpenAI(
    api_key=get_env_var("OPENAI_API_KEY"),
    base_url=get_env_var("OPENAI_BASE_URL")
)

MAX_INPUT_TOKENS = 64000
# 6 000 chars ≈ 4.5-5 k tokens – safe for the chat model's 8k window
CHUNK_SIZE_CHARS = 6000


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


# ---------- DEEPSEEK CALLS ----------
def chat_complete(prompt: str, temperature: float, max_tokens: int = 2048, json_mode: bool = False, model: Optional[str] = None):
    params = {
        "model": model or SLOW_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        params["response_format"] = {"type": "json_object"}
    
    resp = client.chat.completions.create(**params)
    return resp.choices[0].message.content.strip()


def _web_search_function_spec():
    """Return OpenAI function spec for web_search."""
    return [
        {
            "name": "web_search",
            "description": "Search the public web when additional factual information is required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web."
                    }
                },
                "required": ["query"]
            },
        }
    ]


# ---------------- HYBRID SCREENING -----------------

def _screen_with_model(chunk: str, criteria: list[str], model: str, temperature: float, max_rounds: int) -> dict:
    """Internal helper that runs the full tool-calling loop with a given model."""

    system_prompt = (
        "You are a project-diligence analyst.\n"
        "You will be given a document chunk and a list of renewable-energy siting criteria.\n"
        "If the chunk lacks data needed to evaluate a criterion, you may call the `web_search` tool to look up the missing fact.\n"
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

    function_spec = _web_search_function_spec()

    for _ in range(max_rounds):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            functions=function_spec,
            temperature=temperature,
            max_tokens=4096,
        )

        msg = response.choices[0].message

        # If the model wants to call the web_search tool
        if getattr(msg, "tool_calls", None):
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tc.to_dict() for tc in msg.tool_calls],  # type: ignore
            })

            for tc in msg.tool_calls:  # type: ignore
                try:
                    arguments = json.loads(tc.arguments)
                    query = arguments["query"]
                except Exception:
                    query = tc.arguments if isinstance(tc.arguments, str) else str(tc.arguments)

                results_text = web_search(query)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,  # type: ignore
                    "name": tc.name,
                    "content": results_text,
                })
            # go to next round
            continue

        # Otherwise, try to parse the assistant content as JSON answer
        if msg.content:
            try:
                result = json.loads(msg.content)
                if isinstance(result, dict):
                    # Ensure every criterion is present; if missing, mark unknown
                    for crit in criteria:
                        result.setdefault(crit, {"verdict": "unknown", "reason": "Not mentioned"})
                    return result
            except json.JSONDecodeError:
                # fall through to retry
                pass

        # Append the assistant content and retry (may help the model)
        messages.append({"role": "assistant", "content": msg.content or ""})

    # Fallback if all rounds exhausted
    return {c: {"verdict": "unknown", "reason": "Could not evaluate"} for c in criteria}


def deepseek_screen(chunk: str, criteria: list[str], temperature: float = 0.0, max_rounds: int = 4) -> dict:
    """Hybrid strategy: try fast chat model first, fall back to reasoner for unknowns."""

    # 1) fast path
    fast_result = _screen_with_model(chunk, criteria, FAST_MODEL, temperature, max_rounds)

    # If every verdict is yes or no, return immediately
    unknowns = [v for v in fast_result.values() if v.get("verdict") not in ("yes", "no")]
    if not unknowns:
        return fast_result

    # 2) slow path for undecided criteria
    slow_result = _screen_with_model(chunk, criteria, SLOW_MODEL, temperature, max_rounds)

    # Merge – prefer fast verdicts unless they are unknown/error
    merged = {}
    for crit in criteria:
        merged[crit] = fast_result.get(crit) or {"verdict": "unknown", "reason": ""}
        if merged[crit].get("verdict") not in ("yes", "no"):
            merged[crit] = slow_result.get(crit, merged[crit])
    return merged


def deepseek_blurb(verdict_json: dict, temperature: float = 0.6) -> str:
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
    p = argparse.ArgumentParser(description="DeepSeek screening + marketing helper")
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
        verdicts = {f"chunk_{i}": deepseek_screen(c, criteria) for i, c in enumerate(chunks, 1)}

        # Save
        with open(args.out, "w") as f:
            json.dump(verdicts, f, indent=2)
        print(f"Saved screening verdict → {args.out}")

    elif args.mode == "blurb":
        with open(args.verdict_json, encoding="utf-8") as f:
            verdict = json.load(f)
        blurb = deepseek_blurb(verdict, temperature=args.temp)
        print("\n" + blurb + "\n")


if __name__ == "__main__":
    main()
