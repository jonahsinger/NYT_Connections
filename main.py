import os
import re
import json
import time
import random
import sys
import csv
from datetime import datetime
from typing import Set, FrozenSet, Optional
import pandas as pd
from openai import OpenAI

# --------------------------------------------------
# 0.  CONFIGURATION
# --------------------------------------------------

YOUR_API_KEY = "sk-your-api-key-here"
MODEL = "o3"  # "o3" or "o3-mini"
START_DATE = pd.Timestamp("2025-04-17")  # Format: YYYY-MM-DD
END_DATE = pd.Timestamp("2025-05-03")  # Format: YYYY-MM-DD

# File paths
JSON_PATH = "connections.json"  # New puzzles data set
RESULTS_PATH = "new_puzzle_results.csv"  # Output file for results

# Use the specified API key
OPENAI_API_KEY = YOUR_API_KEY
if not OPENAI_API_KEY or OPENAI_API_KEY == "sk-your-api-key-here":
    print("ERROR")
    print("API key not set")
    exit(1)

DEBUG = True

# Initialize client
client = OpenAI(api_key=OPENAI_API_KEY)

print(f"\nNYT Connections Solver (JSON Version)")
print(f"Configuration:")
print(f"  - Model: {MODEL}")
print(f"  - Date range: {START_DATE.date()} to {END_DATE.date()}")
print(f"  - API key: {OPENAI_API_KEY[:5]}...{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 8 else ''}")
print(f"  - Input file: {JSON_PATH}")
print(f"  - Results file: {RESULTS_PATH}")
print("-" * 60)

# --------------------------------------------------
# 1.  LOAD DATA
# --------------------------------------------------
print("Loading data from JSON...")
try:
    with open(JSON_PATH, 'r') as f:
        puzzles_data = json.load(f)
    print(f"Successfully loaded {len(puzzles_data)} puzzles from {JSON_PATH}")
    
    # Analyze available dates
    all_dates = [pd.Timestamp(puzzle['date']) for puzzle in puzzles_data]
    min_date = min(all_dates)
    max_date = max(all_dates)
    print(f"Date range in dataset: {min_date.date()} to {max_date.date()}")
    
    # Check if our selected date range is within the available dates
    if END_DATE < min_date or START_DATE > max_date:
        print(f"WARNING: Selected date range ({START_DATE.date()} to {END_DATE.date()}) is outside the available data range")
except Exception as e:
    print(f"Error loading JSON data: {e}")
    exit(1)

# ONLY process puzzles within the specified date range
puzzles = []
for puzzle in puzzles_data:
    puzzle_date = pd.Timestamp(puzzle['date'])
    if START_DATE <= puzzle_date <= END_DATE:
        # Extract all words from all groups to get the 16 words
        all_words = []
        for answer in puzzle['answers']:
            all_words.extend(answer['members'])
        
        # Store the puzzle with its date and words
        puzzles.append({
            'date': puzzle_date,
            'words': all_words,
            'answers': puzzle['answers']
        })

if not puzzles:
    print(f"No puzzles found in the date range {START_DATE.date()} to {END_DATE.date()}")
    exit(1)

print(f"Found {len(puzzles)} puzzles in the specified date range")

for puzzle in puzzles:
    print(f"Will process puzzle from {puzzle['date'].date()}")
    print(f"Total words: {len(puzzle['words'])}")
    print(f"Number of groups: {len(puzzle['answers'])}")

# --------------------------------------------------
# 2.  HELPERS
# --------------------------------------------------
def gold_groups(puzzle) -> Set[FrozenSet[str]]:
    """Return the correct answer as a *set* of 4 frozensets (order-free)."""
    return {
        frozenset(answer['members']) 
        for answer in puzzle['answers']
    }

GROUP_RE = re.compile(
    r"^\s*group\s*\d+\s*:\s*(.+)$",
    flags=re.IGNORECASE,
)

def parse_groups(text: str) -> Optional[Set[FrozenSet[str]]]:
    """Extract exactly four groups of four words from the model reply."""
    groups = []
    for line in text.splitlines():
        m = GROUP_RE.match(line.strip())
        if not m:
            continue
        words = [w.strip() for w in re.split(r"[;,]", m.group(1)) if w.strip()]
        if len(words) == 4:
            groups.append(frozenset(words))
    return set(groups) if len(groups) == 4 else None


def build_prompt(words: list[str], prompt_type='zero-shot') -> str:
    """
    Build a prompt based on the specified type.
    
    Args:
        words: List of words for the puzzle
        prompt_type: Type of prompt ('zero-shot', 'few-shot', or 'cot')
        
    Returns:
        Formatted prompt string
    """
    # Shuffle words to randomize order
    words_copy = words.copy()
    random.shuffle(words_copy)
    words_str = ', '.join(words_copy)
    
    if prompt_type == 'zero-shot':
        return (
            "You are given a Connections puzzle with 16 words and must divide them into 4 related groups of 4. "
            "Each puzzle has exactly one solution. Output **only** the 4 lines, exactly in this format, nothing else:\n"
            "group 1: w1, w2, w3, w4\n"
            "group 2: w1, w2, w3, w4\n"
            "group 3: w1, w2, w3, w4\n"
            "group 4: w1, w2, w3, w4\n"
            f"Here are the 16 words: {words_str}"
        )
    elif prompt_type == 'few-shot':
        return (
            "You are given a Connections puzzle with 16 words and must divide them into 4 related groups of 4. "
            "Each puzzle has exactly one solution.\n"
            "Here are some examples of correct groupings:\n"
            "Example 1 words: 'DART', 'HEM', 'PLEAT', 'SEAM', 'CAN', 'CURE', 'DRY', 'FREEZE', 'BITE', 'EDGE', 'PUNCH', 'SPICE', 'CONDO', 'HAW', 'HERO', 'LOO'\n"
            "Example 1 correct output:\n"
            "group 1: DART, HEM, PLEAT, SEAM\n"
            "group 2: CAN, CURE, DRY, FREEZE\n"
            "group 3: BITE, EDGE, PUNCH, SPICE\n"
            "group 4: CONDO, HAW, HERO, LOO\n"
            "Example 2 words: COLLECTIVE, COMMON, JOINT, MUTUAL, CLEAR, DRAIN, EMPTY, FLUSH, CIGARETTE, PENCIL, TICKET, TOE, AMERICAN, FEVER, LUCID, PIPE\n"
            "Example 2 correct output:\n"
            "group 1: COLLECTIVE, COMMON, JOINT, MUTUAL\n"
            "group 2: CLEAR, DRAIN, EMPTY, FLUSH\n"
            "group 3: CIGARETTE, PENCIL, TICKET, TOE\n"
            "group 4: AMERICAN, FEVER, LUCID, PIPE\n"
            "Output **only** the 4 lines, exactly in this format, nothing else:\n"
            "group 1: w1, w2, w3, w4\n"
            "group 2: w1, w2, w3, w4\n"
            "group 3: w1, w2, w3, w4\n"
            "group 4: w1, w2, w3, w4\n"
            f"Here are the 16 words: {words_str}"
        )
    elif prompt_type == 'cot':
        return (
            "You are given a Connections puzzle with 16 words and must divide them into 4 related groups of 4. "
            "Each puzzle has exactly one solution.\n"
            "List & Link: Write out the 16 items and note any obvious semantic ties.\n\n"
            "Group Obvious: Pull 2–4 items that clearly share a feature and tentatively group them.\n\n"
            "Label & Test: Name the group; ensure every member fits, else adjust.\n\n"
            "Remove & Repeat: Drop confirmed groups and repeat on the leftovers.\n\n"
            "Check: Confirm you end up with four non-overlapping groups of four.\n\n"
            "Output **only** the 4 lines, exactly in this format, nothing else:\n"
            "group 1: w1, w2, w3, w4\n"
            "group 2: w1, w2, w3, w4\n"
            "group 3: w1, w2, w3, w4\n"
            "group 4: w1, w2, w3, w4\n"
            f"Here are the 16 words: {words_str}"
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

def export_results(date, llm_text, correct, prompt_type, accuracy, parse_error=False, token_count=0, thinking_time=0):
    """
    Export results to CSV file.
    
    Args:
        date: Date of the puzzle
        llm_text: Raw response from the model
        correct: Whether the model got the answer correct
        prompt_type: Type of prompt used ('zero-shot', 'few-shot', or 'cot')
        accuracy: Number of groups correctly identified (0-4)
        parse_error: Whether there was a parsing error
        token_count: Total number of tokens used (prompt + completion)
        thinking_time: Time taken for API call (in seconds)
    """
    file_exists = os.path.isfile(RESULTS_PATH)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(RESULTS_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            writer.writerow([
                'Timestamp', 
                'Puzzle_Date', 
                'Model',
                'Prompt_Type',
                'Correct',
                'Accuracy',
                'Parse_Error',
                'Token_Count',
                'Thinking_Time_Sec'
            ])
        
        # Write result row
        writer.writerow([
            timestamp,
            date.strftime("%Y-%m-%d"),
            MODEL,
            prompt_type,
            'Yes' if correct else 'No',
            accuracy,
            'Yes' if parse_error else 'No',
            token_count,
            thinking_time
        ])
    
    print(f"\nResults for {prompt_type} prompt exported to {RESULTS_PATH}")

# --------------------------------------------------
# 3.  MAIN LOOP
# --------------------------------------------------
solved = 0
total_attempts = 0
prompt_types = ['zero-shot', 'few-shot', 'cot']

print(f"\nStarting to process {len(puzzles)} puzzles within date range {START_DATE.date()} to {END_DATE.date()}")

for puzzle in puzzles:
    date = puzzle['date']
    words = puzzle['words']
    answer = gold_groups(puzzle)
    
    print(f"\n{'='*60}")
    print(f"PUZZLE DATE: {date.date()}")
    print(f"{'='*60}")
    print(f"WORDS: {', '.join(words)}")
    
    # Try each prompt type
    for prompt_type in prompt_types:
        total_attempts += 1
        print(f"\n{'-'*30}")
        print(f"PROMPT TYPE: {prompt_type}")
        print(f"{'-'*30}")
        
        prompt = build_prompt(words, prompt_type)
        print(f"\nPrompt: {prompt}")
        
        try:
            print(f"\nGetting model's solution using {prompt_type} prompt...")
            start_time = time.time()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            end_time = time.time()
            thinking_time = end_time - start_time
            
            print("API call successful!")
            print(f"Raw response: {resp}")
            print(f"Response type: {type(resp)}")
            
            # Extract token usage
            token_count = 0
            if hasattr(resp, 'usage') and resp.usage:
                token_count = resp.usage.total_tokens
                print(f"Token usage: {token_count} total tokens")
                if hasattr(resp.usage, 'prompt_tokens'):
                    print(f"  - Prompt tokens: {resp.usage.prompt_tokens}")
                if hasattr(resp.usage, 'completion_tokens'):
                    print(f"  - Completion tokens: {resp.usage.completion_tokens}")
            else:
                print("No token usage information available in response")
            
            print(f"Thinking time: {thinking_time:.2f} seconds")
            
            if hasattr(resp, 'choices') and resp.choices:
                print(f"Choices: {resp.choices}")
                if hasattr(resp.choices[0], 'message'):
                    print(f"Message: {resp.choices[0].message}")
                    llm_text = resp.choices[0].message.content.strip()
                else:
                    print("No message attribute in first choice")
                    llm_text = "Error: No message in response"
            else:
                print("No choices in response")
                llm_text = "Error: No choices in response"
        except Exception as e:
            end_time = time.time()
            thinking_time = end_time - start_time
            token_count = 0
            print(f"\nError: {e}")
            print(f"Error type: {type(e)}")
            print(f"Thinking time before error: {thinking_time:.2f} seconds")
            llm_text = f"Error: {str(e)}"

        parsed = parse_groups(llm_text)
        correct = False
        accuracy = 0
        parse_error = False
        
        if parsed is None:
            # Could not extract 4 groups of 4 words
            parse_error = True
            print("\nPARSING ERROR: Could not extract 4 groups of 4 words from the model's response")
        else:
            # Check if the answer is correct
            correct = parsed == answer
            if correct:
                solved += 1
                
            # Calculate accuracy (number of correctly identified groups)
            if parsed and answer:
                for group in parsed:
                    if group in answer:
                        accuracy += 1

        print(f"\nRESULT ({prompt_type}): {'✅ SOLVED' if correct else '❌ INCORRECT'}{' (PARSING ERROR)' if parse_error else ''}")
        print("\nMODEL REPLY:")
        print(llm_text)
        
        print("\nCORRECT GROUPS:")
        group_number = 1
        for answer_group in puzzle['answers']:
            print(f"  Group {group_number} ({answer_group['group']}): {', '.join(sorted(answer_group['members']))}")
            group_number += 1
        
        if not correct and parsed:
            print("\nPARSED GROUPS (INCORRECT):")
            for i, g in enumerate(parsed, 1):
                print(f"  Group {i}: {', '.join(sorted(g))}")

        # Export results to CSV for this prompt type
        export_results(date, llm_text, correct, prompt_type, accuracy, parse_error, token_count, thinking_time)

# --------------------------------------------------
# 4.  SUMMARY
# --------------------------------------------------
print(f"\n{'='*60}")
print(f"FINAL RESULTS: Solved {solved}/{total_attempts}  ({solved/total_attempts:.1%})")
print(f"Results saved to: {RESULTS_PATH}")
print(f"{'='*60}")
