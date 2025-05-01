"""
NYT-Connections solver (Python 3.9-compatible)

  • Robust date filtering with pandas.Timestamp
  • Compact regex parser for the model's four output lines
  • Works with OpenAI o3-mini model
"""

import os
import re
import json
import time
import random
import sys
import argparse
import csv
from datetime import datetime
from typing import Set, FrozenSet, Optional

import pandas as pd
from openai import OpenAI

# Parse command line arguments
parser = argparse.ArgumentParser(description='NYT Connections solver using OpenAI API')
parser.add_argument('--api-key', help='Your OpenAI API key')
args = parser.parse_args()

# --------------------------------------------------
# 0.  CONFIGURATION
# --------------------------------------------------
MODEL              = "o3"
START_DATE         = pd.Timestamp("2025-02-04")
END_DATE           = pd.Timestamp("2025-02-17")
CSV_PATH           = "Connections_Data.csv"
RESULTS_PATH       = "connections_results.csv"
# Get API key from command line, environment, or prompt user
OPENAI_API_KEY = args.api_key or os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("\n======================= IMPORTANT =======================")
    print("You need a valid OpenAI API key to use this script.")
    print("Get your API key from: https://platform.openai.com/account/api-keys")
    print("==========================================================\n")
    OPENAI_API_KEY = input("Please enter your OpenAI API key (starts with 'sk-'): ")

if OPENAI_API_KEY and not OPENAI_API_KEY.startswith('sk-'):
    print("\nWARNING: The API key you provided doesn't start with 'sk-', which is unusual for OpenAI API keys.")
    proceed = input("Do you want to proceed anyway? (y/n): ")
    if proceed.lower() != 'y':
        print("Exiting script.")
        exit()

DEBUG              = True  # Set to True for verbose output

# Initialize client
client = OpenAI(
    api_key=OPENAI_API_KEY
)

# --------------------------------------------------
# 1.  LOAD DATA
# --------------------------------------------------
print("Loading data from CSV...")
df = pd.read_csv(CSV_PATH, parse_dates=["Puzzle Date"])
mask = (df["Puzzle Date"] >= START_DATE) & (df["Puzzle Date"] <= END_DATE)
filtered_df = df.loc[mask]
puzzle_dates = sorted(filtered_df["Puzzle Date"].unique())

if not puzzle_dates:
    raise ValueError("No puzzles found in the requested date range.")

# --------------------------------------------------
# 2.  HELPERS
# --------------------------------------------------
def gold_groups(puzzle_day: pd.DataFrame) -> Set[FrozenSet[str]]:
    """Return the correct answer as a *set* of 4 frozensets (order-free)."""
    return {
        frozenset(words)
        for _, words in (
            puzzle_day.groupby("Group Level")["Word"]
            .apply(list)
            .sort_index()
            .items()
        )
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

def export_results(date, llm_text, correct, prompt_type, accuracy, parse_error=False):
    """
    Export results to CSV file.
    
    Args:
        date: Date of the puzzle
        llm_text: Raw response from the model
        correct: Whether the model got the answer correct
        prompt_type: Type of prompt used ('zero-shot', 'few-shot', or 'cot')
        accuracy: Number of groups correctly identified (0-4)
        parse_error: Whether there was a parsing error
    """
    # Check if file exists to determine if we need to write headers
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
                'Parse_Error'
            ])
        
        # Write result row
        writer.writerow([
            timestamp,
            date.strftime("%Y-%m-%d"),
            MODEL,
            prompt_type,
            'Yes' if correct else 'No',
            accuracy,
            'Yes' if parse_error else 'No'
        ])
    
    print(f"\nResults for {prompt_type} prompt exported to {RESULTS_PATH}")

# --------------------------------------------------
# 3.  MAIN LOOP
# --------------------------------------------------
solved = 0
total_attempts = 0
prompt_types = ['zero-shot', 'few-shot', 'cot']

for date in puzzle_dates:
    puzzle_df = filtered_df[filtered_df["Puzzle Date"] == date]
    words     = puzzle_df["Word"].tolist()
    answer    = gold_groups(puzzle_df)
    
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
            resp = client.chat.completions.create(
                model="o3",
                messages=[{"role": "user", "content": prompt}]
            )
            print("API call successful!")
            print(f"Raw response: {resp}")
            print(f"Response type: {type(resp)}")
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
            print(f"\nError: {e}")
            print(f"Error type: {type(e)}")
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
        for i, g in enumerate(answer, 1):
            print(f"  Group {i}: {', '.join(sorted(g))}")
        
        if not correct and parsed:
            print("\nPARSED GROUPS (INCORRECT):")
            for i, g in enumerate(parsed, 1):
                print(f"  Group {i}: {', '.join(sorted(g))}")

        # Export results to CSV for this prompt type
        export_results(date, llm_text, correct, prompt_type, accuracy, parse_error)

# --------------------------------------------------
# 4.  SUMMARY
# --------------------------------------------------
print(f"\n{'='*60}")
print(f"FINAL RESULTS: Solved {solved}/{total_attempts}  ({solved/total_attempts:.1%})")
print(f"{'='*60}")
