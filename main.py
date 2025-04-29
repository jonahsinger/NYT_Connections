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
START_DATE         = pd.Timestamp("2025-01-05")
END_DATE           = pd.Timestamp("2025-01-05")
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


def build_prompt(words: list[str]) -> str:
    return (
        "You are given a NYT Connections puzzle with 16 words and must "
        "divide them into 4 related groups of 4.  Output **only** the 4 lines, "
        "exactly in this format, nothing else:\n"
        "group 1: w1, w2, w3, w4\n"
        "group 2: w1, w2, w3, w4\n"
        "group 3: w1, w2, w3, w4\n"
        "group 4: w1, w2, w3, w4\n\n"
        f"Here are the 16 words: {', '.join(words)}"
    )

# --------------------------------------------------
# 3.  MAIN LOOP
# --------------------------------------------------
solved = 0
for date in puzzle_dates:
    puzzle_df = filtered_df[filtered_df["Puzzle Date"] == date]
    words     = puzzle_df["Word"].tolist()
    answer    = gold_groups(puzzle_df)
    
    print(f"\n{'='*60}")
    print(f"PUZZLE DATE: {date.date()}")
    print(f"{'='*60}")
    print(f"WORDS: {', '.join(words)}")
    
    prompt = build_prompt(words)
    print(f"\nPrompt: {prompt}")
    
    try:
        print("\nGetting model's solution...")
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
    correct = parsed == answer
    solved += bool(correct)

    print(f"\nRESULT: {'✅ SOLVED' if correct else '❌ INCORRECT'}")
    print("\nMODEL REPLY:")
    print(llm_text)
    
    print("\nCORRECT GROUPS:")
    for i, g in enumerate(answer, 1):
        print(f"  Group {i}: {', '.join(sorted(g))}")
    
    if not correct and parsed:
        print("\nPARSED GROUPS (INCORRECT):")
        for i, g in enumerate(parsed, 1):
            print(f"  Group {i}: {', '.join(sorted(g))}")

    # Export results to CSV
    export_results(date, llm_text, correct)

# --------------------------------------------------
# 4.  SUMMARY
# --------------------------------------------------
total = len(puzzle_dates)
print(f"\n{'='*60}")
print(f"FINAL RESULTS: Solved {solved}/{total}  ({solved/total:.1%})")
print(f"{'='*60}")

def export_results(date, llm_text, correct):
    # Implement CSV export logic here
    pass
