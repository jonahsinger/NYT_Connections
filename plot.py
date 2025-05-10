"""
Plot analysis of Connections results

This script reads connections_results.csv and generates visualizations
to analyze model performance by model type and prompt strategy.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from datetime import datetime

# Configuration
RESULTS_PATH = "connections_results.csv"
OUTPUT_DIR = "plots"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Read the data
print(f"Reading data from {RESULTS_PATH}...")
if not os.path.exists(RESULTS_PATH):
    print(f"Error: {RESULTS_PATH} not found. Run main.py first to generate results.")
    exit(1)

df = pd.read_csv(RESULTS_PATH)

# Basic data cleaning
df['Correct'] = df['Correct'].map({'Yes': True, 'No': False})
df['Parse_Error'] = df['Parse_Error'].map({'Yes': True, 'No': False})
df['Accuracy'] = pd.to_numeric(df['Accuracy'])

# Handle optional columns
if 'Token_Count' in df.columns:
    df['Token_Count'] = pd.to_numeric(df['Token_Count'])
else:
    print("Note: 'Token_Count' column not found in data")
    
if 'Thinking_Time_Sec' in df.columns:
    df['Thinking_Time_Sec'] = pd.to_numeric(df['Thinking_Time_Sec'])
else:
    print("Note: 'Thinking_Time_Sec' column not found in data")

# Print basic statistics
print("\nBasic Statistics:")
print(f"Total records: {len(df)}")
model_types = df['Model'].unique()
print(f"Models found in data: {', '.join(model_types)}")
print(f"Prompt types: {', '.join(df['Prompt_Type'].unique())}")
print(f"Puzzle dates: {len(df['Puzzle_Date'].unique())}")
print(f"Overall accuracy: {df['Accuracy'].mean():.2f}/4 groups correct")
print(f"Overall correct solutions: {df['Correct'].sum()}/{len(df)} ({df['Correct'].mean()*100:.1f}%)")

# Add model performance comparison
if 'o3-mini' in model_types and 'o3' in model_types:
    print("\n=== Model Performance Comparison ===")
    o3_mini_acc = df[df['Model'] == 'o3-mini']['Accuracy'].mean()
    o3_acc = df[df['Model'] == 'o3']['Accuracy'].mean()
    o3_mini_count = len(df[df['Model'] == 'o3-mini'])
    o3_count = len(df[df['Model'] == 'o3'])
    print(f"o3-mini: {o3_mini_acc:.2f}/4 groups correct (from {o3_mini_count} samples)")
    print(f"o3:      {o3_acc:.2f}/4 groups correct (from {o3_count} samples)")
    print(f"Difference: {abs(o3_mini_acc - o3_acc):.2f} groups ({max(o3_mini_acc, o3_acc):.2f} vs {min(o3_mini_acc, o3_acc):.2f})")
    if o3_mini_acc > o3_acc:
        print("o3-mini is performing better")
    elif o3_acc > o3_mini_acc:
        print("o3 is performing better")
    else:
        print("Both models are performing equally")

# Calculate and print average accuracy for each model-prompt combination
print("\n=== Average Groups Correct by Model & Prompt Type ===")
# Count samples for each combination
model_prompt_counts = df.groupby(['Model', 'Prompt_Type']).size().reset_index(name='Count')
# Calculate average accuracy
model_prompt_accuracy = df.groupby(['Model', 'Prompt_Type'])['Accuracy'].mean().reset_index()
model_prompt_accuracy['Accuracy'] = model_prompt_accuracy['Accuracy'].round(2)
# Merge count and accuracy
model_prompt_stats = pd.merge(model_prompt_accuracy, model_prompt_counts, on=['Model', 'Prompt_Type'])
# Sort by accuracy (descending)
model_prompt_stats = model_prompt_stats.sort_values('Accuracy', ascending=False)

# Print the results in a clean format
print(f"{'Model':<15} {'Prompt Type':<15} {'Avg. Groups Correct':<20} {'Sample Size':<15}")
print("-" * 65)
for _, row in model_prompt_stats.iterrows():
    print(f"{row['Model']:<15} {row['Prompt_Type']:<15} {row['Accuracy']:<20.2f} {row['Count']:<15}")

# Also save to a CSV file
accuracy_file = f"{OUTPUT_DIR}/model_prompt_accuracy_{timestamp}.csv"
model_prompt_stats.to_csv(accuracy_file, index=False)
print(f"\nSaved detailed accuracy table to: {accuracy_file}")

# Generate timestamp for file names
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Accuracy by Prompt Type
plt.figure(figsize=(10, 6))
sns.barplot(x='Prompt_Type', y='Accuracy', data=df)
plt.title('Average Accuracy by Prompt Type')
plt.ylabel('Average Accuracy (0-4 groups)')
plt.ylim(0, 4)
plt.tight_layout()
filename = f"{OUTPUT_DIR}/accuracy_by_prompt_type_{timestamp}.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 2. Accuracy by Model
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=df)
plt.title('Average Accuracy by Model')
plt.ylabel('Average Accuracy (0-4 groups)')
plt.ylim(0, 4)
plt.tight_layout()
filename = f"{OUTPUT_DIR}/accuracy_by_model_{timestamp}.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 3. Correct Solutions % by Prompt Type
plt.figure(figsize=(10, 6))
correct_by_prompt = df.groupby('Prompt_Type')['Correct'].mean() * 100
sns.barplot(x=correct_by_prompt.index, y=correct_by_prompt.values)
plt.title('Percentage of Correct Solutions by Prompt Type')
plt.ylabel('Correct Solutions (%)')
plt.ylim(0, 100)
plt.tight_layout()
filename = f"{OUTPUT_DIR}/correct_solutions_by_prompt_type_{timestamp}.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 4. Token Usage by Prompt Type (if data available)
if 'Token_Count' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Prompt_Type', y='Token_Count', data=df)
    plt.title('Average Token Usage by Prompt Type')
    plt.ylabel('Average Token Count')
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/token_usage_by_prompt_type_{timestamp}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()
else:
    print("Skipping token usage plot (no data available)")

# 5. Thinking Time by Prompt Type (if data available)
if 'Thinking_Time_Sec' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Prompt_Type', y='Thinking_Time_Sec', data=df)
    plt.title('Average Thinking Time by Prompt Type')
    plt.ylabel('Average Time (seconds)')
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/thinking_time_by_prompt_type_{timestamp}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()
else:
    print("Skipping thinking time plot (no data available)")

# 6. Performance Heatmap (Model vs Prompt Type)
plt.figure(figsize=(12, 8))
heatmap_data = df.groupby(['Model', 'Prompt_Type'])['Accuracy'].mean().unstack()
sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', vmin=0, vmax=4, fmt='.2f')
plt.title('Average Accuracy by Model and Prompt Type')
plt.tight_layout()
filename = f"{OUTPUT_DIR}/heatmap_model_vs_prompt_{timestamp}.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# Additional heatmap for correct solution rate
plt.figure(figsize=(12, 8))
correct_heatmap = df.groupby(['Model', 'Prompt_Type'])['Correct'].mean().unstack() * 100
sns.heatmap(correct_heatmap, annot=True, cmap='RdYlGn', vmin=0, vmax=100, fmt='.1f')
plt.title('Percentage of Completely Correct Solutions by Model and Prompt Type')
plt.tight_layout()
filename = f"{OUTPUT_DIR}/heatmap_correct_solutions_{timestamp}.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 7. Combined Performance Metrics
has_token_data = 'Token_Count' in df.columns
has_time_data = 'Thinking_Time_Sec' in df.columns

if has_token_data or has_time_data:
    # Determine number of subplots needed
    num_metrics = 1  # Accuracy is always included
    if has_token_data:
        num_metrics += 1
    if has_time_data:
        num_metrics += 1
        
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 6))
    
    # If only one subplot, axes needs to be in a list
    if num_metrics == 1:
        axes = [axes]
    
    # Plot index
    plot_idx = 0
    
    # Always plot Accuracy
    sns.barplot(x='Prompt_Type', y='Accuracy', data=df, ax=axes[plot_idx])
    axes[plot_idx].set_title('Accuracy')
    axes[plot_idx].set_ylim(0, 4)
    plot_idx += 1
    
    # Plot Token Usage if available
    if has_token_data:
        sns.barplot(x='Prompt_Type', y='Token_Count', data=df, ax=axes[plot_idx])
        axes[plot_idx].set_title('Token Usage')
        plot_idx += 1
    
    # Plot Thinking Time if available
    if has_time_data:
        sns.barplot(x='Prompt_Type', y='Thinking_Time_Sec', data=df, ax=axes[plot_idx])
        axes[plot_idx].set_title('Thinking Time (s)')
    
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/combined_metrics_by_prompt_{timestamp}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()
else:
    # If neither token nor time data is available, just plot Accuracy
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Prompt_Type', y='Accuracy', data=df)
    plt.title('Accuracy by Prompt Type')
    plt.ylabel('Average Accuracy (0-4 groups)')
    plt.ylim(0, 4)
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/combined_metrics_accuracy_{timestamp}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

# 8. Parse Error Rates
plt.figure(figsize=(10, 6))
error_by_prompt = df.groupby('Prompt_Type')['Parse_Error'].mean() * 100
sns.barplot(x=error_by_prompt.index, y=error_by_prompt.values)
plt.title('Parse Error Rate by Prompt Type')
plt.ylabel('Error Rate (%)')
plt.ylim(0, 100)
plt.tight_layout()
filename = f"{OUTPUT_DIR}/parse_error_by_prompt_type_{timestamp}.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 9. Scatter Plot: Token Count vs Accuracy (if data available)
if 'Token_Count' in df.columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Token_Count', y='Accuracy', hue='Prompt_Type', style='Model', data=df)
    plt.title('Token Count vs Accuracy')
    plt.xlabel('Token Count')
    plt.ylabel('Accuracy (0-4 groups)')
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/token_vs_accuracy_{timestamp}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()
else:
    print("Skipping token vs accuracy scatter plot (no token data available)")

# Add direct model comparison by prompt type
if 'o3-mini' in model_types and 'o3' in model_types:
    # Direct model comparison for each prompt type
    plt.figure(figsize=(12, 6))
    model_prompt_pivot = pd.pivot_table(df, values='Accuracy', index='Prompt_Type', columns='Model')
    model_prompt_pivot.plot(kind='bar', ax=plt.gca())
    plt.title('o3 vs o3-mini Performance by Prompt Type')
    plt.ylabel('Average Accuracy (0-4 groups)')
    plt.ylim(0, 4)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model')
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/model_comparison_by_prompt_{timestamp}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()
    
    # Success rate comparison
    plt.figure(figsize=(12, 6))
    success_pivot = pd.pivot_table(df, values='Correct', index='Prompt_Type', columns='Model', aggfunc='mean') * 100
    success_pivot.plot(kind='bar', ax=plt.gca())
    plt.title('o3 vs o3-mini Success Rate by Prompt Type')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Model')
    plt.tight_layout()
    filename = f"{OUTPUT_DIR}/model_success_rate_comparison_{timestamp}.png"
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

print(f"\nAll plots have been saved to the '{OUTPUT_DIR}' directory.")
print("Use these visualizations to compare model and prompt performance.") 