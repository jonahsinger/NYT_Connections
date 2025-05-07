"""
This script reads the new_puzzle_results.csv file and generates visualizations
comparing token usage, accuracy, and response time for different model+prompt combinations.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

# Configuration - UPDATED OUTPUT DIRECTORY
RESULTS_FILE = "new_puzzle_results.csv"
OUTPUT_DIR = "cur_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Load results data
print(f"Reading results from {RESULTS_FILE}...")
if not os.path.exists(RESULTS_FILE):
    print(f"Error: {RESULTS_FILE} not found. Run main.py first to generate results.")
    exit(1)

# Read and prepare data
df = pd.read_csv(RESULTS_FILE)

# Basic data cleaning and conversion
df['Correct'] = df['Correct'].map({'Yes': True, 'No': False})
df['Parse_Error'] = df['Parse_Error'].map({'Yes': True, 'No': False})
df['Accuracy'] = pd.to_numeric(df['Accuracy'])

# Ensure Token_Count and Thinking_Time_Sec are numeric
if 'Token_Count' in df.columns:
    df['Token_Count'] = pd.to_numeric(df['Token_Count'])
else:
    print("Warning: Token_Count column not found")
    df['Token_Count'] = 0

if 'Thinking_Time_Sec' in df.columns:
    df['Thinking_Time_Sec'] = pd.to_numeric(df['Thinking_Time_Sec'])
else:
    print("Warning: Thinking_Time_Sec column not found")
    df['Thinking_Time_Sec'] = 0

# Create Model+Prompt combination column for easier analysis
df['Model_Prompt'] = df['Model'] + ' + ' + df['Prompt_Type']

# Define consistent color mapping for prompt types
prompt_types = sorted(df['Prompt_Type'].unique())
prompt_colors = dict(zip(prompt_types, sns.color_palette("husl", len(prompt_types))))


def model_sort_key(model_name):
    if model_name == 'o3':
        return 'z' + model_name 
    return model_name

# Print basic statistics
print("\nBasic Statistics:")
print(f"Total records: {len(df)}")
print(f"Models: {', '.join(df['Model'].unique())}")
print(f"Prompt types: {', '.join(df['Prompt_Type'].unique())}")
print(f"Puzzle dates: {len(df['Puzzle_Date'].unique())}")
print(f"Overall accuracy: {df['Accuracy'].mean():.2f}/4 groups correct")
print(f"Overall correct solutions: {df['Correct'].sum()}/{len(df)} ({df['Correct'].mean()*100:.1f}%)")

# Generate performance table by model and prompt
model_prompt_stats = df.groupby(['Model', 'Prompt_Type']).agg({
    'Accuracy': 'mean',
    'Correct': 'mean',
    'Token_Count': 'mean',
    'Thinking_Time_Sec': 'mean',
    'Parse_Error': 'mean',
    'Model': 'count'
}).rename(columns={'Model': 'Count'}).reset_index()

# Format for display
model_prompt_stats['Accuracy'] = model_prompt_stats['Accuracy'].round(2)
model_prompt_stats['Correct'] = (model_prompt_stats['Correct'] * 100).round(1).astype(str) + '%'
model_prompt_stats['Token_Count'] = model_prompt_stats['Token_Count'].round(0).astype(int)
model_prompt_stats['Thinking_Time_Sec'] = model_prompt_stats['Thinking_Time_Sec'].round(2)
model_prompt_stats['Parse_Error'] = (model_prompt_stats['Parse_Error'] * 100).round(1).astype(str) + '%'

# Print performance table
print("\n=== Performance by Model and Prompt ===")
print(model_prompt_stats.to_string(index=False))

# Save to CSV
stats_file = f"{OUTPUT_DIR}/model_prompt_performance.csv"
model_prompt_stats.to_csv(stats_file, index=False)
print(f"\nSaved performance stats to: {stats_file}")

# Convert back to numeric for plotting
for col in ['Correct', 'Parse_Error']:
    model_prompt_stats[col] = model_prompt_stats[col].str.rstrip('%').astype(float)

# ----- VISUALIZATIONS -----

sorted_models = sorted(df['Model'].unique(), key=model_sort_key)

# 1. Accuracy by Model and Prompt Type
plt.figure(figsize=(12, 6))
g = sns.catplot(
    x='Model', y='Accuracy', hue='Prompt_Type', 
    data=df, kind='bar', height=6, aspect=2,
    palette=prompt_colors, order=sorted_models
)
g.fig.suptitle('Number of Groups Correct by Model and Prompt Type')
g.set(ylabel='Number of Groups Correct (0-4)', ylim=(0, 4))
g.ax.grid(axis='y', alpha=0.3)
g.ax.legend(title='Prompt Type')
filename = f"{OUTPUT_DIR}/accuracy_by_model_prompt.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 2. Success Rate by Model and Prompt Type
plt.figure(figsize=(12, 6))
success_data = df.groupby(['Model', 'Prompt_Type'])['Correct'].mean() * 100
success_data = success_data.reset_index()
g = sns.catplot(
    x='Model', y='Correct', hue='Prompt_Type', 
    data=success_data, kind='bar', height=6, aspect=2,
    palette=prompt_colors, order=sorted_models
)
g.fig.suptitle('First Try 100% Solve Rate by Model and Prompt Type')
g.set(ylabel='Success Rate (%)', ylim=(0, 100))
g.ax.grid(axis='y', alpha=0.3)
g.ax.legend(title='Prompt Type')
filename = f"{OUTPUT_DIR}/success_rate_by_model_prompt.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 3. Token Usage by Model and Prompt Type
plt.figure(figsize=(12, 6))
g = sns.catplot(
    x='Model', y='Token_Count', hue='Prompt_Type', 
    data=df, kind='bar', height=6, aspect=2,
    palette=prompt_colors, order=sorted_models
)
g.fig.suptitle('Average Token Usage by Model and Prompt Type')
g.set(ylabel='Average Token Count')
g.ax.grid(axis='y', alpha=0.3)
g.ax.legend(title='Prompt Type')
filename = f"{OUTPUT_DIR}/token_usage_by_model_prompt.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 4. Thinking Time by Model and Prompt Type
plt.figure(figsize=(12, 6))
g = sns.catplot(
    x='Model', y='Thinking_Time_Sec', hue='Prompt_Type', 
    data=df, kind='bar', height=6, aspect=2,
    palette=prompt_colors, order=sorted_models
)
g.fig.suptitle('Average Response Time by Model and Prompt Type')
g.set(ylabel='Average Time (seconds)')
g.ax.grid(axis='y', alpha=0.3)
g.ax.legend(title='Prompt Type')
filename = f"{OUTPUT_DIR}/response_time_by_model_prompt.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 5. Combined Performance Metrics
# Sort the Model_Prompt combinations to maintain order with o3 after o3-mini
df['model_for_sort'] = df['Model'].apply(model_sort_key)
df_sorted = df.sort_values(['model_for_sort', 'Prompt_Type'])
model_prompt_order = df_sorted['Model_Prompt'].unique()

metrics = ['Accuracy', 'Token_Count', 'Thinking_Time_Sec']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, metric in enumerate(metrics):
    sns.barplot(
        x='Model_Prompt', y=metric, 
        data=df, ax=axes[i], 
        order=model_prompt_order,
        hue='Prompt_Type', palette=prompt_colors, dodge=False
    )
    if metric == 'Accuracy':
        axes[i].set_title('Number of Groups Correct')
        axes[i].set_ylim(0, 4)
    else:
        axes[i].set_title(metric.replace('_', ' '))
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
    axes[i].grid(axis='y', alpha=0.3)
    axes[i].legend().set_visible(False)

plt.tight_layout()
filename = f"{OUTPUT_DIR}/combined_metrics.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 6. Heatmap: Model vs Prompt Type for Accuracy
plt.figure(figsize=(10, 8))
# Sort models for heatmap
model_order = sorted(df['Model'].unique(), key=model_sort_key)
heatmap_data = df.pivot_table(values='Accuracy', index='Model', columns='Prompt_Type')
heatmap_data = heatmap_data.reindex(model_order)  # Reorder rows
sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', vmin=0, vmax=4, fmt='.2f')
plt.title('Number of Groups Correct Heatmap by Model and Prompt Type')
plt.tight_layout()
filename = f"{OUTPUT_DIR}/accuracy_heatmap.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 7. Heatmap: Model vs Prompt Type for Token Usage
plt.figure(figsize=(10, 8))
token_heatmap = df.pivot_table(values='Token_Count', index='Model', columns='Prompt_Type')
token_heatmap = token_heatmap.reindex(model_order)  # Reorder rows
sns.heatmap(token_heatmap, annot=True, cmap='Blues', fmt='.0f')
plt.title('Token Usage Heatmap by Model and Prompt Type')
plt.tight_layout()
filename = f"{OUTPUT_DIR}/token_usage_heatmap.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 8. Efficiency Plot: Accuracy vs Token Usage
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='Token_Count', 
    y='Accuracy', 
    hue='Prompt_Type', 
    style='Model', 
    s=100, 
    data=df,
    palette=prompt_colors
)
plt.title('Efficiency: Accuracy vs Token Usage')
plt.xlabel('Token Count')
plt.ylabel('Accuracy (0-4 groups)')
plt.grid(alpha=0.3)
plt.tight_layout()
filename = f"{OUTPUT_DIR}/efficiency_plot.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 9. Bubble Chart: Accuracy vs Time vs Tokens
plt.figure(figsize=(12, 8))

agg_df = df.groupby(['Model', 'Prompt_Type']).agg({
    'Thinking_Time_Sec': 'mean',
    'Accuracy': 'mean',
    'Token_Count': 'mean'
}).reset_index()

# Create Model_Prompt column for the aggregated data
agg_df['Model_Prompt'] = agg_df['Model'] + ' + ' + agg_df['Prompt_Type']

# Sort by model to ensure consistent order
agg_df['model_for_sort'] = agg_df['Model'].apply(model_sort_key)
agg_df = agg_df.sort_values('model_for_sort')

scatter = plt.scatter(
    x=agg_df['Thinking_Time_Sec'],
    y=agg_df['Accuracy'],
    s=agg_df['Token_Count'] / 10,
    c=[prompt_colors[prompt] for prompt in agg_df['Prompt_Type']],
    alpha=0.7
)

# Add labels for each point
for i, row in agg_df.iterrows():
    plt.annotate(
        row['Model_Prompt'],
        (row['Thinking_Time_Sec'], row['Accuracy']),
        fontsize=9,
        xytext=(5, 5),
        textcoords='offset points'
    )

# Add a legend for prompt types
for prompt in prompt_colors:
    plt.scatter([], [], color=prompt_colors[prompt], label=prompt)
plt.legend(title='Prompt Type')

plt.title('Performance Bubble Chart')
plt.xlabel('Response Time (seconds)')
plt.ylabel('Accuracy (0-4 groups)')
plt.grid(alpha=0.3)
plt.tight_layout()
filename = f"{OUTPUT_DIR}/performance_bubble_chart.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# 10. Model Comparison
plt.figure(figsize=(14, 8))

model_comparison_data = df.groupby(['Model', 'Prompt_Type']).agg({
    'Accuracy': 'mean',
    'Correct': 'mean'
}).reset_index()

model_comparison_data['model_for_sort'] = model_comparison_data['Model'].apply(model_sort_key)
model_comparison_data = model_comparison_data.sort_values('model_for_sort')

g = sns.catplot(
    x='Prompt_Type', 
    y='Accuracy', 
    hue='Model', 
    data=model_comparison_data,
    kind='bar',
    height=8, aspect=1.75,
    hue_order=sorted_models
)
g.fig.suptitle('Model Comparison: Number of Groups Correct by Prompt Type', fontsize=16)
g.set(ylabel='Number of Groups Correct (0-4)', ylim=(0, 4))
g.set(xlabel='Prompt Type')
g.ax.set_xlabel('Prompt Type', fontsize=12)
g.ax.grid(axis='y', alpha=0.3)
g.ax.legend(title='Model')

filename = f"{OUTPUT_DIR}/model_comparison.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

# Success rate comparison
plt.figure(figsize=(14, 8))
g = sns.catplot(
    x='Prompt_Type', 
    y='Correct', 
    hue='Model', 
    data=model_comparison_data,
    kind='bar',
    height=8, aspect=1.75,
    hue_order=sorted_models
)
g.fig.suptitle('First Try 100% Solve Rate by Prompt Type', fontsize=16)
g.set(ylabel='Success Rate (proportion)', ylim=(0, 1))
g.set(xlabel='Prompt Type')
g.ax.set_xlabel('Prompt Type', fontsize=12)
g.ax.grid(axis='y', alpha=0.3)
g.ax.legend(title='Model')

filename = f"{OUTPUT_DIR}/success_rate_comparison.png"
plt.savefig(filename)
print(f"Saved: {filename}")
plt.close()

print(f"\nAll analysis complete! Results saved to the '{OUTPUT_DIR}' directory.")
print(f"Summary stats file: {stats_file}") 