# This script is to conduct BERTopic visualization on Reddit data

import sys
print("--- print the path ---")
print(sys.executable)
print("-------------------------------------")

# Import necessary libraries
import pandas as pd
import re
import os
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Set working directory
current_directory = os.getcwd()
print(f"Current working directory (os.getcwd()): {current_directory}")
new_directory_path = "/Users/mshun/Desktop/class_project"  # Change this to your target directory
os.chdir(new_directory_path)

# Create function to extract topic coeffiecients and p-values from regression results
def parse_regression_summary(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    start_index = -1
    end_index = -1
    for i, line in enumerate(lines):
        if '==================================================================' in line and start_index == -1:
            start_index = i + 2 
        elif start_index != -1 and '==================================================================' in line:
            end_index = i
            break

    if start_index == -1 or end_index == -1:
        print("Error: Could not find regression results in the file.")
        return pd.DataFrame()
    
    # Extract relevant lines
    topic_lines = []
    for line in lines[start_index:end_index]:
        # Check if the line contains topic coefficients
        if line.strip().startswith('Topic_'):
            topic_lines.append(line.strip())

    data = []
    pattern = re.compile(r'(\S+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.]+)')
    
    for line in topic_lines:
        match = pattern.search(line)
        if match:
            row = [match.group(1), float(match.group(2)), float(match.group(3)), float(match.group(4)), float(match.group(5))]
            data.append(row)
    
    # Output DataFrame
    df = pd.DataFrame(data, columns=['Variable', 'Coefficient', 'Std.Err.', 't_value', 'p_value'])
    df = df.set_index('Variable')
    
    return df

FILE_PATH = 'bertopic_regression_summary.txt' 
results_df = parse_regression_summary(FILE_PATH)

# Convert columns to numeric types
results_df['Coefficient'] = pd.to_numeric(results_df['Coefficient'])
results_df['p_value'] = pd.to_numeric(results_df['p_value'])
results_df['Std.Err.'] = pd.to_numeric(results_df['Std.Err.'])




# Visualize significant topic coefficients
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Extract only significant topics
# Change colors based on coefficient sign
topic_results = results_df[results_df.index.str.startswith('Topic_')]

print("\n---HEAD---")
print(topic_results[['Coefficient', 'p_value']].head())

topic_results_sig = topic_results[topic_results['p_value'] < 0.05].copy()
colors = ['firebrick' if c < 0 else 'royalblue' for c in topic_results_sig['Coefficient']]

# Check number of significant topics
print(f"Number of significant topics to be plotted: {len(topic_results_sig)}")

# Plot horizontal bar chart
topic_results_sig['Coefficient'].plot(
    kind='barh', 
    ax=ax, 
    color=colors,
    xerr=topic_results_sig['Std.Err.'] 
)

# Add zero line (boundary line)
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

# Set title and labels
ax.set_title('Impact of Context on Upvotes (Log-Transformed) - Significant Topics', fontsize=14)
ax.set_xlabel('Regression Coefficient (log(Upvotes) Effect)', fontsize=12)
ax.set_ylabel('Political Topic (Context)', fontsize=12)

# Adjust and display the plot
plt.tight_layout()
plt.show()

# Save as PNG for use in slides
plt.savefig('significant_topic_coefficients.png')
print("Saved significant topic coefficients graph: significant_topic_coefficients.png")