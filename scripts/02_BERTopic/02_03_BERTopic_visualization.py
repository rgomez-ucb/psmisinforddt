# This script is to conduct BERTopic visualization on Reddit data
# Make bar chart to visualize significant topic coefficients from regression results

import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Set working directory
current_directory = os.getcwd()
print(f"Current working directory (os.getcwd()): {current_directory}")
new_directory_path = "/Users/mshun/Desktop/class_project"  # Change this to your target directory
os.chdir(new_directory_path)

# Load regression results
results_csv_path = "regression_coefficients_with_pvalue.csv"
df_results = pd.read_csv(results_csv_path, index_col=0)

# Assing raw and column names for clarity
COMPOUND_INDEX_NAME = 'vader_compound'
COEF_COL = 'Coef.'
P_VALUE_COL = 'P>|t|'

df_final_coef = df_results[
    (df_results.index.str.startswith('Topic_') & df_results.index.str.endswith('[T.True]')) |
    (df_results.index == COMPOUND_INDEX_NAME) 
].copy()

# Filter for statistically significant results (p < 0.05)
df_sig = df_final_coef[df_final_coef[P_VALUE_COL] < 0.05].copy()
print("✅ Completed filtering significant variables (topics/sentiments) with p < 0.05")
print(f"Number of significant variables (topics/sentiments) extracted: {len(df_sig)}")

# Output the significant topics with their coefficients and p-values
df_sig_sorted = df_sig.sort_values(by=COEF_COL, ascending=True)

print("\n--- Final Presentation: Significant Topics List (Sorted by Impact) ---")
print(df_sig_sorted[[COEF_COL, P_VALUE_COL]])

# Save the significant topics to CSV
df_sig_sorted.to_csv("significant_topics_percent_change.csv", index=True)
print("✅ Significant topics saved to CSV: significant_topics_percent_change.csv")

# Create a function to convert log-transformed coefficients to percent change
def convert_log_coef_to_percent_change(df_data, coef_col=COEF_COL):
    percent_change = (np.exp(df_data[coef_col]) - 1) * 100
    return percent_change

# Apply the conversion function to significant topics
df_sig_sorted['Percent_Change'] = convert_log_coef_to_percent_change(df_sig_sorted, coef_col=COEF_COL)

# Extract indices of significant topics for labeling
SIGNIFICANT_INDICES = [
    'compound', 
    'Topic_3[T.True]', 'Topic_6[T.True]','Topic_10[T.True]','Topic_17[T.True]','Topic_29[T.True]', 
    'Topic_1[T.True]', 'Topic_7[T.True]', 'Topic_13[T.True]', 'Topic_15[T.True]','Topic_16[T.True]','Topic_25[T.True]'
]

# Define mapping for better topic names
TOPIC_MAPPING = {
    'compound': 'Universal Sentiment Score',
    "Topic_3[T.True]": 'Topic 3: Investigation',
    'Topic_6[T.True]': 'Topic 6: Abortion',
    'Topic_10[T.True]': 'Topic 10: Supreme Court',
    'Topic_17[T.True]': 'Topic 17: Student Debt',
    'Topic_29[T.True]': 'Topic 29: Crime & Policing',
    'Topic_1[T.True]': 'Topic 1: Fiscal Policy',
    'Topic_7[T.True]': 'Topic 7: Gun Control',
    'Topic_13[T.True]': 'Topic 13: International Relations',
    'Topic_15[T.True]': 'Topic 15: Figurehead Rivalry',
    'Topic_16[T.True]': 'Topic 16: Expired Election',
    'Topic_25[T.True]': 'Topic 25: Israel/Conflict',
}

# Map topic names for better readability
df_plot = df_sig_sorted[df_sig_sorted.index.isin(SIGNIFICANT_INDICES)].copy()

# Add display names for plotting
df_plot['Display_Name'] = df_plot.index.map(TOPIC_MAPPING)

# Sort the final dataframe for plotting
df_plot_final = df_plot.sort_values(by='Percent_Change', ascending=True)

# Visualization: Bar chart of significant topics
print("=== Generating Bar Chart for Topic Coefficients ===")

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 6))

# Set bar colors based on positive or negative impact
colors = ['royalblue' if c > 0 else 'firebrick' for c in df_plot_final['Percent_Change']]

# Draw horizontal bar chart
ax.barh(
    df_plot_final['Display_Name'], 
    df_plot_final['Percent_Change'], 
    color=colors
)

# Add vertical line at x=0
ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)

# Set title and labels
ax.set_title('Impact of Context on Upvotes: Promotion and Suppression (p < 0.05)', fontsize=16)
ax.set_xlabel('Percent Change in Upvotes (%)', fontsize=12)
ax.set_ylabel(None)

plt.tight_layout()

# Save for presentation
plt.savefig('topic_engagement_contrast_final.png', dpi=300)
print("✅ Final graph saved: topic_engagement_contrast_final.png")

plt.show() 