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

# Create a function to convert log-transformed coefficients to percent change
def convert_log_coef_to_percent_change(df_data, coef_col='Coefficient'):
    percent_change = (np.exp(df_data[coef_col]) - 1) * 100
    return percent_change

# Prapare the dataset for plotting
data_for_plot = {
    'Variable': ['vader_compound', 'Topic_7 (Abortion)', 'Topic_9 (Gun Control)', 'Topic_13 (International Relations)', "Topic_15(2016 Election)", "Topic_16(Student Loans)"],
    'Coefficient': [5.8539, 0.1057, -0.2242, -0.1448, -0.1664, 0.1644], 
    'Std.Err.': [0.018, 0.044, 0.048, 0.057 ,0.061, 0.067] 
}
df_plot = pd.DataFrame(data_for_plot).set_index('Variable')


# Apply the conversion function
df_plot['Percent_Change'] = convert_log_coef_to_percent_change(
    df_plot, 
    coef_col='Coefficient'
)

# Print the final results
print("--- Final Results: Percent Change in Upvotes (%) ---")
print(df_plot[['Coefficient', 'Percent_Change']])

print("=== Generating Bar Chart for Topic Coefficients ===")

data_for_graph = {
    'Topic': [
        'Topic 7: Abortion', 
        'Topic 9: Gun Control', 
        'Topic 13: Intl. Relations',
        "Topic 15: 2016 Election",
        "Topic 16: Student Loans"
    ],
    'Percent_Change': [11.148838, -20.084471, -13.480469, -15.329252, 17.868570], 
    'Effect': ['Promotion', 'Suppression', 'Suppression', 'Suppression', 'Promotion']
}
df_plot_final = pd.DataFrame(data_for_graph).set_index('Topic')


# Sort by absolute value of impact (strongest impact at the top)
df_plot_final = df_plot_final.sort_values(
    by='Percent_Change', 
    ascending=False
).copy()

# Set colors based on effect
colors = ['royalblue' if c > 0 else 'firebrick' for c in df_plot_final['Percent_Change']]

# Create horizontal bar chart
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(9, 5))

# Draw bar chart
df_plot_final['Percent_Change'].plot(
    kind='barh', 
    ax=ax, 
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