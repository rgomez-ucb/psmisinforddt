# Regresion analysis related to VADER scores and the number of upvotes equivalent to a like or thumbs up("score_submission")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
df=pd.read_csv("./data/01_VADER/25_pct_merged_PoliticalDiscussion_comments_llm.csv")
X = df['llm_score']
y = df['score_submission']

X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()
print(model.summary())
# Output regression results to a text file
with open("./data/llm/llm_regression_results.txt", "w") as f:
    f.write(model.summary().as_text())
print("Regression analysis completed and results saved.")

# Visualization
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='vader_compound', y='score_submission', data=df)
plt.title('VADER Compound Score vs. Number of Upvotes')
plt.xlabel('LLM Score')
plt.ylabel('Number of Upvotes')
output_png = "./data/llm/llm_scorevs_upvotes.png" # Output as a PNG file
plt.savefig(output_png)
print("Plot saved as", output_png)
plt.show()
print("Visualization completed.")