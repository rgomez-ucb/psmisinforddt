import os
import pandas as pd
import ollama
from tqdm import tqdm

# -------------------------------
# 文件路径
input_csv = "./data/01_VADER/25_pct_merged_PoliticalDiscussion_comments_vader.csv"
checkpoint_csv = "./data/llm/llm_misinfo_narrative_checkpoint.csv"
output_csv = "./data/llm/PoliticalDiscussion_with_llm_labels.csv"
# -------------------------------

# 读取数据
df = pd.read_csv(input_csv)
client = ollama.Client(host='http://127.0.0.1:11434')

# ---- 尝试加载 checkpoint ----
start_index = 0
if os.path.exists(checkpoint_csv):
    checkpoint_df = pd.read_csv(checkpoint_csv)
    if len(checkpoint_df) <= len(df):
        df.loc[:len(checkpoint_df)-1, ["misinfo_label", "narrative_score"]] = checkpoint_df[["misinfo_label", "narrative_score"]]
        start_index = len(checkpoint_df)
        print(f"Checkpoint loaded. Resuming from row {start_index}.")
    else:
        print(" Ignoring checkpoint.")
else:
    print("Starting from row 0.")

# ---- LLM 调用函数 ----
def qwen_label(text):
    prompt = f"""
You are an expert annotator for political discussion content.

TASKS:
1. Determine if this text contains misinformation.
   Output MISINFO: 0 or 1

2. Evaluate how narrative/opinion/emotional the text is.
   Output NARRATIVE_SCORE: a number between 0 and 1

STRICT OUTPUT FORMAT:
MISINFO: <0 or 1>
NARRATIVE_SCORE: <float>

Text:
{text}
"""
    response = client.chat(
        model="llama3.1",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0,
            "max_tokens": 50,
            "top_p": 0.9,
        },
        stream=False
    )

    content = response["message"]["content"]

    # ---- Parse ----
    misinfo = None
    narrative_score = None
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("MISINFO"):
            try:
                misinfo = int(line.split(":")[1].strip())
            except:
                misinfo = None
        elif line.startswith("NARRATIVE_SCORE"):
            try:
                narrative_score = float(line.split(":")[1].strip())
            except:
                narrative_score = None

    return pd.Series([misinfo, narrative_score])


if "misinfo_label" not in df.columns:
    df["misinfo_label"] = None
if "narrative_score" not in df.columns:
    df["narrative_score"] = None


for i, text in tqdm(df["body"].items()):
    if i < start_index:
        continue  # Skip rows already processed in checkpoint

    if i % 50 == 0:
        print(f"Processing row {i}/{len(df)}...")

    misinfo, narrative = qwen_label(text)
    df.at[i, "misinfo_label"] = misinfo
    df.at[i, "narrative_score"] = narrative

    # saving checkpoint every 1000 rows
    if i > 0 and i % 1000 == 0:
        df.loc[:i, ["misinfo_label", "narrative_score"]].to_csv(checkpoint_csv, index=False)
        print(f"Checkpoint saved at row {i}.")


df.to_csv(output_csv, index=False)
print("\n DONE. Saved full dataset to:", output_csv)
