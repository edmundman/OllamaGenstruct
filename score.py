import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Ensure CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model, and move the model to CUDA
tokenizer = AutoTokenizer.from_pretrained('OpenAssistant/reward-model-deberta-v3-large-v2')
model = AutoModelForSequenceClassification.from_pretrained('OpenAssistant/reward-model-deberta-v3-large-v2', torch_dtype=torch.bfloat16).to(device)

def extract_pair(row):
    inst = row['user']
    resp = row['assistant']
    return inst, resp

def score(inst, resp):
    with torch.no_grad():
        inputs = tokenizer(inst, resp, return_tensors='pt').to(device)
        output = model(**inputs).logits[0].cpu()
        score = float(output)
        return score

# Load the data
df = pd.read_csv('outpoot.csv')

# Iterate through each row, check for NaN or empty, score if valid, otherwise skip and set score to -2
for index, row in df.iterrows():
    if pd.isnull(row['user']) or pd.isnull(row['assistant']) or row['user'] == '' or row['assistant'] == '':
        df.at[index, 'score'] = -2
        print(f"Skipped {index + 1}/{len(df)} due to missing values")
    elif pd.isnull(row.get('score')):  # Proceed if score is not already calculated
        inst, resp = extract_pair(row)
        row_score = score(inst, resp)
        df.at[index, 'score'] = row_score
        print(f"Processed {index + 1}/{len(df)}: Score = {row_score}")

# Save the updated dataframe to a new CSV file to preserve the original
df.to_csv('outpoot_scored.csv', index=False)

print("All entries processed and scores updated.")
