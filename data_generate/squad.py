import os
from datasets import load_dataset
import pandas as pd

OUTDIR = "../data"
OUTPATH = os.path.join(OUTDIR, "squad.csv")
NUM_DATA = 1024

os.makedirs(OUTDIR, exist_ok=True)

data = load_dataset("squad_v2", split="validation")
data = pd.DataFrame.from_records(data)
data = data.sample(NUM_DATA + NUM_DATA // 10)


prompt_texts = []
for i, item in data.iterrows():
    prompt = item['context'] + "\n" + item['question']
    prompt += "\n### Response: "
    prompt_texts.append(prompt)
    

data = data.drop(columns=['question','context', 'title'])
data["prompt_text"] = prompt_texts


data = data.sort_values(by='prompt_text', key = lambda col: col.apply(len))[:NUM_DATA]
data.to_csv(OUTPATH, index=False)