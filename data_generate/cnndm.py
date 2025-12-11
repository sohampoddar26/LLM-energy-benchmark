import os
from datasets import load_dataset
import pandas as pd

OUTDIR = "../data"
OUTPATH = os.path.join(OUTDIR, "cnndm.csv")
NUM_DATA = 1024

os.makedirs(OUTDIR, exist_ok=True)

prompt1 = "Summarize the following news article in about 50 words:"
prompt2 = "\nInput: "
prompt3 = "\nOutput: "



data = load_dataset("cnn_dailymail", '3.0.0', split = "train")
data = pd.DataFrame.from_records(data)
data = data.sample(NUM_DATA + NUM_DATA // 10)


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + item['article'] + prompt3
    
    prompt_texts.append(prompt)
    

data = data.drop(columns=['article'])
data["prompt_text"] = prompt_texts


data = data.sort_values(by='prompt_text', key = lambda col: col.apply(len))[:NUM_DATA]
data.to_csv(OUTPATH, index=False)