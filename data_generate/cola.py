import os
from datasets import load_dataset
import pandas as pd

OUTDIR = "../data"
OUTPATH = os.path.join(OUTDIR, "cola.csv")
NUM_DATA = 1024

os.makedirs(OUTDIR, exist_ok=True)

prompt1 = 'Answer 1 if sentence is acceptable. Answer 0 if sentence is not acceptable.'
prompt2 = "\nSentence: "
prompt3 = "\nAnswer 0/1: "



data = load_dataset("nyu-mll/glue", 'cola', split="train")
data = pd.DataFrame.from_records(data)
data = data.sample(NUM_DATA + NUM_DATA//10)


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + item['sentence'] + prompt3 
    prompt_texts.append(prompt)
    

data = data.drop(columns=['sentence'])
data["prompt_text"] = prompt_texts


data = data.sort_values(by='prompt_text', key = lambda col: col.apply(len))[:NUM_DATA]
data.to_csv(OUTPATH, index=False)