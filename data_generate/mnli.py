import os
from datasets import load_dataset
import pandas as pd

OUTDIR = "../data"
OUTPATH = os.path.join(OUTDIR, "mnli.csv")
NUM_DATA = 1024

os.makedirs(OUTDIR, exist_ok=True)

prompt1 = 'Select the stance of the premise towards the hypothesis: Entailment (0), Neutral (1), Contradiction (2)'
prompt2 = "\nPremise: "
prompt3 = "\nHypothesis: "
prompt4 = "\n Answer 0/1/2: "


data = load_dataset("nyu-mll/glue", 'mnli', split="train")
data = pd.DataFrame.from_records(data)
data = data.sample(NUM_DATA + NUM_DATA//10)


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + item['premise'] + prompt3 + item['hypothesis'] + prompt4
    prompt_texts.append(prompt)
    

data = data.drop(columns=['premise', 'hypothesis'])
data["prompt_text"] = prompt_texts


data = data.sort_values(by='prompt_text', key = lambda col: col.apply(len))[:NUM_DATA]
data.to_csv(OUTPATH, index=False)