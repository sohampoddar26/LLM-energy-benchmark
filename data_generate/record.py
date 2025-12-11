import os
from datasets import load_dataset
import pandas as pd

OUTDIR = "../data"
OUTPATH = os.path.join(OUTDIR, "record.csv")
NUM_DATA = 1024

os.makedirs(OUTDIR, exist_ok=True)

prompt1 = 'Read the passage and find the entity replacing "@placeholder" inside the query'
prompt2 = "\npassage: "
prompt3 = "\nquery: "
prompt4 = "\n@placeholder: "



data = load_dataset("super_glue", 'record', split="train")
data = pd.DataFrame.from_records(data)
data = data.sample(NUM_DATA + NUM_DATA//10)


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + item['passage']\
     + prompt3 + item['query'] + prompt4
    prompt_texts.append(prompt)
    

data = data.drop(columns=['query','passage', 'entities', 'entity_spans'])
data["prompt_text"] = prompt_texts


data = data.sort_values(by='prompt_text', key = lambda col: col.apply(len))[:NUM_DATA]
data.to_csv(OUTPATH, index=False)