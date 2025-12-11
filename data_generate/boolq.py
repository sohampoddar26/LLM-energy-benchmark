import os
from datasets import load_dataset
import pandas as pd

OUTDIR = "../data"
OUTPATH = os.path.join(OUTDIR, "boolq.csv")
NUM_DATA = 1024

os.makedirs(OUTDIR, exist_ok=True)

prompt1 = 'Read the passage and answer the question with True or False.'
prompt2 = "\nquestion: "
prompt3 = "\npassage: "
prompt4 = "\nAnswer: "



data = load_dataset("super_glue", 'boolq', split="train")
data = pd.DataFrame.from_records(data)
data = data.sample(NUM_DATA + NUM_DATA // 10)


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + item['question']\
     + prompt3 + item['passage'] + prompt4
    prompt_texts.append(prompt)
    

data = data.drop(columns=['question','passage'])
data["prompt_text"] = prompt_texts


data = data.sort_values(by='prompt_text', key = lambda col: col.apply(len))[:NUM_DATA]
data.to_csv(OUTPATH, index=False)
