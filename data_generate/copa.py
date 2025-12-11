import os
from datasets import load_dataset
import pandas as pd

OUTDIR = "../data"
OUTPATH = os.path.join(OUTDIR, "copa.csv")
NUM_DATA = 1024

os.makedirs(OUTDIR, exist_ok=True)

prompt1 = 'Select choice1 as 0 or choice2 as 1 as an answer to the question posed in the context of the given premise.'
prompt2 = "\npremise: "
prompt3 = "\nchoice1: "
prompt4 = "\nchoice2: "
prompt5 = "\nquestion: "
prompt6 = "\nAnswer 0/1: "



datasets = []
for split in ["train", "validation"]:
    data = load_dataset("super_glue", "copa", split=split)
    datasets.append(pd.DataFrame.from_records(data))

data = pd.concat(datasets)
print(f"Total samples: {len(data)}")


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + item['premise']\
     + prompt3 + item['choice1'] + prompt4 + \
     item['choice2'] + prompt5 +  item['question'] + prompt6
    prompt_texts.append(prompt)
    

data = data.drop(columns=['question','premise','choice1','choice2'])
data["prompt_text"] = prompt_texts


data = data.sort_values(by='prompt_text', key = lambda col: col.apply(len))[:NUM_DATA]
data.to_csv(OUTPATH, index=False)