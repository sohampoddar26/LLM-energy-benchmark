import os
import html
import re
import pandas as pd

DATADIR = '../data/vax_raw/'
OUTDIR = "../data"
OUTPATH = os.path.join(OUTDIR, "vax.csv")
NUM_DATA = 1024

os.makedirs(OUTDIR, exist_ok=True)


def preprocess(text):
    text = html.unescape(html.unescape(text)).lower()
    text = re.sub(r"https://\S+", "HTTPURL", text)
    text = re.sub(r"\s+", " ", text)
    return text


prompt1 = 'Classify the following tweet into one of the following three vaccine stance categories: Pro-Vaccine, Anti-Vaccine or Neutral'    
prompt2 = "\nInput: "
prompt3 = "\nOutput: "


data = pd.read_csv(os.path.join(DATADIR, 'dataset/final_data.csv'))
if len(data) > NUM_DATA:
    data = data.sample(NUM_DATA + NUM_DATA//10)


prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + preprocess(item['tweet']) + prompt3
    
    prompt_texts.append(prompt)
    

data = data.drop(columns=['tweet'])
data["prompt_text"] = prompt_texts

data = data.sort_values(by='prompt_text', key = lambda col: col.apply(len))[:NUM_DATA]
data.to_csv(OUTPATH, index=False)