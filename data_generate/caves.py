import os
import html
import re
import pandas as pd

DATADIR = '../data/caves_raw/'
OUTDIR = "../data"
OUTPATH = os.path.join(OUTDIR, "caves.csv")
NUM_DATA = 1024

os.makedirs(OUTDIR, exist_ok=True)

label_desc = {
    'unnecessary': 'The tweet indicates vaccines are unnecessary, or that alternate cures are better.',
    'mandatory': 'Against mandatory vaccination — The tweet suggests that vaccines should not be made mandatory.',
    'pharma': 'Against Big Pharma — The tweet indicates that the Big Pharmaceutical companies are just trying to earn money, or the tweet is against such companies in general because of their history.',
    'conspiracy': 'Deeper Conspiracy — The tweet suggests some deeper conspiracy, and not just that the Big Pharma want to make money (e.g., vaccines are being used to track people, COVID is a hoax).',
    'political': 'Political side of vaccines — The tweet expresses concerns that the governments / politicians are pushing their own agenda though the vaccines.',
    'country': 'Country of origin — The tweet is against some vaccine because of the country where it was developed / manufactured.',
    'rushed': 'Untested / Rushed Process — The tweet expresses concerns that the vaccines have not been tested properly or that the published data is not accurate.',
    'ingredients': 'Vaccine Ingredients / technology — The tweet expresses concerns about the ingredients present in the vaccines (eg. fetal cells, chemicals) or the technology used (e.gmRNA vaccines can change your DNA)',
    'side-effect': 'Side Effects / Deaths — The tweet expresses concerns about the side effects of the vaccines, including deaths caused.',
    'ineffective': 'Vaccine is ineffective — The tweet expresses concerns that the vaccines are not effective enough and are useless.',
    'religious': 'Religious Reasons — The tweet is against vaccines because of religious reasons',
    'none': 'No specific reason stated in the tweet, or some reason other than the given ones.'
}

def preprocess(text):
    text = html.unescape(html.unescape(text)).lower()
    text = re.sub(r"https://\S+", "HTTPURL", text)
    text = re.sub(r"\s+", " ", text)
    return text


prompt1 = "Classify into one or more of these anti-vax classes: unnecessary, mandatory, pharma, conspiracy, political, country, rushed, ingredients, side-effect, ineffective, religious, none"
prompt2 = "\nInput: "
prompt3 = "\nOutputs: "


data = pd.read_csv(os.path.join(DATADIR, 'test.csv'))
if len(data) > NUM_DATA:
    data = data.sample(NUM_DATA + NUM_DATA//10)



prompt_texts = []
for i, item in data.iterrows():
    prompt = prompt1 + prompt2 + preprocess(item['text']) + prompt3
    prompt_texts.append(prompt)

data = data.drop(columns=['text'])
data["prompt_text"] = prompt_texts


data = data.sort_values(by='prompt_text', key = lambda col: col.apply(len))[:NUM_DATA]
data.to_csv(OUTPATH, index=False)