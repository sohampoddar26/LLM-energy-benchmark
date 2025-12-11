import json
import pandas as pd
import numpy as np
import os
import math
from transformers import AutoTokenizer
from carbontracker import parser
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, classification_report

DATADIR = '../data/'
RESDIR = '../outputs/'
OUTDIR = '../results/energy_dataset_wise/'

datasets = ['boolq', 'copa', 'cola', 'mnli', 'sst2', 'cnndm', 'samsum', 'caves', 'vax', 'squad']

model_map = {
    "Llama": "meta-llama",
    "Meta": "meta-llama",
    "Phi": "microsoft",
    "Mistral": "mistralai",
    "TinyLlama": "TinyLlama",
    "flan": "google",
    "gpt": "EleutherAI"
}

os.makedirs(OUTDIR, exist_ok=True)

for datasetname in datasets:
    print('*' * 50, '\n\n')
    print(datasetname)
    print('*' * 50, '\n\n')
    
    golddata = pd.read_csv(os.path.join(DATADIR, '%s.csv' % datasetname)).prompt_text
    golddata = golddata.sort_values(key=lambda col: col.apply(len), ascending=False)

    combinedout = []
    outdataall = []

    for dn in sorted([dn for dn in next(os.walk(RESDIR))[1] if datasetname in dn]):
        modelname = dn[dn.rfind('_') + 1:]
        
        print(modelname)
        modelpath = model_map[modelname[:modelname.find("-")]] + '/' + modelname
        tokenizer = AutoTokenizer.from_pretrained(modelpath, trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        try:
            df = pd.read_csv(os.path.join(RESDIR, dn, 'emissions.csv'))
            timeCC = {row['project_name']: row['duration'] for i, row in df.iterrows()}
            energyCC = {row['project_name']: row['energy_consumed'] for i, row in df.iterrows()}
            energyGPU = {row['project_name']: row['gpu_energy'] for i, row in df.iterrows()}

            with open(os.path.join(RESDIR, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)

        except Exception as e:
            print("SKIPPING:", e)
            continue

        if not (len(golddata) == len(preddata)):
            print("IO len mismatch %d and %d and %s" % (len(golddata), len(preddata), dn))
            continue

        inputs = [len(tokenizer.encode(mi, add_special_tokens=False, truncation=True)) for mi, mo in zip(golddata, preddata)]
        avginlen = np.mean(inputs)
        outputs = [len(tokenizer.encode(mo, add_special_tokens=False)) for mo in preddata]
        avgoutlen = np.mean([x if 'flan-t5' in modelname else x - inp for x, inp in zip(outputs, inputs)])

        outdata = [
            modelname,
            round(timeCC[dn] / len(preddata) * 1000),
            round(avginlen, 1),
            round(avgoutlen, 1),
            round(energyCC[dn] / len(preddata) * 1e6, 2),
            round(energyGPU[dn] / len(preddata) * 1e6, 2)
        ]
        combinedout.append(outdata)

    df2 = pd.DataFrame(combinedout, columns=["name", "avg_response_time(ms)", "avg_input_len", "avg_output_len", "energy_CC(mWh)", "energy_GPU(mWh)"])
    df2.to_csv(os.path.join(OUTDIR, "%s_average.csv" % datasetname), index=False)



