import json
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import math
from transformers import AutoTokenizer
from carbontracker import parser
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, classification_report

DATADIR = '../data/'
RESDIR = '../outputs/'
OUTDIR = '../results/energy_model_wise/'

datasets = ['boolq', 'copa', 'cola', 'sst2', 'mnli', 'record', 'squad', 'vax', 'caves']
datasets2 = ['cnndm', 'samsum']

model_map = {
    "Llama": "meta-llama",
    "Mistral": "mistralai",
    "TinyLlama": "TinyLlama",
    "flan": "google"
}

modelnames = [
    'TinyLlama-1.1B-Chat-v1.0', 'Mistral-7B-Instruct-v0.2',
    'Llama-2-7b-chat-hf', 'Llama-2-13b-chat-hf',
    'flan-t5-base', 'flan-t5-large', 'flan-t5-xl', 'flan-t5-xxl'
]

os.makedirs(OUTDIR, exist_ok=True)


for modelname in tqdm(modelnames):
    combinedout = []
    outdataall = []
    
    for datasetname in datasets + datasets2:
        BATCH_SIZE = 8 if datasetname not in datasets2 else 4
        dn = datasetname + "_" + modelname

        golddata = pd.read_csv(os.path.join(DATADIR, '%s.csv' % datasetname))
        golddata = golddata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)

        outdata = []
        modelpath = model_map[modelname[:modelname.find("-")]] + '/' + modelname
        tokenizer = AutoTokenizer.from_pretrained(modelpath)

        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

        try:
            df = pd.read_csv(os.path.join(RESDIR, dn, 'emissions.csv'))
            energyCC = {row['project_name']: row['energy_consumed'] for i, row in df.iterrows()}
            energyCCGPU = {row['project_name']: row['gpu_energy'] for i, row in df.iterrows()}

            ctdata = parser.parse_all_logs(os.path.join(RESDIR, dn, "carbon_tracker/"))
            energyCT = [x['actual']['energy (kWh)'] for x in ctdata]
            energyCTGPU = [x['components']['gpu']['avg_energy_usages (J)'][0][0] / 1000 for x in ctdata]

            with open(os.path.join(RESDIR, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)

        except Exception as e:
            print(e)
            continue

        if not (len(golddata) == len(preddata) and math.ceil(len(golddata) / BATCH_SIZE) == len(timestamps)):
            print("IO len mismatch %d and %d and %d %s" % (len(golddata), len(preddata), len(timestamps), dn))
            continue

        for i in range(math.ceil(len(golddata) / BATCH_SIZE)):
            outdata.append([datasetname + "_%d" % i])

            try:
                outdata[-1].append(round(timestamps[i][1] - timestamps[i][0], 3) * 1000)

                inputs = [len(tokenizer.encode(x, add_special_tokens=False, truncation=True)) for x in golddata.prompt_text[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]]
                avginlen = np.mean(inputs)

                outputs = [len(tokenizer.encode(x, add_special_tokens=False)) for x in preddata[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]]
                avgoutlen = np.mean([x if 'flan-t5' in modelname else x - inp for x, inp in zip(outputs, inputs)])

                outdata[-1].append(round(avginlen, 1))
                outdata[-1].append(round(avgoutlen, 1))

                outdata[-1].append(round(energyCT[i] / BATCH_SIZE * 10**6, 2))
                outdata[-1].append(round(energyCC[dn + "_%d" % i] / BATCH_SIZE * 10**6, 2))
                outdata[-1].append(round(energyCTGPU[i] / BATCH_SIZE * 10**6, 2))
                outdata[-1].append(round(energyCCGPU[dn + "_%d" % i] / BATCH_SIZE * 10**6, 2))

            except Exception as e:
                print(e)
                outdata.pop(-1)
                continue

        outdataall.extend(outdata)

        vals = np.array([x[1:] for x in outdata])
        combinedout.append([datasetname] + np.mean(vals, axis=0).tolist())

    df = pd.DataFrame(outdataall, columns=["name", "response_time(ms)", "avg_input_len", "avg_output_len", "energy_CT(mWh)", "energy_CC(mWh)", "energy_CTGPU(mWh)", "energy_CCGPU(mWh)"])
    df.to_csv(os.path.join(OUTDIR, "%s.csv" % modelname), index=False)

    df2 = pd.DataFrame(combinedout, columns=["name", "response_time(ms)", "avg_input_len", "avg_output_len", "energy_CT(Wh)", "energy_CC(Wh)", "energy_CTGPU(mWh)", "energy_CCGPU(mWh)"])
    df2.to_csv(os.path.join(OUTDIR, "%s_average.csv" % modelname), index=False)



