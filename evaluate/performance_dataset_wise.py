import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import pandas as pd
import numpy as np
import ast
import math
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, classification_report, accuracy_score
import evaluate
import sys

datasets = ['boolq', 'copa', 'cola', 'sst2', 'mnli', 'caves', 'vax', 'squad', 'cnndm', 'samsum']
datanames = datasets

INDIR = '../data/'
PREDDIR = '../outputs/'
OUTDIR = '../results/performance/'
CHANGEPATH = None

PRINT = False

print(INDIR)
print(PREDDIR)
print(OUTDIR)

#########################################################################################
#########################################################################################
def eval_cnndm(datadir, preddir, outdir, name='cnndm'):
    metrics = evaluate.load('rouge')

    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    gold = rawdata['highlights'].apply(str.lower).values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        if 'gpt-neo' in modelname:
            print("skipping", modelname)
            continue

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)

            if 'flan' not in modelname:
                preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
            preddata = map(str.lower, preddata)
            pred = list(preddata)

        except Exception as args:
            print("Skipping", dn, args)
            continue

        results = metrics.compute(predictions=pred, references=gold)

        outdata.append([modelname])
        outdata[-1].append(round(results['rouge1'] * 100, 1))
        outdata[-1].append(round(results['rouge2'] * 100, 1))
        outdata[-1].append(round(results['rougeL'] * 100, 1))

        avg = (results['rouge1'] + results['rouge2'] + results['rougeL']) * 100 / 3
        outdata[-1].append(round(avg, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "ROUGE1", "ROUGE2", "ROUGEL", "avgROUGE"])
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2



#########################################################################################
#########################################################################################
def eval_samsum(datadir, preddir, outdir, name="samsum"):
    metrics = evaluate.load('rouge')

    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    gold = rawdata['summary'].apply(str.lower).values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)

            if 'flan' not in modelname:
                preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
            preddata = map(str.lower, preddata)
            pred = list(preddata)

        except Exception as args:
            print("Skipping", dn, args)
            continue

        results = metrics.compute(predictions=pred, references=gold)

        outdata.append([modelname])
        outdata[-1].append(round(results['rouge1'] * 100, 1))
        outdata[-1].append(round(results['rouge2'] * 100, 1))
        outdata[-1].append(round(results['rougeL'] * 100, 1))

        avg = (results['rouge1'] + results['rouge2'] + results['rougeL']) * 100 / 3
        outdata[-1].append(round(avg, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "ROUGE1", "ROUGE2", "ROUGEL", "avgROUGE"])
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2



#########################################################################################
#########################################################################################
def eval_qnli(datadir, preddir, outdir, name=None):
    rawdata = pd.read_csv(os.path.join(datadir, "qnli.csv"))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    gold = rawdata['label'].values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'qnli' in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata)
                pred = [1 if "1" in pr or "not entail" in pr else 0 for pr in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "qnli.csv"), index=False)
    return df2



#########################################################################################
#########################################################################################
def eval_wnli(datadir, preddir, outdir):
    rawdata = pd.read_csv(os.path.join(datadir, "wnli.csv"))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    gold = rawdata['label'].values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'wnli' in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata)
                pred = [0 if "0" in pr or "not entail" in pr else 1 for pr in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "wnli.csv"), index=False)
    return df2




#########################################################################################
#########################################################################################
def eval_mnli(datadir, preddir, outdir, name='mnli'):
    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    gold = rawdata['label'].values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata)
                pred = [1 if "1" in pr or "neutral" in pr else 2 if "2" in pr or 'contradict' in pr else 0 for pr in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2



#########################################################################################
#########################################################################################
def eval_cola(datadir, preddir, outdir, name="cola"):
    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    gold = rawdata['label'].values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata)
                pred = [0 if "0" in pr or "not accept" in pr else 1 for pr in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2



#########################################################################################
#########################################################################################
def eval_sst2(datadir, preddir, outdir, name="sst2"):
    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    gold = rawdata['label'].values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata)
                pred = [1 if "1" in pr or "positive" in pr else 0 for pr in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        outdata[-1].append(round(accuracy_score(gold, pred) * 100, 1))
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "Acc", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2
    return df2





#########################################################################################
#########################################################################################
def eval_boolq(datadir, preddir, outdir, name='boolq'):
    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    gold = rawdata['label'].values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata)
                pred = [1 if "true" in pr or "yes" in pr else 0 for pr in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2



#########################################################################################
#########################################################################################
def eval_copa(datadir, preddir, outdir, name=None):
    rawdata = pd.read_csv(os.path.join(datadir, "copa.csv"))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    gold = rawdata['label'].values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if 'copa' in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata)
                pred = [0 if "0" in pr or "choice1" in pr else 1 for pr in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "copa.csv"), index=False)
    return df2



#########################################################################################
#########################################################################################
def eval_ag_news(datadir, preddir, outdir, name='ag-news'):
    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    gold = rawdata['label'].values.tolist()

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata)
                pred = [0 if ("0" in pr or "world" in pr) else 1 if ("1" in pr or "sports" in pr) else 2 if ("2" in pr or "business" in pr) else 3 for pr in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2


#########################################################################################
#########################################################################################
def eval_hatex(datadir, preddir, outdir, name='hate'):
    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    golddata = rawdata['label'].values.tolist()
    gold = [0 if ("normal" in pr) else 1 if ("hatespeech" in pr) else 2 for pr in golddata]

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata)
                pred = [0 if ("normal" in pr) else 1 if ("hatespeech" in pr) else 2 for pr in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2



#########################################################################################
#########################################################################################
def eval_vax(datadir, preddir, outdir, name="vax"):
    gold_label_list = ['ProVax', 'AntiVax', 'Neutral']
    pred_label_list = ['pro-vaccine', 'anti-vaccine', 'neutral']

    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    golddata = rawdata['label'].values.tolist()
    golddata = ['All' if isinstance(x, float) and math.isnan(x) else x for x in golddata]
    gold = [[int(lab in row) for lab in gold_label_list] for row in golddata]
    gold2 = [[int(lab in row) for lab in gold_label_list] for row in golddata]

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = map(str.lower, preddata)
                pred = [[int(lab in row) for lab in pred_label_list] for row in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])

        if len(pred) < len(gold):
            gold2 = gold
            gold = gold[:len(pred)]

        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

        gold = gold2[:]

    df2 = pd.DataFrame(outdata, columns=["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2


#########################################################################################
#########################################################################################
def eval_caves(datadir, preddir, outdir, name='caves'):
    labellist = ["unnecessary", "mandatory", "pharma", "conspiracy", "political", "country", "rushed", "ingredients", "side-effect", "ineffective", "religious", "none"]

    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    golddata = rawdata['labels'].values.tolist()
    gold = [[int(lab in row) for lab in labellist] for row in golddata]

    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, timestamps = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = list(map(str.lower, preddata))
                pred = [[int(lab in row) for lab in labellist] for row in preddata]

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])

    df2 = pd.DataFrame(outdata, columns=["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2


        except Exception as args:
            print("Skipping", dn, args)
            continue         
            
        outdata.append([modelname])
        outdata[-1].append(round(precision_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(recall_score(gold, pred, average="macro") * 100, 1))
        outdata[-1].append(round(f1_score(gold, pred, average="macro") * 100, 1))

        print(outdata[-1])
        
    df2 = pd.DataFrame(outdata, columns = ["Model", "M-Pre", "M-Rec", "M-F1"])
    df2.to_csv(os.path.join(outdir, "%s.csv"%name), index = False)
    return df2



#########################################################################################
#########################################################################################
def eval_squad(datadir, preddir, outdir, name='squad'):
    squad_v2_metric = evaluate.load("squad_v2")
    rawdata = pd.read_csv(os.path.join(datadir, "%s.csv" % name))
    rawdata = rawdata.sort_values('prompt_text', key=lambda col: col.apply(len), ascending=False)
    answers = rawdata['answers'].values.tolist()
    index = rawdata['id'].values.tolist()

    answers = list(map(ast.literal_eval, answers))
    outdata = []

    for dn in sorted([dn for dn in next(os.walk(preddir))[1] if name in dn]):
        modelname = dn[dn.rfind('_') + 1:]

        try:
            with open(os.path.join(preddir, dn, "output.json")) as fp:
                preddata, _ = json.load(fp)
                if 'flan' not in modelname:
                    preddata = [pr[len(raw):] for pr, raw in zip(preddata, rawdata.prompt_text)]
                preddata = list(map(str.lower, preddata))

                predictions = []
                references = []
                for id, ans, pr in zip(index, answers, preddata):
                    references.append({'answers': ans, 'id': id})
                    predictions.append({'prediction_text': pr, 'id': id, 'no_answer_probability': 0.})

        except Exception as args:
            print("Skipping", dn, args)
            continue

        outdata.append([modelname])
        metrics = squad_v2_metric.compute(predictions=predictions, references=references)
        for key in ['HasAns_f1', 'NoAns_f1', 'f1']:
            outdata[-1].append(round(metrics[key], 2))

        print(outdata[-1])

    columns = ['Model', 'HasAns_f1', 'NoAns_f1', 'f1']

    df2 = pd.DataFrame(outdata, columns=columns)
    df2.to_csv(os.path.join(outdir, "%s.csv" % name), index=False)
    return df2


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalized_accuracy(scores, outdir, type=None):
    total = None
    for data, res in scores.items():
        if type == 'zscore':
            scores[data] = sigmoid((res - res.mean()) / res.std())
        if type == 'minmax':
            scores[data] = (res - res.min()) / (res.max() - res.min())

        if total is None:
            total = scores[data].copy()
        else:
            total += scores[data]

    total *= 100 / len(scores)
    total = total.round(2)

    total = total.to_frame()
    for data, res in scores.items():
        total[data] = (100 * res).round(2)

    total.to_csv(os.path.join(OUTDIR, "NA.csv"))
    print_latex(total)


set_of_models = [
    'flan-t5-base', 'flan-t5-large', 'flan-t5-xl', 'flan-t5-xxl',
    'TinyLlama-1.1B-Chat-v1.0', 'Phi-3-mini-4k-instruct',
    'Mistral-7B-Instruct-v0.2', 'Llama-2-7b-chat-hf', 'Meta-Llama-3-8B-Instruct', 'Llama-2-13b-chat-hf'
]

model_map = {
    'TinyLlama-1.1B-Chat-v1.0': 'TinyLlama-1.1B',
    'Mistral-7B-Instruct-v0.2': 'Mistral-7B',
    'Llama-2-7b-chat-hf': 'Llama-2-7b',
    'Llama-2-13b-chat-hf': 'Llama-2-13b',
    'Meta-Llama-3-8B-Instruct': 'Llama-3-8B',
    'Phi-3-mini-4k-instruct': 'Phi-3-mini'
}


def print_latex(final):
    print(*[final.loc[model] for model in set_of_models], sep=" & ", end="\\\\ \n")


if __name__ == '__main__':
    os.makedirs(OUTDIR, exist_ok=True)

    scores = {}

    if PRINT:
        df = pd.read_csv(os.path.join(OUTDIR, 'NA.csv')).set_index('Model')
        print_latex(df)
        sys.exit()

    for data, name in zip(datasets, datanames):
        print("\n\n" + "#" * 50, data, name)
        func = eval("eval_" + data)
        res = func(INDIR, PREDDIR, OUTDIR, name)
        res = res[res.Model != 'Meta-Llama-3-8B']
        res = res.set_index("Model")[res.columns[-1]]

        scores[data] = res

        print(data, end=' & ')
        print_latex(res)
        print()

    if not CHANGEPATH:
        normalized_accuracy(scores, OUTDIR, 'zscore')
    else:
        diff = {}
        for data, name in zip(datasets, datanames):
            df = pd.read_csv(CHANGEPATH + name + '.csv')
            df = df[df.Model != 'Meta-Llama-3-8B']
            df = df.set_index("Model")[df.columns[-1]]

            diff[data] = (scores[data] - df) / df

        normalized_accuracy(diff, OUTDIR)









    

