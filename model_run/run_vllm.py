############################################################################################################
# VLLM was not used for this paper experiments
# but the code is kept for reference
############################################################################################################

import os
import pandas as pd
from time import time
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from codecarbon import OfflineEmissionsTracker


def argparse():
    parser = ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("--out_dir", type=str, default='../outputs/')
    
    parser.add_argument("--max_gen_tokens", default=500, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--rep_penalty", default=1.0, type=float)
    parser.add_argument("--num_beams", default=1, type=int)
    
    parser.add_argument("--quantize", action='store_true')
    parser.add_argument("--finetune_path", type=str, default=None)
    
    return parser.parse_args()


############################################################################################################
# SET UP PARAMS

args = argparse()

MODEL_PATH = args.model_path
DATA_PATH = args.data_path
OUT_DIR_BASE = args.out_dir

MAXGENTOKENS = args.max_gen_tokens
TEMPERATURE = args.temperature
REPPENALTY = args.rep_penalty
QUANTIZATION = args.quantize
NUM_BEAMS = args.num_beams

FT_PATH = os.path.join(args.finetune_path, MODEL_PATH.split("/")[-1]) if args.finetune_path else None

if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    raise Exception("CUDA_VISIBLE_DEVICES not set")

TRACK_GPU = os.environ['CUDA_VISIBLE_DEVICES']

print("\n")
print("#" * 50)
print("#" * 50)

MODEL_NAME = MODEL_PATH[MODEL_PATH.rindex('/') + 1:]
DATA_NAME = DATA_PATH[DATA_PATH.rindex('/') + 1:DATA_PATH.rindex('.')]
OUT_DIR = os.path.join(OUT_DIR_BASE, '%s_%s' % (DATA_NAME, MODEL_NAME))

print(OUT_DIR)
print("#" * 50)

os.makedirs(OUT_DIR, exist_ok=True)


############################################################################################################
# LOAD DATA
rawdata = pd.read_csv(DATA_PATH)
if 'response' in rawdata.columns:
    if 'answer' not in rawdata.columns:
        rawdata['answer'] = rawdata['response']
    else:
        raise Exception("ERROR: answer and response columns already present")

data = rawdata.prompt_text

if "deepseek" in MODEL_PATH.lower():
    data = [x + "<think>\n" for x in data]
    rawdata['prompt_text'] = data

results = []

############################################################################################################
# RUN LLM

if FT_PATH:
    if not os.path.exists(os.path.join(FT_PATH, "merged_model")):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        tokenizer.padding_token = tokenizer.eos_token
        tokenizer.padding_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True)
        model.resize_token_embeddings(len(tokenizer))

        model = PeftModel.from_pretrained(model, FT_PATH)
        model = model.merge_and_unload()

        MODEL_PATH = os.path.join(FT_PATH, "merged_model")
        model.save_pretrained(MODEL_PATH)
        tokenizer.save_pretrained(MODEL_PATH)

        print("SAVED MERGED MODEL TO", MODEL_PATH)
    else:
        MODEL_PATH = os.path.join(FT_PATH, "merged_model")
        print("USING MERGED MODEL FROM", MODEL_PATH)

if QUANTIZATION:
    model = LLM(model=MODEL_PATH, gpu_memory_utilization=0.9, trust_remote_code=True, quantization="bitsandbytes", load_format="bitsandbytes")
else:
    model = LLM(model=MODEL_PATH, gpu_memory_utilization=0.9, trust_remote_code=True, max_model_len=74900)

sampling_params = SamplingParams(temperature=TEMPERATURE, top_p=0.95, max_tokens=MAXGENTOKENS, repetition_penalty=REPPENALTY, best_of=NUM_BEAMS)

t0 = time()
with OfflineEmissionsTracker(
    project_name="%s_%s" % (DATA_NAME, MODEL_NAME), country_iso_code="IND", log_level="error",
    allow_multiple_runs=True, tracking_mode="process", output_dir=OUT_DIR, measure_power_secs=15, gpu_ids=TRACK_GPU
) as tracker2:
    outputs = model.generate(data, sampling_params)

t1 = time()

print("TIME TAKEN:", round((t1 - t0) / 60, 1), "mins")

############################################################################################################
# DUMP RESULTS
rawdata['response'] = [x.outputs[0].text for x in outputs]
rawdata.to_csv(os.path.join(OUT_DIR, "output.csv"))
