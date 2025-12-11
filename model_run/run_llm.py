import json
import torch
import os
import sys
import gc
import pandas as pd
from tqdm import tqdm
from time import time, sleep
import torch.utils.data as data_utils
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from peft import PeftModel
from codecarbon import EmissionsTracker, OfflineEmissionsTracker
from carbontracker.tracker import CarbonTracker
from argparse import ArgumentParser
from transformers import BitsAndBytesConfig


def argparse():
    parser = ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("data_path", type=str)
    parser.add_argument("--out_dir", type=str, default='../outputs/')
    
    parser.add_argument("--finetune_path", type=str, default='')
    parser.add_argument("--assistant_model", default=None, type=str)
    
    parser.add_argument("--bs", default=1, type=int)
    parser.add_argument("--max_gen_tokens", default=10, type=int)
    
    parser.add_argument("--quantization", type=str, default='', choices=['', '4bit', '8bit'])
    parser.add_argument("--input_size", type=int, default=None)
    
    return parser.parse_args()


############################################################################################################
# SET UP PARAMS

torch.cuda.empty_cache()

args = argparse()

MODEL_PATH = args.model_path
DATA_PATH = args.data_path
OUT_DIR_BASE = args.out_dir
FT_PATH = args.finetune_path
BATCH_SIZE = args.bs
MAXGENTOKENS = args.max_gen_tokens
QUANTIZATION = args.quantization
INPUT_SIZE = args.input_size
ASSISTANT_MODEL = args.assistant_model

DEVICE = 'cuda:0'

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
# LOAD TOKENIZER AND MODEL

if QUANTIZATION == '4bit':
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
elif QUANTIZATION == '8bit':
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    bnb_config = None


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if 'flan-t5' in MODEL_PATH:
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_PATH, device_map=DEVICE, torch_dtype=torch.float16, quantization_config=bnb_config
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, device_map=DEVICE, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    assistant_model = None
    if ASSISTANT_MODEL:
        assistant_model = AutoModelForCausalLM.from_pretrained(
            ASSISTANT_MODEL, device_map=DEVICE, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        assistant_model.config.pad_token_id = tokenizer.eos_token_id

    if FT_PATH:
        model = PeftModel.from_pretrained(model, FT_PATH)
        
        lora_params = {n: p for n, p in model.named_parameters() if "lora_B" in n}
        for n, p in lora_params.items():
            print(n, p.sum())

        model = model.merge_and_unload()
        print("LOADED PEFT PARAMS")

model.eval()


############################################################################################################
# LOAD DATA
rawdata = pd.read_csv(DATA_PATH)

data_loader = data_utils.DataLoader(rawdata.prompt_text.values.tolist(), batch_size=BATCH_SIZE)

results = []
timestamps = []


############################################################################################################
# TOKENIZE DATA AND RUN LLM
def optimal_generate(inp, attn, tokenizer):
    while True:
        tmp = model.generate(
            inp, attention_mask=attn, max_new_tokens=MAXGENTOKENS,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            cache_implementation="quantized",
            repetition_penalty=1.2,
        )
        return tmp


t0 = time()

for idx, batch in enumerate(tqdm(data_loader, ncols=50)):
    try:
        if INPUT_SIZE:
            batchdata = tokenizer(batch, return_tensors="pt", padding='max_length', truncation=True, max_length=INPUT_SIZE)
        else:
            batchdata = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)

        inp = batchdata.input_ids.to(DEVICE)
        attn = batchdata.attention_mask.to(DEVICE)

        outputs = optimal_generate(inp, attn, tokenizer)

        results.extend(outputs)

        gc.collect()
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        print("*" * 100)
        print("ERROR in batch", idx, e)
        results.extend([[tokenizer.eos_token_id]] * BATCH_SIZE)
    

        
############################################################################################################
# DUMP RESULTS
print("### TIME:", round(time() - t0, 2))

responses = tokenizer.batch_decode(results, skip_special_tokens=True)
rawdata['response'] = responses
rawdata.to_csv(os.path.join(OUT_DIR, "output.csv"))

torch.cuda.empty_cache()

