#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# Example models: google/flan-t5-base google/flan-t5-large google/flan-t5-xl google/flan-t5-xxl
#                 TinyLlama/TinyLlama-1.1B-Chat-v1.0 mistralai/Mistral-7B-Instruct-v0.2
#                 meta-llama/Llama-2-7b-chat-hf meta-llama/Llama-2-13b-chat-hf
#                 meta-llama/Meta-Llama-3-8B-Instruct microsoft/Phi-3-mini-4k-instruct

for llm in google/flan-t5-large meta-llama/Meta-Llama-3-8B-Instruct;
do
    # Discriminative tasks (short output)
    for bs in 8;
    do
        for dataset in boolq copa cola mnli sst2;
        do
            python run_llm.py $llm ../data/$dataset.csv --out_dir ../outputs/ --max_gen_tokens 25 --bs $bs
        done
    done

    # Generative tasks (longer output)
    for bs in 4;
    do
        for dataset in cnndm samsum;
        do
            python run_llm.py $llm ../data/$dataset.csv --out_dir ../outputs/ --max_gen_tokens 500 --bs $bs
        done
    done
done






