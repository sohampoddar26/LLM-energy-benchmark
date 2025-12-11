# LLM Energy Benchmark

Codes used for the paper **"Towards Sustainable NLP: Insights from Benchmarking Inference Energy in Large Language Models"**, accepted at NAACL 2025.

This repository provides tools for benchmarking energy consumption during LLM inference across various NLP tasks. The codebase enables:

- **Data Generation**: Create prompt datasets from standard NLP benchmarks (SuperGLUE, GLUE, CNN/DailyMail, SamSum, SQuAD, etc.)
- **Model Inference**: Run LLM inference with energy tracking using CodeCarbon and CarbonTracker
- **Evaluation**: Compute both task performance metrics (accuracy, F1, ROUGE) and energy efficiency metrics


## Installation

```bash
pip install -r requirements.txt
```

### Energy Tracking Setup

For accurate energy measurements, additional setup is required:
- Ensure `nvidia-smi` is accessible in your PATH for tracking GPU power.
- Permissions to Intel RAPL are needed for CPU power monitoring. See the [CodeCarbon RAPL documentation](https://mlco2.github.io/codecarbon/rapl.html) for details.

## Usage

### 1. Generate Datasets

Generate prompt datasets from HuggingFace datasets:

```bash
cd data_generate
python boolq.py      # Generate BoolQ dataset
python cnndm.py      # Generate CNN/DailyMail dataset
# ... run other dataset scripts as needed
```

This creates CSV files in the `data/` directory with prompts and ground truth labels.

### 2. Run LLM Inference

Run inference with energy tracking:

```bash
cd model_run

# Single run example
python run_llm.py <model_path> <data_path> [options]
```

**Command-line arguments:**
- `model_path`: HuggingFace model path (e.g., `google/flan-t5-large`)
- `data_path`: Path to input CSV file with prompts
- `--out_dir`: Output directory (default: `../outputs/`)
- `--bs`: Batch size (default: 1)
- `--max_gen_tokens`: Maximum tokens to generate (default: 10)
- `--quantization`: Quantization mode (`4bit`, `8bit`, or empty for none)
- `--assistant_model`: Path to assistant model for speculative decoding
- `--finetune_path`: Path to LoRA/PEFT adapter weights

**Run batch experiments:**
```bash
bash run-all_exp.sh
```

### 3. Evaluate Results

Compute performance and energy metrics:

```bash
cd evaluate

# Task performance (accuracy, F1, ROUGE, etc.)
python performance_dataset_wise.py

# Energy consumption per dataset
python energy_dataset_wise.py

# Energy consumption per model
python energy_model_wise.py
```

Results are saved to the `results/` directory.

## Output Format

Each inference run creates a directory in `outputs/` containing:
- `output.json`: Model predictions and timestamps
- `emissions.csv`: Energy consumption data from CodeCarbon
- `carbontracker/`: Detailed energy logs from CarbonTracker

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{poddar2025towards,
  title={Towards sustainable nlp: Insights from benchmarking inference energy in large language models},
  author={Poddar, Soham and Koley, Paramita and Misra, Janardan and Ganguly, Niloy and Ghosh, Saptarshi},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics},
  year={2025}
}
```

