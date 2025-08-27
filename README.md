# The Metric Mirage - Main Algorithm Implementation

This repository contains the implementation of our core algorithm **The Metric Mirage** for generating semantically similar but embedding-distant text through iterative embedding distance maximization and sampling-based rephrasing.

## Core Algorithm: The Metric Mirage

Our method **The Metric Mirage** combines two key components:

1. **GCG-like Embedding Distance Maximization** - Uses gradient-based controlled generation to maximize distance from original dataset embeddings while maintaining semantic relevance
2. **Sampling-based Rephrasing** - Balances preference for finetuned model with likelihood preservation to ensure generated text remains contextually appropriate

### Algorithm Overview

The algorithm takes an original model θ, unlearning dataset D_u, and iteratively:
- Generates next sentences using the original model
- Maximizes embedding distance from the original dataset via GCG optimization
- Applies sampling-based rephrasing with weighted objectives
- Filters results based on three criteria (G1, G2, G3) to ensure quality

### Key Innovation

**The Metric Mirage** addresses the challenge of creating text that appears semantically similar but is maximally distant in embedding space, effectively creating a "mirage" where text maintains contextual relevance while being maximally separated from the original dataset in the embedding manifold.

## Quick Start

### Installation
```bash
pip install -r requirements.txt
huggingface-cli login
```

### Workflow

**The Metric Mirage** requires a two-step process:

1. **Precompute Embeddings** - First, compute embeddings for the unlearning dataset
2. **Run Main Algorithm** - Then execute the main algorithm using the precomputed embeddings

### Step 1: Precompute Embeddings

Before running the main algorithm, you must precompute embeddings for your unlearning dataset:

```bash
python precompute_embed.py \
  --model-dir [path-to-model-to-be-unlearned] \
  --data [unlearning-dataset-name] \
  --tokenize sentence_improved \
  --mean-pooling \
  --batch-size 16
```

**Required Parameters:**
- `--model-dir`: Path to the model for computing embeddings
- `--data`: Name of the unlearning dataset
- `--tokenize`: Tokenization method (default: "sentence_improved")
- `--mean-pooling`: Use mean pooling for embeddings (recommended)
- `--batch-size`: Batch size for embedding computation (default: 16)

**Optional Parameters:**
- `--early-layer`: Use early layer embeddings (layer -20)
- `--max-sents`: Maximum number of sentences to process
- `--ckpt-interval`: Checkpoint interval for saving progress

### Step 2: Run The Metric Mirage Algorithm

After precomputing embeddings, run the main algorithm:

```bash
python method.py \
  --original-model [path-to-model-to-be-unlearned] \
  --finetune-model [path-to-model-finetuned-on-unlearning-dataset] \
  --mean-pooling \
  --ll-inc-percentile 25 \
  --ll-percentile 25 \
  --strategy continue \
  --only-embed-increase \
  --embed-ratio 1.5
```

### Complete Example Workflow

```bash
# Step 1: Precompute embeddings for wmdp_bio_percent1 dataset
python precompute_embed.py \
  --model-dir meta-llama/Meta-Llama-3-8B \
  --data wmdp_bio_percent1 \
  --tokenize sentence_improved \
  --mean-pooling \
  --batch-size 16

# Step 2: Run the main algorithm
python method.py \
  --original-model meta-llama/Meta-Llama-3-8B \
  --finetune-model concept-unlearning/Meta-Llama-3-8B_ft_lora_wmdp_bio_v5_ft \
  --data wmdp_bio_percent1 \
  --retain-data wikitext_train \
  --w-po 0.7 \
  --w-llo 0.3 \
  --w-l2 -1 \
  --mean-pooling \
  --tokenize sentence_improved \
  --num-needed 100 \
  --iter 10 \
  --ll-inc-percentile 25 \
  --ll-percentile 25 \
  --strategy continue \
  --only-embed-increase \
  --embed-ratio 1.5
```

## Key Files

- `method.py` - Main algorithm implementation of The Metric Mirage
- `precompute_embed.py` - **Required preprocessing script** to compute embeddings for the unlearning dataset
- `method_utils.py` - Core utility functions for multi-objective optimization
- `gcg.py` - Gradient-based Controlled Generation for embedding distance maximization
- `data_utils.py` - Data processing and dataset management
- `model_utils.py` - Model loading and embedding computation
- `utils.py` - General utility functions

## Algorithm Parameters

### Core Parameters
- `--original-model`: Path to the model to be unlearned
- `--finetune-model`: Path to the model finetuned on the unlearning dataset
- `--strategy`: Generation strategy (default: "continue")
- `--data`: Dataset name for unlearning (default: "wmdp_bio_percent1")
- `--retain-data`: Dataset name for retention (default: "wikitext_train")

### Optimization Weights
- `--w-po`: Weight for preference optimization (default: 0.7)
- `--w-llo`: Weight for original model likelihood (default: 0.3)
- `--w-l2`: Weight for L2 distance maximization (default: -1)

### Filtering Parameters
- `--ll-percentile`: Likelihood threshold percentile for filtering (default: 25)
- `--ll-inc-percentile`: Likelihood increase threshold percentile (default: 25)
- `--embed-ratio`: Threshold ratio for embedding distance filtering (default: 1.5)
- `--only-embed-increase`: Only consider sentences with increased embedding distance

### Generation Parameters
- `--iter`: Number of iterations for embedding distance maximization (default: 10)
- `--num-needed`: Target number of sentences to generate (default: 100)
- `--mean-pooling`: Use mean pooling for embeddings
- `--tokenize`: Tokenization method (default: "sentence_improved")

## Requirements

- Python 3.8+
- PyTorch with CUDA support
- Transformers library
- Access to Llama models (requires Hugging Face login)

## Citation

If you use this implementation, please cite our paper "The Metric Mirage" (forthcoming).
