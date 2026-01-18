# 🎓 SLMS Training Guide

This guide walks you through the complete process of training your own Small Language Model using the SLMS framework.

---

## 📋 Prerequisites

Before starting, ensure you have:

- **Python 3.10+** installed
- **CUDA-capable GPU** (NVIDIA RTX 3090/4090 recommended for 1B parameter models)
- **Hugging Face Account** with read/write tokens
- **Weights & Biases Account** (optional, for training visualization)
- **Minimum Disk Space**: ~100GB for datasets and checkpoints

---

## 🚀 Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/g-hano/slms.git
cd slms
```

### Step 2: Install Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

**Required packages:**
- `snntorch` - Spiking Neural Network support
- `transformers` - Hugging Face Transformers library
- `datasets` - Hugging Face Datasets library
- `accelerate` - Multi-GPU training support
- `wandb` - Training metrics visualization
- `pyyaml` - Configuration file parsing

**Additional recommended packages:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install flash-attn --no-build-isolation
```

---

## ⚙️ Configuration Setup

### Step 3: Configure Your Environment

Edit `config.yaml` and add your credentials:

```yaml
# --- HUGGING FACE TOKENS ---
HF_WRITE_TOKEN: "hf_xxxxxxxxxxxxxxxxxxxx"  # For uploading datasets/models
HF_READ_TOKEN: "hf_xxxxxxxxxxxxxxxxxxxx"   # For downloading datasets

# Optional: Weights & Biases (for tracking)
WANDB_API_KEY: "your_wandb_api_key"
```

**Important**: Keep your tokens secure and never commit them to version control!

---

## 📦 Data Pipeline

The SLMS framework uses a **four-stage data pipeline**:

### Step 4: Install Raw Data

Download raw text data from Hugging Face:

```bash
python data/install_data.py
```

**What it does:**
- Downloads specified shards from **HuggingFaceFW/fineweb** (or your configured source)
- Saves to `data/fineweb/` directory
- You can customize the source in `config.yaml` under `data_pipeline.install`

**Configuration options:**

```yaml
data_pipeline:
  install:
    source_repo: "HuggingFaceFW/fineweb"
    repo_type: "dataset"
    local_dir: "data/fineweb"
    allow_patterns:
      - "data/CC-MAIN-2025-26/000_0000*"
      - "data/CC-MAIN-2025-26/000_0001*"
```

### Step 5: Prepare & Clean Data

Process the raw data (filtering, cleaning, deduplication):

```bash
python data/prepare_data.py
```

**What it does:**
- Filters out low-quality text
- Removes duplicates
- Structures data for tokenization
- Outputs to the directory specified in `config.yaml` (`prepare.output_dir`)

### Step 6: Train Your Tokenizer

Before tokenizing the dataset, you need a custom tokenizer:

```bash
python tokenizer/train_tokenizer.py
```

**What it does:**
- Trains a **Byte-Pair Encoding (BPE)** tokenizer on your corpus
- Default vocabulary size: **32,768 tokens**
- Saves to `./custom_tokenizer/`
- Optimized for English text

**Configuration:**

```yaml
data_pipeline:
  tokenizer_training:
    raw_data_path: "D:/fineweb2-train/CC-MAIN-2025-26"
    output_dir: "./custom_tokenizer"
    vocab_size: 32768
    sample_count: 1000000  # Number of samples to train on
```

### Step 7: Tokenize Data

Convert text data to token sequences:

```bash
python data/tokenize_data.py
```

**What it does:**
- Tokenizes all prepared text
- **Packs sequences** into fixed-length contexts (256, 1024, 2048)
- Creates **three phases** of training data:
  - `phase1_256/` - Short context (256 tokens)
  - `phase2_1024/` - Medium context (1024 tokens)
  - `phase3_2048/` - Long context (2048 tokens)
- Saves as efficient `.parquet` files

### Step 8: Upload Datasets to Hugging Face (Optional)

If you want to share your datasets or train from remote storage:

```bash
python data/upload_data.py
```

**What it does:**
- Uploads each phase dataset to Hugging Face Hub
- Configured in `config.yaml` under `data_pipeline.upload`

**Example configuration:**

```yaml
data_pipeline:
  upload:
    phase1_256:
      repo_id: "YOUR_USERNAME/phase1-256"
      folder_path: "D:/fineweb2-packed/phase1_256"
      private: true
```

---

## 🏋️ Model Training

### Step 9: Configure Accelerate

Set up multi-GPU or distributed training:

```bash
accelerate config
```

**Recommended settings:**
- Compute environment: `This machine`
- Distributed type: `multi-GPU` (if you have multiple GPUs) or `no`
- Mixed precision: `bf16` (for RTX 30xx/40xx GPUs)
- Number of GPUs: Enter your available GPU count

### Step 10: Configure Training Parameters

Edit `config.yaml` to set your training hyperparameters:

```yaml
# --- TRAINING CONFIG ---
project_name: "snn_fineweb_training"
output_dir: "./checkpoints"

# Model Architecture
model:
  type: "regular"  # 'regular' (Transformer) or 'spiking' (SNN)
  model_id: "YOUR_USERNAME/flow-pulse-tokenizer"  # Your tokenizer
  d_model: 2048          # Hidden dimension
  n_heads: 16            # Number of attention heads
  n_kv_heads: 8          # Grouped Query Attention heads
  num_layers: 18         # Transformer layers
  max_seq_len: 1024      # Maximum sequence length
  vocab_size: 32768      # Must match tokenizer

# Three-Phase Curriculum Learning
phases:
  phase1_256:
    repo_id: "YOUR_USERNAME/phase1-256"
    steps: 10000
    batch_size: 32
  
  phase2_1024:
    repo_id: "YOUR_USERNAME/phase2-1024"
    steps: 6000
    batch_size: 16
  
  phase3_2048:
    repo_id: "YOUR_USERNAME/phase3-2048"
    steps: 3000
    batch_size: 8

# Optimizer (Muon for 2D params, AdamW for 1D)
optimizer:
  muon_lr: 0.02          # Learning rate for Muon
  adam_lr: 0.0006        # Learning rate for AdamW
  weight_decay: 0.01
  warmup_ratio: 0.05
  max_grad_norm: 1.0
  grad_accumulation_steps: 1

# Logging
logging:
  save_steps: 1000       # Save checkpoint every N steps
  log_steps: 100         # Log metrics every N steps
  eval_steps: 1000       # Evaluate every N steps
```

### Step 11: Start Training

Launch the training script:

```bash
accelerate launch multigpu_train.py
```

**What happens:**
1. **Phase 1 (256 tokens)**: Model learns basic language patterns
2. **Phase 2 (1024 tokens)**: Model extends to medium-context reasoning
3. **Phase 3 (2048 tokens)**: Model learns long-context dependencies

**Training will automatically:**
- Load checkpoints from previous phases
- Use gradient checkpointing for memory efficiency
- Log metrics to Weights & Biases (if configured)
- Save checkpoints to `./checkpoints/`

### Step 12: Monitor Training

If you're using Weights & Biases:

```bash
# Login to W&B first
wandb login

# Your training metrics will appear at:
# https://wandb.ai/YOUR_USERNAME/snn_fineweb_training
```

**Key metrics to watch:**
- **Training Loss**: Should decrease from ~7.0 to ~4.0
- **Validation Loss**: Final target around **4.1** for decent perplexity
- **Learning Rate**: Follows cosine annealing schedule

---

## 🎯 Knowledge Distillation (Optional)

Knowledge distillation allows smaller models to learn from larger, more capable "teacher" models. This improves reasoning, accuracy, and task-specific performance without increasing model size.

### Architecture Components

**1. Distillation-Compatible Layers** (`models/transformer/layers.py`)

Both regular and spiking models support knowledge distillation through specialized activation functions:

- **`SwiGLU`**: Gated Linear Unit with Swish activation for regular transformers
- **`SpikingSwiGLU`**: Spiking version with Leaky Integrate-and-Fire (LIF) neurons that processes activations through temporal spiking dynamics
- **`ImprovedSpikingSwiGLU`**: Enhanced spiking layer with surrogate gradients and residual connections

The spiking layers use temporal coding to simulate neural spike trains, which adds biological plausibility and energy efficiency to the model.

### Step 13: Generate Teacher Outputs

Generate synthetic training data from a larger teacher model (e.g., Qwen2.5-Math-1.5B) for distillation:

```bash
python distill/generate_distillation_data.py \
  --teacher_model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
  --dataset "gsm8k" \
  --dataset_config "main" \
  --dataset_split "test" \
  --output_file "./distill_data/gsm8k_teacher_outputs.jsonl" \
  --batch_size 4 \
  --max_new_tokens 1024
```

**Available Arguments:**

- `--teacher_model`: HuggingFace model ID (default: `Qwen/Qwen2.5-Math-1.5B-Instruct`)
- `--dataset`: Dataset name from HuggingFace (e.g., `gsm8k`, `math_qa`)
- `--dataset_config`: Dataset configuration (default: `main`)
- `--dataset_split`: Dataset split to use (default: `test`)
- `--output_file`: Path to save JSONL output (default: `./distill_data/teacher_outputs.jsonl`)
- `--batch_size`: Inference batch size (default: `4`, reduce if OOM)
- `--max_new_tokens`: Max response tokens (default: `1024`)
- `--system_prompt`: Custom system prompt (default: `"You are a helpful assistant."`)
- `--max_samples`: Limit number of samples (default: `None` for all)

**Example with custom settings:**

```bash
# Generate data for only 100 samples with larger batch size
python distill/generate_distillation_data.py \
  --teacher_model "Qwen/Qwen2.5-Math-1.5B-Instruct" \
  --dataset "gsm8k" \
  --output_file "./distill_data/gsm8k_small.jsonl" \
  --batch_size 8 \
  --max_samples 100
```

**What it does:**
- Loads the teacher model and dataset
- Generates step-by-step reasoning for each problem
- Saves responses in JSONL format with questions, ground truth, and teacher outputs
- Displays progress with detailed logging

### Step 14: Fine-tune with Distillation

Train your student model on the generated teacher outputs using supervised fine-tuning:

```bash
accelerate launch distill/distil.py \
  --cache_file "./distill_data/gsm8k_teacher_outputs.jsonl" \
  --student_checkpoint "./checkpoints/phase3_final/pytorch_model.bin" \
  --tokenizer_id "YOUR_USERNAME/flow-pulse-tokenizer" \
  --output_dir "./checkpoints_distilled" \
  --model_type "regular" \
  --max_length 1024 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --epochs 3
```

**Available Arguments:**

- `--cache_file`: Path to teacher outputs JSONL file (default: `./teacher_outputs.jsonl`)
- `--student_checkpoint`: Path to pretrained student model weights (default: `./model_weights/best_phase2_1024/pytorch_model.bin`)
- `--tokenizer_id`: HuggingFace tokenizer ID (default: `Chan-Y/flow-pulse-tokenizer`)
- `--output_dir`: Directory to save trained models (default: `./flow_trained_model`)
- `--model_type`: Model architecture type: `"regular"` or `"spiking"` (default: `spiking`)
- `--max_length`: Maximum sequence length (default: `1024`)
- `--batch_size`: Training batch size per device (default: `2`)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: `4`)
- `--learning_rate`: Learning rate (default: `5e-5`)
- `--epochs`: Number of training epochs (default: `3`)
- `--vocab_size`: Vocabulary size (default: `32768`)
- `--d_model`: Model hidden dimension (default: `2048`)
- `--n_heads`: Number of attention heads (default: `16`)
- `--n_kv_heads`: Number of key-value heads for GQA (default: `8`)
- `--num_layers`: Number of transformer layers (default: `18`)

**What it does:**
- Loads your pretrained student model (Flow-1B or Pulse-1B)
- Fine-tunes on teacher-generated reasoning traces
- Uses supervised fine-tuning with cross-entropy loss
- Saves best and final checkpoints based on training loss
- Supports both regular Transformer and Spiking Neural Network architectures

**Training Tips:**
- Use `--model_type "spiking"` for Pulse-1B (SNN) models
- Use `--model_type "regular"` for Flow-1B (standard Transformer) models
- Reduce `--batch_size` if you encounter OOM errors
- Increase `--gradient_accumulation_steps` to maintain effective batch size

---

## 💾 Saving & Sharing Your Model

### Step 15: Upload to Hugging Face

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your trained model
model = AutoModelForCausalLM.from_pretrained('./checkpoints/phase3_final')
tokenizer = AutoTokenizer.from_pretrained('./custom_tokenizer')

# Upload to HF Hub
model.push_to_hub('YOUR_USERNAME/Flow-1B-custom')
tokenizer.push_to_hub('YOUR_USERNAME/Flow-1B-custom')
"
```

### Step 16: Convert to CoreML (For iOS/Edge Deployment)

Convert your trained model to CoreML format for deployment on Apple devices (iPhone, iPad, Mac):

**For Flow-1B (Regular Transformer):**
```bash
python convert2coreml.py \
  --model_path "./checkpoints_hf_streaming/phase3_final/pytorch_model.bin" \
  --model_type regular \
  --output flow_1b
```

**For Pulse-1B (Spiking Neural Network):**
```bash
python convert2coreml.py \
  --model_path "./checkpoints_hf_streaming/phase3_final/pytorch_model.bin" \
  --model_type spiking \
  --output pulse_1b \
  --num_steps 3
```

**Available Arguments:**
- `--model_path`: Path to PyTorch model weights (`.bin` or `.safetensors`)
- `--model_type`: Model architecture (`regular` or `spiking`)
- `--output`: Output filename (saves as `.mlpackage`)
- `--ios_version`: iOS deployment target (`iOS15`, `iOS16`, `iOS17`, `iOS18`) - default: `iOS18`
- `--compute_units`: Compute units (`ALL`, `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE`) - default: `ALL`
- `--max_seq_len`: Maximum sequence length (default: `2048`)
- `--num_steps`: Spiking timesteps for SNN models (default: `3`)

The generated `.mlpackage` file can be dragged directly into your Xcode project for on-device inference.

---

## 🐛 Troubleshooting

### Out of Memory (OOM) Errors

**Solution 1**: Reduce batch size in `config.yaml`:
```yaml
phases:
  phase1_256:
    batch_size: 16  # Reduce from 32
```

**Solution 2**: Increase gradient accumulation:
```yaml
optimizer:
  grad_accumulation_steps: 2  # Effective batch size = batch_size * 2
```

**Solution 3**: Enable gradient checkpointing (already enabled by default)

### Slow Training Speed

- **Use mixed precision**: Ensure `accelerate config` has `bf16` enabled
- **Increase `num_workers`**: In `config.yaml`, set `streaming.num_workers: 8`
- **Use faster storage**: Move datasets to SSD if on HDD

### Token Mismatch Errors

Make sure `model.vocab_size` in `config.yaml` matches your tokenizer's vocabulary size:

```bash
python -c "from transformers import AutoTokenizer; print(AutoTokenizer.from_pretrained('./custom_tokenizer').vocab_size)"
```

---

## 📊 Expected Results

After following this guide, you should achieve:

- ✅ **Perplexity**: ~4.1 on validation set
- ✅ **Model Size**: ~1B parameters
- ✅ **Training Time**: ~48-72 hours on RTX 4090 (all phases)
- ✅ **Total Cost**: ~$50-100 on vast.ai spot instances

---

## 🎉 Next Steps

Once training is complete:

1. **Evaluate** your model on benchmarks (GSM8K, MMLU, etc.)
2. **Fine-tune** for specific tasks or domains
3. **Deploy** to edge devices using CoreML conversion
4. **Share** your model and datasets with the community!

For questions or issues, please open an issue on GitHub.

Happy training! 🚀

---

## 👨‍💻 Author

This training guide was created by **Cihan Yalçın (Chan-Y)**.

**Connect with me:**
- 💼 LinkedIn: [linkedin.com/in/chanyalcin](https://www.linkedin.com/in/chanyalcin/)
- 🐙 GitHub: [github.com/g-hano](https://github.com/g-hano)
- 🤗 Hugging Face: [huggingface.co/Chan-Y](https://huggingface.co/Chan-Y)

